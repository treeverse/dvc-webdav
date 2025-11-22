import logging
import shlex
import threading
import sys
import httpx

import subprocess
from typing import List, Optional, Union, Protocol

logger = logging.getLogger("dvc")


def _log_with_thread(level: int, msg: str, *args) -> None:
    """
    Universal helper to inject thread identity into logs.
    Output format: [Thread-Name] [BearerAuthClient] Message...
    """
    if logger.isEnabledFor(level):
        thread_name = threading.current_thread().name
        logger.log(level, f"[{thread_name}] {msg}", *args)


def execute_command(command: Union[List[str], str], timeout: int = 10) -> str:
    """Executes a command to retrieve the token."""
    if isinstance(command, str):
        command = shlex.split(command)

    try:
        result = subprocess.run(
            command,
            shell=False,
            capture_output=True,
            text=True,
            check=True,
            timeout=timeout,
            encoding="utf-8",
        )
        token = result.stdout.strip()
        if not token:
            raise ValueError("Command executed successfully but returned an empty token.")
        return token

    except (FileNotFoundError, subprocess.TimeoutExpired, subprocess.CalledProcessError, ValueError, Exception) as e:
        error_header = "\n" + "=" * 60
        error_msg = f"{error_header}\n[CRITICAL] Bearer Token Retrieval Failed.\n" \
                    f"DVC may misinterpret this as 'File Not Found' and skip files.\n" \
                    f"Command: {command}\n" \
                    f"Error: {e}"

        if isinstance(e, subprocess.CalledProcessError):
            error_msg += f"\nStderr: {e.stderr.strip()}"

        error_msg += f"\n{error_header}\n"

        logger.critical(error_msg)
        sys.stderr.write(error_msg)
        sys.stderr.flush()

        # Re-raise the exception so the caller knows it failed.
        # DVC might catch this and swallow it, but we've done our duty to notify.
        raise e


class TokenSaver(Protocol):
    """Protocol defining the token persistence interface"""

    def __call__(self, token: Optional[str]) -> None: ...


def safe_callback(cb: Optional[TokenSaver], value: Optional[str], operation: str) -> None:
    """Safely execute callback function with error handling"""
    if not cb:
        return

    try:
        cb(value)
    except Exception as e:
        _log_with_thread(logging.WARNING, "[BearerAuthClient] Failed to %s token: %s", operation, e)


class BearerAuthClient(httpx.Client):
    """HTTPX client that adds Bearer token authentication using a command.

    Args:
        bearer_token_command: The command to run to get the Bearer token.
        save_token_cb: Optional callback to persist the token.
        token: Optional initial token to use.
        **kwargs: Additional arguments to pass to the httpx.Client constructor.
    """

    def __init__(
            self,
            bearer_token_command: str,
            save_token_cb: Optional[TokenSaver] = None,
            **kwargs,
    ):
        super().__init__(**kwargs)
        if not isinstance(bearer_token_command, str) or not bearer_token_command.strip():
            raise ValueError("[BearerAuthClient] bearer_token_command must be a non-empty string")
        self.bearer_token_command = bearer_token_command
        self.save_token_cb = save_token_cb
        self._token: Optional[str] = None
        self._lock = threading.Lock()

    def _refresh_token_locked(self) -> None:
        """Execute token command and update state."""
        _log_with_thread(logging.DEBUG, "[BearerAuthClient] Refreshing token via command...")

        try:
            new_token = execute_command(self.bearer_token_command)
            if not new_token:
                raise ValueError(f"Bearer token command {self.bearer_token_command} returned empty token")

            self._token = new_token
            self.headers["Authorization"] = f"Bearer {new_token}"
            safe_callback(self.save_token_cb, new_token, "save")

            _log_with_thread(logging.DEBUG, "[BearerAuthClient] Token refreshed successfully.")
        except:
            # Clean up state on failure
            self._token = None
            # Clear persisted token but don't fail the refresh operation
            safe_callback(self.save_token_cb, None, "clear")
            raise

    def _ensure_token(self) -> None:
        """Ensure a token exists before making requests"""
        if self._token:
            return

        with self._lock:
            if not self._token:
                self._refresh_token_locked()

    def update_token(self, token: Optional[str]) -> None:
        """Update the token with a new one"""
        if not token:
            return

        with self._lock:
            if self._token != token:
                self._token = token
                self.headers["Authorization"] = f"Bearer {token}"

    def request(self, *args, **kwargs) -> httpx.Response:
        """Wraps httpx.request with auto-refresh logic for 401 Unauthorized."""
        self._ensure_token()
        response = super().request(*args, **kwargs)

        if response.status_code != 401:
            return response

        _log_with_thread(logging.DEBUG, "[BearerAuthClient] Received 401. Attempting recovery.")
        sent_auth_header = response.request.headers.get("Authorization")

        try:
            with self._lock:
                current_auth_header = self.headers.get("Authorization")
                if sent_auth_header == current_auth_header:
                    self._refresh_token_locked()
                else:
                    _log_with_thread(logging.DEBUG,
                                     "[BearerAuthClient] Token already refreshed by another thread. Retrying.")
        except Exception as e:
            logger.error(f"[BearerAuthClient] Recovery failed: Token refresh threw exception: {e}")
            return response

        # Retry the request with the new valid token
        # We must close the old 401 response to free connections
        response.close()
        return super().request(*args, **kwargs)
