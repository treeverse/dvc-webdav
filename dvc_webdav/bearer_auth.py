import logging
import os
import shlex
import subprocess
import threading
from collections.abc import Generator
from typing import Optional

import httpx

from dvc.exceptions import DvcException

logger = logging.getLogger("dvc")


def _log_with_thread(level: int, msg: str, *args) -> None:
    """
    Universal helper to inject thread identity into logs.
    Output format: [Thread-Name] Message...
    """
    if logger.isEnabledFor(level):
        thread_name = threading.current_thread().name
        log_fmt = f"[{thread_name}] " + msg
        logger.log(level, log_fmt, *args)


def _execute_command(command: list[str], timeout: int = 30) -> str:
    """Executes a command to retrieve the token."""
    try:
        # shell=False ensures safety against injection, but requires valid args list.
        result = subprocess.run(  # noqa: S603
            command,
            shell=False,
            capture_output=True,
            text=True,
            check=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        cmd_str = " ".join(shlex.quote(arg) for arg in command)
        _log_with_thread(
            logging.DEBUG,
            "Bearer Token Retrieval Failed.\nCommand: %s\nStdout: %s\nStderr: %s",
            cmd_str,
            e.stdout,
            e.stderr,
        )

        details = (
            f"Bearer Token Retrieval Failed.\n"
            f"Error Type: {type(e).__name__}\n"
            f"Exit Code: {getattr(e, 'returncode', 'Unknown')}\n"
            f"Run with '-v' to see full command output and error details in debug logs."
        )
        raise DvcException(details) from e

    except Exception as e:
        raise DvcException(f"Unexpected error executing token command: {e}") from e

    token = result.stdout.strip()
    if not token:
        raise DvcException("Bearer token command returned an empty token.")
    return token


class BearerAuth(httpx.Auth):
    """HTTPX Auth class that adds Bearer token authentication using a command.

    Handles 401 Unauthorized retries with thread-safe token refreshing.
    """

    def __init__(self, bearer_token_command: str, shell_timeout: int):
        """Initializes BearerAuth with a command to fetch the token.

        Args:
            bearer_token_command: Command string to execute for token retrieval.
            shell_timeout: Timeout in seconds for the command execution.
        """
        if (
            not isinstance(bearer_token_command, str)
            or not bearer_token_command.strip()
        ):
            raise DvcException(
                "[BearerAuth] bearer_token_command must be a non-empty string"
            )
        is_posix = os.name == "posix"
        self.command_args = shlex.split(bearer_token_command, posix=is_posix)
        self.shell_timeout = shell_timeout
        self._token: Optional[str] = None
        self._lock = threading.Lock()

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["_lock"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = threading.Lock()

    def _fetch_bearer_token(self) -> str:
        _log_with_thread(logging.DEBUG, "[BearerAuth] Refreshing token via command...")
        try:
            self._token = _execute_command(self.command_args, self.shell_timeout)
            _log_with_thread(
                logging.DEBUG, "[BearerAuth] Token refreshed successfully."
            )
            return self._token
        except:
            self._token = None
            raise

    def _ensure_token(self) -> str:
        """Returns the current token, initializing it if necessary."""
        if self._token:
            return self._token

        with self._lock:
            if not self._token:
                return self._fetch_bearer_token()
            return self._token  # type: ignore[unreachable]

    def _refresh_token_if_needed(self, failed_token: str) -> str:
        """Thread-safe token refresh logic."""
        with self._lock:
            # If the token has changed since the failure AND is valid, use it.
            if self._token != failed_token and self._token is not None:
                _log_with_thread(
                    logging.DEBUG,
                    "[BearerAuth] Token already refreshed by another thread.",
                )
                return self._token

            return self._fetch_bearer_token()

    def auth_flow(
        self, request: httpx.Request
    ) -> Generator[httpx.Request, httpx.Response, None]:
        token = self._ensure_token()
        request.headers["Authorization"] = f"Bearer {token}"

        response = yield request

        if response.status_code == 401:
            _log_with_thread(
                logging.DEBUG, "[BearerAuth] Received 401. Attempting recovery."
            )

            token = self._refresh_token_if_needed(failed_token=token)
            request.headers["Authorization"] = f"Bearer {token}"
            yield request
