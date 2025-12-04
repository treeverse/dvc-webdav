import pickle
import subprocess
import threading
import time
from typing import Callable
from unittest import mock

import httpx
import pytest

from dvc.exceptions import DvcException
from dvc_webdav import get_bearer_auth
from dvc_webdav.bearer_auth import BearerAuth


# Helper for constructing mock requests
def make_request():
    return httpx.Request("GET", "https://example.com")


def test_init_validation():
    """Ensure invalid commands raise DvcException."""
    with pytest.raises(DvcException, match="must be a non-empty string"):
        BearerAuth(bearer_token_command="", shell_timeout=30)

    with pytest.raises(DvcException, match="must be a non-empty string"):
        BearerAuth(bearer_token_command="   ", shell_timeout=30)

    with pytest.raises(DvcException, match="must be a non-empty string"):
        BearerAuth(bearer_token_command=None, shell_timeout=30)


def test_command_execution_success(mocker):
    """Test successful token retrieval via subprocess."""
    mock_run = mocker.patch("subprocess.run")
    mock_run.return_value.returncode = 0
    mock_run.return_value.stdout = "my-secret-token\n"

    # Use flags to verify shlex parsing works
    auth = BearerAuth("get-token-cmd --flag", shell_timeout=5)
    token = auth._ensure_token()

    assert token == "my-secret-token"

    # Verify subprocess arguments
    args, kwargs = mock_run.call_args
    # On POSIX shlex.split results in a list
    assert args[0] == ["get-token-cmd", "--flag"]
    assert kwargs["timeout"] == 5
    assert kwargs["shell"] is False
    assert kwargs["capture_output"] is True


def test_command_execution_errors(mocker):
    """Test handling of subprocess errors and timeouts."""
    mock_run = mocker.patch("subprocess.run")

    auth = BearerAuth("cmd", 30)

    # Case 1: Non-zero exit code
    mock_run.side_effect = subprocess.CalledProcessError(1, ["cmd"], stderr="denied")
    with pytest.raises(DvcException, match="Bearer Token Retrieval Failed") as exc:
        auth._ensure_token()
    assert "Bearer Token Retrieval Failed" in str(exc.value)
    assert isinstance(exc.value.__cause__, subprocess.CalledProcessError)

    # Case 2: Timeout
    mock_run.side_effect = subprocess.TimeoutExpired(["cmd"], 30)
    with pytest.raises(DvcException, match="Bearer Token Retrieval Failed"):
        auth._ensure_token()
    assert "Bearer Token Retrieval Failed" in str(exc.value)

    # Case 3: Empty token returned
    mock_run.side_effect = None
    mock_run.return_value.returncode = 0
    mock_run.return_value.stdout = "   "
    with pytest.raises(DvcException, match="returned an empty token"):
        auth._ensure_token()


def test_auth_flow_initial(mocker):
    """Test the happy path where the token is fetched immediately."""
    mock_run = mocker.patch("subprocess.run")
    mock_run.return_value.stdout = "token-v1"

    auth = BearerAuth("cmd", 30)
    req = make_request()

    # Start the generator flow
    flow = auth.auth_flow(req)
    yielded_req = next(flow)

    assert yielded_req.headers["Authorization"] == "Bearer token-v1"
    # Verify the token command runs exactly once,
    # ensuring the token is fetched without redundant executions.
    mock_run.assert_called_once()


@pytest.fixture
def active_auth_flow(mocker):
    """
    Setup fixture that:
    1. Mocks subprocess to return 'token-v1' first, then 'token-v2'.
    2. Initializes the BearerAuth flow.
    3. Advances the generator to the first yield (request with token-v1).
    Returns tuple: (flow_generator, auth_instance, mock_subprocess, first_request)
    """
    mock_run = mocker.patch("subprocess.run")
    mock_run.side_effect = [
        mock.Mock(stdout="token-v1"),
        mock.Mock(stdout="token-v2"),
    ]

    auth = BearerAuth("cmd", 30)
    req = make_request()
    flow = auth.auth_flow(req)

    # Prime the generator: executes until the first 'yield'
    req1 = next(flow)
    assert req1.headers["Authorization"] == "Bearer token-v1"

    return flow, auth, mock_run, req1


def test_auth_flow_refresh_401(active_auth_flow):
    """Test that receiving 401 triggers a token refresh and retry."""
    flow, auth, mock_run, req1 = active_auth_flow

    # Simulate 401 response from server
    response_401 = httpx.Response(401, request=req1)

    # Resume generator; it should refresh token and yield request again
    req2 = flow.send(response_401)

    # Assertions specific to the refresh logic
    assert req2.headers["Authorization"] == "Bearer token-v2"
    assert mock_run.call_count == 2
    assert auth._token == "token-v2"


def test_auth_flow_failure_after_refresh(active_auth_flow):
    """Test that the auth flow terminates if the refreshed token is also rejected."""
    flow, _, _, req1 = active_auth_flow

    # Trigger the first refresh (reusing logic implicitly tested above)
    response_401 = httpx.Response(401, request=req1)
    req2 = flow.send(response_401)

    # Server returns 401 AGAIN (for the new token-v2)
    resp_401_again = httpx.Response(401, request=req2)

    # Generator must terminate (raising StopIteration),
    # indicating no further retries will be attempted.
    with pytest.raises(StopIteration):
        flow.send(resp_401_again)


def _run_concurrent_token_test(
    mocker,
    token_prefix: str,
    setup_auth: Callable[[BearerAuth], None],
    worker_action: Callable[[BearerAuth], str],
    expected_token: str,
    thread_count: int = 5,
    add_delay: bool = False,
):
    """
    Helper function to test concurrent token operations.

    Args:
        mocker: pytest-mock fixture
        token_prefix: Prefix for generated tokens (e.g., "token-refresh", "token-init")
        setup_auth: Function to set up initial auth state
        worker_action: Function each thread executes to get a token
        expected_token: The token all threads should receive
        thread_count: Number of concurrent threads
        add_delay: Whether to add artificial delay in mock
    """
    mock_run = mocker.patch("subprocess.run")

    call_count = {"count": 0}

    def side_effect(*args, **kwargs):
        if add_delay:
            time.sleep(0.01)
        call_count["count"] += 1
        return mock.Mock(stdout=f"{token_prefix}-{call_count['count']}")

    mock_run.side_effect = side_effect

    auth = BearerAuth("cmd", 30)
    setup_auth(auth)

    results = []

    def worker():
        token = worker_action(auth)
        results.append(token)

    threads = [threading.Thread(target=worker) for _ in range(thread_count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Verify: command was called only once (locking works)
    assert mock_run.call_count == 1, f"Expected 1 call, got {mock_run.call_count}"

    # Verify: all threads received the same token
    assert len(set(results)) == 1, f"Expected all same tokens, got {set(results)}"
    assert results[0] == expected_token, (
        f"Expected '{expected_token}', got '{results[0]}'"
    )


def test_concurrent_multiple_threads_refreshing(mocker):
    """
    Test that when multiple threads simultaneously encounter a 401 error,
    only one thread actually executes the token refresh command.
    All threads should receive the same new token, not different tokens
    from separate refresh attempts.
    """
    _run_concurrent_token_test(
        mocker=mocker,
        token_prefix="token-refresh",
        setup_auth=lambda auth: setattr(auth, "_token", "token-old"),
        worker_action=lambda auth: auth._refresh_token_if_needed(
            failed_token="token-old"
        ),
        expected_token="token-refresh-1",
    )


def test_ensure_token_race_condition(mocker):
    """
    Test race condition when multiple threads call _ensure_token for the first time.
    Verify that only one thread executes the token acquisition command,
    while others wait and reuse the same result.
    """
    _run_concurrent_token_test(
        mocker=mocker,
        token_prefix="token-init",
        setup_auth=lambda _: None,  # No setup needed, _token is already None
        worker_action=lambda auth: auth._ensure_token(),
        expected_token="token-init-1",
        add_delay=True,  # Add delay to increase chance of race condition
    )


def test_high_concurrency_token_refresh(mocker):
    """
    High concurrency stress test: simulate production environment where
    many concurrent requests encounter 401 simultaneously.
    """
    _run_concurrent_token_test(
        mocker=mocker,
        token_prefix="token-concurrent",
        setup_auth=lambda auth: setattr(auth, "_token", "token-stale"),
        worker_action=lambda auth: auth._refresh_token_if_needed(
            failed_token="token-stale"
        ),
        expected_token="token-concurrent-1",
        thread_count=50,
        add_delay=True,
    )


def test_pickle_support():
    """Ensure BearerAuth can be pickled (lock must be excluded)."""
    auth = BearerAuth("cmd", 10)
    auth._token = "preserved-token"

    dumped = pickle.dumps(auth)
    loaded = pickle.loads(dumped)

    assert loaded.command_args == auth.command_args
    assert loaded.shell_timeout == 10
    assert loaded._token == "preserved-token"
    # Ensure lock is restored as a new lock instance
    assert loaded._lock is not auth._lock
    assert isinstance(loaded._lock, type(threading.Lock()))


@pytest.mark.parametrize(
    "special_token",
    [
        "token-with-quotes'\"",
        "token\nwith\nnewlines",
        "token\twith\ttabs",
        "token with spaces",
        "token;with;semicolons",
        "token|with|pipes",
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.e30.secret",
    ],
)
def test_special_characters_in_token(mocker, special_token):
    """
    Test handling of tokens containing special characters.
    The token should be properly stripped of leading/trailing whitespace,
    but all internal characters (including quotes, newlines, tabs, spaces, etc.)
    must be preserved as-is.
    """
    mock_run = mocker.patch("subprocess.run")
    mock_run.return_value.stdout = f"  {special_token}  \n"

    auth = BearerAuth("cmd", 30)
    token = auth._ensure_token()

    # Whitespace should be stripped, but all special characters preserved
    assert token == special_token.strip()


def test_memoization_integrity():
    """
    Verifies that get_bearer_auth correctly memoizes instances based on
    command string and timeout. Existing tests mocked this out, hiding
    potential cache-key bugs.
    """
    cmd_a = "get-token"
    timeout_a = 30

    auth_1 = get_bearer_auth(cmd_a, timeout_a)
    auth_2 = get_bearer_auth(cmd_a, timeout_a)
    auth_3 = get_bearer_auth(cmd_a, 60)  # Different timeout

    # Identical config should return same object reference
    assert auth_1 is auth_2
    # Different config should return new object
    assert auth_1 is not auth_3
