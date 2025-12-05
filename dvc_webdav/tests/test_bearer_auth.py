import subprocess
from unittest import mock

import httpx
import pytest

from dvc.exceptions import DvcException
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


def test_auth_flow_refresh_401(mocker):
    """Test that receiving 401 triggers a token refresh and retry."""
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

    # Simulate 401 response from server
    response_401 = httpx.Response(401, request=req1)

    # Resume generator; it should refresh token and yield request again
    req2 = flow.send(response_401)

    # Assertions specific to the refresh logic
    assert req2.headers["Authorization"] == "Bearer token-v2"
    assert mock_run.call_count == 2
    assert auth._token == "token-v2"
