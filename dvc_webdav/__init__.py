import logging
import keyring
import threading
from getpass import getpass
from typing import ClassVar, Optional

from funcy import memoize, wrap_prop, wrap_with

from dvc.repo import Repo
from dvc.utils.objects import cached_property
from dvc_objects.fs.base import FileSystem

from dvc_webdav.bearer_auth_client import BearerAuthClient

logger = logging.getLogger("dvc")


@wrap_with(threading.Lock())
@memoize
def ask_password(host, user):
    return getpass(f"Enter a password for host '{host}' user '{user}':\n")


@wrap_with(threading.Lock())
@memoize
def get_bearer_auth_client(bearer_token_command: str):
    logger.debug("Bearer token command provided, using BearerAuthClient, command: %s", bearer_token_command, )
    return BearerAuthClient(bearer_token_command)


class WebDAVFileSystem(FileSystem):  # pylint:disable=abstract-method
    protocol = "webdav"
    root_marker = ""
    REQUIRES: ClassVar[dict[str, str]] = {"webdav4": "webdav4"}
    PARAM_CHECKSUM = "etag"

    def __init__(self, **config):
        super().__init__(**config)

        cert_path = config.get("cert_path")
        key_path = config.get("key_path")
        self.prefix = config.get("prefix", "")
        self.fs_args.update(
            {
                "base_url": config["url"],
                "cert": cert_path if not key_path else (cert_path, key_path),
                "verify": config.get("ssl_verify", True),
                "timeout": config.get("timeout", 30),
            }
        )
        if bearer_token_command := config.get("bearer_token_command"):
            client = get_bearer_auth_client(bearer_token_command)
            client.save_token_cb = self._save_token
            if token := config.get("token"):
                client.update_token(token)
            self.fs_args["http_client"] = client

    def unstrip_protocol(self, path: str) -> str:
        return self.fs_args["base_url"] + "/" + path

    @staticmethod
    def _normalize_url(url):
        from urllib.parse import urlparse, urlunparse

        parsed = urlparse(url)
        scheme = parsed.scheme.replace("webdav", "http")
        path = parsed.path.rstrip("/")
        return urlunparse((scheme, parsed.netloc, path, None, None, None))

    @classmethod
    def _get_kwargs_from_urls(cls, urlpath):
        from urllib.parse import urlparse, urlunparse

        parsed = urlparse(urlpath)
        scheme = parsed.scheme.replace("webdav", "http")
        return {
            "prefix": parsed.path.rstrip("/"),
            "host": urlunparse((scheme, parsed.netloc, "", None, None, None)),
            "url": cls._normalize_url(urlpath),
            "user": parsed.username,
        }

    def _find_remote_name(self) -> Optional[str]:
        """Find the remote name for the current filesystem."""
        repo = Repo()
        base_url = self.fs_args["base_url"]
        for remote_name, remote_config in repo.config["remote"].items():
            remote_url = remote_config.get("url")
            if not remote_url:
                continue

            normalized_remote_url = self._normalize_url(remote_url)
            if normalized_remote_url == base_url:
                return remote_name
        return None

    @wrap_with(threading.Lock())
    def _save_token(self, token: Optional[str]) -> None:
        """Save or unset the token in the local DVC config."""
        remote_name = self._find_remote_name()
        if not remote_name:
            logger.warning("Skipping token persistence - Could not find remote name to save token.")
            return

        with Repo().config.edit("local") as conf:
            remote_conf = conf.setdefault("remote", {}).setdefault(remote_name, {})
            if token:
                if remote_conf.get("token") != token:
                    remote_conf["token"] = token
                    logger.debug("Saved token for remote '%s'", remote_name)
            elif "token" in remote_conf:
                del remote_conf["token"]
                logger.debug("Unset token for remote '%s'", remote_name)

    def _prepare_credentials(self, **config):
        user = config.get("user")
        password = config.get("password")
        headers = {}
        auth = None
        if token := config.get("token"):
            headers.update({"Authorization": f"Bearer {token}"})
        elif user:
            if not password and config.get("ask_password"):
                password = ask_password(config["host"], user)
            auth = (user, password)
        elif custom_auth_header := config.get("custom_auth_header"):
            if not password and config.get("ask_password"):
                password = ask_password(config["host"], custom_auth_header)
            headers.update({custom_auth_header: password})

        return {"headers": headers, "auth": auth}

    @wrap_prop(threading.Lock())
    @cached_property
    def fs(self):
        from webdav4.fsspec import WebdavFileSystem

        return WebdavFileSystem(**self.fs_args)

    def upload_fobj(self, fobj, to_info, **kwargs):
        size = kwargs.get("size")
        # using upload_fileobj to directly upload fileobj rather than buffering
        # and using overwrite=True to avoid check for an extra exists call,
        # as caller should ensure that the file does not exist beforehand.
        return self.fs.upload_fileobj(fobj, to_info, overwrite=True, size=size)


class WebDAVSFileSystem(WebDAVFileSystem):  # pylint:disable=abstract-method
    protocol = "webdavs"
