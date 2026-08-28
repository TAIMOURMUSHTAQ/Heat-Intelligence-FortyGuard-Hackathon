"""Production entrypoint for Render and other WSGI hosts."""

from src.api import app

__all__ = ["app"]
