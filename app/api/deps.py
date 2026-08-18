"""FastAPI dependency providers."""

from __future__ import annotations

from fastapi import Request

from app.services import Services


def get_services(request: Request) -> Services:
    """Yield the application's service container from request state."""
    return request.app.state.services
