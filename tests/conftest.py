"""Configuration pytest — fixtures partagées."""

import pytest


@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"
