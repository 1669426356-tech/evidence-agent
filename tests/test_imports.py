from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def test_package_import_requires_no_credentials_and_creates_no_provider_client():
    source = Path(__file__).resolve().parents[1] / "src"
    environment = dict(os.environ)
    for key in list(environment):
        if any(token in key.upper() for token in ("API_KEY", "BASE_URL", "TOKEN", "SECRET")):
            environment.pop(key)
    environment["PYTHONPATH"] = str(source)
    script = """
import socket
import sys
def no_network(*args, **kwargs):
    raise AssertionError('Import attempted a network connection')
socket.socket.connect = no_network
socket.create_connection = no_network
import agent_framework
from agent_framework.models import LangChainDecisionModel
assert 'langchain_openai' not in sys.modules
assert 'serial' not in sys.modules
print('offline import ok')
"""
    completed = subprocess.run([sys.executable, "-c", script], env=environment,
                               capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
    assert "offline import ok" in completed.stdout

