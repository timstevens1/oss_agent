import pytest

# Updated import to reflect the actual location of the utility function within the
# package hierarchy. The function ``run_zsh_command`` resides in ``oss_agent.utils``.
from oss_agent.utils import run_zsh_command


def test_run_zsh_command_stdout():
    """A simple command should capture stdout and have empty stderr."""
    out, err = run_zsh_command(["echo", "hello world"])
    # echo adds a trailing newline
    assert out == "hello world\n"
    assert err == ""


def test_run_zsh_command_stderr():
    """A command that writes to stderr should capture it."""
    # "python -c" program prints to stderr
    out, err = run_zsh_command(["python", "-c", "import sys; sys.stderr.write('error')"])
    assert out == ""
    assert err == "error"


def test_run_zsh_command_not_found():
    """Running a non‑existent command should raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        run_zsh_command(["nonexistent_command_xyz"])
