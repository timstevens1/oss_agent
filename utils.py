"""Utility module providing simplified imports for test compatibility.

This module re‑exports the :func:`run_zsh_command` function from the
``oss_agent.utils`` package so that test code can import it directly via
``from utils import run_zsh_command``.
"""

from oss_agent.utils import run_zsh_command

__all__ = ["run_zsh_command"]