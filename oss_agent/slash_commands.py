"""Utility module for handling slash commands in a modular, object‑oriented way.

The original implementation lived in ``llm_cli.py`` and also included a REPL
loop.  The REPL functionality is not required for the library use‑case, so this
module provides only the command‑management facilities.

Usage example::

    from slash_commands import dispatch_special, add_slash_command

    def hello():
        print("Hello world")

    add_slash_command("/hello", hello)
    dispatch_special("/hello")   # prints "Hello world"

The module maintains a global ``slash_commands`` instance that stores the
registry.  Existing code that imported ``dispatch_special`` from ``llm_cli`` can be
updated to import it from this module without any behavioural change.
"""

import os
import readline

# ---------------------------------------------------------------------------
# In‑memory message store (unchanged from the original file)
# ---------------------------------------------------------------------------
MESSAGE_STORE = []


def add_to_store(msg: str) -> None:
    """Store a user/agent message into the persistent cache."""
    MESSAGE_STORE.append(msg)


def clear_message_history() -> None:
    """Delete everything that is kept in the message store."""
    global MESSAGE_STORE
    MESSAGE_STORE.clear()
    print("All messages have been cleared from the cache.")

def smooth_exit():
    raise KeyboardInterrupt



class SlashCommands:
    """Manage registration and dispatch of slash commands.

    The class provides a simple registry mapping command strings (including the
    leading ``/``) to callables.  It is deliberately lightweight – the goal is to
    allow external code to add custom commands without touching the core REPL logic.
    """

    def __init__(self) -> None:
        self._routes: dict[str, callable] = {}
        # Built‑in commands
        self.register("/clear_history", clear_message_history)
        # ``/quit`` triggers a KeyboardInterrupt to exit the REPL cleanly.
        self.register("/quit", lambda: smooth_exit())

    def register(self, command: str, func: callable) -> None:
        """Add *command* to the registry.

        ``command`` should include the leading ``/`` (e.g. ``"/mycmd"``).
        ``func`` is a zero‑argument callable that will be executed when the command is
        invoked.
        """
        self._routes[command] = func

    def dispatch(self, command: str) -> None:
        """Execute the callable associated with *command*.

        Raises ``RuntimeError`` if the command is unknown.
        """
        cmd = command.strip()
        if cmd in self._routes:
            self._routes[cmd]()
        else:
            raise RuntimeError(f"Unknown special command: {command!r}")


# Global instance used by the REPL loop (if any) and by the helper functions
slash_commands = SlashCommands()


def dispatch_special(command: str) -> None:
    """Legacy wrapper that forwards to the :class:`SlashCommands` instance.

    Existing code imports ``dispatch_special`` directly, so we keep a thin wrapper for
    backward compatibility.
    """
    slash_commands.dispatch(command)


def add_slash_command(command: str, func: callable) -> None:
    """Convenient helper to register a new slash command at runtime.

    Example::

        def my_cmd():
            print("Running my command")
        add_slash_command("/mycmd", my_cmd)
    """
    slash_commands.register(command, func)
