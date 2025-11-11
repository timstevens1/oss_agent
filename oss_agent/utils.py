
import re
from typing import List, Optional, Type, TypeVar, Tuple

# ----------------------------------------------------------------------
# Expected shape of the message model (the real model lives in openai_harmony)
# ----------------------------------------------------------------------
from openai_harmony import Message as _BaseMessage, Author, Role, TextContent  # type: 
import subprocess



def run_zsh_command(command: List[str]) -> Tuple[str, str]:
    """
    Execute an arbitrary Zsh command with optional positional arguments.

    Args:
        command (List[str]):
            the command and arguments to run where each argument is a separate list element.
             The command is the first element of this list. The command must be available in
            the system ``PATH`` or be a valid absolute/relative path.
            e.g. (['ls', '-l'] for 'ls -l', or ['echo', 'this is an argument with spaces'])


    Returns:
        Tuple[str, str]:
            A two‑element tuple ``(stdout, stderr)`` where:
            * ``stdout`` – captured standard‑output as a UTF‑8 decoded string.
            * ``stderr`` – captured standard‑error output as a UTF‑8 decoded
              string (empty if the command produced no error output).

    Raises:
        FileNotFoundError:
            Raised when the specified *command* cannot be located on the system.

    Example:
        >>> out, err = run_zsh_command("ls -l /tmp")
        >>> print(out)
        total 0
        drwxr-xr-x  2 user  staff   64 Sep 10 12:34 example_dir
        -rw-r--r--  1 user  staff   13 Sep 10 12:35 file.txt
        >>> print(err)   # No error output in this case
        ''
    """
    # Build the full command list (command + any additional args)

    # Run the command, capture stdout and stderr, raise on non‑zero exit.
    completed = subprocess.run(command,          # Let CalledProcessError be raised on failure
        text=True,            # Return output as str (decoded using default encoding, UTF‑8)
        capture_output=True   # Capture both stdout and stderr
    )

    return completed.stdout, completed.stderr

