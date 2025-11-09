from mcp.server import FastMCP

import subprocess


mcp = FastMCP(name='cli')

@mcp.tool()
def run_zsh_command(command: str, *args: str) -> Tuple[str, str]:
    """
    Execute an arbitrary Zsh command with optional positional arguments.

    Args:
        command (str):
            The base command to execute (e.g., ``"ls"``, ``"git"``, ``"echo"``,
            or a full path to an executable). The command must be available in
            the system ``PATH`` or be a valid absolute/relative path.

        *args (str):
            Zero or more additional arguments that will be passed to *command*.
            Each argument is treated as a distinct token; quoting/escaping is
            handled automatically by the underlying subprocess call.

    Returns:
        Tuple[str, str]:
            A two‑element tuple ``(stdout, stderr)`` where:
            * ``stdout`` – captured standard‑output as a UTF‑8 decoded string.
            * ``stderr`` – captured standard‑error output as a UTF‑8 decoded
              string (empty if the command produced no error output).

    Raises:
        FileNotFoundError:
            Raised when the specified *command* cannot be located on the system.

        subprocess.CalledProcessError:
            Raised if the command exits with a non‑zero return code.  The
            exception’s ``returncode``, ``stdout`` and ``stderr`` attributes
            contain the details of the failure.

    Example:
        >>> out, err = run_zsh_command("ls", "-l", "/tmp")
        >>> print(out)
        total 0
        drwxr-xr-x  2 user  staff   64 Sep 10 12:34 example_dir
        -rw-r--r--  1 user  staff   13 Sep 10 12:35 file.txt
        >>> print(err)   # No error output in this case
        ''
    """
    # Build the full command list (command + any additional args)
    cmd = [command, *args]

    # Run the command, capture stdout and stderr, raise on non‑zero exit.
    completed = subprocess.run(
        cmd,
        check=True,            # Let CalledProcessError be raised on failure
        text=True,            # Return output as str (decoded using default encoding, UTF‑8)
        capture_output=True   # Capture both stdout and stderr
    )

    return completed.stdout, completed.stderr

if __name__ == '__main__':
    mcp.run(transport='stdio')
