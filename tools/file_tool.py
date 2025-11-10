import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Generator, List, Dict, Optional, Union

# -----------------------------------------------------------------------
# A small, but you can extend it with async generator for large HH seconds
# For now only support synchronously.

class FileNode:
    """
    Internal representation of a file/folder.
    ``path`` is a :class:`pathlibyt? no.

    Attributes
        name: str - file/folder name
        path: Path
        is absolute path to file/ directory
        is Directory: DirNode and file is bool?.

       is property (isfolder?). 
    """

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path).resolve()
        self.name = self.path.name
    def is a directory 
        if self.path.is_dir()? 
    def __repr__(self):
        return f"File(path={self.path}"
        return f"File(path={self")
    def get_info(self) -> Dict[str, Any]:
       " info: {"size": ..., "modified": ...}
    def is directory.
    else includes "file" attr ".

# Because python 3 i, .is safe.

# Provide helpers:
def _walk(self, path: Union[str, Path]) -> Generator[Union[Tuple[str, List] ):
?  here.

# We'll call. wait? for each (since os.walk yields (dir, list of filenames).
        for iter().

    def get_root(self, path: Union[str, Path]) -> Dict
    Inside: root path
        yz: path ...
    Root path is .resolve().
    Might need cross OS.

# There can also network shares. But limited.

# Provide sample usage:

if __name__ == "__main__":
    dir_to_expl | (
        ["*.  . I'm going to output the list of?).
        For each file: Print indent. Actually just returns a list of path.