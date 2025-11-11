"""Agent module for the OSS Agent.

The original implementation lived at the repository root as ``agent.py``.  This
file mirrors that implementation but uses package‑relative imports so that the
module can be imported as ``oss_agent.agent`` after the refactor.
"""

from datetime import datetime

from openai_harmony import (
    SystemContent,
    Message,
    TextContent,
    Conversation,
    Role,
    load_harmony_encoding,
    HarmonyEncodingName,
    ReasoningEffort,
    StreamableParser,
    StreamState,
    DeveloperContent,
    ToolNamespaceConfig,
    Author,
)

from pathlib import Path
import json
from .tools.tools import Tools
from mlx_lm import load, stream_generate, generate
from mlx_lm.sample_utils import make_sampler

from typing import Union, List
import re

# Relative import for the model definitions
from .model import BaseModel
from .slash_commands import clear_message_history, smooth_exit
from rich.console import Console

developer_prompt = """you're a general purpose assistant. You have filesystem access, \nand internet search access. Your job is to answer user questions in an informed way and to\nhandle their requests such as for coding or summarization or text generation. Accuracy and brevity are\nyour top priorities. Verify your answers with the tools you have access to. Simply try to meet \nthe user's expectations with your answer, and allow them to prompt you with\nfollow up queries if they would like more from you. For example:\n\nTools are provided to you grouped into namespaces. Make sure to use the correct namespace when calling the tool.\nMake sure the message to the tool is a valid json dict containing the arguments you'd like to pass to the tool.\n"""

GREY = "\33[90m"
BOLD = "\33[1m"
RESET = "\33[0m"
FAIL = '\033[91m'



class Agent:

    def __init__(self, model: BaseModel, developer_message: str = developer_prompt, tools: Tools = None, refining_agent=None):
        self.model = model
        self.tools = tools
        self.encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        self.messages = []
        self.refining_agent = refining_agent
        # ---------------------------------------------------------------------
        # Slash‑command registry (new feature)
        # ---------------------------------------------------------------------
        # The slash command system was previously a separate module.  For a more
        # object‑oriented design we embed the registry directly on each Agent
        # instance.  This allows callers to add custom commands that can interact
        # with the specific Agent instance (e.g. clearing its history).
        self._slash_routes: dict[str, callable] = {}
        # Register built‑in commands that existed in the original implementation.
        # ``/clear_history`` clears the global MESSAGE_STORE defined in
        # ``slash_commands``; we keep the same behaviour for compatibility.
        self._slash_routes["/clear_history"] = self.clear_history
        # ``/quit`` should raise KeyboardInterrupt to terminate the REPL.
        self._slash_routes["/quit"] = lambda: (_ for _ in ()).throw(KeyboardInterrupt)
        developer_content = DeveloperContent.new()
        if developer_message:
            developer_content = developer_content.with_instructions(developer_message)
        if tools is not None:
            for n in self.tools.namespaces:
                developer_content = developer_content.with_tools(n)
        self.messages.append(Message.from_role_and_content(Role.DEVELOPER, developer_content))

    def clear_history(self):
        """
        Messages[:2] is the system and developer message
        """
        self.messages = self.messages[:2]

    # ---------------------------------------------------------------------
    # Slash‑command public API
    # ---------------------------------------------------------------------
    def add_slash_command(self, command: str, func: callable) -> None:
        """Register a new slash command.

        ``command`` must include the leading ``/`` (e.g. ``"/mycmd"``).  The
        ``func`` is a zero‑argument callable that will be invoked when the command
        is dispatched.
        """
        self._slash_routes[command] = func

    def dispatch_slash(self, command: str) -> None:
        """Execute the callable associated with *command*.

        Raises ``RuntimeError`` if the command is not recognised.
        """

        cmd = command.strip().split(' ') 

        if cmd[0] in self._slash_routes:
            if len(cmd) == 1:
                return self._slash_routes[cmd[0]]()
            else:
                args = cmd[1:]
                cmd = cmd[0]
                return self._slash_routes[cmd](*args)
        else:
            raise RuntimeError(f"Unknown special command: {command!r}")

    async def _invoke(self, message: dict, print_analysis=True):
        self.messages.append(message)
        new_messages = await self.model.complete(self.messages, print_analysis)
        self.messages.extend(new_messages)
        while self.messages[-1].recipient is not None:
            yield self.messages[-1]
            new_messages = await self.tools.handle_tool_message(self.messages[-1])
            self.messages.extend(new_messages)
            yield self.messages[-1]
            new_messages = await self.model.complete(self.messages, print_analysis)
            self.messages.extend(new_messages)
        yield self.messages[-1]

    async def __call__(self, message: str, print_analysis=True):
        if message.startswith("/"):
            return self.dispatch_slash(message)
        message_obj = Message.from_role_and_content(Role.USER, message)
        async for _ in self._invoke(message_obj, print_analysis=print_analysis):
            continue
        return self.messages[-1].content[0].text
