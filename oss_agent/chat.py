"""Command‑line entry point for the OSS Agent.

The original script ``llm_chat.py`` lived at the repository root and was meant to be
executed directly.  After refactoring we expose the same functionality as a module
that can be invoked with ``python -m oss_agent.chat``.
"""

import getpass
import os
import platform
import readline
import asyncio

from .agent import Agent
from .model import RemoteOssModel
from .tools.tools import Tools
from .utils import run_zsh_command
from rich.console import Console
from rich.markdown import Markdown

GREY = "\33[90m"
BOLD = "\33[1m"
RESET = "\33[0m"

# The original script used a hard‑coded workspace path; preserve the variable for
# backward compatibility but it is not used elsewhere in the module.
root = "/Users/timstevens/projects/qwen_agent/workspace"

developer_prompt = """you're a general purpose assistant. You have filesystem access, \nand internet search access. Your job is to answer user questions in an informed way and to\nhandle their requests such as for coding or summarization or text generation. Accuracy and brevity are\nyour top priorities. Verify your answers with the tools you have access to. Simply try to meet \nthe user's expectations with your answer, and allow them to prompt you with\nfollow up queries if they would like more from you.\n\nTools are provided to you grouped into namespaces. Make sure to use the correct namespace when calling the tool.\nMake sure the message to the tool is a valid json dict containing the arguments you'd like to pass to the tool.\n"""

prompt = "User:"
GREEN = "\033[32m"
BLUE = "\033[34m"
RED = "\033[31m"
RESET = "\033[0m"

computername = platform.node()
workdir = os.getcwd()
username = getpass.getuser()
prompt = (
    f"{GREEN}{username}@{BLUE}{workdir}{RESET} "
    f"{RED}(llm_chat) >{RESET} "
)

console = Console()

async def main(_: None = None):
    # History handling (unchanged from original script)
    hist_file = os.path.expanduser("~/.llm_cli_history")
    if os.path.exists(hist_file):
        readline.read_history_file(hist_file)
    else:
        readline.write_history_file(hist_file)

    tools = Tools(tools=[run_zsh_command], filename=f"{os.path.expanduser('~')}/.config/oss_agent/servers.json")
    await tools.init_mcp_connections()
    model = RemoteOssModel("ws://localhost:8999") 
    agent = Agent(model, tools=tools, developer_message=developer_prompt)
    while True:
        try:
            message = input(prompt).rstrip()
            if not message:
                continue
            readline.add_history(message)
            if message.startswith('/'):
                # Slash commands are now handled by the Agent instance.
                try:
                    agent.dispatch_slash(message)
                except RuntimeError as e:
                    console.print(f"[red]Error:[/red] {e}")
                continue
            response = await agent(message)
            console.print(Markdown(response))
        except KeyboardInterrupt:
            print("\nBye! Welcome again.")
            break
    readline.write_history_file(hist_file)

if __name__ == "__main__":
    asyncio.run(main())
