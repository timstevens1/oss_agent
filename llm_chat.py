#!/usr/bin/env python
import getpass

import argparse
import sys
import os
import platform
import readline

from datetime import datetime


from agent import Agent
from model import OssModel


from openai_harmony import ToolDescription, Conversation, load_harmony_encoding, HarmonyEncodingName
from tools.mcp_client import StdIOClient, clients_from_json

from tools.tools import Tools

import asyncio

from typing import Union
from pathlib import Path

from glob import glob

from utils import run_zsh_command
from subprocess import run
from rich.console import Console
from rich.markdown import Markdown

from llm_cli import dispatch_special


GREY = "\33[90m"
BOLD = "\33[1m"
RESET = "\33[0m"


root = "/Users/timstevens/projects/qwen_agent/workspace"




developer_prompt = """you're a general purpose assistant. You have filesystem access, 
and internet search access. Your job is to answer user questions in an informed way and to
handle their requests such as for coding or summarization or text generation. Accuracy and brevity are
your top priorities. Verify your answers with the tools you have access to. Simply try to meet 
the user's expectations with your answer, and allow them to prompt you with
follow up queries if they would like more from you.

Format your responses in markdown. Make sure to double check your final output for spelling, grammar,
and formatting mistakes. 

Tools are provided to you grouped into namespaces. Call tools with the format to=<namespace>.<name>
Make sure the message to the tool is a valid json dict containing the arguments you'd like to pass to the function.

You have the ability to run  arbitrary zsh commands with the run_zsh_command  function. Use this to execute code or perform actions not enabled by the other servers.

If a tool execution fails, you will receive the error message from that tool. 
Take advantage of your internet access and internal documation (such as man pages or python docstrings) to fix your tool broken tool executions.

A few hints:
 -  If you would like to patch a file with a zsh command. Use `patch`.
 - The current directory path is ".", the parent directory is ".." 

"""


reducer_prompt = """
You are a a supervisor to an LLM. You will receive a conversation history between a user, an assistant, and tools. 

Rules:
 - Summarize the conversation into a single message.
 - Your summary should include all the relevant information from the conversation to help the agent respond
 correctly. The intention is to prune repetitive messages, thought loops, and information that is no longer relevant. 
 - Only remove information from the conversation if you are certain that it is erroneous to the task at hand. 
 - Be specific, and include exact quotes when necessary.
 - At the top of your message, write THIS IS A SUMMARY OF YOUR CONVERSATION HISTORY

 """


editor_prompt = """
You are an editor agent. You will be given a conversation history between a user, tools and an agent. The last message of the conversation will be the final output from the agent
to the user. Your job is to edit that final message to ensure that it is free of spelling and grammar mistakes, formatting errors, or deviates from the data collected by the tools
or the user's inital request. 

Respond as if you were the original agent. Do not include any prelude or additional meta commentary, just output the improved version of the agent's response. Your response must
be in markdown format.

The conversation history might include references to tools, calls to tools or tool responses. You do NOT have access to any tools and under no circumstances should you attempt to
call any tools.
"""


prompt = "User:"
# Terminal‑style prompt (green, blue, red)
GREEN  = "\033[32m"
BLUE   = "\033[34m"
RED    = "\033[31m"
RESET  = "\033[0m"

computername = platform.node()            # returns the host name
workdir     = os.getcwd()
username = getpass.getuser()
# Compose coloured string: "<green username>@<reset><blue computername><reset><red>(llm_chat) >"
prompt = (
    f"{GREEN}{username}@{BLUE}{workdir}{RESET} "
    f"{RED}(llm_chat) >{RESET} "
)


console = Console()
async def main(args):

    hist_file = os.path.expanduser("~/.llm_cli_history")                                                                                                                                       
    if os.path.exists(hist_file):                                                                                                                                                              
        readline.read_history_file(hist_file)   
    else:
        readline.write_history_file(hist_file)   

    tools = Tools(tools=[run_zsh_command], filename="servers.json")

    await tools.init_mcp_connections()
    
    model = OssModel.from_size(20)
    agent = Agent(model, tools=tools, developer_message=developer_prompt)

    while True:
        try:
            message = input(prompt).rstrip()
            if not message:                                                                                                                                                                 
                continue                                                                                                                                                                       
                                                                                                                                                                                                
                # Store in the builtin revision list                                                                                                                                               
            readline.add_history(message)                                                                                                                                              
                                                                                                                                                                                                
            # Handle special commands                                                                                                                                                          
            if message.startswith("/"):                                                                                                                                                     
                dispatch_special(message)                                                                                                                                                   
                continue   
            response = await agent(message)
            good_markdown = Markdown(response)
            console.print(good_markdown)
        except KeyboardInterrupt:                                                                                                                                                              
            print("\nBye! Welcome again.") 
            break
    readline.write_history_file(hist_file)
        


if __name__ == "__main__":
    asyncio.run(main(None))
        
