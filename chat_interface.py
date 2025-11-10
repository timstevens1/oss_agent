import cmd

#!/usr/bin/env python
import getpass

import os
import platform
import readline



from agent import Agent
from model import OssModel


from tools.tools import Tools

import asyncio


from utils import run_zsh_command
from rich.console import Console
from rich.markdown import Markdown

from llm_cli import dispatch_special


GREY = "\33[90m"
BOLD = "\33[1m"

GREEN  = "\033[32m"
BLUE   = "\033[34m"
RED    = "\033[31m"
RESET  = "\033[0m"


class AgentChat(cmd.Cmd):
    intro = 'Welcome to AgentChat with your agent!'
    prompt = (
    f"{GREEN}{getpass.getuser()}@{BLUE}{os.getcwd()}{RESET} "
    f"{RED}(llm_chat) >{RESET} "
)
    def __init__(self, agent):
        super().__init()
        self.agent = agent

    def default(*args):
        prompt = ' '.join(args)
        
        




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
        
