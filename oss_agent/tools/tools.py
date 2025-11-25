
import asyncio


from openai_harmony import (
    Message,
    Role,
    ToolDescription,
    ToolNamespaceConfig,
    Author
)

from importlib.resources import read_text, files

from .mcp_client import MCPClient, clients_from_json
from .tool import Tool, BrowserTool

import types
import json

import re

import inspect
from typing import Union
from pathlib import Path


from gpt_oss.tools.apply_patch import apply_patch as apply_patch_tool

from gpt_oss.tools.python_docker.docker_tool import PythonTool

import gpt_oss.tools.apply_patch


        



def clean_func_name(recp):
    func_str = re.sub('<[^>]+>', ' ', recp).split(' ')[0]
    print(func_str)
    func_list = func_str.split('.')
    if len(func_list) == 1:
        return func_list[0], func_list[0]
    namespace = func_list[0]
    name = func_list[1]
    return namespace, name


class Tools:
    def  __init__(self, 
                  tools: list[types.FunctionType | Tool] = None,
                  servers: list[MCPClient] = None,
                  filename: Union[Path, str] = None,
                  python_tool = True,
                  browser_tool = True,
                  apply_patch = True):
        
        self.namespaces = []
        self.tools = {}
        if tools is not None:
            func_tools = [Tool.from_function(f) for f in tools]
            self.tools = self.tools | {f.name: f for f in func_tools}

        if browser_tool:
            self.browser_tool = BrowserTool()
            self.tools['search'] = self.browser_tool
            self.tools['find'] = self.browser_tool
            self.tools['open'] = self.browser_tool
            self.tools['browser'] = self.browser_tool

        else:
            self.browser_tool = None

        if apply_patch:
            current_file_path = files(gpt_oss.tools.apply_patch) / 'apply_patch.md'
            with open(current_file_path, 'r') as file:
                instructions = file.read()
            name = 'apply_patch'
            
            parameters= {
                    "type": "object",
                    "properties": {'text': {'type': 'str'}},
                    "required": ['text']
            }
            self.tools[name] = Tool(apply_patch_tool, name, instructions, parameters)
                
        if servers is None and filename is None:
            self.namespaces.append(ToolNamespaceConfig(name="functions", description=None, tools=descriptions))

        self.server_dict = {}
        if servers is not None:
            self.server_dict = {s.name: s for s in servers}
            

        if filename is not None:
            additional_servers = {s.name: s for s in clients_from_json(filename)}
        
            self.server_dict = self.server_dict | additional_servers

    def get_tools(self):
        return [t.get_config() for t in self.tools.values() if t.include_in_prompt]
        

    
    async def init_mcp_connections(self):
        for k, s in self.server_dict.items():
            if k == 'browser' or k == 'python':
                continue
            await s.connect()
            self.tools = self.tools | s.get_tools()

        
   
    async def handle_tool_message(self, msg: dict):
        print("\n message: " + str(msg))
        try:
            name = msg['function']['name']
            args = json.loads(msg['function']['arguments'])
            result = await self.tools[name](**args)
            return result
            
        except FileExistsError as e:
            print(f'\n ERROR: {e}')
            return [{'role': 'tool', 'content': str(e)}]
    


