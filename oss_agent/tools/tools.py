
import asyncio


from openai_harmony import (
    Message,
    Role,
    ToolDescription,
    ToolNamespaceConfig,
    Author
)

from importlib.resources import read_text

from .mcp_client import MCPClient, clients_from_json

import types
import json

import re

import inspect
from typing import Union
from pathlib import Path


from gpt_oss.tools.apply_patch import apply_patch as apply_patch_tool
from .duck_backend import DuckBackend
from gpt_oss.tools.simple_browser.simple_browser_tool import SimpleBrowserTool

from gpt_oss.tools.python_docker.docker_tool import PythonTool








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
                  tools: list[types.FunctionType] = None,
                  servers: list[MCPClient] = None,
                  filename: Union[Path, str] = None,
                  python_tool = True,
                  browser_tool = True,
                  apply_patch = True):
        
        self.namespaces = []
       
        self.tool_dict = {}
        if tools is not None:
            self.tool_dict = { f.__name__: f for f in tools}
            descriptions = [Tools.tool_from_func(f) for f in tools]

            if python_tool:
                self.python_tool = PythonTool(execution_backend="dangerously_use_local_jupyter")
            else:
                self.python_tool = None

            if browser_tool:
                self.browser_tool = SimpleBrowserTool(DuckBackend(source=""))
            else:
                self.browser_tool = None

            if apply_patch:
                current_file_path = Path(__file__).resolve().parent / 'apply_patch' / 'apply_patch.md'
                with open(current_file_path, 'r') as file:
                    instructions = file.read()
                name = 'apply_patch'
                
                parameters= {
                        "type": "object",
                        "properties": {'text': {'type': 'str'}},
                        "required": ['text'], 
                }
                self.tool_dict[name] = apply_patch_tool
                descriptions.append(ToolDescription.new(name=name, description=instructions, parameters=parameters))
                

            self.namespaces.append(ToolNamespaceConfig(name="functions", description=None, tools=descriptions))


        self.server_dict = {}
        if servers is not None:
            self.server_dict = {s.name: s for s in servers}
            

        if filename is not None:
            additional_servers = {s.name: s for s in clients_from_json(filename)}
        
            self.server_dict = self.server_dict | additional_servers
        

    
    async def init_mcp_connections(self):
        for k, s in self.server_dict.items():
            if k == 'browser' or k == 'python':
                continue
            await s.connect()
            self.namespaces.append(s.get_namespace())

    @staticmethod
    def tool_from_func(f):
        inspection = inspect.getfullargspec(f)
        args = inspection.args
        param_dict = {}
        required = []
        signature = inspect.signature(f)
        for a in args:
            param = signature.parameters[a]
            name, dic, req =  Tools.format_param(param)
            param_dict[name] = dic
            if req:
                required.append(name)
        parameters={
            "type": "object",
            "properties": param_dict,
            "required": required,
        }
        
        return ToolDescription.new(
            f.__name__, f.__doc__, parameters=parameters
        )

    @staticmethod
    def format_param(p):
        param_dict = {}
        required = False
        if p.annotation  is not inspect.Parameter.empty:
            param_dict['type'] = str(p.annotation)
        if p.default is not inspect.Parameter.empty:
            param_dict['default'] = p.default
        else:
            required = True
        
        return p.name, param_dict, required
    
    async def call_mcp_tool(self, namespace, name, args):

        server = self.server_dict[namespace]
        res = await server.call_tool(name, args)
        return res.content[0].text
    
    async def call_function_tool(self, name, args):
        tool = self.tool_dict[name]
        if inspect.iscoroutinefunction(tool):
            result = await tool(**args)
        else:
            result = await asyncio.to_thread(tool, **args)

        return result

        
   
    async def handle_tool_message(self, msg: Message):
        print("\n message: " + str(msg))
        try:
            namespace, name = clean_func_name(msg.recipient)
            content = msg.content[0].text
            args = json.loads(content)

            if namespace == 'functions':
                result = await self.call_function_tool(name, args)
            elif namespace == 'browser':
                result = []
                async for m in self.browser_tool.process(msg):
                    result.append(m)
                return result
            elif namespace == 'python':
                result = []
                async for m in self.python_tool.process(msg):
                    result.append(m)
                return result
            else:
                result = await self.call_mcp_tool(namespace, name, args)
                

            return	[Message.from_author_and_content(
                    Author.new(Role.TOOL, '.'.join([namespace, name])),
                str(result)
                ).with_channel("commentary")]
        except FileNotFoundError as e:
            print(f'\n ERROR: {e}')
            return [Message.from_author_and_content(Author.new(Role.TOOL,name=msg.recipient), content=str(e)).with_channel("comentary")]
    


