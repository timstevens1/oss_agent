
import asyncio


from openai_harmony import (
    Message,
    Role,
    ToolNamespaceConfig,
    ToolDescription,
    Author
)

from typing import Optional, Any, Union
from pathlib import Path
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.types import ListToolsResult

from mcp.client.stdio import stdio_client
import json

import re
import httpx



def clients_from_json(file: Union[Path, str]):
    config = json.load(open(file, 'r'))
    clients = []
    for server, conf in config['mcpServers'].items():


        clients.append(StdIOClient(
            name = server,
            command = conf['command'],
            args = conf['args'],
            env = conf.get('env', None)
        ))
    return clients




def clean_func_name(recp):
    func_str = re.sub('<[^>]+>', ' ', recp).split(' ')[0]
    print(func_str)
    func_list = func_str.split('.')
    namespace = func_list[0]
    name = '.'.join(func_list[1:])
    return namespace, name



class MCPClient:

    def __init__(self, name):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.tools = None
        self.init_response = None
        self.name = name

    async def get_client(self):
        return "nothing"
        
    async def connect(self):
        if self.init_response is not None:
            return
        stdio_transport = await self.get_client()
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        self.init_response = await self.session.initialize()
        response = await self.session.list_tools()
        self.tools = response


    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()


    def trim_schema(self, schema: dict) -> dict:
        # Turn JSON Schema from MCP generated into Harmony's variant.
        if "title" in schema:
            del schema["title"]
        if "default" in schema and schema["default"] is None:
            del schema["default"]
        if "anyOf" in schema:
            # Turn "anyOf": [{"type": "type-1"}, {"type": "type-2"}] into "type": ["type-1", "type-2"]
            # if there's more than 1 types, also remove "null" type as Harmony will just ignore it
            types = [
                type_dict["type"] for type_dict in schema["anyOf"]
                if type_dict["type"] != 'null'
            ]
            schema["type"] = types
            del schema["anyOf"]
        if "properties" in schema:
            schema["properties"] = {
                k: self.trim_schema(v)
                for k, v in schema["properties"].items()
            }
        return schema


    def post_process_tools_description(self,
            list_tools_result: ListToolsResult) -> ListToolsResult:
        # Adapt the MCP tool result for Harmony
        for tool in list_tools_result.tools:
            tool.inputSchema = self.trim_schema(tool.inputSchema)

        # Some tools schema don't need to be part of the prompt (e.g. simple text in text out for Python)
        list_tools_result.tools = [
            tool for tool in list_tools_result.tools
            if getattr(tool.annotations, "include_in_prompt", True)
        ]
        return list_tools_result
    
    async def call_tool(self,tool_name, tool_args):
        print(tool_args)
        response = await self.session.call_tool(tool_name, tool_args)
        return response
    
    
    def get_namespace(self):

        tool_from_mcp = ToolNamespaceConfig(
            name=self.name,
            description=self.init_response.instructions,
            tools=self.get_harmony_tools())
        return tool_from_mcp

    def get_tools(self):
        return [t.inputSchema for t in self.tools.tools]
    
    def get_harmony_tools(self):
        list_tools_response = self.post_process_tools_description(self.tools)
        tools=[
                ToolDescription.new(name=tool.name,
                                    description=tool.description,
                                    parameters=tool.inputSchema)
                for tool in list_tools_response.tools
            ]
        return tools


class StdIOClient(MCPClient):
    def __init__(self, name, command: str, args:list[str] = [], env=None):
        # Initialize session and client objects

        super().__init__(name)
        self.server_params = StdioServerParameters(
            command=command,
            args=args,
            env=env
        )

    async def get_client(self):
        client = await self.exit_stack.enter_async_context(stdio_client(self.server_params))
        return client





class SSEClient(MCPClient):

    def __init__(self,
                 name,
                url: str,
            headers: dict[str, Any] | None = None,
            timeout: float = 5,
            sse_read_timeout: float = 60 * 5,
            auth: httpx.Auth | None = None,):

        super().__init__(name)
        self.url = url
        self.headers = headers
        self.timeout = timeout
        self.sse_read_timeout = sse_read_timeout
        self.auth = auth
    
    async def get_client(self):
        client = await self.exit_stack.enter_async_context(sse_client(self.url, self.headers, self.timeout, self.sse_read_timeout, self.auth))
        return client



