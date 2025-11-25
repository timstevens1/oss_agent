
from openai_harmony import ToolDescription
import inspect
import asyncio

from .duck_backend import DuckBackend
from gpt_oss.tools.simple_browser.simple_browser_tool import SimpleBrowserTool

class Tool:
    def __init__(self, func, name='', description='', params={}):
        self.tool = func
        self.name = name
        self.description = description
        self.params = params
        self.include_in_prompt = True

        if not self.name or not self.description:
            self.include_in_prompt = False


    def get_config(self):
        return {'type': 'function', 'function': {'name': self.name, 'description': self.description, 'parameters': self.params}}

    def get_harmony_description(self):
        return ToolDescription(self.name, self.description, self.params)
    

    @staticmethod
    def from_function(f):
        inspection = inspect.getfullargspec(f)
        args = inspection.args
        param_dict = {}
        required = []
        signature = inspect.signature(f)
        for a in args:
            param = signature.parameters[a]
            name, dic, req =  Tool.format_param(param)
            param_dict[name] = dic
            if req:
                required.append(name)
        parameters={
            "type": "object",
            "properties": param_dict,
            "required": required,
        }
        
        return Tool(f, f.__name__, f.__doc__, parameters)
    
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
    
    
    async def __call__(self, **args):
        if inspect.iscoroutinefunction(self.tool):
            result = await self.tool(**args)
        else:
            result = await asyncio.to_thread(self.tool, **args)

        return [{'role': 'tool', 'content': str(result)}]
    

class BrowserTool(Tool):

    def __init__(self):
        super().__init__(self.__call__)
        self.base_tool = SimpleBrowserTool(DuckBackend(source=""))

    async def search(self, **args):
        messages = []
        async for m in self.base_tool.search(**args):
            messages.append(m.to_dict())
        return messages

    async def find(self, **args):
        messages = []
        async for m in self.base_tool.find(**args):
            messages.append(m.to_dict())
        return messages

    async def open(self, **args):
        messages = []
        async for m in self.base_tool.open(**args):
            messages.append(m.to_dict())
        return messages  
    
    async def __call__(self, **args):

        if 'query' in args.keys():
            return await self.search(**args)
        
        if 'pattern' in args.keys():
            return await self.find(**args)
        
        else:
            return await self.open(**args)
    

    