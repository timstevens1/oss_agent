import json
from typing import List, Optional
from .tools.tools import Tools


class Agent:
    def __init__(self, model, tools: Tools = None, developer_message=None):
        self.model = model
        self.tools = tools
        self.messages = [{'role': 'developer', 'content': developer_message}] if developer_message is not None else []

    async def __call__(self, text: str):
        self.messages.append({'role': 'user', 'content': text})
        # send messages list to model model.complete
        tools = True
        while tools:
            new_messages = await self.model.complete(conversation=self.messages, tools = self.tools.get_tools())
            # parse response messages
            for resp in new_messages:
                self.messages.append(resp.to_dict())
        # handle tool messages
            tools = False
            if new_messages[-1].tool_calls:
                tools = True
                for tool_call in new_messages[-1].tool_calls:
                    new_msgs = await self.tools.handle_tool_message(tool_call.to_dict())
                    self.messages.extend(new_msgs)
        return self.messages[-1]['content']
    


        
