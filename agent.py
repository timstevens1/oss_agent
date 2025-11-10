from datetime import datetime


from openai_harmony import (SystemContent,
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
                            Author)



from pathlib import Path
import json
from tools.tools import Tools
from mlx_lm import load, stream_generate, generate

from mlx_lm.sample_utils import make_sampler


from typing import Union, List

import re

from model import BaseModel, OssModel
from rich.console import Console




developer_prompt = """you're a general purpose assistant. You have filesystem access, 
and internet search access. Your job is to answer user questions in an informed way and to
handle their requests such as for coding or summarization or text generation. Accuracy and brevity are
your top priorities. Verify your answers with the tools you have access to. Simply try to meet 
the user's expectations with your answer, and allow them to prompt you with
follow up queries if they would like more from you. For example:

Tools are provided to you grouped into namespaces. Make sure to use the correct namespace when calling the tool.
Make sure the message to the tool is a valid json dict containing the arguments you'd like to pass to the tool.
"""

GREY = "\33[90m"
BOLD = "\33[1m"
RESET = "\33[0m"
FAIL = '\033[91m'





def verbose(response):
    print()
    print("=" * 10)
    print(
    f"{FAIL}Prompt: {response.prompt_tokens} tokens, "
    f"{response.prompt_tps:.3f} tokens-per-sec"
    )
    print(
        f"Generation: {response.generation_tokens} tokens, "
        f"{response.generation_tps:.3f} tokens-per-sec"
    )
    print(f"Peak memory: {response.peak_memory:.3f} GB{RESET}")





class Agent:

    def __init__(self, model:BaseModel, 
                 developer_message: str=developer_prompt, 
                 tools: Tools=None,
                 refining_agent=None):


        self.model = model
        self.tools = tools
        self.encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        self.messages = []
        self.refining_agent = refining_agent
        developer_content = DeveloperContent.new()
        if developer_message:
            developer_content = developer_content.with_instructions(developer_message)
        if tools is not None:
            for n in self.tools.namespaces:
                developer_content = developer_content.with_tools(n)

        self.messages.append(Message.from_role_and_content(Role.DEVELOPER, developer_content))



    async def _invoke(self, message: dict, print_analysis=True):
        self.messages.append(message)
        new_messages = self.model.complete(self.messages, print_analysis)
        self.messages.extend(new_messages)
        
        while self.messages[-1].recipient is not None:
            yield self.messages[-1]
            new_messages = await self.tools.handle_tool_message(self.messages[-1])
            self.messages.extend(new_messages)
            yield message
            new_messages = self.model.complete(self.messages, print_analysis)
            self.messages.extend(new_messages)


        yield self.messages[-1]
			
		
        

    async def __call__(self, message: str, print_analysis=True):
        message = Message.from_role_and_content(Role.USER, message)
		
        async for r in self._invoke(message, print_analysis=print_analysis):
            continue
        return r.content[0].text
    
    
