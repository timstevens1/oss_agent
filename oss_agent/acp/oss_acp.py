"""
Obsidian → Python → your OssAgent

* reads ACP JSON‑RPC messages from stdin
* extracts the prompt (the list of content blocks that the plugin sends)
* calls `OssAgent.generate(prompt_text)`
* sends the answer back as a `PromptResponse`

All other ACP messages (initialisation, new‑session, permissions, etc.) are
handled automatically by the SDK – we only need to implement the three
callbacks that the protocol requires.
"""

import asyncio
import sys
from pathlib import Path
from ..tools.tools import Tools
from ..utils import run_zsh_command
from uuid import uuid4

import logging
from typing import Any



logger = logging.getLogger()


# ---------------------------------------------------------------
#   1️⃣  Install the Python ACP SDK (only needed once)
# ---------------------------------------------------------------
# pip install agent-client-protocol

# ---------------------------------------------------------------
#   2️⃣  Import the SDK classes we need
# ---------------------------------------------------------------
from acp import (
    Agent,
    AgentSideConnection,
    AuthenticateRequest,
    AuthenticateResponse,
    CancelNotification,
    InitializeRequest,
    InitializeResponse,
    LoadSessionRequest,
    LoadSessionResponse,
    NewSessionRequest,
    NewSessionResponse,
    PromptRequest,
    PromptResponse,
    SetSessionModeRequest,
    SetSessionModeResponse,
    session_notification,
    stdio_streams,
    text_block,
    update_agent_message,
    PROTOCOL_VERSION,
)

from acp.schema import AgentCapabilities, AgentMessageChunk, Implementation

# ---------------------------------------------------------------
#   3️⃣  Import your local agent implementation
# ---------------------------------------------------------------
# The file `agent.py` must be in the same directory and define a class
# called `OssAgent` with a method `generate(self, prompt: str) -> str`.
from agent import OssAgent


class BridgeAgent(Agent):
    """Minimal object that the SDK asks to do the three lifecycle actions.

    The SDK gives us a `_conn` object that we use to send notifications back to
    the client (Obsidian).
    """

    def __init__(self, conn: AgentSideConnection):
        self._conn = conn
        self._oss = None
        self._next_session_id = 0
        self._sessions: set[str] = set()

    async def _send_agent_message(self, session_id: str, content: Any) -> None:
        update = content if isinstance(content, AgentMessageChunk) else update_agent_message(content)
        await self._conn.sessionUpdate(session_notification(session_id, update))


    # ------------------------------------------------------------------
    # 1️⃣  Initialise the protocol version – we just echo what the client sent.
    # ------------------------------------------------------------------
    async def initialize(self, params: InitializeRequest) -> InitializeResponse:
        return InitializeResponse(
            protocolVersion=PROTOCOL_VERSION,
            agentCapabilities=AgentCapabilities(),
            agentInfo=Implementation(name="example-agent", title="Example Agent", version="0.1.0"))

    # ------------------------------------------------------------------
    # 2️⃣  Create a new session – the client tells us the cwd, we return an ID.
    # ------------------------------------------------------------------
    async def newSession(self, params: NewSessionRequest) -> NewSessionResponse:  # noqa: ARG002
        logging.info("Received new session request")
        session_id = str(self._next_session_id)
        self._next_session_id += 1
        self._sessions.add(session_id)
        return NewSessionResponse(sessionId=session_id, modes=None)

    async def loadSession(self, params: LoadSessionRequest) -> LoadSessionResponse | None:  # noqa: ARG002
        logging.info("Received load session request %s", params.sessionId)
        self._sessions.add(params.sessionId)
        return LoadSessionResponse()

    async def setSessionMode(self, params: SetSessionModeRequest) -> SetSessionModeResponse | None:  # noqa: ARG002
        logging.info("Received set session mode request %s -> %s", params.sessionId, params.modeId)
        return SetSessionModeResponse()

    async def authenticate(self, params: AuthenticateRequest) -> AuthenticateResponse | None:  # noqa: ARG002
        logging.info("Received authenticate request %s", params.methodId)
        return AuthenticateResponse()
    # ------------------------------------------------------------------
    # 3️⃣  Handle the actual user prompt.
    # ------------------------------------------------------------------
    async def prompt(self, params: PromptRequest) -> PromptResponse:
        """Extract the user's text, ask the OssAgent, and send the answer.

        The `params.prompt` field is a list of *content blocks* – for a simple
        chat integration we just concatenate the `text` fields.
        """
        
        if params.sessionId not in self._sessions:
            self._sessions.add(params.sessionId)

        # ----- 1️⃣  Pull out plain‑text from the blocks -----------------
        if self._oss is None:
            tools = Tools(tools=[run_zsh_command], filename="/Users/timstevens/projects/qwen_agent/servers.json")

            await tools.init_mcp_connections()
            # Initialise your own agent once – this will be reused for every turn.
            self._oss = OssAgent(model=120, tools=tools)
        parts = []
        for block in params.prompt:
            # Most Obsidian messages are simple "text" blocks.
            parts.append(block.text)
        user_prompt = "\n".join(parts).strip()

        # ----- 2️⃣  Ask your local OssAgent ----------------------------
        reply = await self._oss(user_prompt)
        await self._send_agent_message(params.sessionId, text_block(reply))
        # Tell the client that the turn is finished.
        return PromptResponse(stopReason="end_turn")
    


async def main() -> None:
    # The SDK provides a helper that turns stdin/stdout into async streams.
    reader, writer = await stdio_streams()
    # The connection object will call our BridgeAgent for every incoming RPC.
    logger.debug('here')
    AgentSideConnection(
        BridgeAgent,   # factory that builds the handler
        writer,
        reader,
    )
    logger.debug('here2')
    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
