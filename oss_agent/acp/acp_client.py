import asyncio
import sys
from pathlib import Path

from acp import spawn_agent_process, text_block, PROTOCOL_VERSION
from acp.interfaces import Client
from acp.schema import InitializeRequest, NewSessionRequest, PromptRequest, SessionNotification


class SimpleClient(Client):
    async def requestPermission(self, params):  # pragma: no cover - minimal stub
        return {"outcome": {"outcome": "cancelled"}}

    async def sessionUpdate(self, params: SessionNotification) -> None:
        print("update:", params.sessionId, params.update)


async def main() -> None:
    script = Path("/Users/timstevens/projects/oss_agent/oss_acp.py")
    get_cli = lambda _agent: SimpleClient()
    async with spawn_agent_process(get_cli, sys.executable, str(script)) as (conn, _proc):
        print(_proc.pid)
        await conn.initialize(InitializeRequest(protocolVersion=PROTOCOL_VERSION))
        print('initted')
        session = await conn.newSession(NewSessionRequest(cwd=str(script.parent), mcpServers=[]))
        print('run')
        await conn.prompt(
            PromptRequest(
                sessionId=session.sessionId,
                prompt=[text_block("Respond back the word 'hello'")],
            )
        )
        print('hmm')

asyncio.run(main())