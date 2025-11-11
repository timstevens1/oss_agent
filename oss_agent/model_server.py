"""Simple WebSocket server for OSS GPT models using mlx.

The server loads an OSS model (size 20B or 120B) directly using ``mlx_lm.load``
and reproduces the minimal ``NoOpDetokenizer`` functionality that originally lived
in ``oss_agent/model.py``.  It exposes a single endpoint ``/generate/`` that accepts a
JSON payload containing an array of integer token IDs and returns a JSON payload with
the generated token IDs.

Running the module directly starts the server::

    python -m oss_agent.model_server --size 20 --port 8000 --timeout 300

The implementation relies on the ``websockets`` library which is lightweight and
compatible with the standard ``asyncio`` event loop.  All heavy MLX work is
performed asynchronously in the handler to keep the server responsive.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import List
import logging


logging.basicConfig(level=logging.INFO) 


import websockets

from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tokenizer_utils import TokenizerWrapper, StreamingDetokenizer


# ---------------------------------------------------------------------------
# Helper to run generation given a model instance.
# ---------------------------------------------------------------------------
def _generate_tokens(
    model, input_tokens: List[int], max_new_tokens: int = 128000, sampler=None
 ) -> List[int]:
    """Run ``stream_generate`` on *input_tokens* and return the generated token IDs.

    The function mirrors the ``generate`` helper in ``oss_agent/model.py`` but
    returns only the list of new token IDs (excluding the prompt tokens).

    *sampler* can be provided to override the model's default sampler – this is
    used to support per‑request temperature overrides.
    """
    generated: List[int] = []
    # Choose sampler: use provided override or the model's default.
    effective_sampler = sampler if sampler is not None else model.sampler
    # ``stream_generate`` yields ``TokenInfo`` objects with a ``token`` attribute.
    for info in stream_generate(
        model.backend,
        model.tokenizer,
        input_tokens,
        max_tokens=max_new_tokens,
        sampler=effective_sampler,
    ):
        generated.append(info.token)
    logging.info(
        f"Prompt: {info.prompt_tokens} tokens, "
        f"{info.prompt_tps:.3f} tokens-per-sec"
    )
    logging.info(
        f"Generation: {info.generation_tokens} tokens, "
        f"{info.generation_tps:.3f} tokens-per-sec"
    )
    logging.info(f"Peak memory: {info.peak_memory:.3f} GB")
    return generated



async def _handler(websocket, model, timeout: float):  # pragma: no cover
    """WebSocket handler for the ``/generate/`` endpoint.

    The client must send a JSON object ``{"tokens": [int, …]}``.  The server
    responds with ``{"tokens": [int, …]}`` containing the newly generated token
    IDs.  If no message is received within *timeout* seconds the connection is
    closed silently.
    """
    try:
        raw = await asyncio.wait_for(websocket.recv(), timeout=timeout)
    except asyncio.TimeoutError:
        await websocket.close()
        return
    request = json.loads(raw)
    prompt_tokens: List[int] = request.get("tokens", [])
    # Optional temperature override – "temp" may be omitted.
    temp = request.get("temp")
    # Determine sampler: if a temperature is supplied and differs from the model's
    # default, create a new sampler with that temperature.  ``make_sampler``
    # defaults to temperature 0 (greedy) when not overridden.
    if temp is not None:
        try:
            temp_val = float(temp)
        except Exception:
            temp_val = None
    else:
        temp_val = None

    # If a temperature value is provided, build a sampler; otherwise use the model's.
    sampler = None
    if temp_val is not None:
        sampler = make_sampler(temp=temp_val)

    # Run generation synchronously – the underlying MLX functions are CPU/GPU bound
    # and may be CPU‑intensive, but they are quick for the typical token limits.
    generated = _generate_tokens(model, prompt_tokens, sampler=sampler)
    response = json.dumps({"tokens": generated})
    await websocket.send(response)



def _parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WebSocket server for OSS GPT models")
    parser.add_argument(
        "--size",
        type=int,
        choices=[20, 120],
        default=20,
        help="Model size (20 or 120 billion parameters)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the WebSocket server",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Idle timeout (seconds) for a connection before closing",
    )
    return parser.parse_args(argv)



async def _serve(model, port: int, timeout: float):  # pragma: no cover
    # ``websockets.serve`` will invoke the handler for each new connection.
    async with websockets.serve(
        lambda ws : _handler(ws, model, timeout), "0.0.0.0", port
    ) as server:
        await server.wait_closed()



def main(argv: List[str] | None = None):  # pragma: no cover
    args = _parse_args(argv)
    # Load the model using the helper from ``oss_agent.model``.
    # Map model size to hub path (mirroring ``OssModel.from_size``)
    model_path_map = {
        20: "mlx-community/gpt-oss-20b-MXFP4-Q4",
        120: "mlx-community/gpt-oss-120b-MXFP4-Q4",
    }
    if args.size not in model_path_map:
        raise ValueError(f"Unsupported size {args.size}. Choose 20 or 120.")
    backend, tokenizer_raw = load(model_path_map[args.size])
    print("model_loaded")
    # Constants for end‑of‑sequence tokens (same values used in ``model.py``)
    END = 200007
    RETURN = 200002
    CALL = 200012
    # Define NoOpDetokenizer locally (copied from ``model.py``)
    class NoOpDetokenizer(StreamingDetokenizer):
        def __init__(self, tokenizer):
            pass

        def reset(self):
            pass

        def add_token(self, token):
            pass

        def finalize(self):
            pass

        @property
        def last_segment(self):
            return ""

    tokenizer = TokenizerWrapper(tokenizer_raw._tokenizer, NoOpDetokenizer, eos_token_ids={RETURN, CALL})
    sampler = make_sampler(temp=1.0)
    # Bundle needed objects into a simple namespace for the handler.
    class _ModelBundle:
        def __init__(self, backend, tokenizer, sampler):
            self.backend = backend
            self.tokenizer = tokenizer
            self.sampler = sampler

    model = _ModelBundle(backend, tokenizer, sampler)
    # Start the asynchronous server.
    try:
        logging.info(f"serving {args.size}b gpt-oss model on port {args.port}")
        asyncio.run(_serve(model, args.port, args.timeout))
    except KeyboardInterrupt:
        # Graceful shutdown on Ctrl‑C.
        sys.exit(0)



if __name__ == "__main__":  # pragma: no cover
    main()
