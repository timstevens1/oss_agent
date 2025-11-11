# acp_content.py
"""Utility functions for handling Agent Client Protocol (ACP) content blocks.

The ACP defines a set of *ContentBlock* types that a client can send to an
agent.  This module provides a simple, LLM‑oriented handler that converts the
supported blocks into a text prompt that can be fed to a language model.

Supported block types (as defined by the protocol):
- ``text``          – plain‑text messages.
- ``image``         – image data (base‑64).  The LLM cannot process images, so we
                      return an informative placeholder.
- ``audio``         – audio data (base‑64).  Likewise unsupported for a plain LLM.
- ``resource``      – an embedded resource; may contain either a ``text`` field or a
                      binary ``blob``.  Text resources are extracted, blobs are not
                      supported.
- ``resource_link`` – a URI that points to an external resource.  This
                      implementation does not perform network/file I/O, so it reports
                      the link as unavailable.

Each function receives a dictionary representing a single content block (the JSON
object that would appear in an ACP message) and returns a string that can be
concatenated into a final prompt.  If a block cannot be processed, a helpful
message is returned instead of raising an exception – this keeps the overall
prompt generation robust.
"""

from __future__ import annotations
from typing import Dict, Any


def handle_text(block: Dict[str, Any]) -> str:
    """Return the plain‑text payload.

    Expected keys:
        - ``type`` (must be ``"text"``)
        - ``text`` (the message string)
    """
    return block.get("text", "")


def handle_image(block: Dict[str, Any]) -> str:
    """Return a placeholder for unsupported image content.

    The LLM cannot interpret raw image data, so we acknowledge the limitation.
    """
    mime = block.get("mimeType", "image")
    return f"[UNABLE TO PROCESS IMAGE ({mime})]"


def handle_audio(block: Dict[str, Any]) -> str:
    """Return a placeholder for unsupported audio content.

    The LLM cannot interpret audio data directly.
    """
    mime = block.get("mimeType", "audio")
    return f"[UNABLE TO PROCESS AUDIO ({mime})]"


def _extract_resource_text(resource: Dict[str, Any]) -> str:
    """Extract text from an embedded ``resource`` block.

    The ``resource`` field may contain either a ``text`` attribute (plain
    source code / document) or a ``blob`` attribute (binary data).  Only the
    text variant can be turned into a prompt.
    """
    if "text" in resource:
        return resource["text"]
    # ``blob`` is base‑64 binary; we cannot turn it into a useful prompt.
    return "[UNABLE TO PROCESS BINARY RESOURCE]"


def handle_resource(block: Dict[str, Any]) -> str:
    """Handle an embedded resource block.

    The block structure is ``{"type": "resource", "resource": {...}}``.  If the
    embedded resource contains a ``text`` field, we return that text; otherwise we
    return a placeholder indicating the content cannot be used.
    """
    resource = block.get("resource", {})
    return _extract_resource_text(resource)


def handle_resource_link(block: Dict[str, Any]) -> str:
    """Retrieve the resource at the given ``uri`` if possible and return its
    content for inclusion in the prompt.  Currently only ``file://`` URIs are
    supported – the function reads the referenced file as UTF-8 text.  For any
    other scheme (e.g., http/https) or on error, a placeholder message is
    returned.
    """
    uri = block.get("uri", "")
    name = block.get("name", "<unnamed>")
    # Simple handling for local file URIs
    if uri.startswith("file://"):
        path = uri[7:]
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            return f"[UNABLE TO READ FILE RESOURCE ({path}): {e}]"
    # Other URI schemes not supported in this simple implementation
    return f"[UNABLE TO FETCH RESOURCE LINK: {name} ({uri})]"


def dispatch_content(block: Dict[str, Any]) -> str:
    """Dispatch a content block to the appropriate handler.

    Parameters
    ----------
    block: dict
        The content‑block JSON object as received from an ACP message.

    Returns
    -------
    str
        The string that should be incorporated into the final LLM prompt.
    """
    block_type = block.get("type")
    if block_type == "text":
        return handle_text(block)
    if block_type == "image":
        return handle_image(block)
    if block_type == "audio":
        return handle_audio(block)
    if block_type == "resource":
        return handle_resource(block)
    if block_type == "resource_link":
        return handle_resource_link(block)
    # Fallback for unknown types
    return f"[UNSUPPORTED CONTENT TYPE: {block_type}]"


def build_prompt(blocks: list[Dict[str, Any]]) -> str:
    """Combine a list of content blocks into a single prompt string.

    Each block is passed through :func:`dispatch_content`.  The resulting
    strings are concatenated with a single space between them.
    """
    parts = [dispatch_content(b) for b in blocks]
    # Filter out empty strings (e.g., a text block with no ``text`` field)
    parts = [p for p in parts if p]
    return " ".join(parts)
