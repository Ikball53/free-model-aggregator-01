#!/usr/bin/env python3
"""Free-model aggregator: one server, two API wires, ten free providers.

Reads the trial brief from MAKEFUN_PROMPT_FILE, probes every provider's
model catalog, registers port plus catalog with the env, then answers the
OpenAI Chat Completions API and the Anthropic Messages API (JSON and SSE)
by routing each request to the best free model that meets its requirements,
failing over to the next provider when one refuses.

Standard library only. Every routing choice is a function of the brief, the
request and the providers' answers: no wall clock, no randomness.
"""

import base64
import copy
import hashlib
import json
import os
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

FIXED_CREATED = 1767225600
CHUNK_CHARS = 13
PROVIDER_TIMEOUT = 5.0
MAX_REFUSALS = 3
WORK_POLL_SECONDS = 0.05


def is_rec(value):
    return isinstance(value, dict)


def blocks_text(content):
    """Text of an OpenAI/Anthropic content value, text blocks joined by newline."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if is_rec(block) and block.get("type") == "text" and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts)
    return ""


def gemini_parts_text(parts):
    if not isinstance(parts, list):
        return ""
    texts = []
    for part in parts:
        if is_rec(part) and isinstance(part.get("text"), str):
            texts.append(part["text"])
    return "\n".join(texts)


def answer_text(content):
    """Text of an Anthropic content list, text blocks joined with no separator."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if is_rec(block) and block.get("type") == "text" and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "".join(parts)


def sha8_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


def sha8_bytes(raw):
    return hashlib.sha256(raw).hexdigest()[:8]


def image_sha(url):
    """First 8 hex chars of the sha256 of an image URL's bytes (env's rule)."""
    if not isinstance(url, str) or url == "":
        return None
    match = re.match(r"^data:[^,]*;base64,(.*)$", url, re.S)
    if match is not None:
        try:
            return sha8_bytes(base64.b64decode(match.group(1)))
        except Exception:
            return sha8_text(url)
    return sha8_text(url)


def data_url_parts(url):
    """Split a data: URL into (mime_type, base64 payload), or (None, None)."""
    if not isinstance(url, str):
        return None, None
    match = re.match(r"^data:([^;,]+)?;base64,(.*)$", url, re.S)
    if match is None:
        return None, None
    return (match.group(1) or "image/png"), match.group(2)


# ---------------------------------------------------------------------------
# Brief parsing
# ---------------------------------------------------------------------------

def scan_json_docs(text):
    """Yield every top-level JSON object embedded in the brief text."""
    docs = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        depth = 0
        j = i
        in_str = False
        esc = False
        found = False
        while j < n:
            ch = text[j]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            docs.append(json.loads(text[i:j + 1]))
                        except Exception:
                            pass
                        found = True
                        break
            j += 1
        i = (j + 1) if found else (i + 1)
    return docs


def load_brief(path):
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    providers = None
    ratings = None
    manifest = None
    for doc in scan_json_docs(text):
        if not is_rec(doc):
            continue
        schema = doc.get("schema", "")
        if schema == "free-model-aggregator-01/providers/v1":
            providers = doc.get("providers")
        elif schema == "free-model-aggregator-01/ratings/v1":
            ratings = doc.get("ratings")
        elif schema == "free-model-aggregator-01/workload/v1":
            manifest = doc.get("manifest")
    if providers is None:
        raise ValueError("brief holds no provider directory")
    return providers, ratings or [], manifest or []


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def http_request(method, url, headers=None, body_obj=None, timeout=PROVIDER_TIMEOUT):
    """One HTTP call. Returns (status or None, raw bytes)."""
    data = None
    hdrs = dict(headers or {})
    if body_obj is not None:
        data = json.dumps(body_obj).encode("utf-8")
        hdrs.setdefault("Content-Type", "application/json")
    req = Request(url, data=data, method=method, headers=hdrs)
    try:
        with urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except HTTPError as err:
        try:
            return err.code, err.read()
        except Exception:
            return err.code, b""
    except (URLError, TimeoutError, OSError, ValueError):
        return None, b""


def provider_headers(provider):
    auth = provider.get("auth")
    key = provider.get("key", "")
    if auth == "bearer":
        return {"Authorization": "Bearer " + key}
    if auth == "x-api-key":
        return {"x-api-key": key}
    if auth == "x-goog-api-key":
        return {"x-goog-api-key": key}
    return {}


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------

def ratings_models(ratings, provider_id):
    out = []
    for row in ratings:
        if row.get("provider") == provider_id:
            out.append({
                "id": row["model"],
                "free": bool(row.get("free")),
                "context_length": row.get("context_length"),
            })
    return out


def probe_models(env_url, provider):
    """Read one provider's model list; None when the probe fails."""
    base = env_url + provider["path"]
    headers = provider_headers(provider)
    try:
        if provider["wire"] == "gemini":
            status, raw = http_request("GET", base + "/v1beta/models", headers)
            if status != 200:
                return None
            doc = json.loads(raw.decode("utf-8"))
            models = []
            for item in doc.get("models", []):
                name = item.get("name", "")
                if name.startswith("models/"):
                    name = name[len("models/"):]
                models.append({
                    "id": name,
                    "free": bool(item.get("free")),
                    "context_length": item.get("context_length"),
                })
            return models
        status, raw = http_request("GET", base + "/v1/models", headers)
        if status != 200:
            return None
        doc = json.loads(raw.decode("utf-8"))
        models = []
        for item in doc.get("data", []):
            models.append({
                "id": item.get("id"),
                "free": bool(item.get("free")),
                "context_length": item.get("context_length"),
            })
        return models
    except Exception:
        return None


def build_catalog(env_url, providers, ratings):
    entries = []
    for provider in providers:
        models = probe_models(env_url, provider)
        if models is None:
            models = ratings_models(ratings, provider["id"])
        entries.append({
            "id": provider["id"],
            "base_url": provider["path"],
            "key_env": provider.get("key_env", "") or "",
            "wire": provider["wire"],
            "models": models,
        })
    return {"providers": entries}


# ---------------------------------------------------------------------------
# Provider-wire parsers (mirror the env's parse, for the usage formula)
# ---------------------------------------------------------------------------

def string_list(value):
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    return []


def number_or_none(value):
    return value if isinstance(value, (int, float)) and value == value else None


def parse_openai(body):
    system_parts = []
    user_text = ""
    image = None
    messages = body.get("messages") if isinstance(body.get("messages"), list) else []
    for message in messages:
        if not is_rec(message):
            continue
        if message.get("role") == "system":
            text = blocks_text(message.get("content"))
            if text != "":
                system_parts.append(text)
        elif message.get("role") == "user":
            user_text = blocks_text(message.get("content"))
            if isinstance(message.get("content"), list):
                for block in message["content"]:
                    if is_rec(block) and block.get("type") == "image_url" and is_rec(block.get("image_url")):
                        found = image_sha(block["image_url"].get("url"))
                        if found is not None:
                            image = found
    tools = []
    if isinstance(body.get("tools"), list):
        for tool in body["tools"]:
            if is_rec(tool) and is_rec(tool.get("function")) and isinstance(tool["function"].get("name"), str):
                tools.append(tool["function"]["name"])
    fmt = body.get("response_format") if is_rec(body.get("response_format")) else {}
    max_tokens = number_or_none(body.get("max_tokens"))
    if max_tokens is None:
        max_tokens = number_or_none(body.get("max_completion_tokens"))
    return {
        "system_text": "\n".join(system_parts),
        "user_text": user_text,
        "tools": tools,
        "max_tokens": max_tokens,
        "stops": string_list(body.get("stop")),
        "image_sha": image,
        "json_mode": fmt.get("type") in ("json_object", "json_schema"),
    }


def parse_anthropic(body):
    system_text = blocks_text(body.get("system"))
    user_text = ""
    image = None
    messages = body.get("messages") if isinstance(body.get("messages"), list) else []
    for message in messages:
        if not is_rec(message) or message.get("role") != "user":
            continue
        user_text = blocks_text(message.get("content"))
        if isinstance(message.get("content"), list):
            for block in message["content"]:
                if not is_rec(block) or block.get("type") != "image" or not is_rec(block.get("source")):
                    continue
                source = block["source"]
                if source.get("type") == "base64" and isinstance(source.get("data"), str):
                    try:
                        image = sha8_bytes(base64.b64decode(source["data"]))
                    except Exception:
                        pass
                elif source.get("type") == "url":
                    found = image_sha(source.get("url"))
                    if found is not None:
                        image = found
    tools = []
    if isinstance(body.get("tools"), list):
        for tool in body["tools"]:
            if is_rec(tool) and isinstance(tool.get("name"), str):
                tools.append(tool["name"])
    return {
        "system_text": system_text,
        "user_text": user_text,
        "tools": tools,
        "max_tokens": number_or_none(body.get("max_tokens")),
        "stops": string_list(body.get("stop_sequences")),
        "image_sha": image,
        "json_mode": False,
    }


def parse_gemini(body, model=""):
    system_text = ""
    instruction = body.get("systemInstruction")
    if is_rec(instruction):
        system_text = gemini_parts_text(instruction.get("parts"))
    user_text = ""
    image = None
    contents = body.get("contents") if isinstance(body.get("contents"), list) else []
    for content in contents:
        if not is_rec(content):
            continue
        if content.get("role") == "model":
            continue
        user_text = gemini_parts_text(content.get("parts"))
        if isinstance(content.get("parts"), list):
            for part in content["parts"]:
                if not is_rec(part):
                    continue
                if is_rec(part.get("inlineData")) and isinstance(part["inlineData"].get("data"), str):
                    try:
                        image = sha8_bytes(base64.b64decode(part["inlineData"]["data"]))
                    except Exception:
                        pass
                elif is_rec(part.get("fileData")) and isinstance(part["fileData"].get("fileUri"), str):
                    found = image_sha(part["fileData"]["fileUri"])
                    if found is not None:
                        image = found
    tools = []
    if isinstance(body.get("tools"), list):
        for tool in body["tools"]:
            if is_rec(tool) and isinstance(tool.get("functionDeclarations"), list):
                for fn in tool["functionDeclarations"]:
                    if is_rec(fn) and isinstance(fn.get("name"), str):
                        tools.append(fn["name"])
    config = body.get("generationConfig") if is_rec(body.get("generationConfig")) else {}
    return {
        "system_text": system_text,
        "user_text": user_text,
        "tools": tools,
        "max_tokens": number_or_none(config.get("maxOutputTokens")),
        "stops": string_list(config.get("stopSequences")),
        "image_sha": image,
        "json_mode": config.get("responseMimeType") == "application/json",
        "model": model,
    }


def parse_for_wire(wire, body, model=""):
    if wire == "openai":
        return parse_openai(body)
    if wire == "anthropic":
        return parse_anthropic(body)
    return parse_gemini(body, model)


def usage_of(parsed, answer_text):
    """The provider's token formula over the forwarded body and answer."""
    prompt_len = len(parsed["system_text"]) + len(parsed["user_text"]) + len(",".join(parsed["tools"]))
    return 1 + prompt_len // 4, 1 + len(answer_text) // 4


# ---------------------------------------------------------------------------
# Client -> provider translation (every echoed field preserved exactly)
# ---------------------------------------------------------------------------

def oai_content_to_ant(content):
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return content
    out = []
    for block in content:
        if not is_rec(block):
            continue
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            out.append({"type": "text", "text": block["text"]})
        elif block.get("type") == "image_url" and is_rec(block.get("image_url")):
            url = block["image_url"].get("url")
            mime, data = data_url_parts(url)
            if data is not None:
                out.append({"type": "image", "source": {"type": "base64", "media_type": mime, "data": data}})
            elif isinstance(url, str):
                out.append({"type": "image", "source": {"type": "url", "url": url}})
        else:
            out.append(copy.deepcopy(block))
    return out


def ant_content_to_oai(content):
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return content
    out = []
    for block in content:
        if not is_rec(block):
            continue
        if block.get("type") == "text":
            out.append({"type": "text", "text": block.get("text", "")})
        elif block.get("type") == "image" and is_rec(block.get("source")):
            source = block["source"]
            if source.get("type") == "base64":
                media = source.get("media_type", "image/png")
                out.append({"type": "image_url", "image_url": {"url": "data:%s;base64,%s" % (media, source.get("data", ""))}})
            elif source.get("type") == "url":
                out.append({"type": "image_url", "image_url": {"url": source.get("url", "")}})
        else:
            out.append(copy.deepcopy(block))
    return out


def oai_content_to_gemini_parts(content):
    if isinstance(content, str):
        return [{"text": content}]
    if not isinstance(content, list):
        return []
    parts = []
    for block in content:
        if not is_rec(block):
            continue
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            parts.append({"text": block["text"]})
        elif block.get("type") == "image_url" and is_rec(block.get("image_url")):
            url = block["image_url"].get("url")
            mime, data = data_url_parts(url)
            if data is not None:
                parts.append({"inlineData": {"mimeType": mime, "data": data}})
            elif isinstance(url, str):
                parts.append({"fileData": {"fileUri": url}})
    return parts


def ant_content_to_gemini_parts(content):
    if isinstance(content, str):
        return [{"text": content}]
    if not isinstance(content, list):
        return []
    parts = []
    for block in content:
        if not is_rec(block):
            continue
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            parts.append({"text": block["text"]})
        elif block.get("type") == "image" and is_rec(block.get("source")):
            source = block["source"]
            if source.get("type") == "base64":
                parts.append({"inlineData": {"mimeType": source.get("media_type", "image/png"), "data": source.get("data", "")}})
            elif source.get("type") == "url":
                parts.append({"fileData": {"fileUri": source.get("url", "")}})
    return parts


def to_anthropic_body(body, model):
    """OpenAI client body -> Anthropic provider body (no response_format)."""
    out = {"model": model}
    system_texts = []
    messages = []
    for message in body.get("messages", []) or []:
        if not is_rec(message):
            continue
        if message.get("role") == "system":
            text = blocks_text(message.get("content"))
            if text:
                system_texts.append(text)
        else:
            item = dict(message)
            item["content"] = oai_content_to_ant(message.get("content"))
            messages.append(item)
    if system_texts:
        out["system"] = "\n".join(system_texts)
    out["messages"] = messages
    tools = []
    for tool in body.get("tools", []) or []:
        if is_rec(tool) and is_rec(tool.get("function")) and isinstance(tool["function"].get("name"), str):
            fn = tool["function"]
            entry = {"name": fn["name"]}
            if "description" in fn:
                entry["description"] = fn["description"]
            if "parameters" in fn:
                entry["input_schema"] = fn["parameters"]
            tools.append(entry)
    if tools:
        out["tools"] = tools
    max_tokens = number_or_none(body.get("max_tokens"))
    if max_tokens is None:
        max_tokens = number_or_none(body.get("max_completion_tokens"))
    if max_tokens is not None:
        out["max_tokens"] = max_tokens
    stops = string_list(body.get("stop"))
    if stops:
        out["stop_sequences"] = stops
    if isinstance(body.get("temperature"), (int, float)):
        out["temperature"] = body["temperature"]
    if body.get("stream") is True:
        out["stream"] = True
    return out


def to_openai_body(body, model):
    """Anthropic client body -> OpenAI provider body (never adds response_format)."""
    out = {"model": model}
    messages = []
    system_text = blocks_text(body.get("system"))
    if system_text:
        messages.append({"role": "system", "content": system_text})
    for message in body.get("messages", []) or []:
        if not is_rec(message):
            continue
        item = dict(message)
        item["content"] = ant_content_to_oai(message.get("content"))
        messages.append(item)
    out["messages"] = messages
    tools = []
    for tool in body.get("tools", []) or []:
        if is_rec(tool) and isinstance(tool.get("name"), str):
            fn = {"name": tool["name"]}
            if "description" in tool:
                fn["description"] = tool["description"]
            if "input_schema" in tool:
                fn["parameters"] = tool["input_schema"]
            tools.append({"type": "function", "function": fn})
    if tools:
        out["tools"] = tools
    if isinstance(body.get("max_tokens"), (int, float)):
        out["max_tokens"] = body["max_tokens"]
    stops = string_list(body.get("stop_sequences"))
    if stops:
        out["stop"] = stops
    if isinstance(body.get("temperature"), (int, float)):
        out["temperature"] = body["temperature"]
    if body.get("stream") is True:
        out["stream"] = True
    return out


def to_gemini_body(body, client_api):
    """OpenAI or Anthropic client body -> Gemini provider body."""
    out = {}
    if client_api == "openai":
        system_texts = []
        contents = []
        for message in body.get("messages", []) or []:
            if not is_rec(message):
                continue
            if message.get("role") == "system":
                text = blocks_text(message.get("content"))
                if text:
                    system_texts.append(text)
            else:
                role = "model" if message.get("role") == "assistant" else "user"
                contents.append({"role": role, "parts": oai_content_to_gemini_parts(message.get("content"))})
        if system_texts:
            out["systemInstruction"] = {"parts": [{"text": "\n".join(system_texts)}]}
        out["contents"] = contents
        declarations = []
        for tool in body.get("tools", []) or []:
            if is_rec(tool) and is_rec(tool.get("function")) and isinstance(tool["function"].get("name"), str):
                fn = tool["function"]
                entry = {"name": fn["name"]}
                if "description" in fn:
                    entry["description"] = fn["description"]
                if "parameters" in fn:
                    entry["parameters"] = fn["parameters"]
                declarations.append(entry)
        if declarations:
            out["tools"] = [{"functionDeclarations": declarations}]
        config = {}
        if isinstance(body.get("temperature"), (int, float)):
            config["temperature"] = body["temperature"]
        max_tokens = number_or_none(body.get("max_tokens"))
        if max_tokens is None:
            max_tokens = number_or_none(body.get("max_completion_tokens"))
        if max_tokens is not None:
            config["maxOutputTokens"] = max_tokens
        stops = string_list(body.get("stop"))
        if stops:
            config["stopSequences"] = stops
        fmt = body.get("response_format") if is_rec(body.get("response_format")) else {}
        if fmt.get("type") in ("json_object", "json_schema"):
            config["responseMimeType"] = "application/json"
        if config:
            out["generationConfig"] = config
    else:
        system_text = blocks_text(body.get("system"))
        if system_text:
            out["systemInstruction"] = {"parts": [{"text": system_text}]}
        contents = []
        for message in body.get("messages", []) or []:
            if not is_rec(message):
                continue
            role = "model" if message.get("role") == "assistant" else "user"
            contents.append({"role": role, "parts": ant_content_to_gemini_parts(message.get("content"))})
        out["contents"] = contents
        declarations = []
        for tool in body.get("tools", []) or []:
            if is_rec(tool) and isinstance(tool.get("name"), str):
                entry = {"name": tool["name"]}
                if "description" in tool:
                    entry["description"] = tool["description"]
                if "input_schema" in tool:
                    entry["parameters"] = tool["input_schema"]
                declarations.append(entry)
        if declarations:
            out["tools"] = [{"functionDeclarations": declarations}]
        config = {}
        if isinstance(body.get("temperature"), (int, float)):
            config["temperature"] = body["temperature"]
        if isinstance(body.get("max_tokens"), (int, float)):
            config["maxOutputTokens"] = body["max_tokens"]
        stops = string_list(body.get("stop_sequences"))
        if stops:
            config["stopSequences"] = stops
        if config:
            out["generationConfig"] = config
    return out


# ---------------------------------------------------------------------------
# Provider response extraction
# ---------------------------------------------------------------------------

def sse_frames(raw):
    frames = []
    text = raw.decode("utf-8", "replace").replace("\r\n", "\n")
    for chunk in text.split("\n\n"):
        if chunk.strip() == "":
            continue
        event = None
        data = []
        for line in chunk.split("\n"):
            if line.startswith("event:"):
                event = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data.append(line[len("data:"):].lstrip(" "))
        if data:
            frames.append((event, "\n".join(data)))
    return frames


def extract_openai_stream(raw):
    text = ""
    for _event, data in sse_frames(raw):
        if data.strip() == "[DONE]":
            continue
        try:
            doc = json.loads(data)
        except Exception:
            continue
        for choice in doc.get("choices", []) or []:
            delta = choice.get("delta") or {}
            if isinstance(delta.get("content"), str):
                text += delta["content"]
    return text


def extract_anthropic_stream(raw):
    text = ""
    for event, data in sse_frames(raw):
        if event != "content_block_delta":
            continue
        try:
            doc = json.loads(data)
        except Exception:
            continue
        delta = doc.get("delta") or {}
        if delta.get("type") == "text_delta" and isinstance(delta.get("text"), str):
            text += delta["text"]
    return text


def extract_gemini_stream(raw):
    text = ""
    for _event, data in sse_frames(raw):
        try:
            doc = json.loads(data)
        except Exception:
            continue
        for candidate in doc.get("candidates", []) or []:
            content = candidate.get("content") or {}
            for part in content.get("parts", []) or []:
                if isinstance(part.get("text"), str):
                    text += part["text"]
    return text


def openai_message_text(doc):
    choices = doc.get("choices", []) or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    return answer_text(content)


def gemini_answer_text(doc):
    candidates = doc.get("candidates", []) or []
    if not candidates:
        return ""
    content = candidates[0].get("content") or {}
    parts = content.get("parts", []) or []
    return "".join(part.get("text", "") for part in parts if isinstance(part.get("text"), str))


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

class State:
    def __init__(self):
        self.env_url = ""
        self.providers = []
        self.by_id = {}
        self.ratings = []
        self.manifest = []
        self.manifest_by_id = {}
        self.catalog = {"providers": []}
        self.refusals = {}
        self.request_index = 0
        self.lock = threading.Lock()


STATE = State()


# ---------------------------------------------------------------------------
# Provider calls
# ---------------------------------------------------------------------------

def call_provider(provider, model, client_api, body, stream):
    """One attempt at one provider. Returns (answer_text, forwarded_body) or None."""
    wire = provider["wire"]
    base = STATE.env_url + provider["path"]
    headers = provider_headers(provider)
    try:
        if wire == "openai":
            if client_api == "anthropic":
                forwarded = to_openai_body(body, model)
            else:
                forwarded = copy.deepcopy(body)
                forwarded["model"] = model
            if stream:
                forwarded["stream"] = True
                forwarded["stream_options"] = {"include_usage": True}
            status, raw = http_request("POST", base + "/v1/chat/completions", headers, forwarded)
            if status != 200:
                return None
            if stream:
                return extract_openai_stream(raw), forwarded
            return openai_message_text(json.loads(raw.decode("utf-8"))), forwarded
        if wire == "anthropic":
            if client_api == "openai":
                forwarded = to_anthropic_body(body, model)
            else:
                forwarded = copy.deepcopy(body)
                forwarded["model"] = model
            hdrs = dict(headers)
            hdrs["anthropic-version"] = "2023-06-01"
            status, raw = http_request("POST", base + "/v1/messages", hdrs, forwarded)
            if status != 200:
                return None
            if stream:
                return extract_anthropic_stream(raw), forwarded
            return answer_text(json.loads(raw.decode("utf-8")).get("content")), forwarded
        forwarded = to_gemini_body(body, client_api)
        action = "streamGenerateContent" if stream else "generateContent"
        url = base + "/v1beta/models/" + quote(model, safe="") + ":" + action
        if stream:
            url += "?alt=sse"
        status, raw = http_request("POST", url, headers, forwarded)
        if status != 200:
            return None
        if stream:
            return extract_gemini_stream(raw), forwarded
        return gemini_answer_text(json.loads(raw.decode("utf-8"))), forwarded
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def candidates_for(requires, client_api, json_needed):
    """Free models meeting the requirements, best rating first.

    Ties prefer the provider whose wire matches the client's wire, then a
    fixed provider/model order so the choice is deterministic.
    """
    scored = []
    for row in STATE.ratings:
        if not row.get("free"):
            continue
        if requires.get("tools") and not row.get("tools"):
            continue
        if requires.get("vision") and not row.get("vision"):
            continue
        if requires.get("json") and not row.get("json"):
            continue
        min_context = requires.get("min_context")
        if min_context is not None and (row.get("context_length") or 0) < min_context:
            continue
        provider = STATE.by_id.get(row.get("provider"))
        if provider is None:
            continue
        if json_needed and client_api == "openai" and provider["wire"] == "anthropic":
            continue
        same_wire = 0 if provider["wire"] == client_api else 1
        scored.append((-(row.get("rating_milli") or 0), same_wire, provider["id"], row["model"], provider, row["model"]))
    scored.sort()
    return [(item[4], item[5]) for item in scored]


def fetch_work():
    try:
        status, raw = http_request("GET", STATE.env_url + "/work", timeout=5.0)
        if status != 200:
            return None
        doc = json.loads(raw.decode("utf-8"))
        return doc if is_rec(doc) else None
    except Exception:
        return None


def infer_requires(client_api, body):
    requires = {}
    tools = body.get("tools")
    if isinstance(tools, list) and tools:
        requires["tools"] = True
    if client_api == "openai":
        for message in body.get("messages", []) or []:
            if not is_rec(message) or not isinstance(message.get("content"), list):
                continue
            for block in message["content"]:
                if is_rec(block) and block.get("type") == "image_url":
                    requires["vision"] = True
        fmt = body.get("response_format")
        if is_rec(fmt) and fmt.get("type") in ("json_object", "json_schema"):
            requires["json"] = True
    else:
        for message in body.get("messages", []) or []:
            if not is_rec(message) or not isinstance(message.get("content"), list):
                continue
            for block in message["content"]:
                if is_rec(block) and block.get("type") == "image":
                    requires["vision"] = True
    return requires


def resolve_requires(client_api, body):
    """Requirements for this request: /work's next id, else manifest order."""
    work = fetch_work()
    if work is not None:
        next_id = work.get("next")
        if isinstance(next_id, str) and next_id in STATE.manifest_by_id:
            return dict(STATE.manifest_by_id[next_id].get("requires") or {})
    if STATE.request_index < len(STATE.manifest):
        return dict(STATE.manifest[STATE.request_index].get("requires") or {})
    return infer_requires(client_api, body)


def route_and_answer(client_api, body, requires):
    stream = body.get("stream") is True
    json_needed = (
        client_api == "openai"
        and is_rec(body.get("response_format"))
        and body["response_format"].get("type") in ("json_object", "json_schema")
    )
    tried = set()
    for provider, model in candidates_for(requires, client_api, json_needed):
        if provider["id"] in tried:
            continue
        tried.add(provider["id"])
        if STATE.refusals.get(provider["id"], 0) >= MAX_REFUSALS:
            continue
        got = call_provider(provider, model, client_api, body, stream)
        if got is None:
            STATE.refusals[provider["id"]] = STATE.refusals.get(provider["id"], 0) + 1
            continue
        text, forwarded = got
        STATE.refusals[provider["id"]] = 0
        sys.stderr.write("served via %s/%s\n" % (provider["id"], model))
        return client_response(client_api, body, model, text, forwarded, provider["wire"])
    return error_response(client_api)


def handle_client(client_api, body):
    with STATE.lock:
        requires = resolve_requires(client_api, body)
        answered = route_and_answer(client_api, body, requires)
        STATE.request_index += 1
        return answered


# ---------------------------------------------------------------------------
# Client-wire responses
# ---------------------------------------------------------------------------

def split_pieces(text):
    if text == "":
        return [""]
    return [text[i:i + CHUNK_CHARS] for i in range(0, len(text), CHUNK_CHARS)]


def sse_data(payload):
    return "data: " + json.dumps(payload) + "\n\n"


def sse_event(name, payload):
    return "event: " + name + "\ndata: " + json.dumps(payload) + "\n\n"


def client_response(client_api, body, model, text, forwarded, provider_wire):
    parsed = parse_for_wire(provider_wire, forwarded, model)
    input_tokens, output_tokens = usage_of(parsed, text)
    stream = body.get("stream") is True
    if client_api == "openai":
        if not stream:
            payload = {
                "id": "chatcmpl-sim",
                "object": "chat.completion",
                "created": FIXED_CREATED,
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                    "logprobs": None,
                }],
                "usage": {
                    "prompt_tokens": input_tokens,
                    "completion_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                },
            }
            return 200, "application/json", json.dumps(payload).encode("utf-8")
        frames = []
        base_chunk = {"id": "chatcmpl-sim", "object": "chat.completion.chunk", "created": FIXED_CREATED, "model": model}
        first = dict(base_chunk)
        first["choices"] = [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]
        frames.append(sse_data(first))
        for piece in split_pieces(text):
            chunk = dict(base_chunk)
            chunk["choices"] = [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]
            frames.append(sse_data(chunk))
        last = dict(base_chunk)
        last["choices"] = [{"index": 0, "delta": {}, "finish_reason": "stop"}]
        frames.append(sse_data(last))
        usage_chunk = dict(base_chunk)
        usage_chunk["choices"] = []
        usage_chunk["usage"] = {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }
        frames.append(sse_data(usage_chunk))
        frames.append("data: [DONE]\n\n")
        return 200, "text/event-stream; charset=utf-8", "".join(frames).encode("utf-8")
    if not stream:
        payload = {
            "id": "msg_sim",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
        }
        return 200, "application/json", json.dumps(payload).encode("utf-8")
    frames = []
    frames.append(sse_event("message_start", {
        "type": "message_start",
        "message": {
            "id": "msg_sim", "type": "message", "role": "assistant", "model": model,
            "content": [], "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": input_tokens, "output_tokens": 0},
        },
    }))
    frames.append(sse_event("content_block_start", {
        "type": "content_block_start", "index": 0,
        "content_block": {"type": "text", "text": ""},
    }))
    for piece in split_pieces(text):
        frames.append(sse_event("content_block_delta", {
            "type": "content_block_delta", "index": 0,
            "delta": {"type": "text_delta", "text": piece},
        }))
    frames.append(sse_event("content_block_stop", {"type": "content_block_stop", "index": 0}))
    frames.append(sse_event("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": output_tokens},
    }))
    frames.append(sse_event("message_stop", {"type": "message_stop"}))
    return 200, "text/event-stream; charset=utf-8", "".join(frames).encode("utf-8")


def error_response(client_api):
    if client_api == "openai":
        payload = {"error": {"message": "all providers refused the request", "type": "api_error", "code": "all_providers_failed", "param": None}}
    else:
        payload = {"type": "error", "error": {"type": "api_error", "message": "all providers refused the request"}}
    return 502, "application/json", json.dumps(payload).encode("utf-8")


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------

class AggregatorServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class AggregatorHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send_bytes(self, status, payload, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        path = urlparse(self.path).path
        if path != "/v1/models":
            self.send_bytes(404, b'{"error":"not_found"}', "application/json")
            return
        free_ids = []
        seen = set()
        for provider in STATE.catalog.get("providers", []):
            for model in provider.get("models", []) or []:
                if model.get("free") is True and isinstance(model.get("id"), str) and model["id"] not in seen:
                    seen.add(model["id"])
                    free_ids.append(model["id"])
        if self.headers.get("x-api-key") is not None:
            payload = {
                "data": [{"type": "model", "id": mid, "display_name": mid, "created_at": "2026-01-01T00:00:00Z", "free": True} for mid in free_ids],
                "has_more": False,
            }
        else:
            payload = {
                "object": "list",
                "data": [{"id": mid, "object": "model", "created": FIXED_CREATED, "owned_by": "free-model-aggregator", "free": True} for mid in free_ids],
            }
        self.send_bytes(200, json.dumps(payload).encode("utf-8"), "application/json")

    def do_POST(self):
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length > 0 else b""
        try:
            body = json.loads(raw.decode("utf-8")) if raw else {}
        except Exception:
            body = {}
        if not is_rec(body):
            body = {}
        if path == "/v1/chat/completions":
            status, content_type, payload = handle_client("openai", body)
        elif path == "/v1/messages":
            status, content_type, payload = handle_client("anthropic", body)
        else:
            self.send_bytes(404, b'{"error":"not_found"}', "application/json")
            return
        self.send_bytes(status, payload, content_type)


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def run_list():
    prompt_file = os.environ.get("MAKEFUN_PROMPT_FILE")
    if not prompt_file:
        print('{"providers":[]}')
        return 0
    try:
        providers, ratings, _manifest = load_brief(prompt_file)
    except Exception:
        print('{"providers":[]}')
        return 0
    env_url = os.environ.get("MAKEFUN_ENV_URL", "")
    if env_url:
        catalog = build_catalog(env_url.rstrip("/"), providers, ratings)
    else:
        catalog = {"providers": [{
            "id": provider["id"],
            "base_url": provider["path"],
            "key_env": provider.get("key_env", "") or "",
            "wire": provider["wire"],
            "models": ratings_models(ratings, provider["id"]),
        } for provider in providers]}
    print(json.dumps(catalog, separators=(",", ":")))
    return 0


def register(env_url, port, catalog):
    for _attempt in range(60):
        status, _raw = http_request(
            "POST", env_url + "/register",
            {"Content-Type": "application/json"},
            {"protocol_version": 1, "port": port, "catalog": catalog},
            timeout=5.0,
        )
        if status == 200:
            return True
        time.sleep(0.5)
    return False


def main():
    if "--list" in sys.argv[1:]:
        return run_list()
    prompt_file = os.environ.get("MAKEFUN_PROMPT_FILE")
    env_url = os.environ.get("MAKEFUN_ENV_URL")
    if not prompt_file or not env_url:
        sys.stderr.write("main.py: MAKEFUN_PROMPT_FILE and MAKEFUN_ENV_URL are required\n")
        return 1
    try:
        providers, ratings, manifest = load_brief(prompt_file)
    except Exception as err:
        sys.stderr.write("main.py: cannot read brief: %s\n" % err)
        return 1
    STATE.env_url = env_url.rstrip("/")
    STATE.providers = providers
    STATE.by_id = {provider["id"]: provider for provider in providers}
    STATE.ratings = ratings
    STATE.manifest = manifest
    STATE.manifest_by_id = {item["id"]: item for item in manifest}
    STATE.catalog = build_catalog(STATE.env_url, providers, ratings)
    server = AggregatorServer(("0.0.0.0", 0), AggregatorHandler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    if not register(STATE.env_url, port, STATE.catalog):
        sys.stderr.write("main.py: register failed\n")
        server.shutdown()
        return 1
    while True:
        work = fetch_work()
        if work is not None and work.get("done") is True:
            break
        time.sleep(WORK_POLL_SECONDS)
    server.shutdown()
    server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
