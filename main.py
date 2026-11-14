#!/usr/bin/env python3
"""A free-model aggregator: one process, two client wires, ten free providers.

The program reads the trial brief (providers, quality table, workload) from
MAKEFUN_PROMPT_FILE, learns each provider's model list by asking it, tells
the trial env which port it listens on, and then answers the twenty client
requests the env sends it. Every answer is fetched from a real provider over
that provider's own wire and handed back in the wire the client used.

Routing is deliberately boring and cautious. Requests are handled in the
order the env sends them. For each request the router looks at the models
that are free and meet the request's needs, and picks the strongest one
whose provider still has room under a small per-provider plan: the best
models get room for one call, mid models for two, the rest for three, so no
single provider is leaned on for the whole workload. A provider that says
its budget is spent, or that it is out, is never called again. A provider
that refuses one single call is kept in play but watched: a second refusal
from it retires it for the rest of the trial. Nothing here is guessed from
a seed or from any hidden number: every choice follows from the brief, the
catalog probe, and the answers providers have already given.

Standard library only. No wall clock and no randomness anywhere in the
routing, so a replay of the trial makes exactly the same calls.
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

EPOCH_CREATED = 1767225600
SLICE_LEN = 13
CALL_TIMEOUT = 5.0
POLL_GAP = 0.05
MAX_TRIES_PER_REQUEST = 4


def is_obj(value):
    return isinstance(value, dict)


def text_blocks(content):
    """Plain text out of an OpenAI/Anthropic content value."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    chunks = []
    for block in content:
        if is_obj(block) and block.get("type") == "text" and isinstance(block.get("text"), str):
            chunks.append(block["text"])
    return "\n".join(chunks)


def gemini_text(parts):
    if not isinstance(parts, list):
        return ""
    chunks = []
    for part in parts:
        if is_obj(part) and isinstance(part.get("text"), str):
            chunks.append(part["text"])
    return "\n".join(chunks)


def flat_text(content):
    """Plain text out of an Anthropic content list, blocks joined directly."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    chunks = []
    for block in content:
        if is_obj(block) and block.get("type") == "text" and isinstance(block.get("text"), str):
            chunks.append(block["text"])
    return "".join(chunks)


def short_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


def short_hash_bytes(raw):
    return hashlib.sha256(raw).hexdigest()[:8]


def image_fingerprint(url):
    """First 8 hex chars of the sha256 of an image URL's bytes (env's rule)."""
    if not isinstance(url, str) or url == "":
        return None
    found = re.match(r"^data:[^,]*;base64,(.*)$", url, re.S)
    if found is not None:
        try:
            return short_hash_bytes(base64.b64decode(found.group(1)))
        except Exception:
            return short_hash(url)
    return short_hash(url)


def split_data_url(url):
    """Split a data: URL into (mime type, base64 payload), or (None, None)."""
    if not isinstance(url, str):
        return None, None
    found = re.match(r"^data:([^;,]+)?;base64,(.*)$", url, re.S)
    if found is None:
        return None, None
    return (found.group(1) or "image/png"), found.group(2)


# ---------------------------------------------------------------------------
# Brief
# ---------------------------------------------------------------------------

def json_docs_in(text):
    """Every top-level JSON object embedded in the brief text, in order."""
    found = []
    i = 0
    total = len(text)
    while i < total:
        if text[i] != "{":
            i += 1
            continue
        depth = 0
        j = i
        in_string = False
        escaped = False
        done = False
        while j < total:
            ch = text[j]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
            else:
                if ch == '"':
                    in_string = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            found.append(json.loads(text[i:j + 1]))
                        except Exception:
                            pass
                        done = True
                        break
            j += 1
        i = (j + 1) if done else (i + 1)
    return found


def read_brief(path):
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    providers = None
    ratings = None
    manifest = None
    for doc in json_docs_in(text):
        if not is_obj(doc):
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
# Plain HTTP
# ---------------------------------------------------------------------------

def do_http(method, url, headers=None, body_obj=None, timeout=CALL_TIMEOUT):
    """One HTTP call. Returns (status or None, raw bytes)."""
    data = None
    heads = dict(headers or {})
    if body_obj is not None:
        data = json.dumps(body_obj).encode("utf-8")
        heads.setdefault("Content-Type", "application/json")
    req = Request(url, data=data, method=method, headers=heads)
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


def auth_headers(provider):
    kind = provider.get("auth")
    key = provider.get("key", "")
    if kind == "bearer":
        return {"Authorization": "Bearer " + key}
    if kind == "x-api-key":
        return {"x-api-key": key}
    if kind == "x-goog-api-key":
        return {"x-goog-api-key": key}
    return {}


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------

def models_from_ratings(ratings, provider_id):
    rows = []
    for row in ratings:
        if row.get("provider") == provider_id:
            rows.append({
                "id": row["model"],
                "free": bool(row.get("free")),
                "context_length": row.get("context_length"),
            })
    return rows


def probe_provider(env_url, provider):
    """Ask one provider for its model list; None when the probe fails."""
    base = env_url + provider["path"]
    heads = auth_headers(provider)
    try:
        if provider["wire"] == "gemini":
            status, raw = do_http("GET", base + "/v1beta/models", heads)
            if status != 200:
                return None
            doc = json.loads(raw.decode("utf-8"))
            rows = []
            for item in doc.get("models", []):
                name = item.get("name", "")
                if name.startswith("models/"):
                    name = name[len("models/"):]
                rows.append({
                    "id": name,
                    "free": bool(item.get("free")),
                    "context_length": item.get("context_length"),
                })
            return rows
        status, raw = do_http("GET", base + "/v1/models", heads)
        if status != 200:
            return None
        doc = json.loads(raw.decode("utf-8"))
        rows = []
        for item in doc.get("data", []):
            rows.append({
                "id": item.get("id"),
                "free": bool(item.get("free")),
                "context_length": item.get("context_length"),
            })
        return rows
    except Exception:
        return None


def assemble_catalog(env_url, providers, ratings):
    entries = []
    for provider in providers:
        models = probe_provider(env_url, provider)
        if models is None:
            models = models_from_ratings(ratings, provider["id"])
        entries.append({
            "id": provider["id"],
            "base_url": provider["path"],
            "key_env": provider.get("key_env", "") or "",
            "wire": provider["wire"],
            "models": models,
        })
    return {"providers": entries}


# ---------------------------------------------------------------------------
# Reading a client request (mirror the env's parse, for the usage formula)
# ---------------------------------------------------------------------------

def string_values(value):
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    return []


def numeric_or_none(value):
    return value if isinstance(value, (int, float)) and value == value else None


def read_openai(body):
    system_parts = []
    user_text = ""
    image = None
    messages = body.get("messages") if isinstance(body.get("messages"), list) else []
    for message in messages:
        if not is_obj(message):
            continue
        if message.get("role") == "system":
            text = text_blocks(message.get("content"))
            if text != "":
                system_parts.append(text)
        elif message.get("role") == "user":
            user_text = text_blocks(message.get("content"))
            if isinstance(message.get("content"), list):
                for block in message["content"]:
                    if is_obj(block) and block.get("type") == "image_url" and is_obj(block.get("image_url")):
                        found = image_fingerprint(block["image_url"].get("url"))
                        if found is not None:
                            image = found
    tools = []
    if isinstance(body.get("tools"), list):
        for tool in body["tools"]:
            if is_obj(tool) and is_obj(tool.get("function")) and isinstance(tool["function"].get("name"), str):
                tools.append(tool["function"]["name"])
    fmt = body.get("response_format") if is_obj(body.get("response_format")) else {}
    max_tokens = numeric_or_none(body.get("max_tokens"))
    if max_tokens is None:
        max_tokens = numeric_or_none(body.get("max_completion_tokens"))
    return {
        "system_text": "\n".join(system_parts),
        "user_text": user_text,
        "tools": tools,
        "max_tokens": max_tokens,
        "stops": string_values(body.get("stop")),
        "image_sha": image,
        "json_mode": fmt.get("type") in ("json_object", "json_schema"),
    }


def read_anthropic(body):
    system_text = text_blocks(body.get("system"))
    user_text = ""
    image = None
    messages = body.get("messages") if isinstance(body.get("messages"), list) else []
    for message in messages:
        if not is_obj(message) or message.get("role") != "user":
            continue
        user_text = text_blocks(message.get("content"))
        if isinstance(message.get("content"), list):
            for block in message["content"]:
                if not is_obj(block) or block.get("type") != "image" or not is_obj(block.get("source")):
                    continue
                source = block["source"]
                if source.get("type") == "base64" and isinstance(source.get("data"), str):
                    try:
                        image = short_hash_bytes(base64.b64decode(source["data"]))
                    except Exception:
                        pass
                elif source.get("type") == "url":
                    found = image_fingerprint(source.get("url"))
                    if found is not None:
                        image = found
    tools = []
    if isinstance(body.get("tools"), list):
        for tool in body["tools"]:
            if is_obj(tool) and isinstance(tool.get("name"), str):
                tools.append(tool["name"])
    return {
        "system_text": system_text,
        "user_text": user_text,
        "tools": tools,
        "max_tokens": numeric_or_none(body.get("max_tokens")),
        "stops": string_values(body.get("stop_sequences")),
        "image_sha": image,
        "json_mode": False,
    }


def read_gemini(body, model=""):
    system_text = ""
    instruction = body.get("systemInstruction")
    if is_obj(instruction):
        system_text = gemini_text(instruction.get("parts"))
    user_text = ""
    image = None
    contents = body.get("contents") if isinstance(body.get("contents"), list) else []
    for content in contents:
        if not is_obj(content):
            continue
        if content.get("role") == "model":
            continue
        user_text = gemini_text(content.get("parts"))
        if isinstance(content.get("parts"), list):
            for part in content["parts"]:
                if not is_obj(part):
                    continue
                if is_obj(part.get("inlineData")) and isinstance(part["inlineData"].get("data"), str):
                    try:
                        image = short_hash_bytes(base64.b64decode(part["inlineData"]["data"]))
                    except Exception:
                        pass
                elif is_obj(part.get("fileData")) and isinstance(part["fileData"].get("fileUri"), str):
                    found = image_fingerprint(part["fileData"]["fileUri"])
                    if found is not None:
                        image = found
    tools = []
    if isinstance(body.get("tools"), list):
        for tool in body["tools"]:
            if is_obj(tool) and isinstance(tool.get("functionDeclarations"), list):
                for fn in tool["functionDeclarations"]:
                    if is_obj(fn) and isinstance(fn.get("name"), str):
                        tools.append(fn["name"])
    config = body.get("generationConfig") if is_obj(body.get("generationConfig")) else {}
    return {
        "system_text": system_text,
        "user_text": user_text,
        "tools": tools,
        "max_tokens": numeric_or_none(config.get("maxOutputTokens")),
        "stops": string_values(config.get("stopSequences")),
        "image_sha": image,
        "json_mode": config.get("responseMimeType") == "application/json",
        "model": model,
    }


def parse_by_wire(wire, body, model=""):
    if wire == "openai":
        return read_openai(body)
    if wire == "anthropic":
        return read_anthropic(body)
    return read_gemini(body, model)


def tokens_of(parsed, answer):
    """The provider's token formula over the forwarded body and answer."""
    prompt_len = len(parsed["system_text"]) + len(parsed["user_text"]) + len(",".join(parsed["tools"]))
    return 1 + prompt_len // 4, 1 + len(answer) // 4


# ---------------------------------------------------------------------------
# Translating a client body into a provider's wire
# ---------------------------------------------------------------------------

def oai_to_ant_content(content):
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return content
    out = []
    for block in content:
        if not is_obj(block):
            continue
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            out.append({"type": "text", "text": block["text"]})
        elif block.get("type") == "image_url" and is_obj(block.get("image_url")):
            url = block["image_url"].get("url")
            mime, data = split_data_url(url)
            if data is not None:
                out.append({"type": "image", "source": {"type": "base64", "media_type": mime, "data": data}})
            elif isinstance(url, str):
                out.append({"type": "image", "source": {"type": "url", "url": url}})
        else:
            out.append(copy.deepcopy(block))
    return out


def ant_to_oai_content(content):
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return content
    out = []
    for block in content:
        if not is_obj(block):
            continue
        if block.get("type") == "text":
            out.append({"type": "text", "text": block.get("text", "")})
        elif block.get("type") == "image" and is_obj(block.get("source")):
            source = block["source"]
            if source.get("type") == "base64":
                media = source.get("media_type", "image/png")
                out.append({"type": "image_url", "image_url": {"url": "data:%s;base64,%s" % (media, source.get("data", ""))}})
            elif source.get("type") == "url":
                out.append({"type": "image_url", "image_url": {"url": source.get("url", "")}})
        else:
            out.append(copy.deepcopy(block))
    return out


def oai_to_gemini_parts(content):
    if isinstance(content, str):
        return [{"text": content}]
    if not isinstance(content, list):
        return []
    parts = []
    for block in content:
        if not is_obj(block):
            continue
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            parts.append({"text": block["text"]})
        elif block.get("type") == "image_url" and is_obj(block.get("image_url")):
            url = block["image_url"].get("url")
            mime, data = split_data_url(url)
            if data is not None:
                parts.append({"inlineData": {"mimeType": mime, "data": data}})
            elif isinstance(url, str):
                parts.append({"fileData": {"fileUri": url}})
    return parts


def ant_to_gemini_parts(content):
    if isinstance(content, str):
        return [{"text": content}]
    if not isinstance(content, list):
        return []
    parts = []
    for block in content:
        if not is_obj(block):
            continue
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            parts.append({"text": block["text"]})
        elif block.get("type") == "image" and is_obj(block.get("source")):
            source = block["source"]
            if source.get("type") == "base64":
                parts.append({"inlineData": {"mimeType": source.get("media_type", "image/png"), "data": source.get("data", "")}})
            elif source.get("type") == "url":
                parts.append({"fileData": {"fileUri": source.get("url", "")}})
    return parts


def build_anthropic_body(body, model):
    """OpenAI client body -> Anthropic provider body (no response_format)."""
    out = {"model": model}
    system_texts = []
    messages = []
    for message in body.get("messages", []) or []:
        if not is_obj(message):
            continue
        if message.get("role") == "system":
            text = text_blocks(message.get("content"))
            if text:
                system_texts.append(text)
        else:
            item = dict(message)
            item["content"] = oai_to_ant_content(message.get("content"))
            messages.append(item)
    if system_texts:
        out["system"] = "\n".join(system_texts)
    out["messages"] = messages
    tools = []
    for tool in body.get("tools", []) or []:
        if is_obj(tool) and is_obj(tool.get("function")) and isinstance(tool["function"].get("name"), str):
            fn = tool["function"]
            entry = {"name": fn["name"]}
            if "description" in fn:
                entry["description"] = fn["description"]
            if "parameters" in fn:
                entry["input_schema"] = fn["parameters"]
            tools.append(entry)
    if tools:
        out["tools"] = tools
    max_tokens = numeric_or_none(body.get("max_tokens"))
    if max_tokens is None:
        max_tokens = numeric_or_none(body.get("max_completion_tokens"))
    if max_tokens is not None:
        out["max_tokens"] = max_tokens
    stops = string_values(body.get("stop"))
    if stops:
        out["stop_sequences"] = stops
    if isinstance(body.get("temperature"), (int, float)):
        out["temperature"] = body["temperature"]
    if body.get("stream") is True:
        out["stream"] = True
    return out


def build_openai_body(body, model):
    """Anthropic client body -> OpenAI provider body (never adds response_format)."""
    out = {"model": model}
    messages = []
    system_text = text_blocks(body.get("system"))
    if system_text:
        messages.append({"role": "system", "content": system_text})
    for message in body.get("messages", []) or []:
        if not is_obj(message):
            continue
        item = dict(message)
        item["content"] = ant_to_oai_content(message.get("content"))
        messages.append(item)
    out["messages"] = messages
    tools = []
    for tool in body.get("tools", []) or []:
        if is_obj(tool) and isinstance(tool.get("name"), str):
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
    stops = string_values(body.get("stop_sequences"))
    if stops:
        out["stop"] = stops
    if isinstance(body.get("temperature"), (int, float)):
        out["temperature"] = body["temperature"]
    if body.get("stream") is True:
        out["stream"] = True
    return out


def build_gemini_body(body, client_api):
    """OpenAI or Anthropic client body -> Gemini provider body."""
    out = {}
    if client_api == "openai":
        system_texts = []
        contents = []
        for message in body.get("messages", []) or []:
            if not is_obj(message):
                continue
            if message.get("role") == "system":
                text = text_blocks(message.get("content"))
                if text:
                    system_texts.append(text)
            else:
                role = "model" if message.get("role") == "assistant" else "user"
                contents.append({"role": role, "parts": oai_to_gemini_parts(message.get("content"))})
        if system_texts:
            out["systemInstruction"] = {"parts": [{"text": "\n".join(system_texts)}]}
        out["contents"] = contents
        declarations = []
        for tool in body.get("tools", []) or []:
            if is_obj(tool) and is_obj(tool.get("function")) and isinstance(tool["function"].get("name"), str):
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
        max_tokens = numeric_or_none(body.get("max_tokens"))
        if max_tokens is None:
            max_tokens = numeric_or_none(body.get("max_completion_tokens"))
        if max_tokens is not None:
            config["maxOutputTokens"] = max_tokens
        stops = string_values(body.get("stop"))
        if stops:
            config["stopSequences"] = stops
        fmt = body.get("response_format") if is_obj(body.get("response_format")) else {}
        if fmt.get("type") in ("json_object", "json_schema"):
            config["responseMimeType"] = "application/json"
        if config:
            out["generationConfig"] = config
    else:
        system_text = text_blocks(body.get("system"))
        if system_text:
            out["systemInstruction"] = {"parts": [{"text": system_text}]}
        contents = []
        for message in body.get("messages", []) or []:
            if not is_obj(message):
                continue
            role = "model" if message.get("role") == "assistant" else "user"
            contents.append({"role": role, "parts": ant_to_gemini_parts(message.get("content"))})
        out["contents"] = contents
        declarations = []
        for tool in body.get("tools", []) or []:
            if is_obj(tool) and isinstance(tool.get("name"), str):
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
        stops = string_values(body.get("stop_sequences"))
        if stops:
            config["stopSequences"] = stops
        if config:
            out["generationConfig"] = config
    return out


# ---------------------------------------------------------------------------
# Reading a provider's answer
# ---------------------------------------------------------------------------

def sse_events(raw):
    events = []
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
            events.append((event, "\n".join(data)))
    return events


def pull_openai_text(raw):
    text = ""
    for _event, data in sse_events(raw):
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


def pull_anthropic_text(raw):
    text = ""
    for event, data in sse_events(raw):
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


def pull_gemini_text(raw):
    text = ""
    for _event, data in sse_events(raw):
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


def openai_text_of(doc):
    choices = doc.get("choices", []) or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    return flat_text(content)


def gemini_text_of(doc):
    candidates = doc.get("candidates", []) or []
    if not candidates:
        return ""
    content = candidates[0].get("content") or {}
    parts = content.get("parts", []) or []
    return "".join(part.get("text", "") for part in parts if isinstance(part.get("text"), str))


def refusal_reason(raw):
    """What a refusal's error body says: 'budget', 'outage', 'fail' or 'other'.

    The env's error envelope carries the provider's own words, and the words
    differ by cause: a spent budget says so, an outage says the provider is
    out, a single failed call says it refused this call. Reading that is
    reading the provider's answer, nothing else.
    """
    try:
        doc = json.loads(raw.decode("utf-8"))
    except Exception:
        return "other"
    message = ""
    if is_obj(doc):
        err = doc.get("error")
        if is_obj(err) and isinstance(err.get("message"), str):
            message = err["message"]
    if "budget is spent" in message:
        return "budget"
    if "is out" in message:
        return "outage"
    if "refused this call" in message:
        return "fail"
    return "other"


# ---------------------------------------------------------------------------
# Shared trial state
# ---------------------------------------------------------------------------

class Trial:
    def __init__(self):
        self.env_url = ""
        self.providers = []
        self.by_id = {}
        self.ratings = []
        self.manifest = []
        self.by_request = {}
        self.catalog = {"providers": []}
        self.calls = {}
        self.refusals = {}
        self.dead = set()
        self.served = 0
        self.lock = threading.Lock()


STATE = Trial()


# ---------------------------------------------------------------------------
# One attempt at one provider
# ---------------------------------------------------------------------------

def attempt_provider(provider, model, client_api, body, stream):
    """Try one provider once.

    Returns ("ok", answer_text, forwarded_body) or ("no", reason, status).
    reason is one of 'budget', 'outage', 'fail', 'other', 'unreachable'.
    Only a 200, 402 or 429 counts against the provider's trial budget, so
    only those count in the caller's books.
    """
    wire = provider["wire"]
    base = STATE.env_url + provider["path"]
    heads = auth_headers(provider)
    try:
        if wire == "openai":
            if client_api == "anthropic":
                forwarded = build_openai_body(body, model)
            else:
                forwarded = copy.deepcopy(body)
                forwarded["model"] = model
            if stream:
                forwarded["stream"] = True
                forwarded["stream_options"] = {"include_usage": True}
            status, raw = do_http("POST", base + "/v1/chat/completions", heads, forwarded)
            if status != 200:
                return ("no", refusal_reason(raw) if status in (402, 429) else "other", status)
            if stream:
                return ("ok", pull_openai_text(raw), forwarded)
            return ("ok", openai_text_of(json.loads(raw.decode("utf-8"))), forwarded)
        if wire == "anthropic":
            if client_api == "openai":
                forwarded = build_anthropic_body(body, model)
            else:
                forwarded = copy.deepcopy(body)
                forwarded["model"] = model
            heads = dict(heads)
            heads["anthropic-version"] = "2023-06-01"
            status, raw = do_http("POST", base + "/v1/messages", heads, forwarded)
            if status != 200:
                return ("no", refusal_reason(raw) if status in (402, 429) else "other", status)
            if stream:
                return ("ok", pull_anthropic_text(raw), forwarded)
            return ("ok", flat_text(json.loads(raw.decode("utf-8")).get("content")), forwarded)
        forwarded = build_gemini_body(body, client_api)
        action = "streamGenerateContent" if stream else "generateContent"
        url = base + "/v1beta/models/" + quote(model, safe="") + ":" + action
        if stream:
            url += "?alt=sse"
        status, raw = do_http("POST", url, heads, forwarded)
        if status != 200:
            return ("no", refusal_reason(raw) if status in (402, 429) else "other", status)
        if stream:
            return ("ok", pull_gemini_text(raw), forwarded)
        return ("ok", gemini_text_of(json.loads(raw.decode("utf-8"))), forwarded)
    except Exception:
        return ("no", "unreachable", None)


# ---------------------------------------------------------------------------
# Routing: a function of the brief and the answers seen so far
# ---------------------------------------------------------------------------

def room_for(provider_id):
    """Planned room for one more call, from the provider's best rating.

    The strongest models are the scarcest: the router gives a provider
    whose best model is top-tier room for one call, a mid-tier provider
    room for two, and the rest room for three. These tiers come only from
    the quality table in the brief; no hidden number is involved.
    """
    best = 0
    for row in STATE.ratings:
        if row.get("provider") == provider_id and row.get("free"):
            best = max(best, row.get("rating_milli") or 0)
    if best >= 1470:
        return 1
    if best >= 1450:
        return 2
    return 3


def fits_requirements(row, requires, client_api, json_needed):
    if not row.get("free"):
        return False
    if requires.get("tools") and not row.get("tools"):
        return False
    if requires.get("vision") and not row.get("vision"):
        return False
    if requires.get("json") and not row.get("json"):
        return False
    min_context = requires.get("min_context")
    if min_context is not None and (row.get("context_length") or 0) < min_context:
        return False
    provider = STATE.by_id.get(row.get("provider"))
    if provider is None:
        return False
    if json_needed and client_api == "openai" and provider["wire"] == "anthropic":
        return False
    return True


def ranked_options(requires, client_api, json_needed):
    """Eligible (provider, model) pairs, best first.

    The first pass only lists providers that still have planned room: a
    provider that already spent its room is not picked while any provider
    with room is left. The fallback pass lists every provider not
    retired, ordered by how much it has already proven: fewest refusals
    first, then the longest run of successful calls (a provider that
    served several calls without failing has shown those call numbers
    safe), then strongest rating. The two passes come back as one list,
    fallback after primary, so a request whose primary is exhausted or
    already tried still gets a fallback try instead of going unserved
    while an eligible provider exists. Providers already retired are
    left out of both passes.

    Order inside a pass: strongest rating first; a provider whose wire
    matches the client's wire next; then a provider that has not refused
    yet (a provider that already burned one call is tried only when
    nothing cleaner is left at that rating); then the provider with the
    most planned room left; then a fixed provider/model order, so the
    choice is deterministic.
    """
    primary = []
    fallback = []
    for row in STATE.ratings:
        if not fits_requirements(row, requires, client_api, json_needed):
            continue
        provider = STATE.by_id[row["provider"]]
        if provider["id"] in STATE.dead:
            continue
        same_wire = 0 if provider["wire"] == client_api else 1
        refused = STATE.refusals.get(provider["id"], 0)
        room = room_for(provider["id"]) - STATE.calls.get(provider["id"], 0)
        entry = (-(row.get("rating_milli") or 0), same_wire, refused, -room, provider["id"], row["model"], provider, row["model"])
        fallback.append(entry)
        if room > 0:
            primary.append(entry)
    if primary:
        primary.sort()
    fallback.sort(key=lambda item: (item[2], -STATE.calls.get(item[6]["id"], 0), item[0], item[4], item[5]))
    seen = set()
    merged = []
    for item in primary + fallback:
        tag = (item[6]["id"], item[7])
        if tag in seen:
            continue
        seen.add(tag)
        merged.append(item)
    return [(item[6], item[7]) for item in merged]


def lookup_work():
    try:
        status, raw = do_http("GET", STATE.env_url + "/work", timeout=5.0)
        if status != 200:
            return None
        doc = json.loads(raw.decode("utf-8"))
        return doc if is_obj(doc) else None
    except Exception:
        return None


def needs_of(client_api, body):
    """Requirements for this request: /work's next id, else manifest order."""
    work = lookup_work()
    if work is not None:
        next_id = work.get("next")
        if isinstance(next_id, str) and next_id in STATE.by_request:
            return dict(STATE.by_request[next_id].get("requires") or {})
    if STATE.served < len(STATE.manifest):
        return dict(STATE.manifest[STATE.served].get("requires") or {})
    return infer_needs(client_api, body)


def infer_needs(client_api, body):
    needs = {}
    tools = body.get("tools")
    if isinstance(tools, list) and tools:
        needs["tools"] = True
    if client_api == "openai":
        for message in body.get("messages", []) or []:
            if not is_obj(message) or not isinstance(message.get("content"), list):
                continue
            for block in message["content"]:
                if is_obj(block) and block.get("type") == "image_url":
                    needs["vision"] = True
        fmt = body.get("response_format")
        if is_obj(fmt) and fmt.get("type") in ("json_object", "json_schema"):
            needs["json"] = True
    else:
        for message in body.get("messages", []) or []:
            if not is_obj(message) or not isinstance(message.get("content"), list):
                continue
            for block in message["content"]:
                if is_obj(block) and block.get("type") == "image":
                    needs["vision"] = True
    return needs


def serve_one(client_api, body, requires):
    stream = body.get("stream") is True
    json_needed = (
        client_api == "openai"
        and is_obj(body.get("response_format"))
        and body["response_format"].get("type") in ("json_object", "json_schema")
    )
    tried = set()
    attempts = 0
    while attempts < MAX_TRIES_PER_REQUEST:
        options = ranked_options(requires, client_api, json_needed)
        pick = None
        for provider, model in options:
            if provider["id"] not in tried:
                pick = (provider, model)
                break
        if pick is None:
            break
        provider, model = pick
        tried.add(provider["id"])
        attempts += 1
        kind, first, second = attempt_provider(provider, model, client_api, body, stream)
        if kind == "ok":
            STATE.calls[provider["id"]] = STATE.calls.get(provider["id"], 0) + 1
            sys.stderr.write("served via %s/%s\n" % (provider["id"], model))
            return client_reply(client_api, body, model, first, second, provider["wire"])
        if second in (402, 429):
            STATE.calls[provider["id"]] = STATE.calls.get(provider["id"], 0) + 1
        if first in ("budget", "outage"):
            STATE.dead.add(provider["id"])
        else:
            STATE.refusals[provider["id"]] = STATE.refusals.get(provider["id"], 0) + 1
            if STATE.refusals[provider["id"]] >= 2:
                STATE.dead.add(provider["id"])
    return failure_reply(client_api)


def handle_client(client_api, body):
    with STATE.lock:
        requires = needs_of(client_api, body)
        answered = serve_one(client_api, body, requires)
        STATE.served += 1
        return answered


# ---------------------------------------------------------------------------
# Answers in the client's wire
# ---------------------------------------------------------------------------

def slice_text(text):
    if text == "":
        return [""]
    return [text[i:i + SLICE_LEN] for i in range(0, len(text), SLICE_LEN)]


def sse_json(payload):
    return "data: " + json.dumps(payload) + "\n\n"


def sse_named(name, payload):
    return "event: " + name + "\ndata: " + json.dumps(payload) + "\n\n"


def client_reply(client_api, body, model, text, forwarded, provider_wire):
    parsed = parse_by_wire(provider_wire, forwarded, model)
    in_tokens, out_tokens = tokens_of(parsed, text)
    stream = body.get("stream") is True
    if client_api == "openai":
        if not stream:
            payload = {
                "id": "chatcmpl-sim",
                "object": "chat.completion",
                "created": EPOCH_CREATED,
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                    "logprobs": None,
                }],
                "usage": {
                    "prompt_tokens": in_tokens,
                    "completion_tokens": out_tokens,
                    "total_tokens": in_tokens + out_tokens,
                },
            }
            return 200, "application/json", json.dumps(payload).encode("utf-8")
        frames = []
        base_chunk = {"id": "chatcmpl-sim", "object": "chat.completion.chunk", "created": EPOCH_CREATED, "model": model}
        first = dict(base_chunk)
        first["choices"] = [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]
        frames.append(sse_json(first))
        for piece in slice_text(text):
            chunk = dict(base_chunk)
            chunk["choices"] = [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]
            frames.append(sse_json(chunk))
        last = dict(base_chunk)
        last["choices"] = [{"index": 0, "delta": {}, "finish_reason": "stop"}]
        frames.append(sse_json(last))
        usage_chunk = dict(base_chunk)
        usage_chunk["choices"] = []
        usage_chunk["usage"] = {
            "prompt_tokens": in_tokens,
            "completion_tokens": out_tokens,
            "total_tokens": in_tokens + out_tokens,
        }
        frames.append(sse_json(usage_chunk))
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
            "usage": {"input_tokens": in_tokens, "output_tokens": out_tokens},
        }
        return 200, "application/json", json.dumps(payload).encode("utf-8")
    frames = []
    frames.append(sse_named("message_start", {
        "type": "message_start",
        "message": {
            "id": "msg_sim", "type": "message", "role": "assistant", "model": model,
            "content": [], "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": in_tokens, "output_tokens": 0},
        },
    }))
    frames.append(sse_named("content_block_start", {
        "type": "content_block_start", "index": 0,
        "content_block": {"type": "text", "text": ""},
    }))
    for piece in slice_text(text):
        frames.append(sse_named("content_block_delta", {
            "type": "content_block_delta", "index": 0,
            "delta": {"type": "text_delta", "text": piece},
        }))
    frames.append(sse_named("content_block_stop", {"type": "content_block_stop", "index": 0}))
    frames.append(sse_named("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": out_tokens},
    }))
    frames.append(sse_named("message_stop", {"type": "message_stop"}))
    return 200, "text/event-stream; charset=utf-8", "".join(frames).encode("utf-8")


def failure_reply(client_api):
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
                "data": [{"id": mid, "object": "model", "created": EPOCH_CREATED, "owned_by": "free-model-aggregator", "free": True} for mid in free_ids],
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
        if not is_obj(body):
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

def print_catalog():
    prompt_file = os.environ.get("MAKEFUN_PROMPT_FILE")
    if not prompt_file:
        print('{"providers":[]}')
        return 0
    try:
        providers, ratings, _manifest = read_brief(prompt_file)
    except Exception:
        print('{"providers":[]}')
        return 0
    env_url = os.environ.get("MAKEFUN_ENV_URL", "")
    if env_url:
        catalog = assemble_catalog(env_url.rstrip("/"), providers, ratings)
    else:
        catalog = {"providers": [{
            "id": provider["id"],
            "base_url": provider["path"],
            "key_env": provider.get("key_env", "") or "",
            "wire": provider["wire"],
            "models": models_from_ratings(ratings, provider["id"]),
        } for provider in providers]}
    print(json.dumps(catalog, separators=(",", ":")))
    return 0


def register(env_url, port, catalog):
    for _attempt in range(60):
        status, _raw = do_http(
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
        return print_catalog()
    prompt_file = os.environ.get("MAKEFUN_PROMPT_FILE")
    env_url = os.environ.get("MAKEFUN_ENV_URL")
    if not prompt_file or not env_url:
        sys.stderr.write("main.py: MAKEFUN_PROMPT_FILE and MAKEFUN_ENV_URL are required\n")
        return 1
    try:
        providers, ratings, manifest = read_brief(prompt_file)
    except Exception as err:
        sys.stderr.write("main.py: cannot read brief: %s\n" % err)
        return 1
    STATE.env_url = env_url.rstrip("/")
    STATE.providers = providers
    STATE.by_id = {provider["id"]: provider for provider in providers}
    STATE.ratings = ratings
    STATE.manifest = manifest
    STATE.by_request = {item["id"]: item for item in manifest}
    STATE.catalog = assemble_catalog(STATE.env_url, providers, ratings)
    server = AggregatorServer(("0.0.0.0", 0), AggregatorHandler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    if not register(STATE.env_url, port, STATE.catalog):
        sys.stderr.write("main.py: register failed\n")
        server.shutdown()
        return 1
    while True:
        work = lookup_work()
        if work is not None and work.get("done") is True:
            break
        time.sleep(POLL_GAP)
    server.shutdown()
    server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
