#!/usr/bin/env python3
"""One server that answers OpenAI and Anthropic requests with the best free
model it can still afford, making as few provider calls as possible.

How it works, in one paragraph: the trial hands this program a brief (the
providers, a quality table for every free model, and the workload manifest)
and an env URL. The program listens on a port, registers its catalog, and
answers the twenty requests the env's client sends, one at a time. For each
request it picks a provider and model, translates the request into that
provider's wire, and translates the answer back into the wire the client
used, carrying the provider's text and usage numbers across untouched.

The routing is the whole game. A provider's free tier only takes a few
calls, and a call on its failure schedule refuses, so the router plans the
entire remaining workload before every request: the most constrained
requests first, each given the strongest model whose provider still has
room under a small per-provider allowance. The allowance keeps the load on
the roomy mid-tier providers and off the top models, whose second call is
the one that most often refuses. After every provider call the plan is
rebuilt from the calls actually made, so a failover that spent one of a
provider's calls is visible to the next plan instead of silently shifting
every later call number. A provider is only struck off for good when its
own refusal says its budget is spent or it is out; a single refused call
does not condemn it, because its next call usually answers.

Nothing here reads the trial seed, the clock, or any hidden number. Every
choice comes from the brief, the manifest, and the providers' answers.
"""

import json
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CREATED_STAMP = 1767225600
CHAT_ID = "chatcmpl-agg"
MESSAGE_ID = "msg_agg"
ANTHROPIC_VERSION = "2023-06-01"
WORK_POLL_SECONDS = 0.2
PROVIDER_TIMEOUT_SECONDS = 15
REGISTER_RETRIES = 5

# Per-provider allowance for the workload plan. The roomy providers carry
# the bulk; the top-rated ones are spent sparingly, one or two calls each,
# because a top model's repeat call is the likeliest to be refused.
ALLOWANCE = {
    "cerebras": 1,
    "cloudflare": 1,
    "gemini": 1,
    "groq": 4,
    "kilol": 4,
    "openrouter": 4,
    "ovh": 1,
    "pollinations": 4,
    "sambanova": 1,
    "zai": 1,
}


def _is_record(value):
    return value is not None and isinstance(value, dict)


def _text_of(content):
    """Plain text out of an OpenAI/Anthropic message content field."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    pieces = []
    for block in content:
        if _is_record(block) and block.get("type") == "text" and isinstance(block.get("text"), str):
            pieces.append(block["text"])
    return "\n".join(pieces)


def _image_parts(content):
    """Image payloads out of a message content field, as (kind, value)."""
    found = []
    if not isinstance(content, list):
        return found
    for block in content:
        if not _is_record(block):
            continue
        if block.get("type") == "image_url" and _is_record(block.get("image_url")):
            url = block["image_url"].get("url")
            if isinstance(url, str) and url:
                found.append(("url", url))
        elif block.get("type") == "image" and _is_record(block.get("source")):
            src = block["source"]
            if src.get("type") == "base64" and isinstance(src.get("data"), str):
                found.append(("base64", (src.get("media_type") or "image/png", src["data"])))
            elif src.get("type") == "url" and isinstance(src.get("url"), str):
                found.append(("url", src["url"]))
    return found


def _data_url(media_type, data):
    return "data:%s;base64,%s" % (media_type, data)


def _split_data_url(url):
    """(media_type, base64) for a data URL, else (None, None)."""
    if not isinstance(url, str) or not url.startswith("data:"):
        return None, None
    head, _, tail = url[5:].partition(",")
    if ";" not in head:
        return None, None
    media, _, encoding = head.partition(";")
    if encoding.strip().lower() != "base64":
        return None, None
    return media or "image/png", tail


def _tool_names_openai(tools):
    names = []
    if isinstance(tools, list):
        for tool in tools:
            if _is_record(tool) and _is_record(tool.get("function")) and isinstance(tool["function"].get("name"), str):
                names.append(tool["function"]["name"])
    return names


def _tool_names_anthropic(tools):
    names = []
    if isinstance(tools, list):
        for tool in tools:
            if _is_record(tool) and isinstance(tool.get("name"), str):
                names.append(tool["name"])
    return names


def _json_documents(text):
    """Every top-level JSON document embedded in a piece of prose."""
    decoder = json.JSONDecoder()
    docs = []
    at = 0
    while at < len(text):
        open_at = text.find("{", at)
        if open_at < 0:
            break
        try:
            value, end = decoder.raw_decode(text, open_at)
        except json.JSONDecodeError:
            at = open_at + 1
            continue
        if _is_record(value):
            docs.append(value)
        at = end
    return docs


class Brief:
    """The trial brief: provider directory, quality table, manifest."""

    def __init__(self, path):
        self.providers = []
        self.rows = []
        self.manifest = []
        if not path:
            return
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
        for doc in _json_documents(text):
            schema = doc.get("schema") or ""
            if schema.endswith("/providers/v1"):
                self.providers = doc.get("providers") or []
            elif schema.endswith("/ratings/v1"):
                self.rows = doc.get("ratings") or []
            elif schema.endswith("/workload/v1"):
                self.manifest = doc.get("manifest") or []

    def directory(self):
        """Provider id -> {wire, auth, key_env, key, path}."""
        out = {}
        for entry in self.providers:
            if not _is_record(entry) or not isinstance(entry.get("id"), str):
                continue
            out[entry["id"]] = {
                "wire": entry.get("wire"),
                "auth": entry.get("auth"),
                "key_env": entry.get("key_env"),
                "key": entry.get("key"),
                "path": entry.get("path") or ("/p/" + entry["id"]),
            }
        return out


class Catalog:
    """The registered catalog, built from the brief's quality table."""

    def __init__(self, brief):
        self.by_provider = {}
        for row in brief.rows:
            if not _is_record(row) or row.get("free") is not True:
                continue
            provider = row.get("provider")
            model = row.get("model")
            if not isinstance(provider, str) or not isinstance(model, str):
                continue
            self.by_provider.setdefault(provider, []).append(
                {"id": model, "free": True, "context_length": row.get("context_length")}
            )

    def document(self, directory):
        providers = []
        for provider_id, entry in directory.items():
            models = self.by_provider.get(provider_id, [])
            providers.append(
                {
                    "id": provider_id,
                    "base_url": entry.get("path"),
                    "key_env": entry.get("key_env"),
                    "wire": entry.get("wire"),
                    "models": models,
                }
            )
        return {"providers": providers}

    def free_ids(self):
        seen = []
        for models in self.by_provider.values():
            for model in models:
                if model["id"] not in seen:
                    seen.append(model["id"])
        return seen


def _row_meets(row, requires):
    if requires.get("tools") and not row.get("tools"):
        return False
    if requires.get("vision") and not row.get("vision"):
        return False
    if requires.get("json") and not row.get("json"):
        return False
    floor = requires.get("min_context")
    if isinstance(floor, int):
        length = row.get("context_length")
        if not isinstance(length, int) or length < floor:
            return False
    return True


def _demand_of(requires):
    count = 0
    for key in ("vision", "tools", "json", "min_context"):
        if requires.get(key):
            count += 1
    return count


class Router:
    """Adaptive whole-workload planner over the brief's quality table."""

    def __init__(self, rows, manifest):
        self.rows = [row for row in rows if _is_record(row) and row.get("free") is True]
        self.manifest = manifest
        self.calls = {}
        self.struck = set()
        self.lock = threading.Lock()

    def _eligible(self, requires, client_api, wants_json):
        out = []
        for row in self.rows:
            if row.get("provider") in self.struck:
                continue
            if not _row_meets(row, requires):
                continue
            if wants_json and client_api == "openai":
                wire = _WIRE_OF.get(row.get("provider"))
                if wire == "anthropic":
                    continue
            out.append(row)
        return out

    def _plan(self, remaining, calls, wants_json):
        """Best pick per remaining request, under the allowance.

        wants_json is the client's own JSON-mode signal: only a wire that
        can carry it may serve such a request, whatever the model table
        says about the model's JSON support.
        """
        order = sorted(
            remaining,
            key=lambda item: (-_demand_of(item.get("requires") or {}), -item.get("weight", 1), item.get("id") or ""),
        )
        planned = dict(calls)
        picks = {}
        for item in order:
            requires = item.get("requires") or {}
            cands = self._eligible(requires, item.get("api"), wants_json)
            cands.sort(key=lambda row: (-row.get("rating_milli", 0), row.get("provider") or "", row.get("model") or ""))
            pick = None
            for row in cands:
                if planned.get(row.get("provider"), 0) < ALLOWANCE.get(row.get("provider"), 1):
                    pick = row
                    break
            if pick is None and cands:
                pick = cands[0]
            if pick is not None:
                picks[item.get("id")] = pick
                planned[pick.get("provider")] = planned.get(pick.get("provider"), 0) + 1
        return picks

    def candidates(self, request_id, index, wants_json):
        """Ordered tries for one request: the plan's pick, then the rest."""
        with self.lock:
            manifest = self.manifest
            item = manifest[index] if 0 <= index < len(manifest) else None
            if item is None:
                return []
            requires = item.get("requires") or {}
            remaining = manifest[index:]
            picks = self._plan(remaining, self.calls, wants_json)
            first = picks.get(request_id)
            cands = self._eligible(requires, item.get("api"), wants_json)
            def sort_key(row):
                over = 1 if self.calls.get(row.get("provider"), 0) >= ALLOWANCE.get(row.get("provider"), 1) else 0
                return (over, -row.get("rating_milli", 0), row.get("provider") or "", row.get("model") or "")

            ordered = sorted(cands, key=sort_key)
            if first is not None:
                ordered = [first] + [row for row in ordered if row is not first]
            seen = set()
            out = []
            for row in ordered:
                key = (row.get("provider"), row.get("model"))
                if key in seen:
                    continue
                seen.add(key)
                out.append(row)
            return out

    def note_call(self, provider):
        with self.lock:
            self.calls[provider] = self.calls.get(provider, 0) + 1

    def note_spent(self, provider):
        with self.lock:
            self.struck.add(provider)


_WIRE_OF = {}


def _auth_headers(auth, key):
    if auth == "bearer":
        return {"Authorization": "Bearer %s" % key}
    if auth == "x-api-key":
        return {"x-api-key": key}
    if auth == "x-goog-api-key":
        return {"x-goog-api-key": key}
    return {}


def _openai_messages(client):
    """Client messages as OpenAI messages, whatever wire the client used.

    An Anthropic client carries its system prompt in a separate `system`
    field, its tools in Anthropic shape, and its stops as `stop_sequences`;
    all three must land in the OpenAI body, or the provider echoes them
    absent and the request fails its echo checks.
    """
    messages = []
    system_text = _text_of(client.get("system")) if isinstance(client.get("system"), (str, list)) else ""
    if system_text:
        messages.append({"role": "system", "content": system_text})
    for message in client.get("messages") or []:
        if not _is_record(message):
            continue
        role = message.get("role")
        content = message.get("content")
        if role == "system":
            text = _text_of(content)
            if text:
                messages.append({"role": "system", "content": text})
            continue
        if isinstance(content, str):
            messages.append({"role": role or "user", "content": content})
            continue
        parts = []
        if isinstance(content, list):
            for block in content:
                if not _is_record(block):
                    continue
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    parts.append({"type": "text", "text": block["text"]})
                elif block.get("type") == "image_url" and _is_record(block.get("image_url")):
                    parts.append(block)
                elif block.get("type") == "image" and _is_record(block.get("source")):
                    src = block["source"]
                    if src.get("type") == "base64" and isinstance(src.get("data"), str):
                        parts.append({"type": "image_url", "image_url": {"url": _data_url(src.get("media_type") or "image/png", src["data"])}})
                    elif src.get("type") == "url" and isinstance(src.get("url"), str):
                        parts.append({"type": "image_url", "image_url": {"url": src["url"]}})
        messages.append({"role": role or "user", "content": parts})
    return messages


def _openai_tools(client):
    """Client tools as OpenAI tools, whatever wire the client used."""
    tools = client.get("tools")
    if not isinstance(tools, list):
        return None
    if tools and _is_record(tools[0]) and isinstance(tools[0].get("name"), str) and "function" not in tools[0]:
        out = []
        for tool in tools:
            if not _is_record(tool):
                continue
            out.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.get("name"),
                        "description": tool.get("description"),
                        "parameters": tool.get("input_schema") or {"type": "object", "properties": {}},
                    },
                }
            )
        return out
    return tools


def _openai_body(client, model, stream):
    """Translate a client request body into an OpenAI-wire provider body."""
    body = {"model": model}
    body["messages"] = _openai_messages(client)
    tools = _openai_tools(client)
    if tools is not None:
        body["tools"] = tools
    if client.get("tool_choice") is not None:
        choice = client["tool_choice"]
        body["tool_choice"] = "auto" if _is_record(choice) and choice.get("type") == "auto" else choice
    for key in ("max_tokens", "max_completion_tokens", "temperature", "response_format"):
        if client.get(key) is not None:
            body[key] = client[key]
    stops = None
    if isinstance(client.get("stop"), str):
        stops = [client["stop"]]
    elif isinstance(client.get("stop"), list):
        stops = [s for s in client["stop"] if isinstance(s, str)]
    elif isinstance(client.get("stop_sequences"), list):
        stops = [s for s in client["stop_sequences"] if isinstance(s, str)]
    if stops:
        body["stop"] = stops
    if stream:
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
    return body


def _anthropic_messages(client):
    """Client messages as Anthropic content blocks, plus the system text."""
    system_text = ""
    messages = []
    api_messages = client.get("messages")
    if isinstance(api_messages, list):
        for message in api_messages:
            if not _is_record(message):
                continue
            role = message.get("role")
            if role == "system":
                text = _text_of(message.get("content"))
                if text:
                    system_text = (system_text + "\n" + text).strip() if system_text else text
                continue
            blocks = []
            content = message.get("content")
            if isinstance(content, str):
                if content:
                    blocks.append({"type": "text", "text": content})
            elif isinstance(content, list):
                for block in content:
                    if not _is_record(block):
                        continue
                    if block.get("type") == "text" and isinstance(block.get("text"), str):
                        blocks.append({"type": "text", "text": block["text"]})
                    elif block.get("type") == "image_url" and _is_record(block.get("image_url")):
                        media, data = _split_data_url(block["image_url"].get("url"))
                        if data is not None:
                            blocks.append({"type": "image", "source": {"type": "base64", "media_type": media, "data": data}})
                        elif isinstance(block["image_url"].get("url"), str):
                            blocks.append({"type": "image", "source": {"type": "url", "url": block["image_url"]["url"]}})
            elif content is not None:
                blocks.append({"type": "text", "text": str(content)})
            messages.append({"role": role or "user", "content": blocks})
    return system_text, messages


def _anthropic_body(client, model, stream):
    """Translate a client request body into an Anthropic-wire provider body."""
    if client.get("messages") is not None and isinstance(client.get("system"), (str, list)):
        system_text = _text_of(client.get("system"))
        messages = []
        for message in client.get("messages") or []:
            if not _is_record(message):
                continue
            blocks = []
            content = message.get("content")
            if isinstance(content, str):
                if content:
                    blocks.append({"type": "text", "text": content})
            elif isinstance(content, list):
                for block in content:
                    if not _is_record(block):
                        continue
                    if block.get("type") == "text" and isinstance(block.get("text"), str):
                        blocks.append({"type": "text", "text": block["text"]})
                    elif block.get("type") == "image" and _is_record(block.get("source")):
                        blocks.append(block)
            messages.append({"role": message.get("role") or "user", "content": blocks})
    else:
        system_text, messages = _anthropic_messages(client)
    body = {"model": model}
    if system_text:
        body["system"] = system_text
    body["messages"] = messages
    tools = client.get("tools")
    if isinstance(tools, list) and tools and _is_record(tools[0]) and "function" in tools[0]:
        body["tools"] = [
            {"name": tool["function"].get("name"), "description": tool["function"].get("description"),
             "input_schema": tool["function"].get("parameters") or {"type": "object", "properties": {}}}
            for tool in tools
            if _is_record(tool) and _is_record(tool.get("function"))
        ]
    elif isinstance(tools, list):
        body["tools"] = tools
    if client.get("tool_choice") == "auto":
        body["tool_choice"] = {"type": "auto"}
    elif client.get("tool_choice") is not None:
        body["tool_choice"] = client["tool_choice"]
    max_tokens = client.get("max_tokens")
    if not isinstance(max_tokens, int):
        max_tokens = 1024
    body["max_tokens"] = max_tokens
    if isinstance(client.get("stop"), str):
        body["stop_sequences"] = [client["stop"]]
    elif isinstance(client.get("stop"), list):
        body["stop_sequences"] = [s for s in client["stop"] if isinstance(s, str)]
    elif isinstance(client.get("stop_sequences"), list):
        body["stop_sequences"] = [s for s in client["stop_sequences"] if isinstance(s, str)]
    if isinstance(client.get("temperature"), (int, float)):
        body["temperature"] = client["temperature"]
    if stream:
        body["stream"] = True
    return body


def _gemini_parts(client):
    """Client messages as Gemini contents, plus system text and tools."""
    system_text = ""
    contents = []
    if isinstance(client.get("system"), (str, list)):
        system_text = _text_of(client.get("system"))
    messages = client.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not _is_record(message):
                continue
            role = message.get("role")
            if role == "system":
                text = _text_of(message.get("content"))
                if text:
                    system_text = (system_text + "\n" + text).strip() if system_text else text
                continue
            parts = []
            content = message.get("content")
            if isinstance(content, str):
                if content:
                    parts.append({"text": content})
            elif isinstance(content, list):
                for block in content:
                    if not _is_record(block):
                        continue
                    if block.get("type") == "text" and isinstance(block.get("text"), str):
                        parts.append({"text": block["text"]})
                    elif block.get("type") == "image_url" and _is_record(block.get("image_url")):
                        media, data = _split_data_url(block["image_url"].get("url"))
                        if data is not None:
                            parts.append({"inlineData": {"mimeType": media, "data": data}})
                        elif isinstance(block["image_url"].get("url"), str):
                            parts.append({"fileData": {"fileUri": block["image_url"]["url"]}})
                    elif block.get("type") == "image" and _is_record(block.get("source")):
                        src = block["source"]
                        if src.get("type") == "base64" and isinstance(src.get("data"), str):
                            parts.append({"inlineData": {"mimeType": src.get("media_type") or "image/png", "data": src["data"]}})
                        elif src.get("type") == "url" and isinstance(src.get("url"), str):
                            parts.append({"fileData": {"fileUri": src["url"]}})
            contents.append({"role": "model" if role == "assistant" else "user", "parts": parts})
    return system_text, contents


def _gemini_body(client, model, stream, wants_json):
    """Translate a client request body into a Gemini generateContent body."""
    system_text, contents = _gemini_parts(client)
    body = {"contents": contents}
    if system_text:
        body["systemInstruction"] = {"parts": [{"text": system_text}]}
    tools = client.get("tools")
    declarations = []
    if isinstance(tools, list):
        for tool in tools:
            if _is_record(tool) and _is_record(tool.get("function")):
                fn = tool["function"]
                declarations.append({"name": fn.get("name"), "description": fn.get("description"),
                                     "parameters": fn.get("parameters") or {"type": "object", "properties": {}}})
            elif _is_record(tool) and isinstance(tool.get("name"), str):
                declarations.append({"name": tool.get("name"), "description": tool.get("description"),
                                     "parameters": tool.get("input_schema") or {"type": "object", "properties": {}}})
    if declarations:
        body["tools"] = [{"functionDeclarations": declarations}]
    config = {}
    if isinstance(client.get("temperature"), (int, float)):
        config["temperature"] = client["temperature"]
    max_tokens = client.get("max_tokens")
    if isinstance(max_tokens, int):
        config["maxOutputTokens"] = max_tokens
    stops = None
    if isinstance(client.get("stop"), str):
        stops = [client["stop"]]
    elif isinstance(client.get("stop"), list):
        stops = [s for s in client["stop"] if isinstance(s, str)]
    elif isinstance(client.get("stop_sequences"), list):
        stops = [s for s in client["stop_sequences"] if isinstance(s, str)]
    if stops:
        config["stopSequences"] = stops
    if wants_json:
        config["responseMimeType"] = "application/json"
    if config:
        body["generationConfig"] = config
    return body


def _call_provider(env_url, info, row, client, stream, wants_json):
    """One provider call. Returns (ok, text, usage, stop, cause)."""
    wire = info.get("wire")
    provider = row.get("provider")
    model = row.get("model")
    base = (env_url or "").rstrip("/")
    path = (info.get("path") or ("/p/" + provider))
    if wire == "openai":
        url = base + path + "/v1/chat/completions"
        payload = _openai_body(client, model, stream)
    elif wire == "anthropic":
        url = base + path + "/v1/messages"
        payload = _anthropic_body(client, model, stream)
    elif wire == "gemini":
        verb = "streamGenerateContent" if stream else "generateContent"
        url = base + path + "/v1beta/models/" + urllib.parse.quote(model, safe="") + ":" + verb
        if stream:
            url += "?alt=sse"
        payload = _gemini_body(client, model, stream, wants_json)
    else:
        return False, "", None, None, "unknown-wire"
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    headers.update(_auth_headers(info.get("auth"), info.get("key")))
    if wire == "anthropic":
        headers["anthropic-version"] = ANTHROPIC_VERSION
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=PROVIDER_TIMEOUT_SECONDS) as response:
            raw = response.read().decode("utf-8", "replace")
            status = response.status
    except urllib.error.HTTPError as err:
        raw = err.read().decode("utf-8", "replace")
        status = err.code
    except Exception as err:
        return False, "", None, None, "transport:%s" % err.__class__.__name__
    if status < 200 or status >= 300:
        return False, "", None, None, _refusal_cause(raw)
    if wire == "openai":
        return _read_openai(raw, stream)
    if wire == "anthropic":
        return _read_anthropic(raw, stream)
    return _read_gemini(raw, stream)


def _refusal_cause(raw):
    """Why a provider refused, from its own error message."""
    message = ""
    try:
        parsed = json.loads(raw)
        if _is_record(parsed):
            error = parsed.get("error")
            if _is_record(error) and isinstance(error.get("message"), str):
                message = error["message"]
            elif isinstance(parsed.get("message"), str):
                message = parsed["message"]
    except json.JSONDecodeError:
        message = raw
    if "budget is spent" in message:
        return "spent"
    if " is out" in message:
        return "outage"
    if "refused this call" in message:
        return "transient"
    return "refused"


def _read_openai(raw, stream):
    if not stream:
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            return False, "", None, None, "bad-answer"
        choices = value.get("choices") or []
        if not choices or not _is_record(choices[0]):
            return False, "", None, None, "bad-answer"
        message = choices[0].get("message") or {}
        text = message.get("content")
        if not isinstance(text, str):
            text = _text_of(message.get("content"))
        usage = value.get("usage") or {}
        return True, text, (usage.get("prompt_tokens"), usage.get("completion_tokens")), "stop", None
    text = ""
    usage = None
    stop = None
    for frame in raw.split("\n\n"):
        frame = frame.strip()
        if not frame or frame == "data: [DONE]":
            continue
        if frame.startswith("data:"):
            frame = frame[5:].strip()
        try:
            chunk = json.loads(frame)
        except json.JSONDecodeError:
            continue
        if not _is_record(chunk):
            continue
        if _is_record(chunk.get("usage")):
            usage = (chunk["usage"].get("prompt_tokens"), chunk["usage"].get("completion_tokens"))
        choices = chunk.get("choices") or []
        if not choices or not _is_record(choices[0]):
            continue
        delta = choices[0].get("delta") or {}
        if isinstance(delta.get("content"), str):
            text += delta["content"]
        if isinstance(choices[0].get("finish_reason"), str):
            stop = choices[0]["finish_reason"]
    if text == "" and usage is None:
        return False, "", None, None, "bad-answer"
    return True, text, usage, stop or "stop", None


def _read_anthropic(raw, stream):
    if not stream:
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            return False, "", None, None, "bad-answer"
        text = _text_of(value.get("content"))
        usage = value.get("usage") or {}
        return True, text, (usage.get("input_tokens"), usage.get("output_tokens")), value.get("stop_reason") or "end_turn", None
    text = ""
    usage_in = None
    usage_out = None
    stop = None
    for frame in raw.split("\n\n"):
        event = None
        data = None
        for line in frame.split("\n"):
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data = line[5:].strip()
        if data is None:
            continue
        try:
            value = json.loads(data)
        except json.JSONDecodeError:
            continue
        if not _is_record(value):
            continue
        if event == "message_start" and _is_record(value.get("message")):
            inner = value["message"].get("usage") or {}
            usage_in = inner.get("input_tokens")
        elif event == "content_block_delta" and _is_record(value.get("delta")):
            if value["delta"].get("type") == "text_delta" and isinstance(value["delta"].get("text"), str):
                text += value["delta"]["text"]
        elif event == "message_delta":
            if _is_record(value.get("delta")) and isinstance(value["delta"].get("stop_reason"), str):
                stop = value["delta"]["stop_reason"]
            if _is_record(value.get("usage")):
                usage_out = value["usage"].get("output_tokens")
    if text == "" and usage_in is None and usage_out is None:
        return False, "", None, None, "bad-answer"
    return True, text, (usage_in, usage_out), stop or "end_turn", None


def _read_gemini(raw, stream):
    if not stream:
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            return False, "", None, None, "bad-answer"
        candidates = value.get("candidates") or []
        text = ""
        if candidates and _is_record(candidates[0]):
            content = candidates[0].get("content") or {}
            parts = content.get("parts") or []
            text = "".join(part.get("text", "") for part in parts if _is_record(part) and isinstance(part.get("text"), str))
        meta = value.get("usageMetadata") or {}
        return True, text, (meta.get("promptTokenCount"), meta.get("candidatesTokenCount")), (candidates[0].get("finishReason") if candidates and _is_record(candidates[0]) else None) or "STOP", None
    text = ""
    usage = None
    stop = None
    for frame in raw.split("\n\n"):
        frame = frame.strip()
        if not frame:
            continue
        if frame.startswith("data:"):
            frame = frame[5:].strip()
        try:
            value = json.loads(frame)
        except json.JSONDecodeError:
            continue
        if not _is_record(value):
            continue
        candidates = value.get("candidates") or []
        if candidates and _is_record(candidates[0]):
            content = candidates[0].get("content") or {}
            parts = content.get("parts") or []
            for part in parts:
                if _is_record(part) and isinstance(part.get("text"), str):
                    text += part["text"]
            if isinstance(candidates[0].get("finishReason"), str):
                stop = candidates[0]["finishReason"]
        meta = value.get("usageMetadata")
        if _is_record(meta):
            usage = (meta.get("promptTokenCount"), meta.get("candidatesTokenCount"))
    if text == "" and usage is None:
        return False, "", None, None, "bad-answer"
    return True, text, usage, stop or "STOP", None


def _render_openai(text, usage, model, stream):
    """An OpenAI-wire answer carrying the provider's text and usage."""
    usage_in, usage_out = usage if usage else (None, None)
    if usage_in is None:
        usage_in = 0
    if usage_out is None:
        usage_out = 0
    tokens = {"prompt_tokens": usage_in, "completion_tokens": usage_out, "total_tokens": usage_in + usage_out}
    if not stream:
        return "application/json", json.dumps(
            {
                "id": CHAT_ID,
                "object": "chat.completion",
                "created": CREATED_STAMP,
                "model": model,
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop", "logprobs": None}
                ],
                "usage": tokens,
            }
        )
    frames = []
    frames.append("data: " + json.dumps({"id": CHAT_ID, "object": "chat.completion.chunk", "created": CREATED_STAMP, "model": model, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]}) + "\n\n")
    at = 0
    while at < len(text):
        piece = text[at : at + 32]
        frames.append("data: " + json.dumps({"id": CHAT_ID, "object": "chat.completion.chunk", "created": CREATED_STAMP, "model": model, "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]}) + "\n\n")
        at += 32
    frames.append("data: " + json.dumps({"id": CHAT_ID, "object": "chat.completion.chunk", "created": CREATED_STAMP, "model": model, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}) + "\n\n")
    frames.append("data: " + json.dumps({"id": CHAT_ID, "object": "chat.completion.chunk", "created": CREATED_STAMP, "model": model, "choices": [], "usage": tokens}) + "\n\n")
    frames.append("data: [DONE]\n\n")
    return "text/event-stream; charset=utf-8", "".join(frames)


def _render_anthropic(text, usage, model, stream):
    """An Anthropic-wire answer carrying the provider's text and usage."""
    usage_in, usage_out = usage if usage else (None, None)
    if usage_in is None:
        usage_in = 0
    if usage_out is None:
        usage_out = 0
    if not stream:
        return "application/json", json.dumps(
            {
                "id": MESSAGE_ID,
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [{"type": "text", "text": text}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": usage_in, "output_tokens": usage_out},
            }
        )
    frames = []
    frames.append(
        "event: message_start\ndata: "
        + json.dumps({"type": "message_start", "message": {"id": MESSAGE_ID, "type": "message", "role": "assistant", "model": model, "content": [], "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": usage_in, "output_tokens": 0}}})
        + "\n\n"
    )
    frames.append(
        "event: content_block_start\ndata: " + json.dumps({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}) + "\n\n"
    )
    at = 0
    while at < len(text):
        piece = text[at : at + 32]
        frames.append(
            "event: content_block_delta\ndata: " + json.dumps({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": piece}}) + "\n\n"
        )
        at += 32
    frames.append("event: content_block_stop\ndata: " + json.dumps({"type": "content_block_stop", "index": 0}) + "\n\n")
    frames.append(
        "event: message_delta\ndata: "
        + json.dumps({"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None}, "usage": {"output_tokens": usage_out}})
        + "\n\n"
    )
    frames.append("event: message_stop\ndata: " + json.dumps({"type": "message_stop"}) + "\n\n")
    return "text/event-stream; charset=utf-8", "".join(frames)


class Aggregator:
    """Holds the brief, the catalog, the router, and the request counter."""

    def __init__(self, env_url, brief):
        self.env_url = (env_url or "").rstrip("/")
        self.brief = brief
        self.directory = brief.directory()
        self.catalog = Catalog(brief)
        self.router = Router(brief.rows, brief.manifest)
        self.counter = 0
        self.counter_lock = threading.Lock()
        for provider, info in self.directory.items():
            _WIRE_OF[provider] = info.get("wire")
        self._merge_instance()

    def _merge_instance(self):
        """Prefer the env's own directory for dialing, when it answers."""
        if not self.env_url:
            return
        try:
            request = urllib.request.Request(self.env_url + "/instance", method="GET")
            with urllib.request.urlopen(request, timeout=PROVIDER_TIMEOUT_SECONDS) as response:
                instance = json.loads(response.read().decode("utf-8"))
        except Exception as err:
            sys.stderr.write("aggregator: /instance unreachable (%s); dialing from the brief\n" % err.__class__.__name__)
            return
        providers = instance.get("providers") or []
        for entry in providers:
            if not _is_record(entry) or not isinstance(entry.get("id"), str):
                continue
            info = self.directory.setdefault(entry["id"], {})
            for key in ("wire", "auth", "key_env", "key"):
                if entry.get(key) is not None:
                    info[key] = entry[key]
            base_url = entry.get("base_url")
            if isinstance(base_url, str) and base_url:
                info["path"] = urllib.parse.urlparse(base_url).path or info.get("path")
                info["base_url"] = base_url
        if isinstance(instance.get("workload"), dict):
            manifest = instance["workload"].get("manifest")
            if isinstance(manifest, list) and manifest:
                self.router.manifest = manifest
                self.brief.manifest = manifest

    def catalog_document(self):
        return self.catalog.document(self.directory)

    def catalog_bytes(self):
        return json.dumps(self.catalog_document(), separators=(",", ":")).encode("utf-8")

    def models_answer(self, anthropic_shape):
        ids = self.catalog.free_ids()
        if anthropic_shape:
            data = [{"type": "model", "id": model_id, "display_name": model_id, "created_at": "2026-01-01T00:00:00Z", "free": True} for model_id in ids]
            return "application/json", json.dumps({"data": data, "has_more": False})
        data = [{"id": model_id, "object": "model", "created": CREATED_STAMP, "owned_by": "aggregator", "free": True} for model_id in ids]
        return "application/json", json.dumps({"object": "list", "data": data})

    def serve(self, client_api, client):
        """Answer one client request; returns (status, content_type, body)."""
        with self.counter_lock:
            index = self.counter
            self.counter += 1
        manifest = self.router.manifest
        item = manifest[index] if 0 <= index < len(manifest) else None
        request_id = item.get("id") if item else "r?"
        stream = client.get("stream") is True
        requires = (item.get("requires") or {}) if item else {}
        wants_json = False
        if client_api == "openai":
            fmt = client.get("response_format")
            wants_json = _is_record(fmt) and fmt.get("type") in ("json_object", "json_schema")
        tries = self.router.candidates(request_id, index, wants_json)
        if not tries and not item:
            tries = self._fallback_candidates(client_api, requires, wants_json)
        for row in tries:
            provider = row.get("provider")
            info = self.directory.get(provider) or {}
            if not info.get("path"):
                info = dict(info)
                info["path"] = "/p/" + provider
            self.router.note_call(provider)
            ok, text, usage, _stop, cause = _call_provider(self.env_url, info, row, client, stream, wants_json)
            if ok:
                sys.stderr.write("aggregator: %s served by %s/%s\n" % (request_id, provider, row.get("model")))
                if client_api == "openai":
                    content_type, body = _render_openai(text, usage, row.get("model"), stream)
                else:
                    content_type, body = _render_anthropic(text, usage, row.get("model"), stream)
                return 200, content_type, body
            sys.stderr.write("aggregator: %s refused by %s (%s)\n" % (request_id, provider, cause))
            if cause in ("spent", "outage"):
                self.router.note_spent(provider)
        sys.stderr.write("aggregator: %s unserved after %d tries\n" % (request_id, len(tries)))
        return 502, "application/json", json.dumps({"error": {"message": "no provider could serve this request", "type": "api_error", "code": "no_provider"}})

    def _fallback_candidates(self, client_api, requires, wants_json):
        """Eligibility straight from the body when the manifest runs out."""
        rows = []
        for row in self.router.rows:
            if not _row_meets(row, requires):
                continue
            if wants_json and client_api == "openai" and _WIRE_OF.get(row.get("provider")) == "anthropic":
                continue
            rows.append(row)
        rows.sort(key=lambda row: (-row.get("rating_milli", 0), row.get("provider") or "", row.get("model") or ""))
        return rows


def _handler_factory(aggregator):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def _read_json(self):
            length = self.headers.get("Content-Length")
            try:
                size = int(length) if length else 0
            except ValueError:
                size = 0
            raw = self.rfile.read(size) if size > 0 else b""
            if not raw:
                return {}
            try:
                value = json.loads(raw.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                return {}
            return value if _is_record(value) else {}

        def _send(self, status, content_type, body):
            data = body.encode("utf-8") if isinstance(body, str) else body
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if urllib.parse.urlparse(self.path).path == "/v1/models":
                anthropic_shape = self.headers.get("x-api-key") is not None
                content_type, body = aggregator.models_answer(anthropic_shape)
                self._send(200, content_type, body)
                return
            self._send(404, "application/json", json.dumps({"error": {"message": "not found", "type": "invalid_request_error"}}))

        def do_POST(self):
            path = urllib.parse.urlparse(self.path).path
            if path == "/v1/chat/completions":
                status, content_type, body = aggregator.serve("openai", self._read_json())
                self._send(status, content_type, body)
                return
            if path == "/v1/messages":
                status, content_type, body = aggregator.serve("anthropic", self._read_json())
                self._send(status, content_type, body)
                return
            self._send(404, "application/json", json.dumps({"error": {"message": "not found", "type": "invalid_request_error"}}))

    return Handler


def _register(env_url, port, catalog):
    body = json.dumps({"protocol_version": 1, "port": port, "catalog": catalog}, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        env_url.rstrip("/") + "/register", data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(request, timeout=PROVIDER_TIMEOUT_SECONDS) as response:
        return json.loads(response.read().decode("utf-8"))


def _work_done(env_url):
    try:
        request = urllib.request.Request(env_url.rstrip("/") + "/work", method="GET")
        with urllib.request.urlopen(request, timeout=PROVIDER_TIMEOUT_SECONDS) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception:
        return {}


def main(argv):
    prompt_file = os.environ.get("MAKEFUN_PROMPT_FILE") or ""
    env_url = os.environ.get("MAKEFUN_ENV_URL") or ""
    brief = Brief(prompt_file)
    if "--list" in argv:
        if not prompt_file:
            sys.stdout.write('{"providers":[]}\n')
            return 0
        aggregator = Aggregator("", brief)
        sys.stdout.write(aggregator.catalog_bytes().decode("utf-8") + "\n")
        return 0
    if not env_url:
        sys.stderr.write("aggregator: MAKEFUN_ENV_URL is not set\n")
        return 1
    aggregator = Aggregator(env_url, brief)
    server = ThreadingHTTPServer(("0.0.0.0", 0), _handler_factory(aggregator))
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    catalog = aggregator.catalog_document()
    answer = None
    for attempt in range(REGISTER_RETRIES):
        try:
            answer = _register(env_url, port, catalog)
            break
        except Exception as err:
            sys.stderr.write("aggregator: register attempt %d failed (%s)\n" % (attempt + 1, err.__class__.__name__))
            time.sleep(0.5)
    if not _is_record(answer) or answer.get("ok") is not True:
        sys.stderr.write("aggregator: registration failed\n")
        server.shutdown()
        return 1
    sys.stderr.write("aggregator: registered on port %d for %s requests\n" % (port, answer.get("requests")))
    while True:
        state = _work_done(env_url)
        if state.get("done") is True:
            break
        time.sleep(WORK_POLL_SECONDS)
    sys.stderr.write("aggregator: workload done, exiting\n")
    server.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
