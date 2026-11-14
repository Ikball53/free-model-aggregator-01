# Free-Model Aggregator

One Python server (standard library only) that serves the OpenAI Chat
Completions API and the Anthropic Messages API — JSON and server-sent
events — over ten simulated free-tier model providers, and reports the
catalog it read from them.

## How it works

- **Brief.** The trial brief (`MAKEFUN_PROMPT_FILE`) holds three JSON
  documents: the ten providers (id, wire, auth scheme, base path, key),
  the sealed quality table (rating, context length, tools/vision/json,
  free flag) and the workload manifest (request id, API, streaming,
  weight, requirements). The brief is parsed with a brace scanner that
  respects JSON strings.
- **Catalog.** At startup the server probes every provider's model list
  (`GET /v1/models`, or `/v1beta/models` for Gemini) with that
  provider's own auth header, and registers the result as its catalog.
  If a probe fails, the models for that provider fall back to the
  ratings table in the brief. `python3 main.py --list` prints the same
  catalog document; with no brief it prints `{"providers":[]}`.
- **Routing.** For each request the server asks `GET /work` for the
  request id and takes its requirements from the manifest (falling back
  to manifest order, then to inferring them from the body). Candidates
  are the free models meeting the requirements, ordered by rating
  descending; ties prefer the provider whose wire matches the client's
  wire. An OpenAI JSON-mode request is never sent to an Anthropic-wire
  provider, because that wire cannot carry `response_format`.
- **Translation.** The request is translated into the provider's wire
  without touching any echoed field: system text, user text, tool names
  and order, `max_tokens`, stop sequences, image bytes (sha) and JSON
  mode are preserved across OpenAI, Anthropic and Gemini shapes.
- **Failover.** Candidates are tried in order, one model per provider
  per request. A non-2xx answer or a transport error counts as a
  refusal and the next candidate is tried; a provider with three
  refusals in a row is skipped for the rest of the trial. Exactly one
  successful provider call answers each request.
- **Answers.** The provider's answer text is returned whole in the
  client's wire, with usage recomputed by the provider's own formula
  over the forwarded body (`1 + floor(chars/4)` in, `1 + floor(chars/4)`
  out) and the stop reason mapped to the client's family (`stop` for
  OpenAI, `end_turn` for Anthropic). Streams are re-chunked in
  13-character pieces with the correct event order and a final usage
  chunk (OpenAI) or `message_delta`/`message_stop` (Anthropic).
- **Shutdown.** The server polls `GET /work` every 50 ms and exits 0
  as soon as the env reports the workload done.

## Run

```sh
# print the catalog for the brief in MAKEFUN_PROMPT_FILE
python3 main.py --list

# serve a trial (the env sets MAKEFUN_PROMPT_FILE and MAKEFUN_ENV_URL)
python3 main.py
```

## Files

- `main.py` — the whole server, Python 3 standard library only.
- `README.md` — this file.

## Practice scores (local, kit env + scorer)

| Trial | Score | Quality | Served | Wire | Failover | Catalog | Upstream calls |
|-------|------:|--------:|-------:|-----:|---------:|--------:|---------------:|
| 0 | 99 | 99 | 100 | 100 | 100 | 100 | 27 |
| 1 | 99 | 99 | 100 | 100 | 100 | 100 | 30 |
| 2 | 99 | 99 | 100 | 100 | 100 | 100 | 32 |

Quality 99 (not 100) is the budget ceiling: the best model per request
(openrouter `z-ai/glm-5.3-flash:free`, 1475) has a finite per-trial
budget, so a few requests necessarily fall back to the 1470-rated
models. All 20 requests are served validly in every trial.
