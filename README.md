# Free Model Aggregator

One server that answers OpenAI Chat Completions and Anthropic Messages
requests with the best free model it can still afford, making as few
provider calls as possible.

## Run

```sh
python3 main.py --list    # print the catalog and exit
python3 main.py           # serve; needs MAKEFUN_ENV_URL and MAKEFUN_PROMPT_FILE
```

## Routing

The router plans the whole remaining workload before every request: the
most constrained requests first, each given the strongest model whose
provider still has room under a small per-provider allowance. The plan is
rebuilt from the calls actually made after every provider call, so a
failover that spends one of a provider's calls is visible to the next
plan. A provider is only struck off for good when its own refusal says
its budget is spent or it is out; a single refused call does not condemn
it. Nothing here reads the trial seed, the clock, or any hidden number —
every choice comes from the brief, the manifest, and the providers'
answers.

## Layout

- `main.py` — the server, the router, and the wire translators
- `LICENSE` — MIT
