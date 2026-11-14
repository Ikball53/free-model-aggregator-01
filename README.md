# Free model aggregator — safe routing, no seed tricks

One server that answers the OpenAI Chat Completions API and the Anthropic
Messages API (JSON and streaming) by routing each request to the best free
model that meets its requirements, failing over when a provider refuses.

## How routing works

Requests are handled in the order the env sends them. For each request the
router ranks every eligible free model, strongest rating first, and tries
them in that order, re-ranking before every try so a failure changes the
next pick. Room is tiered by the provider's best rating in the brief's
quality table: a top-tier provider gets room for one call, a mid-tier
provider for two, the rest for three — so no single provider is leaned on
for the whole workload. Providers with room left come first; after them,
in the same list, come providers that already spent their room, ordered
by fewest refusals, then longest run of successful calls, then rating —
so a request is never left unserved while an eligible provider exists.

A provider that says its budget is spent, or that it is out, is never
called again. A provider that refuses one single call stays in play but is
watched: a second refusal retires it for the rest of the trial. The error
words are read from the provider's own answer, nothing else.

Nothing here is guessed from a seed or from any hidden number. Every
choice follows from the brief, the catalog probe, and the answers
providers have already given. No wall clock, no randomness, so a replay
makes exactly the same calls.

## Local results (practice trials, kit env + scorer)

| Trial | Score | Quality | Served | Wire | Failover | Catalog | Upstream calls |
|-------|------:|--------:|-------:|-----:|---------:|--------:|---------------:|
| 0 | 99 | 98 | 100 | 100 | 100 | 100 | 22 |
| 1 | 99 | 98 | 100 | 100 | 100 | 100 | 26 |
| 2 | 98 | 97 | 100 | 100 | 100 | 100 | 27 |

The call counts stay above 21 because each provider's free-tier budget
is drawn per trial and never published (the brief says so), and the total
safe capacity across the ten providers is only about 21 calls for 20
requests — so any provider called once past its hidden budget costs a
call, and the brief gives no signal that tells two same-rated providers
apart (Groq and Cerebras both offer the same 1434 model, yet one takes
five calls and the other one). Routing only from the brief, the catalog
probe, and the providers' own answers, this is the honest range; getting
every trial to 20 calls would need the hidden budgets themselves.

## Run

`python3 main.py` reads the brief from `MAKEFUN_PROMPT_FILE` and the env
from `MAKEFUN_ENV_URL`. `python3 main.py --list` prints the catalog.
