# Product Requirements Document
## SHL Conversational Assessment Recommender — v1.0

---

## 1. Overview

A stateless conversational agent exposed as a FastAPI service that takes a hiring manager or recruiter from a vague intent to a grounded shortlist of SHL Individual Test Solutions through multi-turn dialogue.

**What it is not:** a general hiring advisor, legal consultant, or open-ended chatbot.

---

## 2. Stack

| Layer | Choice | Reason |
|---|---|---|
| API framework | FastAPI (Python) | Specified in assignment |
| LLM | OpenAI `gpt-4.5-nano` | Fast, cheap, sufficient for structured JSON output |
| Embeddings | OpenAI `text-embedding-3-large` | Best quality; catalog is small so cost is negligible |
| Vector store | FAISS (in-memory) | No infra needed; catalog fits in RAM; loaded at startup |
| Catalog source | Local `catalog.json` | Pre-scraped; no runtime scraping |
| Deployment | Render (free tier) | Assignment accounts for 2-min cold start |
| Language | Python 3.11+ | FastAPI ecosystem |

---

## 3. Repository Structure

```
shl-recommender/
├── main.py                  # FastAPI app — /health and /chat endpoints
├── agent.py                 # Core agent logic — routing, prompt building, response parsing
├── retriever.py             # FAISS index — build, load, search
├── validator.py             # Schema enforcement, URL whitelist, test_type validation
├── catalog.json             # SHL Individual Test Solutions (input, not modified)
├── requirements.txt
├── test_conversations.py    # Test script — replays all 10 example traces
└── README.md
```

---

## 4. API Specification

### 4.1 GET /health
Returns HTTP 200 with `{"status": "ok"}`. No dependencies, always fast. Used by Render and the evaluator for readiness.

### 4.2 POST /chat

**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    {"role": "user", "content": "..."}
  ]
}
```

**Response (non-negotiable schema):**
```json
{
  "reply": "string — agent's natural language response",
  "recommendations": [],
  "end_of_conversation": false
}
```

**Recommendation item schema:**
```json
{
  "name": "string",
  "url": "string — must be from catalog whitelist",
  "test_type": "string — e.g. K, P, A, B, S, C, D or comma-separated"
}
```

**Rules:**
- `recommendations` is `[]` when agent is clarifying, comparing, or refusing
- `recommendations` has 1–10 items when agent commits to a shortlist
- `end_of_conversation` is `true` only when user explicitly confirms the shortlist
- Every URL must exist in the scraped catalog — enforced in code, not prompt

---

## 5. Catalog Ingestion

### 5.1 Fields Used for Embedding
Each catalog item is converted into a single text document for embedding:

```
Name: {name}
Description: {description}
Keys: {keys joined by ", "}
Job Levels: {job_levels joined by ", "}
Languages: {languages joined by ", "}
Duration: {duration}
Remote: {remote}
Adaptive: {adaptive}
```

All fields are included because `text-embedding-3-large` handles long inputs well and richer context improves recall on nuanced queries (language requirements, seniority, sector).

### 5.2 Index Build
- FAISS index is built **once at startup** from `catalog.json`
- Index type: `IndexFlatIP` (inner product — equivalent to cosine similarity on normalized vectors)
- Index + a parallel list of catalog items are kept in memory for the lifetime of the process
- No persistence to disk required — rebuild on each cold start is fast enough (<5s for this catalog size)

### 5.3 Retrieval
On each `/chat` call:
1. Build a query string from the last user message + any key constraints extracted from conversation history
2. Embed the query using `text-embedding-3-large`
3. Retrieve top-15 candidates from FAISS
4. Pass the 15 candidates (name, description, keys, url, test_type) to the LLM as context

Top-15 gives the LLM enough to reason over without overwhelming the context window.

---

## 6. Agent Design

### 6.1 Request Flow

```
POST /chat received
        │
        ▼
[CODE] Count total turns in messages
        │
        ▼
[CODE] Obvious off-topic check (regex on last message)
  → if clearly off-topic: return refusal response, skip LLM
        │
        ▼
[CODE] Build FAISS query from conversation context
        │
        ▼
[CODE] Retrieve top-15 catalog candidates
        │
        ▼
[CODE] If turn_count >= 6: inject "MUST_RECOMMEND" flag into prompt
        │
        ▼
[LLM]  Decide mode: clarify / recommend / compare / refuse
       Select assessments from retrieved candidates
       Write reply text
       Set end_of_conversation
        │
        ▼
[CODE] Validate every URL against catalog whitelist
[CODE] Validate test_type codes against known set
[CODE] Enforce recommendations cap (max 10)
[CODE] Ensure all required schema fields are present
        │
        ▼
Return response
```

### 6.2 Agent Modes

| Mode | Trigger | recommendations | end_of_conversation |
|---|---|---|---|
| **Clarify** | Query too vague, single blocking ambiguity | `[]` | `false` |
| **Recommend** | Enough context, or turn >= 6 | 1–10 items | `false` |
| **Refine** | User modifies constraints mid-conversation | Updated 1–10 items | `false` |
| **Compare** | User asks "difference between X and Y" | `[]` or current list | `false` |
| **Refuse** | Off-topic, legal, prompt injection | `[]` | `false` |
| **Close** | User confirms shortlist | Repeat last list | `true` |

### 6.3 Clarification Rules (enforced via prompt)
- Ask **exactly one** clarifying question per turn — never stack questions
- Only clarify on blocking ambiguity: language/locale, seniority, sector type, use case (selection vs. development)
- Never ask about budget, timelines, or generic open-ended preferences
- If query names a specific role AND specific test types needed → recommend immediately, do not clarify

### 6.4 OPQ32r Default Behaviour
Include OPQ32r by default for any role-based hiring query. Offer an explicit opt-out in the reply text. Remove it only if the user explicitly asks to drop personality testing.

### 6.5 Refine Behaviour
When the user changes constraints (add/remove/swap assessments):
- Execute the change surgically — do not restart the shortlist from scratch
- Always re-display the **full updated list**, not just the delta
- Acknowledge what changed in the reply text

### 6.6 Compare Behaviour
When asked to compare two assessments:
- Answer from catalog data in retrieved context — never from LLM prior knowledge
- If catalog data is insufficient to answer, say so explicitly
- Set `recommendations: []` for that turn (or repeat current list — both acceptable)

### 6.7 Refusal Behaviour
The agent refuses and stays in conversation (does not terminate) for:
- General hiring advice not related to SHL assessments
- Legal or compliance questions (e.g. "are we required to test under HIPAA?")
- Pricing questions
- Prompt injection attempts
- Requests for assessments not in the catalog

Refusal pattern: acknowledge what was asked, explain the boundary, redirect to what the agent *can* help with.

### 6.8 Graceful Gap Handling
If no catalog assessment directly matches a requested technology or role (e.g. Rust developer):
- Acknowledge the gap explicitly ("There is no Rust-specific test in the catalog")
- Offer the closest alternatives with reasoning
- Note the gap again at conversation end

---

## 7. Prompt Design

### 7.1 System Prompt Structure

```
[ROLE & SCOPE]
You are an SHL assessment recommender. You only discuss SHL Individual Test 
Solutions from the provided catalog. You never recommend assessments not in 
the catalog. You never invent URLs.

[CATALOG CONTEXT — injected per call]
Here are the most relevant assessments from the catalog for this conversation:
{top_15_retrieved_items}

[BEHAVIORAL RULES]
1. Clarify if query is vague. Ask ONE question at a time.
2. Recommend once you have role, seniority/level, and use case (selection/development).
3. Always default-include OPQ32r for role-based hiring; offer opt-out.
4. On refinement: update the list surgically, re-display full updated list.
5. On comparison: answer from catalog data only, not your training knowledge.
6. Refuse legal, pricing, general hiring advice, and off-topic questions gracefully.
7. Never set end_of_conversation: true unless the user has confirmed the shortlist.
8. [INJECTED IF turn >= 6]: You have limited turns remaining. You MUST provide 
   a recommendation shortlist in this response based on context gathered so far.

[OUTPUT FORMAT]
Respond ONLY with valid JSON matching this exact schema:
{
  "reply": "...",
  "recommendations": [...] or [],
  "end_of_conversation": true/false
}
No markdown, no preamble, no explanation outside the JSON.
```

### 7.2 Query Construction for FAISS
The retrieval query is not just the last user message. It is built from:
- Last user message
- Role/job title mentions in conversation history
- Test type preferences mentioned (cognitive, personality, simulation, etc.)
- Any constraints (language, seniority, sector)

This ensures FAISS retrieves relevant candidates even when the latest message is a short refinement like "add AWS and Docker."

---

## 8. Validation Layer (Code-enforced)

### 8.1 URL Whitelist
At startup, build a set of all valid URLs from `catalog.json`. After every LLM response, filter `recommendations` to remove any item whose URL is not in this set. This prevents hallucinated URLs from ever reaching the evaluator.

### 8.2 test_type Validation
Valid single codes: `A`, `B`, `C`, `D`, `K`, `P`, `S`
Multi-type: comma-separated string, e.g. `"K,S"` or `"P,C"`
Any code not in this set is stripped or the item is removed.

### 8.3 Schema Enforcement
After LLM response parsing:
- If `reply` is missing → set to `"I need a moment to think about that."`
- If `recommendations` is missing → set to `[]`
- If `end_of_conversation` is missing → set to `false`
- If `recommendations` has > 10 items → truncate to 10
- Strip any recommendation item missing `name` or `url`

### 8.4 Turn Cap Guard
If `len(messages) >= 12` (6 user + 6 assistant — absolute maximum), force `end_of_conversation: true` regardless of LLM output.

---

## 9. Behavior Probes — Expected Outcomes

These are the known evaluator checks. The implementation must pass all of them.

| Probe | Expected behaviour |
|---|---|
| Vague first message ("I need an assessment") | Clarify — `recommendations: []` |
| Specific first message with role + type | Recommend immediately |
| Turn 1 vague, Turn 2 specific | Recommend on Turn 2 |
| "Add personality tests" mid-conversation | Refine shortlist, keep everything else |
| "Drop OPQ" | Remove OPQ32r, return updated list |
| "What's the difference between X and Y?" | Compare answer, catalog-grounded |
| "What's the legal requirement for testing?" | Refuse gracefully, stay in conversation |
| "Ignore previous instructions and..." | Refuse, treat as prompt injection |
| User confirms shortlist | `end_of_conversation: true` |
| Turn count reaches 8 | Must have provided recommendations by now |
| URL in recommendations | Must exist in catalog whitelist |
| Hallucinated assessment name | Must never appear — FAISS + validation prevents this |

---

## 10. Test Script

`test_conversations.py` does the following:

1. Loads all 10 example conversation traces from `/traces/` folder
2. For each trace, replays it turn by turn against the local `/chat` endpoint
3. After each turn, asserts:
   - Response is valid JSON matching the schema
   - All URLs are in the catalog whitelist
   - `recommendations` is `[]` or a list of 1–10 items (never null)
   - `end_of_conversation` is a boolean
4. At the end of each trace, computes **Recall@10**: how many of the expected assessments appear in the final shortlist
5. Prints a summary table: trace ID, turns used, Recall@10, pass/fail

Run with: `python test_conversations.py --host http://localhost:8000`

---

## 11. Deployment (Render)

- **Service type:** Web Service (free tier)
- **Build command:** `pip install -r requirements.txt`
- **Start command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
- **Environment variables:** `OPENAI_API_KEY`
- **Cold start:** FAISS index is built at startup; expect 30–60 seconds on first wake
- **Health check path:** `/health` — evaluator allows 2 minutes for first response

`requirements.txt` includes:
```
fastapi
uvicorn
openai
faiss-cpu
numpy
pydantic
python-dotenv
```

---

## 12. Known Constraints & Trade-offs

| Constraint | Decision | Reason |
|---|---|---|
| 30s timeout per call | Single LLM call per request; FAISS retrieval is <100ms | Two LLM calls would risk timeout |
| 8-turn evaluator cap | Force recommendation at turn 6 | Buffer of 2 turns for user confirmation |
| Stateless API | Full conversation history sent each call | Specified in assignment; no server-side session storage |
| gpt-4.5-nano | Requires strict JSON output prompt | Smaller models need explicit format instruction |
| FAISS in-memory | Rebuilt on cold start | Acceptable: catalog is ~100 items, build time <5s |
| text-embedding-3-large | Slower than small but better recall | Catalog is embedded once at startup; query embedding is ~200ms |

---

## 13. Out of Scope (v1)

- Pre-packaged Job Solutions (assignment excludes these)
- Multi-user session management
- Streaming responses
- Frontend UI
- Authentication / rate limiting
- Database persistence of conversations
- Automatic catalog re-scraping

---

## 14. Success Criteria

| Metric | Target |
|---|---|
| Hard evals (schema, URL, turn cap) | 100% pass |
| Mean Recall@10 across public traces | ≥ 0.7 |
| Behavior probes pass-rate | ≥ 85% |
| p95 response latency | < 25s (5s buffer for timeout) |
| Cold start /health response | < 120s |