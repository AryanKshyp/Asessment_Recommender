"""
validator.py
------------
Validates and cleans every response that comes back from the LLM
before it is returned to the caller.

Why do we need this?
  LLMs are non-deterministic. Even with a strict prompt, they can:
    - Return malformed JSON
    - Hallucinate URLs that don't exist in the catalog
    - Use invalid test_type codes
    - Include more than 10 recommendations
    - Miss required fields entirely

  This file is the last line of defence. It runs AFTER the LLM responds
  and BEFORE we send anything back to the evaluator.

  The evaluator is automated. A single schema violation = 0 points for that turn.
  So we enforce everything in code, not just in the prompt.

Structure:
  - Constants: valid test_type codes, required response fields
  - parse_llm_response()   : extract JSON from raw LLM text
  - validate_recommendations(): clean the recommendations list
  - validate_response()    : top-level function — call this on every LLM output
"""

import json
import logging
import re

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# All valid single-letter test type codes found across the 10 example traces.
# Multi-type items use comma-separated codes, e.g. "K,S" or "P,C".
# We validate each individual letter after splitting on commas.
VALID_TEST_TYPE_CODES = {"A", "B", "C", "D", "K", "P", "S"}

# Full names that map to each code.
# Used to resolve test_type when the LLM returns a full name instead of a code.
TEST_TYPE_NAME_TO_CODE = {
    "ability & aptitude": "A",
    "ability and aptitude": "A",
    "biodata & situational judgment": "B",
    "biodata and situational judgment": "B",
    "situational judgment": "B",
    "competencies": "C",
    "development & 360": "D",
    "development and 360": "D",
    "knowledge & skills": "K",
    "knowledge and skills": "K",
    "personality & behavior": "P",
    "personality and behavior": "P",
    "personality": "P",
    "simulations": "S",
    "simulation": "S",
}

# Fields every valid response must have.
# If any of these are missing after LLM output, we add safe defaults.
REQUIRED_FIELDS = {"reply", "recommendations", "end_of_conversation"}

# Maximum recommendations per response (assignment requirement).
MAX_RECOMMENDATIONS = 10

# Evaluator cap: at most 8 messages (user + assistant combined) per request.
MAX_MESSAGES_LENGTH = 8


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Parse raw LLM text into a Python dict
# ─────────────────────────────────────────────────────────────────────────────

def parse_llm_response(raw_text: str) -> dict:
    """
    Extract a JSON object from the LLM's raw text output.

    We ask the LLM to return only JSON, but it sometimes:
      - Wraps it in markdown code fences: ```json { ... } ```
      - Adds a preamble: "Here is my response: { ... }"
      - Returns valid JSON directly: { ... }

    We handle all three cases.

    Args:
        raw_text: The raw string returned by the OpenAI chat completion

    Returns:
        Parsed dict if successful

    Raises:
        ValueError: If no valid JSON object can be found in the text
    """
    if not raw_text or not raw_text.strip():
        raise ValueError("LLM returned empty response")

    text = raw_text.strip()

    # Case 1: Try direct parse first — fastest path, works if LLM is well-behaved
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Case 2: Strip markdown code fences
    # Matches ```json ... ``` or ``` ... ```
    fence_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    fence_match = re.search(fence_pattern, text)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    # Case 3: Extract the first { ... } block from the text
    # This handles cases where the LLM adds prose before/after the JSON
    brace_pattern = r"\{[\s\S]*\}"
    brace_match = re.search(brace_pattern, text)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    # Nothing worked
    raise ValueError(
        f"Could not extract valid JSON from LLM response. "
        f"Raw text (first 200 chars): {text[:200]}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Validate and clean test_type field
# ─────────────────────────────────────────────────────────────────────────────

def validate_test_type(test_type_value: str | None) -> str:
    """
    Normalize and validate a test_type string.

    The LLM might return any of:
      - "K"                          → valid, return as-is
      - "K,S"                        → valid multi-type, return as-is
      - "Knowledge & Skills"         → resolve to "K"
      - "Knowledge & Skills, Simulations" → resolve to "K,S"
      - "X"                          → invalid code, return "K" as fallback
      - None or ""                   → return "" (unknown)

    Args:
        test_type_value: Raw test_type string from LLM

    Returns:
        Cleaned, validated test_type string
    """
    if not test_type_value:
        return ""

    raw = str(test_type_value).strip()

    # Split on commas to handle multi-type values like "K,S" or "P, C"
    parts = [p.strip() for p in raw.split(",") if p.strip()]

    resolved_codes = []
    for part in parts:
        # Check if it's already a valid single-letter code
        upper = part.upper()
        if upper in VALID_TEST_TYPE_CODES:
            resolved_codes.append(upper)
            continue

        # Try resolving from full name
        lower = part.lower()
        if lower in TEST_TYPE_NAME_TO_CODE:
            resolved_codes.append(TEST_TYPE_NAME_TO_CODE[lower])
            continue

        # Unknown — log it but don't crash
        logger.warning(f"Unknown test_type code or name: '{part}' — skipping")

    # Deduplicate while preserving order
    seen = set()
    unique_codes = []
    for code in resolved_codes:
        if code not in seen:
            seen.add(code)
            unique_codes.append(code)

    return ",".join(unique_codes)


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Validate the recommendations list
# ─────────────────────────────────────────────────────────────────────────────

def validate_recommendations(
    recommendations: list | None,
    valid_urls: set[str],
) -> list[dict]:
    """
    Clean the recommendations list returned by the LLM.

    Rules enforced here (in code, not just in the prompt):
      1. Must be a list (if null/None, return empty list)
      2. Each item must have 'name' and 'url'
      3. Every 'url' must exist in the catalog whitelist
         (prevents hallucinated URLs from reaching the evaluator)
      4. test_type is validated and normalized
      5. Maximum 10 items

    Args:
        recommendations: Raw recommendations value from LLM output
        valid_urls: Set of all valid catalog URLs (from SHLRetriever.valid_urls)

    Returns:
        Cleaned list of recommendation dicts, each with: name, url, test_type
    """
    # Handle null / missing
    if recommendations is None:
        return []

    # Handle case where LLM returns a non-list (e.g. a dict or a string)
    if not isinstance(recommendations, list):
        logger.warning(
            f"recommendations is not a list (got {type(recommendations).__name__}) "
            f"— converting to empty list"
        )
        return []

    cleaned = []
    for i, item in enumerate(recommendations):
        # Each item must be a dict
        if not isinstance(item, dict):
            logger.warning(f"Recommendation[{i}] is not a dict — skipping")
            continue

        name = item.get("name", "").strip()
        url = item.get("url", "").strip()
        test_type_raw = item.get("test_type", "")

        # Must have a name
        if not name:
            logger.warning(f"Recommendation[{i}] has no name — skipping")
            continue

        # Must have a URL
        if not url:
            logger.warning(f"Recommendation[{i}] '{name}' has no URL — skipping")
            continue

        # URL must be in the catalog whitelist
        # This is the most important check — it prevents hallucinated URLs
        if url not in valid_urls:
            logger.warning(
                f"Recommendation[{i}] '{name}' has URL not in catalog: '{url}' — removing"
            )
            continue

        # Validate and normalize test_type
        test_type = validate_test_type(test_type_raw)

        cleaned.append({
            "name": name,
            "url": url,
            "test_type": test_type,
        })

    # Enforce maximum of 10 recommendations
    if len(cleaned) > MAX_RECOMMENDATIONS:
        logger.warning(
            f"LLM returned {len(cleaned)} recommendations — truncating to {MAX_RECOMMENDATIONS}"
        )
        cleaned = cleaned[:MAX_RECOMMENDATIONS]

    return cleaned


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Top-level validate function — call this on every LLM response
# ─────────────────────────────────────────────────────────────────────────────

def validate_response(
    raw_llm_text: str,
    valid_urls: set[str],
    messages_length: int,
    previous_shortlist: list[dict] | None = None,
    enforce_closing: bool = False,
    is_comparison: bool = False,
    force_no_recommendations: bool = False,
) -> dict:
    """
    Parse and validate a raw LLM response into a clean, schema-compliant dict.

    This is the only function you need to call from agent.py.
    It runs all validation steps in order and always returns a valid response,
    even if the LLM output is completely broken.

    Args:
        raw_llm_text:    The raw string from the OpenAI API response
        valid_urls:      Set of all valid catalog URLs (from retriever)
        messages_length: Number of messages in the conversation so far
                         Used to force end_of_conversation at the turn cap
        previous_shortlist: Last validated recommendations (for closing/refinement)
        enforce_closing: When True, force end_of_conversation and prefer previous list
        is_comparison: When True, enforce comparison rules on recommendations
        force_no_recommendations: When True, strip all recommendations (clarify/refusal)

    Returns:
        A dict with exactly these keys:
          {
            "reply": str,
            "recommendations": list,   # [] or list of 1-10 items
            "end_of_conversation": bool
          }
    """

    # ── Parse JSON from raw text ──────────────────────────────────────────────
    try:
        parsed = parse_llm_response(raw_llm_text)
    except ValueError as e:
        # LLM returned something we cannot parse at all.
        # Return a safe fallback response rather than crashing.
        logger.error(f"Failed to parse LLM response: {e}")
        return _fallback_response(
            "I encountered an issue processing that request. Could you rephrase?"
        )

    # ── Ensure parsed value is a dict ─────────────────────────────────────────
    if not isinstance(parsed, dict):
        logger.error(f"LLM response parsed to {type(parsed).__name__}, not dict")
        return _fallback_response(
            "I encountered an issue processing that request. Could you rephrase?"
        )

    # ── Extract and validate each field ──────────────────────────────────────

    # reply: must be a non-empty string
    reply = parsed.get("reply", "")
    if not isinstance(reply, str) or not reply.strip():
        logger.warning("LLM returned empty or missing 'reply' — using fallback text")
        reply = "I need a moment to think about that. Could you give me more context?"

    # recommendations: validate against catalog whitelist
    raw_recommendations = parsed.get("recommendations", [])
    recommendations = validate_recommendations(raw_recommendations, valid_urls)

    if force_no_recommendations:
        if recommendations:
            logger.info("Clarify/refusal turn: stripping recommendations")
        recommendations = []

    # Comparison: grounded answer in reply; do not introduce new catalog items
    if is_comparison:
        prev_valid = validate_recommendations(previous_shortlist or [], valid_urls)
        if prev_valid:
            allowed_urls = {r["url"] for r in prev_valid}
            recommendations = [r for r in recommendations if r["url"] in allowed_urls]
        else:
            recommendations = []

    # Refinement: if the model wiped the prior list, merge back in
    if (
        previous_shortlist
        and not enforce_closing
        and not is_comparison
        and recommendations
    ):
        prev_valid = validate_recommendations(previous_shortlist, valid_urls)
        if prev_valid:
            prev_urls = {r["url"] for r in prev_valid}
            new_urls = {r["url"] for r in recommendations}
            if prev_urls and not prev_urls & new_urls:
                logger.warning("Refinement restarted shortlist — merging with previous")
                seen: set[str] = set()
                merged: list[dict] = []
                for item in prev_valid + recommendations:
                    url = item["url"]
                    if url not in seen:
                        seen.add(url)
                        merged.append(item)
                recommendations = merged[:MAX_RECOMMENDATIONS]

    # Closing turn: keep the prior shortlist (evaluator expects stability)
    if enforce_closing and previous_shortlist:
        pinned = validate_recommendations(previous_shortlist, valid_urls)
        if pinned:
            recommendations = pinned
            logger.info(
                f"Closing turn: enforced previous shortlist ({len(recommendations)} items)"
            )

    # end_of_conversation: must be a boolean
    eoc = parsed.get("end_of_conversation", False)
    if not isinstance(eoc, bool):
        # Sometimes LLM returns "true"/"false" as strings
        if isinstance(eoc, str):
            eoc = eoc.lower().strip() == "true"
        else:
            eoc = bool(eoc)

    if enforce_closing:
        eoc = True

    # ── Turn cap safety net ───────────────────────────────────────────────────
    if messages_length >= MAX_MESSAGES_LENGTH:
        if not eoc:
            logger.warning(
                f"Forcing end_of_conversation=true: messages_length={messages_length} "
                f">= MAX_MESSAGES_LENGTH={MAX_MESSAGES_LENGTH}"
            )
        eoc = True

    # ── Build final clean response ────────────────────────────────────────────
    return {
        "reply": reply.strip(),
        "recommendations": recommendations,
        "end_of_conversation": eoc,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _fallback_response(message: str) -> dict:
    """
    Return a safe, schema-compliant response when something goes wrong.
    Never crashes, never returns invalid schema.
    """
    return {
        "reply": message,
        "recommendations": [],
        "end_of_conversation": False,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Quick test — run this file directly: python validator.py
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Pretend these are the only two valid URLs in our catalog
    MOCK_VALID_URLS = {
        "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/",
        "https://www.shl.com/products/product-catalog/view/shl-verify-interactive-g/",
        "https://www.shl.com/products/product-catalog/view/core-java-advanced-level-new/",
    }

    print("=" * 60)
    print("TEST 1: Perfect JSON from LLM")
    raw = json.dumps({
        "reply": "Here are the assessments.",
        "recommendations": [
            {
                "name": "SHL Verify Interactive G+",
                "url": "https://www.shl.com/products/product-catalog/view/shl-verify-interactive-g/",
                "test_type": "A"
            },
            {
                "name": "Occupational Personality Questionnaire OPQ32r",
                "url": "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/",
                "test_type": "P"
            }
        ],
        "end_of_conversation": False
    })
    result = validate_response(raw, MOCK_VALID_URLS, messages_length=4)
    print(json.dumps(result, indent=2))

    print("\n" + "=" * 60)
    print("TEST 2: LLM wraps JSON in markdown fences")
    raw = '```json\n{"reply": "Got it.", "recommendations": [], "end_of_conversation": false}\n```'
    result = validate_response(raw, MOCK_VALID_URLS, messages_length=2)
    print(json.dumps(result, indent=2))

    print("\n" + "=" * 60)
    print("TEST 3: Hallucinated URL — should be stripped")
    raw = json.dumps({
        "reply": "Here you go.",
        "recommendations": [
            {
                "name": "Fake Assessment",
                "url": "https://www.shl.com/fake-url-not-in-catalog/",
                "test_type": "K"
            },
            {
                "name": "Core Java (Advanced Level) (New)",
                "url": "https://www.shl.com/products/product-catalog/view/core-java-advanced-level-new/",
                "test_type": "K"
            }
        ],
        "end_of_conversation": False
    })
    result = validate_response(raw, MOCK_VALID_URLS, messages_length=6)
    print(json.dumps(result, indent=2))
    print("→ Only Core Java should remain; Fake Assessment should be removed")

    print("\n" + "=" * 60)
    print("TEST 4: Full test_type name instead of code")
    raw = json.dumps({
        "reply": "Here you go.",
        "recommendations": [
            {
                "name": "SHL Verify Interactive G+",
                "url": "https://www.shl.com/products/product-catalog/view/shl-verify-interactive-g/",
                "test_type": "Ability & Aptitude"   # Full name, not code
            }
        ],
        "end_of_conversation": False
    })
    result = validate_response(raw, MOCK_VALID_URLS, messages_length=4)
    print(json.dumps(result, indent=2))
    print("→ test_type should be resolved to 'A'")

    print("\n" + "=" * 60)
    print("TEST 5: Turn cap hit — force end_of_conversation")
    raw = json.dumps({
        "reply": "Here are your assessments.",
        "recommendations": [
            {
                "name": "SHL Verify Interactive G+",
                "url": "https://www.shl.com/products/product-catalog/view/shl-verify-interactive-g/",
                "test_type": "A"
            }
        ],
        "end_of_conversation": False   # LLM says false, but turn cap overrides
    })
    result = validate_response(raw, MOCK_VALID_URLS, messages_length=8)
    print(json.dumps(result, indent=2))
    print("→ end_of_conversation should be True despite LLM saying False")

    print("\n" + "=" * 60)
    print("TEST 6: Completely broken LLM output")
    raw = "Sorry, I cannot help with that request at this time."
    result = validate_response(raw, MOCK_VALID_URLS, messages_length=4)
    print(json.dumps(result, indent=2))
    print("→ Should return a safe fallback response, not crash")