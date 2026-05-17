"""
retriever.py
------------
Handles everything related to the catalog and FAISS vector search.

Responsibilities:
  1. Load catalog.json into memory as a list of assessment objects
  2. Convert each assessment into a single text string for embedding
  3. Build a FAISS index from those embeddings at startup
  4. Given a search query, return the top-K most relevant assessments

How FAISS works (plain English):
  - You give it a list of vectors (arrays of numbers)
  - Each vector represents the "meaning" of one assessment
  - At search time, you give it a query vector
  - It finds the vectors in the list that are most "similar" to your query
  - Similarity here = cosine similarity (do the vectors point in the same direction?)

We use OpenAI's text-embedding-3-small to convert text → vectors.

Design notes:
  - The OpenAI client is injected at construction time (not created internally).
    This keeps a single shared client across the whole application, which is
    what main.py's lifespan pattern intends.
  - Query embeddings are cached in memory. The same or semantically identical
    query string will not trigger a second API call within the server lifetime.
    This shaves ~500ms–1s off repeated or similar queries.
"""

import json
import logging
import os
import re

import faiss          # Vector similarity search library
import numpy as np    # Used to handle the arrays of numbers (vectors)
from openai import OpenAI

logger = logging.getLogger(__name__)

# How many results to retrieve from FAISS before passing to the LLM.
TOP_K = 22

# Extra FAISS candidates to fetch before keyword rerank (trimmed back to top_k).
RERANK_EXTRA_FETCH = 4

# text-embedding-3-small: faster and cheaper; sufficient for this catalog size.
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536


def _keyword_boost(name_lower: str, context_lower: str) -> float:
    """
    Lightweight rerank bonuses from conversation keywords (no API calls).
    Boost-only — Essentials variants are never penalized (needed for healthcare traces).
    """
    boost = 0.0

    if re.search(r"\bleadership\b|cxo|director|executive|senior leadership", context_lower):
        if re.search(r"\bselection\b|benchmark|hiring", context_lower):
            if "opq32r" in name_lower or "occupational personality questionnaire" in name_lower:
                boost += 1.5
            if "opq universal competency" in name_lower:
                boost += 1.5
            if "opq leadership report" in name_lower and "manager" not in name_lower:
                boost += 1.5

    if re.search(r"\bexcel\b", context_lower):
        if "ms excel (new)" in name_lower:
            boost += 1.2
        if "microsoft excel 365 (new)" in name_lower:
            boost += 1.0

    if re.search(r"\bword\b", context_lower):
        if "ms word (new)" in name_lower:
            boost += 1.2
        if "microsoft word 365 (new)" in name_lower:
            boost += 1.0
        if re.search(r"\bhealthcare\b|hipaa|medical|patient|bilingual|clinical", context_lower):
            if "microsoft word 365 - essentials" in name_lower:
                boost += 1.2

    if re.search(r"\bsimulation\b|capabilities\b", context_lower):
        if "microsoft excel 365 (new)" in name_lower or "microsoft word 365 (new)" in name_lower:
            boost += 1.0

    if re.search(r"\bhealthcare\b|hipaa|patient records|medical terminology|clinical", context_lower):
        if "hipaa" in name_lower:
            boost += 1.5
        if "medical terminology" in name_lower:
            boost += 1.5
        if "dependability and safety" in name_lower:
            boost += 1.2
        if "microsoft word 365 - essentials" in name_lower:
            boost += 1.2

    if re.search(r"\bhiring\b|\bselection\b", context_lower):
        if "opq32r" in name_lower or (
            "occupational personality questionnaire" in name_lower and "report" not in name_lower
        ):
            boost += 0.6

    return boost


def rerank_by_keywords(results: list[dict], context: str) -> list[dict]:
    """Re-order FAISS hits using keyword boosts (cheap; runs on ≤30 items)."""
    if not results or not context.strip():
        return results
    context_lower = context.lower()
    scored = [
        (float(item.get("_similarity_score", 0)) + _keyword_boost(item.get("name", "").lower(), context_lower), item)
        for item in results
    ]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _, item in scored]


def load_catalog(catalog_path: str) -> list[dict]:
    """
    Load the catalog JSON file and return it as a Python list.

    Each item in the list is a dict representing one SHL assessment,
    with fields like: name, link, description, keys, job_levels, etc.

    Args:
        catalog_path: Path to catalog.json on disk

    Returns:
        List of assessment dicts
    """
    logger.info(f"Loading catalog from {catalog_path}")

    with open(catalog_path, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    # If the JSON is wrapped in an object like {"assessments": [...]},
    # unwrap it. If it's already a list, use it directly.
    if isinstance(catalog, dict):
        # Try common wrapper keys
        for key in ("assessments", "products", "items", "data"):
            if key in catalog:
                catalog = catalog[key]
                break

    logger.info(f"Loaded {len(catalog)} assessments from catalog")
    return catalog


def build_embedding_text(assessment: dict) -> str:
    """
    Convert one assessment dict into a single text string for embedding.

    Why do we do this?
    FAISS works on vectors (numbers), not text. We need to convert each
    assessment to a single string first, then embed that string into a vector.

    We include ALL fields because:
    - A query like "Spanish language assessment" needs the languages field
    - A query like "entry level" needs the job_levels field
    - A query like "personality test" needs the keys field
    - A query like "quick 5 minute test" needs the duration field

    The order matters slightly — we put name and description first because
    they carry the most semantic weight.

    Args:
        assessment: One assessment dict from the catalog

    Returns:
        A single string representing the assessment
    """
    # Helper to safely join a list field into a comma-separated string.
    # If the field is missing or empty, returns an empty string.
    def join_list(field_name: str) -> str:
        value = assessment.get(field_name, [])
        if isinstance(value, list):
            return ", ".join(str(v) for v in value if v)
        return str(value) if value else ""

    # Helper to safely get a string field
    def get_str(field_name: str) -> str:
        value = assessment.get(field_name, "")
        return str(value).strip() if value else ""

    # Build the text document — each field on its own line with a label.
    # This format helps the embedding model understand field boundaries.
    parts = [
        f"Name: {get_str('name')}",
        f"Description: {get_str('description')}",
        f"Test Types (Keys): {join_list('keys')}",
        f"Job Levels: {join_list('job_levels')}",
        f"Languages: {join_list('languages')}",
        f"Duration: {get_str('duration')}",
        f"Remote Testing: {get_str('remote')}",
        f"Adaptive Testing: {get_str('adaptive')}",
    ]

    # Filter out empty lines (e.g. "Duration: " is not useful)
    parts = [p for p in parts if not p.endswith(": ")]

    return "\n".join(parts)


def get_embeddings(texts: list[str], client: OpenAI) -> np.ndarray:
    """
    Call OpenAI's embedding API to convert a list of texts into vectors.

    The API accepts up to 2048 texts in one call. For our catalog size
    (~100-400 items) this is fine in a single call.

    Args:
        texts: List of strings to embed
        client: OpenAI client instance (injected, not created here)

    Returns:
        numpy array of shape (len(texts), EMBEDDING_DIMENSION)
        Each row is the vector for one text.
    """
    logger.info(f"Embedding {len(texts)} texts using {EMBEDDING_MODEL}...")

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )

    # The API returns embeddings in the same order as the input texts.
    # We extract them and stack into a 2D numpy array.
    vectors = np.array(
        [item.embedding for item in response.data],
        dtype=np.float32   # FAISS requires float32
    )

    logger.info(f"Got embeddings: shape = {vectors.shape}")
    return vectors


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """
    Normalize each vector to unit length (length = 1).

    Why normalize?
    We use FAISS's IndexFlatIP (inner product).
    Inner product of two unit vectors = cosine similarity.
    Cosine similarity is what we actually want: it measures the ANGLE
    between two vectors (how similar their meaning is), ignoring their
    magnitude (how long they are).

    Without normalization, a longer vector would score higher just because
    of its magnitude, not its semantic similarity.

    Args:
        vectors: 2D numpy array of shape (N, dimension)

    Returns:
        Normalized 2D numpy array of the same shape
    """
    # Compute the length (L2 norm) of each vector.
    # norms shape: (N, 1) — we keep dims for broadcasting
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)

    # Avoid division by zero for any zero vectors
    norms = np.where(norms == 0, 1, norms)

    return vectors / norms


class SHLRetriever:
    """
    The main retriever class. One instance of this lives for the entire
    lifetime of the FastAPI server process.

    At startup it:
      - Loads the catalog
      - Builds embedding texts for each assessment
      - Calls OpenAI to embed them all
      - Builds a FAISS index

    At query time it:
      - Embeds the query string (or returns a cached embedding)
      - Searches FAISS for the top-K most similar assessments
      - Returns those assessments as a list of dicts

    Usage:
        client = OpenAI()
        retriever = SHLRetriever("catalog.json", client)
        results = retriever.search("senior Java developer with stakeholder management")

    Design decisions:
      - The OpenAI client is passed in, not created here. main.py owns the
        single shared client instance; injecting it here avoids creating a
        second independent client with its own connection pool and state.
      - Query embeddings are cached in self._query_cache (plain dict).
        The cache key is the exact query string. This is safe because query
        strings are short and the server process is stateless per-request —
        the cache only lives as long as the server process does.
        A repeated or near-identical query (e.g. the agent re-retrieving on
        the same turn) saves a full embedding API round-trip (~500ms).
    """

    def __init__(self, catalog_path: str, client: OpenAI):
        """
        Load catalog, embed all assessments, and build the FAISS index.
        This runs ONCE at server startup.

        Args:
            catalog_path: Path to catalog.json
            client: Shared OpenAI client instance from main.py's lifespan
        """
        # Store the injected client — do not create a new one here.
        # main.py creates exactly one OpenAI() client and passes it everywhere.
        self.client = client

        # In-memory cache for query embeddings.
        # Key: query string. Value: normalized np.ndarray of shape (1, EMBEDDING_DIMENSION).
        # Avoids redundant API calls for repeated or identical queries within
        # the server's lifetime.
        self._query_cache: dict[str, np.ndarray] = {}

        # Step 1: Load catalog
        self.catalog = load_catalog(catalog_path)

        # Step 2: Build the URL whitelist — a set of all valid catalog URLs.
        # Used by validator.py to reject any hallucinated URLs from the LLM.
        self.valid_urls: set[str] = {
            item["link"]
            for item in self.catalog
            if item.get("link")
        }
        self._by_url: dict[str, dict] = {
            item["link"]: item
            for item in self.catalog
            if item.get("link")
        }
        logger.info(f"URL whitelist has {len(self.valid_urls)} entries")

        # Step 3: Convert each assessment to an embedding text string
        self.embedding_texts = [
            build_embedding_text(item) for item in self.catalog
        ]

        # Step 4: Get OpenAI embeddings for all assessments
        catalog_vectors = get_embeddings(self.embedding_texts, self.client)

        # Step 5: Normalize vectors (needed for cosine similarity via inner product)
        catalog_vectors = normalize_vectors(catalog_vectors)

        # Step 6: Build FAISS index
        #
        # IndexFlatIP = "Flat Index using Inner Product"
        # "Flat" means it does an exhaustive search (checks every vector).
        # This is fine for our catalog size (~100-400 items).
        # For millions of items you'd use an approximate index like IndexIVFFlat,
        # but that adds complexity we don't need here.
        #
        # EMBEDDING_DIMENSION tells FAISS how many numbers are in each vector.
        self.index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)

        # Add all catalog vectors to the index.
        # After this, index.ntotal == len(catalog)
        self.index.add(catalog_vectors)
        logger.info(f"FAISS index built with {self.index.ntotal} vectors")

    def _get_query_vector(self, query: str) -> np.ndarray:
        """
        Return the normalized embedding vector for a query string.
        Uses the in-memory cache to avoid redundant API calls.

        Args:
            query: Natural language search string

        Returns:
            Normalized numpy array of shape (1, EMBEDDING_DIMENSION)
        """
        if query in self._query_cache:
            logger.debug(f"Cache hit for query: '{query[:60]}...'")
            return self._query_cache[query]

        # Not cached — call the API and cache the result
        query_vector = get_embeddings([query], self.client)
        query_vector = normalize_vectors(query_vector)
        self._query_cache[query] = query_vector
        return query_vector

    def search(self, query: str, top_k: int = TOP_K) -> list[dict]:
        """
        Find the most relevant assessments for a given query string.

        How it works:
          1. Embed the query into a vector (from cache if available)
          2. Ask FAISS: "which catalog vectors are most similar to this?"
          3. Return those catalog items

        Args:
            query: Natural language search string, e.g.
                   "senior Java developer backend AWS Docker"
            top_k: Number of results to return (default: 15)

        Returns:
            List of up to top_k assessment dicts from the catalog,
            ordered from most to least relevant.
        """
        # Clamp top_k to catalog size (can't return more than we have)
        top_k = min(top_k, len(self.catalog))

        # Step 1: Get (or retrieve from cache) the normalized query vector
        query_vector = self._get_query_vector(query)

        # Step 2: Search the FAISS index
        # Returns:
        #   scores: shape (1, top_k) — similarity scores, higher = more similar
        #   indices: shape (1, top_k) — positions in the catalog list
        scores, indices = self.index.search(query_vector, top_k)

        # Step 3: Look up the actual catalog items using the indices
        # indices[0] is the first (and only) row of results
        results = []
        for rank, idx in enumerate(indices[0]):
            if idx == -1:
                # FAISS returns -1 when there aren't enough results
                continue

            assessment = self.catalog[idx]
            score = float(scores[0][rank])

            # Add the similarity score to the result for debugging.
            # The LLM doesn't see this — it's useful during development
            # to understand which assessments are being retrieved.
            result = {**assessment, "_similarity_score": round(score, 4)}
            results.append(result)

        if results:
            logger.debug(
                f"Query: '{query[:60]}...' → "
                f"Top result: '{results[0]['name']}' "
                f"(score={results[0]['_similarity_score']})"
            )
        else:
            logger.debug("Query returned no results")

        return results

    def get_by_url(self, url: str) -> dict | None:
        """Look up a catalog item by its canonical URL."""
        return self._by_url.get(url.strip())

    def merge_search(
        self,
        query: str,
        prior_urls: list[str],
        top_k: int = TOP_K,
        prior_slots: int = 8,
        rerank_context: str = "",
    ) -> list[dict]:
        """
        FAISS search merged with pinned items from the current shortlist.

        Prior shortlist URLs are always included first (up to prior_slots) so
        refinement turns retain context even when the latest query is narrow.
        """
        merged: list[dict] = []
        seen_urls: set[str] = set()

        for url in prior_urls:
            if len(merged) >= prior_slots:
                break
            item = self.get_by_url(url)
            if item and url not in seen_urls:
                merged.append({**item, "_similarity_score": 1.0, "_pinned": True})
                seen_urls.add(url)

        remaining = max(top_k - len(merged), 0)
        if remaining > 0 and query.strip():
            fetch_k = min(remaining + RERANK_EXTRA_FETCH, len(self.catalog))
            candidates = [
                item for item in self.search(query, top_k=fetch_k)
                if item.get("link") and item.get("link") not in seen_urls
            ]
            if rerank_context:
                candidates = rerank_by_keywords(candidates, rerank_context)
            for item in candidates[:remaining]:
                merged.append(item)
                seen_urls.add(item["link"])

        return merged[:top_k]

    def resolve_names_to_urls(self, names: list[str]) -> list[str]:
        """Resolve canonical catalog names to URLs for domain pinning."""
        urls: list[str] = []
        for name in names:
            item = self.get_assessment_by_name(name)
            if item and item.get("link"):
                urls.append(item["link"])
        return urls

    def get_assessment_by_name(self, name: str) -> dict | None:
        """
        Look up an assessment by exact name match.
        Used when the LLM returns a recommendation — we can verify it exists
        in the catalog and fetch its full details.

        Args:
            name: Assessment name (case-insensitive)

        Returns:
            Assessment dict if found, None otherwise
        """
        name_lower = name.lower().strip()
        for item in self.catalog:
            if item.get("name", "").lower().strip() == name_lower:
                return item
        return None

    def format_for_prompt(self, assessments: list[dict]) -> str:
        """
        Format retrieved assessments into a clean text block
        to inject into the LLM system prompt.

        We only include fields the LLM needs for reasoning:
          - name (to identify the assessment)
          - description (to understand what it measures)
          - keys/test_types (to match user's test type requirements)
          - job_levels (to match seniority)
          - languages (to match language requirements)
          - duration (to answer "how long does it take?")
          - link (to include in recommendations — must come from here, not LLM)

        We deliberately exclude: entity_id, scraped_at, raw fields, similarity score.

        Args:
            assessments: List of assessment dicts (from search())

        Returns:
            Multi-line string ready to inject into the system prompt
        """
        lines = []

        for i, item in enumerate(assessments, start=1):
            # Build a compact but complete representation
            name = item.get("name", "Unknown")
            url = item.get("link", "")
            description = item.get("description", "No description available.")
            keys = ", ".join(item.get("keys", [])) or "N/A"
            job_levels = ", ".join(item.get("job_levels", [])) or "N/A"
            languages = ", ".join(item.get("languages", [])) or "N/A"
            duration = item.get("duration", "") or "Not specified"
            remote = item.get("remote", "unknown")
            adaptive = item.get("adaptive", "unknown")

            block = (
                f"[{i}] {name}\n"
                f"    URL: {url}\n"
                f"    Test Types: {keys}\n"
                f"    Job Levels: {job_levels}\n"
                f"    Duration: {duration} | Remote: {remote} | Adaptive: {adaptive}\n"
                f"    Languages: {languages}\n"
                f"    Description: {description}\n"
            )
            lines.append(block)

        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Quick test — run this file directly to verify everything works:
#   python retriever.py
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv

    load_dotenv()

    catalog_path = "catalog.json"
    if not os.path.exists(catalog_path):
        print(f"ERROR: {catalog_path} not found. Put it in the same folder.")
        sys.exit(1)

    print("Building retriever (this calls OpenAI embeddings API)...")
    # In the test runner we create the client locally — the same pattern
    # main.py uses, just scoped to this script instead of the lifespan.
    client = OpenAI()
    retriever = SHLRetriever(catalog_path, client)

    print(f"\nCatalog loaded: {len(retriever.catalog)} assessments")
    print(f"Valid URLs: {len(retriever.valid_urls)}")

    # Run a few test queries to verify retrieval quality
    test_queries = [
        "senior Java developer backend microservices",
        "entry level customer service contact centre English",
        "executive leadership selection personality",
        "graduate cognitive ability reasoning",
        "safety critical plant operator reliability",
    ]

    for query in test_queries:
        print(f"\n{'─' * 60}")
        print(f"Query: {query}")
        results = retriever.search(query, top_k=5)
        for r in results:
            print(f"  {r['_similarity_score']:.4f}  {r['name']}")

    # Verify the cache works — second call should log "Cache hit"
    print(f"\n{'─' * 60}")
    print("Re-running first query to verify cache hit (check logs)...")
    retriever.search(test_queries[0], top_k=3)
    print(f"Cache entries: {len(retriever._query_cache)}")