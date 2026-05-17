"""
main.py
-------
The FastAPI application. This is the entry point of the service.

This file is intentionally thin — it only handles HTTP concerns:
  - Defining request/response shapes (Pydantic models)
  - Routing incoming requests to agent.py
  - Startup: initializing the retriever and OpenAI client once

Everything else (retrieval, prompting, validation) lives in the other files.

Endpoints:
  GET  /health  → readiness check, always returns {"status": "ok"}
  POST /chat    → main conversational endpoint

To run locally:
  uvicorn main:app --reload --port 8000

To test:
  curl http://localhost:8000/health
  curl -X POST http://localhost:8000/chat \
    -H "Content-Type: application/json" \
    -d '{"messages": [{"role": "user", "content": "I need to hire a Java developer"}]}'
"""

import logging
import os
import time

from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from openai import OpenAI
from pydantic import BaseModel, Field, field_validator

from agent import get_agent_response
from retriever import SHLRetriever

# ─────────────────────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv()
# ─────────────────────────────────────────────────────────────────────────────
# Pydantic models — define the exact shape of requests and responses
#
# Pydantic does two things for us:
#   1. Validates incoming data — if the request is missing a required field,
#      FastAPI automatically returns a 422 error with a clear message
#   2. Documents the API — FastAPI uses these to auto-generate /docs
# ─────────────────────────────────────────────────────────────────────────────

class Message(BaseModel):
    """A single message in the conversation history."""

    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1)

    class Config:
        # Reject any extra fields not defined here
        extra = "forbid"


class ChatRequest(BaseModel):
    """
    Request body for POST /chat.

    The API is stateless — the caller sends the full conversation
    history on every request. We store nothing server-side.
    """

    messages: list[Message] = Field(
        ...,
        min_length=1,
        description="Full conversation history. Must have at least one message."
    )

    @field_validator("messages")
    @classmethod
    def last_message_must_be_user(cls, messages: list[Message]) -> list[Message]:
        """
        The last message must always be from the user.
        An assistant message at the end would mean the agent is responding
        to itself, which makes no sense.
        """
        if messages and messages[-1].role != "user":
            raise ValueError("The last message in the conversation must be from the user.")
        return messages

    class Config:
        extra = "forbid"


class RecommendationItem(BaseModel):
    """A single assessment recommendation."""

    name: str
    url: str
    test_type: str

    class Config:
        extra = "ignore"   # Allow extra fields (like _similarity_score) to be stripped


class ChatResponse(BaseModel):
    """
    Response body for POST /chat.

    This schema is non-negotiable — the automated evaluator depends on it.
    Do not rename fields. Do not add required fields.
    """

    reply: str = Field(
        ...,
        description="The agent's natural language response to the user."
    )
    recommendations: list[RecommendationItem] = Field(
        default=[],
        description=(
            "Empty list when clarifying/comparing/refusing. "
            "1-10 items when the agent has committed to a shortlist."
        )
    )
    end_of_conversation: bool = Field(
        default=False,
        description="True only when the user has explicitly confirmed the final shortlist."
    )

    class Config:
        extra = "ignore"


class HealthResponse(BaseModel):
    """Response body for GET /health."""
    status: Literal["ok"]


# ─────────────────────────────────────────────────────────────────────────────
# Application state
#
# We use a simple dict to hold shared resources that are created once
# at startup and reused across all requests. This is FastAPI's recommended
# pattern for managing application-level state.
# ─────────────────────────────────────────────────────────────────────────────

app_state: dict = {}


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan — startup and shutdown logic
#
# The @asynccontextmanager lifespan pattern is FastAPI's modern way
# to run code at startup and shutdown. It replaces the older @app.on_event
# decorators.
#
# Everything before `yield` runs at startup.
# Everything after `yield` runs at shutdown.
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: build the FAISS index and initialize the OpenAI client.
    These are expensive operations — we do them once and reuse across requests.

    Shutdown: nothing special needed (FAISS index is in-memory, no cleanup required).
    """
    logger.info("Starting up SHL Assessment Recommender...")

    # Check required environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY environment variable is not set. "
            "Set it before starting the server."
        )

    # Find catalog.json — look in current directory
    catalog_path = os.getenv("CATALOG_PATH", "catalog.json")
    if not os.path.exists(catalog_path):
        raise RuntimeError(
            f"Catalog file not found at '{catalog_path}'. "
            f"Set CATALOG_PATH env var or place catalog.json in the working directory."
        )

    # Initialize OpenAI client (reads OPENAI_API_KEY from environment automatically)
    client = OpenAI()
    app_state["client"] = client
    logger.info("OpenAI client initialized")

    # Build the FAISS retriever — this calls the OpenAI embeddings API
    # to embed the entire catalog. Takes ~10-30 seconds on first startup.
    logger.info("Building FAISS index from catalog (this may take a moment)...")
    start = time.time()
    retriever = SHLRetriever(catalog_path, client)
    elapsed = time.time() - start
    logger.info(
        f"FAISS index ready — {len(retriever.catalog)} assessments indexed in {elapsed:.1f}s"
    )

    app_state["retriever"] = retriever

    # Server is ready to handle requests
    yield

    # Shutdown
    logger.info("Shutting down...")
    app_state.clear()


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app instance
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="SHL Assessment Recommender",
    description=(
        "Conversational agent that recommends SHL Individual Test Solutions "
        "based on hiring needs."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ─────────────────────────────────────────────────────────────────────────────
# Middleware — request logging
#
# Logs every request with method, path, status code, and duration.
# Useful for debugging and monitoring on Render.
# ─────────────────────────────────────────────────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start) * 1000
    logger.info(
        f"{request.method} {request.url.path} "
        f"→ {response.status_code} "
        f"({duration_ms:.0f}ms)"
    )
    return response


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns 200 OK when the service is ready. Used by Render and the evaluator.",
)
async def health():
    """
    Readiness check endpoint.

    The evaluator calls this before running any conversations.
    For cold-start hosting (Render free tier), the first call allows
    up to 2 minutes for the service to wake up.

    This endpoint itself is always fast — the slow part (building the FAISS
    index) happens in the lifespan startup above.
    """
    return {"status": "ok"}


@app.post(
    "/chat",
    response_model=ChatResponse,
    summary="Conversational assessment recommender",
    description=(
        "Takes a full stateless conversation history and returns the next "
        "agent reply plus, when appropriate, a structured shortlist of "
        "SHL assessment recommendations."
    ),
)
async def chat(request: ChatRequest):
    """
    Main conversational endpoint.

    The caller is responsible for maintaining conversation history —
    every request must include the full history from turn 1.
    The server stores no per-conversation state.

    Request body:
        messages: List of {"role": "user"|"assistant", "content": "..."}
                  Must end with a user message.

    Response:
        reply: Agent's natural language response
        recommendations: [] or list of 1-10 SHL assessments
        end_of_conversation: true when user has confirmed the shortlist
    """
    # Convert Pydantic Message objects to plain dicts for agent.py
    # (OpenAI SDK expects plain dicts, not Pydantic models)
    messages_as_dicts = [
        {"role": msg.role, "content": msg.content}
        for msg in request.messages
    ]

    # Delegate to agent.py — all the real logic lives there
    result = get_agent_response(
        messages=messages_as_dicts,
        retriever=app_state["retriever"],
        client=app_state["client"],
    )

    # Pydantic will validate the output shape before sending
    return ChatResponse(**result)


# ─────────────────────────────────────────────────────────────────────────────
# Global exception handler
#
# Catches any unhandled exception that bubbles up through the entire stack.
# Returns a clean JSON error instead of an ugly 500 HTML page.
# ─────────────────────────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "reply": "Something went wrong on my end. Please try again.",
            "recommendations": [],
            "end_of_conversation": False,
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Local development entry point
#
# Run with:  python main.py
# Or better: uvicorn main:app --reload --port 8000
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,   # Auto-reload on file changes during development
        log_level="info",
    )