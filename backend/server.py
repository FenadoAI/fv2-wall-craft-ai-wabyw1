"""FastAPI server exposing AI agent endpoints."""

import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI, HTTPException, Request
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware

from ai_agents import AgentConfig, ChatAgent, SearchAgent, ImageAgent


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).parent


class StatusCheck(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_name: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class StatusCheckCreate(BaseModel):
    client_name: str


class ChatRequest(BaseModel):
    message: str
    agent_type: str = "chat"
    context: Optional[dict] = None


class ChatResponse(BaseModel):
    success: bool
    response: str
    agent_type: str
    capabilities: List[str]
    metadata: dict = Field(default_factory=dict)
    error: Optional[str] = None


class SearchRequest(BaseModel):
    query: str
    max_results: int = 5


class SearchResponse(BaseModel):
    success: bool
    query: str
    summary: str
    search_results: Optional[dict] = None
    sources_count: int
    error: Optional[str] = None


class Wallpaper(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    prompt: str
    image_url: str
    image_data: str
    likes: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class WallpaperCreate(BaseModel):
    prompt: str


class WallpaperResponse(BaseModel):
    success: bool
    wallpaper: Optional[dict] = None
    error: Optional[str] = None


class LikeWallpaperResponse(BaseModel):
    success: bool
    likes: int
    error: Optional[str] = None


def _ensure_db(request: Request):
    try:
        return request.app.state.db
    except AttributeError as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=503, detail="Database not ready") from exc


def _get_agent_cache(request: Request) -> Dict[str, object]:
    if not hasattr(request.app.state, "agent_cache"):
        request.app.state.agent_cache = {}
    return request.app.state.agent_cache


async def _get_or_create_agent(request: Request, agent_type: str):
    cache = _get_agent_cache(request)
    if agent_type in cache:
        return cache[agent_type]

    config: AgentConfig = request.app.state.agent_config

    if agent_type == "search":
        cache[agent_type] = SearchAgent(config)
    elif agent_type == "chat":
        cache[agent_type] = ChatAgent(config)
    elif agent_type == "image":
        cache[agent_type] = ImageAgent(config)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown agent type '{agent_type}'")

    return cache[agent_type]


async def _check_rate_limit(request: Request, client_ip: str) -> bool:
    """Check if IP has exceeded 5 generations per day"""
    db = _ensure_db(request)
    today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)

    count = await db.wallpapers.count_documents({
        "ip_address": client_ip,
        "created_at": {"$gte": today_start}
    })

    return count < 5


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv(ROOT_DIR / ".env")

    mongo_url = os.getenv("MONGO_URL")
    db_name = os.getenv("DB_NAME")

    if not mongo_url or not db_name:
        missing = [name for name, value in {"MONGO_URL": mongo_url, "DB_NAME": db_name}.items() if not value]
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

    client = AsyncIOMotorClient(mongo_url)

    try:
        app.state.mongo_client = client
        app.state.db = client[db_name]
        app.state.agent_config = AgentConfig()
        app.state.agent_cache = {}
        logger.info("AI Agents API starting up")
        yield
    finally:
        client.close()
        logger.info("AI Agents API shutdown complete")


app = FastAPI(
    title="AI Agents API",
    description="Minimal AI Agents API with LangGraph and MCP support",
    lifespan=lifespan,
)

api_router = APIRouter(prefix="/api")


@api_router.get("/")
async def root():
    return {"message": "Hello World"}


@api_router.post("/status", response_model=StatusCheck)
async def create_status_check(input: StatusCheckCreate, request: Request):
    db = _ensure_db(request)
    status_obj = StatusCheck(**input.model_dump())
    await db.status_checks.insert_one(status_obj.model_dump())
    return status_obj


@api_router.get("/status", response_model=List[StatusCheck])
async def get_status_checks(request: Request):
    db = _ensure_db(request)
    status_checks = await db.status_checks.find().to_list(1000)
    return [StatusCheck(**status_check) for status_check in status_checks]


@api_router.post("/chat", response_model=ChatResponse)
async def chat_with_agent(chat_request: ChatRequest, request: Request):
    try:
        agent = await _get_or_create_agent(request, chat_request.agent_type)
        response = await agent.execute(chat_request.message)

        return ChatResponse(
            success=response.success,
            response=response.content,
            agent_type=chat_request.agent_type,
            capabilities=agent.get_capabilities(),
            metadata=response.metadata,
            error=response.error,
        )
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Error in chat endpoint")
        return ChatResponse(
            success=False,
            response="",
            agent_type=chat_request.agent_type,
            capabilities=[],
            error=str(exc),
        )


@api_router.post("/search", response_model=SearchResponse)
async def search_and_summarize(search_request: SearchRequest, request: Request):
    try:
        search_agent = await _get_or_create_agent(request, "search")
        search_prompt = (
            f"Search for information about: {search_request.query}. "
            "Provide a comprehensive summary with key findings."
        )
        result = await search_agent.execute(search_prompt, use_tools=True)

        if result.success:
            metadata = result.metadata or {}
            return SearchResponse(
                success=True,
                query=search_request.query,
                summary=result.content,
                search_results=metadata,
                sources_count=int(metadata.get("tool_run_count", metadata.get("tools_used", 0)) or 0),
            )

        return SearchResponse(
            success=False,
            query=search_request.query,
            summary="",
            sources_count=0,
            error=result.error,
        )
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Error in search endpoint")
        return SearchResponse(
            success=False,
            query=search_request.query,
            summary="",
            sources_count=0,
            error=str(exc),
        )


@api_router.get("/agents/capabilities")
async def get_agent_capabilities(request: Request):
    try:
        search_agent = await _get_or_create_agent(request, "search")
        chat_agent = await _get_or_create_agent(request, "chat")

        return {
            "success": True,
            "capabilities": {
                "search_agent": search_agent.get_capabilities(),
                "chat_agent": chat_agent.get_capabilities(),
            },
        }
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Error getting capabilities")
        return {"success": False, "error": str(exc)}


@api_router.post("/wallpapers/generate", response_model=WallpaperResponse)
async def generate_wallpaper(wallpaper_create: WallpaperCreate, request: Request):
    """Generate a wallpaper using AI image generation"""
    try:
        client_ip = request.client.host if request.client else "unknown"

        # Check rate limit
        if not await _check_rate_limit(request, client_ip):
            raise HTTPException(
                status_code=429,
                detail="Daily generation limit (5) reached. Try again tomorrow."
            )

        # Generate image using ImageAgent
        image_agent = await _get_or_create_agent(request, "image")
        result = await image_agent.execute(
            f"Create a beautiful wallpaper: {wallpaper_create.prompt}",
            use_tools=True
        )

        if not result.success:
            return WallpaperResponse(success=False, error=result.error or "Failed to generate image")

        # Extract image data from metadata
        image_url = result.metadata.get("image_url", "")
        image_data = result.metadata.get("image_data", "")

        if not image_url or not image_data:
            return WallpaperResponse(success=False, error="No image data returned")

        # Save to database
        db = _ensure_db(request)
        wallpaper = Wallpaper(
            prompt=wallpaper_create.prompt,
            image_url=image_url,
            image_data=image_data
        )

        wallpaper_dict = wallpaper.model_dump()
        wallpaper_dict["ip_address"] = client_ip
        await db.wallpapers.insert_one(wallpaper_dict)

        return WallpaperResponse(success=True, wallpaper=wallpaper.model_dump())

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error generating wallpaper")
        return WallpaperResponse(success=False, error=str(exc))


@api_router.get("/wallpapers/top")
async def get_top_wallpapers(request: Request):
    """Get top 5 most liked wallpapers"""
    try:
        db = _ensure_db(request)
        wallpapers = await db.wallpapers.find().sort("likes", -1).limit(5).to_list(5)

        # Remove IP address from response
        for wallpaper in wallpapers:
            wallpaper.pop("ip_address", None)
            wallpaper["_id"] = str(wallpaper.get("_id", ""))

        return {"success": True, "wallpapers": wallpapers}
    except Exception as exc:
        logger.exception("Error fetching top wallpapers")
        return {"success": False, "error": str(exc)}


@api_router.post("/wallpapers/{wallpaper_id}/like", response_model=LikeWallpaperResponse)
async def like_wallpaper(wallpaper_id: str, request: Request):
    """Increment likes for a wallpaper"""
    try:
        db = _ensure_db(request)
        result = await db.wallpapers.update_one(
            {"id": wallpaper_id},
            {"$inc": {"likes": 1}}
        )

        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Wallpaper not found")

        wallpaper = await db.wallpapers.find_one({"id": wallpaper_id})
        return LikeWallpaperResponse(success=True, likes=wallpaper["likes"])

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error liking wallpaper")
        return LikeWallpaperResponse(success=False, likes=0, error=str(exc))


@api_router.get("/wallpapers/{wallpaper_id}")
async def get_wallpaper(wallpaper_id: str, request: Request):
    """Get a specific wallpaper by ID"""
    try:
        db = _ensure_db(request)
        wallpaper = await db.wallpapers.find_one({"id": wallpaper_id})

        if not wallpaper:
            raise HTTPException(status_code=404, detail="Wallpaper not found")

        wallpaper.pop("ip_address", None)
        wallpaper["_id"] = str(wallpaper.get("_id", ""))

        return {"success": True, "wallpaper": wallpaper}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error fetching wallpaper")
        return {"success": False, "error": str(exc)}


@api_router.get("/wallpapers/limit/check")
async def check_rate_limit(request: Request):
    """Check remaining generations for today"""
    try:
        client_ip = request.client.host if request.client else "unknown"
        db = _ensure_db(request)
        today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)

        count = await db.wallpapers.count_documents({
            "ip_address": client_ip,
            "created_at": {"$gte": today_start}
        })

        remaining = max(0, 5 - count)
        return {"success": True, "used": count, "remaining": remaining, "limit": 5}

    except Exception as exc:
        logger.exception("Error checking rate limit")
        return {"success": False, "error": str(exc)}


app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
