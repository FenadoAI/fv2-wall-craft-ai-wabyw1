from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List
import os
import uuid
from dotenv import load_dotenv
import httpx

load_dotenv()

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB
MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "wallpaper_db")
client = AsyncIOMotorClient(MONGO_URL)
db = client[DB_NAME]

# LiteLLM Configuration
LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL", "https://litellm-docker-545630944929.us-central1.run.app")
LITELLM_AUTH_TOKEN = os.getenv("LITELLM_AUTH_TOKEN", "")
CODEXHUB_MCP_AUTH_TOKEN = os.getenv("CODEXHUB_MCP_AUTH_TOKEN", "")

# Rate Limiting
DAILY_LIMIT = 5
rate_limit_store = {}  # In-memory store for rate limiting (IP -> {count, date})

class WallpaperRequest(BaseModel):
    prompt: str

class Wallpaper(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    prompt: str
    image_data: str  # Base64 encoded image
    likes: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)

async def generate_image_with_mcp(prompt: str) -> str:
    """Generate image using CodexHub MCP Image Generation service"""
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            # Call CodexHub MCP Image Generation endpoint
            response = await client.post(
                "https://mcp.codexhub.ai/image/generate",
                json={
                    "prompt": prompt,
                    "aspect_ratio": "16:9",  # Good for wallpapers
                    "megapixels": "1",
                    "output_format": "png"
                },
                headers={
                    "Authorization": f"Bearer {CODEXHUB_MCP_AUTH_TOKEN}",
                    "Content-Type": "application/json"
                }
            )

            if response.status_code == 200:
                result = response.json()
                # The MCP returns a URL, we need to fetch and convert to base64
                image_url = result.get("url")
                if image_url:
                    # Fetch the image
                    img_response = await client.get(image_url)
                    if img_response.status_code == 200:
                        import base64
                        image_base64 = base64.b64encode(img_response.content).decode('utf-8')
                        return f"data:image/png;base64,{image_base64}"

            raise Exception(f"Image generation failed: {response.status_code} - {response.text}")

    except Exception as e:
        print(f"Error generating image: {str(e)}")
        raise

def get_client_ip(request: Request) -> str:
    """Extract client IP from request"""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"

def check_rate_limit(ip: str) -> dict:
    """Check if IP has exceeded daily limit"""
    today = datetime.utcnow().date()

    if ip not in rate_limit_store:
        rate_limit_store[ip] = {"count": 0, "date": today}

    if rate_limit_store[ip]["date"] != today:
        rate_limit_store[ip] = {"count": 0, "date": today}

    used = rate_limit_store[ip]["count"]
    remaining = max(0, DAILY_LIMIT - used)

    return {
        "success": True,
        "used": used,
        "remaining": remaining,
        "limit": DAILY_LIMIT
    }

def increment_rate_limit(ip: str):
    """Increment rate limit counter for IP"""
    today = datetime.utcnow().date()

    if ip not in rate_limit_store or rate_limit_store[ip]["date"] != today:
        rate_limit_store[ip] = {"count": 1, "date": today}
    else:
        rate_limit_store[ip]["count"] += 1

@app.get("/")
async def root():
    return {"message": "AI Wallpaper Studio API"}

@app.get("/api/wallpapers/top")
async def get_top_wallpapers():
    """Get top 5 wallpapers by likes"""
    try:
        wallpapers = await db.wallpapers.find().sort("likes", -1).limit(5).to_list(length=5)

        # Convert MongoDB _id to string id
        for w in wallpapers:
            w["id"] = str(w.pop("_id", w.get("id", "")))

        return {
            "success": True,
            "wallpapers": wallpapers
        }
    except Exception as e:
        print(f"Error fetching wallpapers: {str(e)}")
        return {
            "success": True,
            "wallpapers": []
        }

@app.get("/api/wallpapers/limit/check")
async def check_limit(request: Request):
    """Check rate limit for current IP"""
    ip = get_client_ip(request)
    return check_rate_limit(ip)

@app.post("/api/wallpapers/generate")
async def generate_wallpaper(data: WallpaperRequest, request: Request):
    """Generate a new wallpaper using AI"""
    try:
        # Check rate limit
        ip = get_client_ip(request)
        limit_status = check_rate_limit(ip)

        if limit_status["remaining"] <= 0:
            raise HTTPException(status_code=429, detail="Daily generation limit reached")

        # Generate image
        image_data = await generate_image_with_mcp(data.prompt)

        # Create wallpaper object
        wallpaper = Wallpaper(
            prompt=data.prompt,
            image_data=image_data
        )

        # Save to database
        wallpaper_dict = wallpaper.dict()
        wallpaper_dict["_id"] = wallpaper_dict.pop("id")
        await db.wallpapers.insert_one(wallpaper_dict)

        # Increment rate limit
        increment_rate_limit(ip)

        # Return with id field
        wallpaper_dict["id"] = wallpaper_dict.pop("_id")

        return {
            "success": True,
            "wallpaper": wallpaper_dict
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error generating wallpaper: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/wallpapers/{wallpaper_id}/like")
async def like_wallpaper(wallpaper_id: str):
    """Increment likes for a wallpaper"""
    try:
        result = await db.wallpapers.update_one(
            {"_id": wallpaper_id},
            {"$inc": {"likes": 1}}
        )

        if result.modified_count == 0:
            raise HTTPException(status_code=404, detail="Wallpaper not found")

        return {"success": True}

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error liking wallpaper: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
