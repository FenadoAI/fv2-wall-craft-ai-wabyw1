# Tech Stack

## Backend
FastAPI, Python 3.8+, Motor (AsyncIOMotorClient), MongoDB, Pydantic

### AI Agents
Extensible AI agents built with **LangGraph** and **MCP (Model Context Protocol)** for building verified, intelligent services. Features real-time web search, image generation with HTTP verification, and structured JSON output. See [AI Agents Documentation](./how-to-add-ai-functionality.md) for detailed implementation guide.

**Key Features:**
- LangGraph-powered agent orchestration
- MCP tool integration with verification
- HTTP validation for generated content
- Structured output with Pydantic
- SearchAgent, ImageAgent, ChatAgent

### Installed Packages
fastapi==0.110.1, uvicorn==0.25.0, motor==3.3.1, pymongo==4.5.0, pydantic>=2.6.4, email-validator>=2.2.0, python-jose>=3.3.0, passlib>=1.7.4, pyjwt>=2.10.1, python-dotenv>=1.0.1, requests>=2.31.0, cryptography>=42.0.8, bcrypt

**AI Agent Packages:**
langgraph>=0.6.7, langgraph-checkpoint>=2.1.1, langgraph-prebuilt>=0.6.4, langchain-core>=0.3.76, langchain-openai>=0.3.33, langchain-mcp-adapters>=0.1.9

### API Structure Pattern
```python
from fastapi import FastAPI, APIRouter, HTTPException
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field
import os, uuid
from datetime import datetime

# Setup
app = FastAPI()
api_router = APIRouter(prefix="/api")
client = AsyncIOMotorClient(os.environ['MONGO_URL'])
db = client[os.environ['DB_NAME']]

# Model
class Item(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

# Route
@api_router.post("/items", response_model=Item)
async def create_item(data: ItemCreate):
    item = Item(**data.dict())
    await db.items.insert_one(item.dict())
    return item

# Include router
app.include_router(api_router)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Authentication Pattern
```python
from jose import jwt
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "your-secret-key")

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=30)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm="HS256")
```

## Frontend
React 19, React Router v7, Tailwind CSS, shadcn/ui components, craco

### Installed Packages
react@19, react-dom@19, react-router-dom@7, axios@1, tailwindcss@3, @radix-ui/*, lucide-react, react-hook-form@7, zod@3, clsx, tailwind-merge

### Component Pattern
```javascript
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { useState, useEffect } from 'react'
import axios from 'axios'

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8001'

export function MyComponent() {
  const [data, setData] = useState([])
  
  useEffect(() => {
    axios.get(`${API_URL}/api/items`)
      .then(res => setData(res.data))
      .catch(err => console.error(err))
  }, [])
  
  return (
    <Card>
      {data.map(item => (
        <div key={item.id}>{item.name}</div>
      ))}
    </Card>
  )
}
```

### Frontend Configuration (App.js)

**`MY_HOMEPAGE_URL`** - Base URL where the website is hosted (computed in App.js)
- Automatically derived from `API_BASE` URL or falls back to `window.location.origin`
- Used for generating absolute share links
- All shareable URLs should be relative to this base URL
- Example usage:
  ```javascript
  // In App.js (lines 11-13)
  const MY_HOMEPAGE_URL = API_BASE?.match(/-([a-z0-9]+)\./)?.[1]
    ? `https://${API_BASE?.match(/-([a-z0-9]+)\./)?.[1]}.previewer.live`
    : window.location.origin;
  
  // Use for share links:
  const profileShareUrl = `${MY_HOMEPAGE_URL}/profile/${username}`
  const productShareUrl = `${MY_HOMEPAGE_URL}/product/${productId}`
  ```

### Frontend Environment Variables

**`REACT_APP_API_URL`** - Backend API URL (default: `http://localhost:8001`)
- Used for all API requests to the backend
- Also used to compute `MY_HOMEPAGE_URL` for preview deployments

## Test Pattern
```python
import pytest
from starlette.testclient import TestClient

from server import app


def test_health_check():
    client = TestClient(app)
    response = client.get("/api/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}
```

## Database
MongoDB, collections: users, items, status_checks

## Environment Variables

### Backend
MONGO_URL, DB_NAME, JWT_SECRET_KEY, CORS_ORIGINS, LITELLM_AUTH_TOKEN, CODEXHUB_MCP_AUTH_TOKEN, AI_MODEL_NAME

### Frontend
REACT_APP_API_URL

**Note:** `MY_HOMEPAGE_URL` is not an environment variable - it's computed in `App.js` based on the API URL or `window.location.origin`.

## Run Commands
**Backend:** `cd backend && uvicorn server:app --reload --port 8001`
**Frontend:** `cd frontend && bun start`
**Tests (AI Agents):** `cd backend && python tests/test_agents.py` (no server required)
**Tests (API):** `cd backend && pytest tests/test_api.py -v` (requires running server)

See [AI Agents Documentation](./how-to-add-ai-functionality.md) and [LangGraph MCP Integration](../backend/LANGGRAPH_MCP_INTEGRATION.md) for details.
