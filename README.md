# Project Setup

## Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn server:app --reload
```

### Required Environment
- `MONGO_URL`: MongoDB connection string
- `DB_NAME`: Database name
- `LITELLM_AUTH_TOKEN`, `LITELLM_BASE_URL`, `AI_MODEL_NAME`: LiteLLM configuration
- Optional: `CODEXHUB_MCP_AUTH_TOKEN` for MCP tool access

### Running the API Tests
Unit tests mock external services and use FastAPI's `TestClient`:
```bash
cd backend
pytest
```
Integration smoke scripts that hit live services were removed; reach for manual calls when you need them.

## Frontend  
```bash
cd frontend
bun install
bun start
```

## Environment

### Backend Environment Variables
Create `backend/.env` with:
- `MONGO_URL`: MongoDB connection string
- `DB_NAME`: Database name
- `JWT_SECRET_KEY`: Secret key for JWT tokens
- `CODEXHUB_MCP_AUTH_TOKEN`: Authentication token for MCP services (web search, image generation)
- `LITELLM_AUTH_TOKEN`: Authentication token for LiteLLM API
- `LITELLM_BASE_URL`: LiteLLM API base URL (default: https://litellm-docker-545630944929.us-central1.run.app)
- `AI_MODEL_NAME`: AI model to use (default: gemini-2.5-pro)

### Frontend Environment Variables
- `REACT_APP_API_URL`: Backend API URL (default: http://localhost:8001)
  - For production: Set to your deployed backend URL (e.g., https://api.yourdomain.com)

### Frontend Configuration
- `MY_HOMEPAGE_URL`: Computed base URL for the website (defined in `App.js`)
  - Automatically derived from `REACT_APP_API_URL` for preview deployments or falls back to `window.location.origin`
  - Used for generating shareable links (e.g., profile URLs, product pages)
  - All share links should be relative to this URL: `${MY_HOMEPAGE_URL}/profile/${username}`
  - Example: `https://preview-abc123.previewer.live` or `https://www.yourdomain.com`

## AI Agents

This project includes extensible AI agents built with **LangGraph** and **MCP (Model Context Protocol)**:

### Available Agents

1. **SearchAgent** - Web search and research with real-time information
   - Uses web search MCP tools
   - Verifies tool usage to prevent fabricated responses
   - Ideal for current events, weather, news

2. **ImageAgent** - Image generation from text prompts
   - Generates real images via CodexHub Image MCP
   - HTTP verification ensures images are accessible
   - Returns persistent Google Cloud Storage URLs

3. **ChatAgent** - General conversation and assistance
4. **BaseAgent** - Base class for creating custom agents

### Usage Examples

#### Image Generation
```python
from ai_agents import ImageAgent, AgentConfig
import asyncio

async def generate_image():
    config = AgentConfig()
    agent = ImageAgent(config)
    
    response = await agent.execute(
        "Generate an image of a sunset over mountains",
        use_tools=True
    )
    
    if response.success and response.metadata.get('tools_used'):
        print(response.content)  # Real GCS URL
        # https://storage.googleapis.com/fenado-ai-farm-public/generated/{uuid}.webp

asyncio.run(generate_image())
```

#### Web Search
```python
from ai_agents import SearchAgent, AgentConfig
import asyncio

async def search_web():
    config = AgentConfig()
    agent = SearchAgent(config)
    
    response = await agent.execute(
        "Use web search to find today's weather in Tokyo",
        use_tools=True
    )
    
    if response.success and response.metadata.get('tools_used'):
        print(response.content)  # Real-time search results

asyncio.run(search_web())
```

### Documentation

- 📖 [How to Add AI Functionality](docs/how-to-add-ai-functionality.md) - Complete guide to AI agents, image generation, and web search
- 📖 [LangGraph + MCP Integration](backend/LANGGRAPH_MCP_INTEGRATION.md) - Technical guide
- 📖 [Tech Stack](docs/techstack.md) - Technology stack overview
- 📖 [How to Test](HOW_TO_TEST.md) - Testing guide with troubleshooting tips

## Test

### Quick Test - Run All Tests
```bash
# Run all tests with pytest
cd backend && python -m pytest tests/ -v

# Or run specific test files
cd backend && python -m pytest tests/test_agents.py -v  # AI agents only
cd backend && python -m pytest tests/test_api.py -v     # API endpoints only
```

### AI Agents Tests (No Server Required)
```bash
# Option 1: Run with pytest
cd backend && python -m pytest tests/test_agents.py -v -s

# Option 2: Run directly with formatted output
cd backend && python tests/test_agents.py
```

### API Integration Tests (Requires Running Server)
```bash
# Terminal 1: Start the server
cd backend && uvicorn server:app --reload --port 8001

# Terminal 2: Run API tests
cd backend && python -m pytest tests/test_api.py -v
```

### Expected Test Results

**All Tests (pytest):**
```bash
6 passed in ~50s

tests/test_agents.py::test_search_agent PASSED           [ 16%]  ✅
tests/test_agents.py::test_image_agent PASSED            [ 33%]  ✅
tests/test_api.py::test_root_endpoint PASSED             [ 50%]  ✅
tests/test_api.py::test_chat_endpoint PASSED             [ 66%]  ✅
tests/test_api.py::test_search_endpoint PASSED           [ 83%]  ✅
tests/test_api.py::test_capabilities_endpoint PASSED     [100%] ✅
```

**AI Agents Tests (direct execution):**
```
🤖 AI AGENTS TEST SUITE
============================================================
🔍 Testing SearchAgent...
  ✅ Search Agent PASSED
     Tools used: True
     Tool calls: 1

🎨 Testing ImageAgent...
  ✅ Image Agent PASSED
     Tools used: True
     Tool calls: 1
     Image URL: https://storage.googleapis.com/...
     HTTP Status: 200

🎉 ALL TESTS PASSED!
```

**What the tests verify:**
- ✅ MCP tools are actually invoked (`tools_used: True`)
- ✅ Real web search results (not from training data)
- ✅ Real image URLs from Google Cloud Storage
- ✅ HTTP 200 verification for image accessibility
- ✅ FastAPI endpoints work correctly
- ✅ Agent capabilities are properly reported

### Recent Fixes

**Test Suite Improvements:**
- ✅ Fixed authentication error handling (401 Unauthorized) - requires valid `LITELLM_AUTH_TOKEN` starting with 'sk-'
- ✅ Fixed test assertion bug in `test_search_endpoint` (string comparison issue)
- ✅ Fixed async test compatibility with pytest (added `@pytest.mark.asyncio` decorators)
- ✅ Added enhanced logging for MCP tool usage tracking (`tool_call_count` in metadata)
- ✅ Improved test output with formatted results and verification details
- ✅ Comprehensive troubleshooting section in `HOW_TO_TEST.md`

All 6 tests now pass consistently with both pytest and direct execution. See `HOW_TO_TEST.md` for detailed troubleshooting guide.
