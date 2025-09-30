# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Backend (FastAPI)
```bash
cd backend
pip install -r requirements.txt
uvicorn server:app --reload
```

### Frontend (React)
```bash
cd frontend
bun install
bun start
```

### Testing

**AI Agents Tests (No Server Required):**
```bash
cd backend && python tests/test_agents.py
```

**API Integration Tests (Requires Running Server):**
```bash
# Terminal 1: Start server first
cd backend && uvicorn server:app --reload --port 8001

# Terminal 2: Run API tests
cd backend && python -m pytest tests/test_api.py -v
```

### Code Quality
```bash
# Backend linting and formatting
cd backend
black .
isort .
flake8 .
mypy .

# Frontend testing
cd frontend
bun test
```

## Architecture Overview

### Backend Structure
- **FastAPI** application with AsyncIOMotorClient for MongoDB
- **AI Agents**: Extensible AI agents library with LangChain and MCP support
- **LiteLLM Integration**: Unified proxy for multiple AI models (Gemini, Claude)
- **Authentication**: JWT tokens with bcrypt password hashing
- **API Pattern**: All routes under `/api` prefix using APIRouter
- **Environment**: Requires `.env` with `MONGO_URL`, `DB_NAME`, `LITELLM_AUTH_TOKEN`
- **CORS**: Configured for all origins in development

### Frontend Structure
- **React 19** with React Router v7
- **UI Framework**: shadcn/ui components built on Radix UI
- **Styling**: Tailwind CSS with custom configuration via craco
- **API Communication**: Axios with `REACT_APP_API_URL` (defaults to http://localhost:8001)
- **Components**: Located in `/src/components/ui/` with consistent import pattern `@/components/ui/`
- **Configuration in App.js**:
  - `MY_HOMEPAGE_URL`: Computed base URL for shareable links (derived from `API_BASE` or `window.location.origin`)
  - Use for share links: `${MY_HOMEPAGE_URL}/profile/${username}`
- **Environment Variables**:
  - `REACT_APP_API_URL`: Backend API URL

### Database
- **MongoDB** with collections: users, items, status_checks
- **Connection**: AsyncIOMotorClient with environment-based configuration

### AI Agents Implementation

#### Architecture
- **BaseAgent**: Extensible foundation class with LangChain integration
- **SearchAgent**: Pre-built agent with web search MCP capabilities  
- **ChatAgent**: Conversational agent for general assistance
- **AgentConfig**: Environment-based configuration management

#### LiteLLM Integration
- **Proxy Service**: `https://litellm-docker-545630944929.us-central1.run.app` (hardcoded)
- **Authentication**: 
  - `LITELLM_AUTH_TOKEN` - For AI model access (workflow input)
  - `CODEXHUB_MCP_AUTH_TOKEN` - For CodexHub MCP services only (workflow input)
- **Provider Compatibility**: Automatic mapping to OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY
- **Supported Models**:
  - `gemini-2.5-pro` - Complex reasoning and analysis
  - `gemini-2.5-flash` - Fast general responses
  - `gemini-2.5-flash-lite` - Quick simple tasks
  - `claude-4-sonnet` - Balanced performance
  - `claude-4-haiku` - Fast simple tasks

#### MCP (Model Context Protocol)
- **CodexHub Web Search**: Integrated via `https://mcp.codexhub.ai/web/mcp`
- **Authentication**: Uses dedicated `CODEXHUB_MCP_AUTH_TOKEN`
- **Extensible**: Support for custom MCP servers (HTTP/stdio)

#### API Endpoints
- `POST /api/chat` - Chat with AI agents (chat or search type)
- `POST /api/search` - Direct web search with AI summarization
- `GET /api/agents/capabilities` - List available agent capabilities

#### Usage Pattern
```python
from ai_agents import SearchAgent, ChatAgent, AgentConfig

# Create agent
config = AgentConfig()  # Loads from .env
agent = SearchAgent(config)  # Auto-configures web search

# Use agent
response = await agent.execute("What is the capital of France?", use_tools=True)
```

#### Testing
- **AI Functionality**: `cd backend && python tests/test_agents.py` - Tests real web search and image generation (standalone, no server required)
- **API Endpoints**: `cd backend && python -m pytest tests/test_api.py -v` - Tests FastAPI integration (requires server running on port 8001)
- **Real Tests**: All tests can fail if functionality is broken (no fake passes)

## Documentation Structure

### Core Documentation Files

#### `docs/techstack.md`
Complete technical stack reference including:
- **Backend Stack**: FastAPI, Python 3.8+, Motor (AsyncIOMotorClient), MongoDB, Pydantic
- **AI Agents Section**: Overview of extensible AI agents library with LangChain and MCP support
- **Frontend Stack**: React 19, React Router v7, Tailwind CSS, shadcn/ui components
- **API Patterns**: Standard FastAPI patterns with AsyncIOMotorClient and Pydantic models
- **Authentication Patterns**: JWT tokens with bcrypt password hashing examples
- **Database Patterns**: MongoDB collections and connection management
- **Environment Variables**: Required configuration for all components
- **Run Commands**: Standard commands for backend, frontend, and testing
- **LiteLLM Integration**: Links to AI agents documentation for detailed implementation

#### `docs/how-to-add-ai-functionality.md` 
Comprehensive AI agents library documentation including:
- **Architecture Overview**: SOLID principles, extensible design patterns
- **LiteLLM Integration**: Detailed proxy configuration and model selection
- **Supported Models**: Complete list of Gemini and Claude models with use cases
- **Environment Setup**: All required environment variables and dependencies
- **Agent Types**: ChatAgent, SearchAgent, ImageAgent, and custom agent development
- **MCP Integration**: Web search and image generation with CodexHub MCP services
- **Image Generation**: LangGraph-powered workflow with guaranteed tool execution
- **API Endpoints**: FastAPI integration with chat, search, and capabilities endpoints
- **Best Practices**: Prompt engineering, error handling, tool verification
- **Troubleshooting**: Common issues and solutions for MCP, LangGraph, and tool usage
- **API Reference**: Complete class documentation with examples
- **Testing**: Test patterns and verification strategies

#### `HOW_TO_TEST.md`
Testing methodology and patterns including:
- **Testing Philosophy**: Real tests that can fail vs fake tests that always pass
- **Test Types**: API integration, business logic, AI agents, database, endpoints
- **Anti-Patterns**: What NOT to do when writing tests
- **Good Practices**: Proper assertion patterns and failure scenarios
- **Troubleshooting Section**: Common test failures and solutions
  - Authentication errors (401 Unauthorized)
  - Test assertion bugs (string comparison)
  - MCP tool usage patterns (when tools are invoked)
  - Connection errors and graceful degradation
  - Environment verification commands
  - Debug logging for tool usage

## Environment Configuration

### .env File Management

**Important**: `.env` files are **committed to the repository** for easy setup and configuration sharing.

#### `backend/.env` Structure
```bash
# Database Configuration
MONGO_URL="mongodb://localhost:27017"
DB_NAME="your_database_name"

# LiteLLM Configuration (DO NOT MODIFY)
LITELLM_BASE_URL="https://litellm-docker-545630944929.us-central1.run.app"
LITELLM_AUTH_TOKEN=""
CODEXHUB_MCP_AUTH_TOKEN=""

# Provider Variables (DO NOT MODIFY - automatically set from LITELLM)
OPENAI_API_BASE="${LITELLM_BASE_URL}"
OPENAI_API_KEY="${LITELLM_AUTH_TOKEN}"
# ... other provider configs

# AI Model Selection (can be updated)
AI_MODEL_NAME="gemini-2.5-pro"

# Other Services (can be added/updated freely)
# Example: External API integrations
STRIPE_API_KEY="sk-your-stripe-key"
SENDGRID_API_KEY="SG.your-sendgrid-key"
REDIS_URL="redis://localhost:6379"
```

#### Guidelines for .env Updates

**✅ Safe to Update:**
- Database URLs and credentials
- Third-party service API keys (Stripe, SendGrid, etc.)
- Custom service configurations
- AI model names (choose from supported models)
- Application-specific settings

**❌ Do NOT Modify:**
- `LITELLM_BASE_URL` - Hardcoded proxy endpoint
- `LITELLM_AUTH_TOKEN` - Workflow input parameter
- `CODEXHUB_MCP_AUTH_TOKEN` - Workflow input parameter for CodexHub only
- Provider base URLs (OPENAI_API_BASE, etc.) - Auto-mapped to LiteLLM

#### Adding New Services

When integrating new external services, add their configuration directly to `.env`:

```bash
# Example: Adding a new service
NEW_SERVICE_API_URL="https://api.newservice.com"
NEW_SERVICE_API_KEY="your-api-key"
NEW_SERVICE_TIMEOUT=30
```

Then use in your code:
```python
import os
from dotenv import load_dotenv

load_dotenv()

NEW_SERVICE_URL = os.getenv("NEW_SERVICE_API_URL")
NEW_SERVICE_KEY = os.getenv("NEW_SERVICE_API_KEY")
```

### Development Patterns
- Backend models use Pydantic with automatic UUID generation and datetime fields
- Frontend components follow React functional pattern with hooks
- API responses follow consistent JSON structure
- Authentication handled via JWT tokens in request headers
- AI agents follow extensible inheritance pattern for customization
- All tests are located in `backend/tests/` directory and can actually fail when functionality is broken
- Environment variables are committed and can be updated for new service integrations