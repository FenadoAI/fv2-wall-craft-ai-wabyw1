# How to Add AI Functionality

Complete guide for integrating AI agents with LangChain, LangGraph, and MCP services in the Fenado template.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Environment Setup](#environment-setup)
- [Supported Models](#supported-models)
- [Agent Types](#agent-types)
- [MCP Integration](#mcp-integration)
- [Image Generation](#image-generation)
- [API Endpoints](#api-endpoints)
- [Testing](#testing)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [API Reference](#api-reference)

## Architecture Overview

The AI agents library provides extensible AI capabilities through a clean class hierarchy:

- **BaseAgent**: Core class with LangChain integration and MCP support
- **ChatAgent**: Conversational assistant for general queries
- **SearchAgent**: Web search capabilities via CodexHub Web MCP
- **ImageAgent**: LangGraph-powered image generation with guaranteed tool execution
- **AgentConfig**: Environment-based configuration management

```python
from ai_agents import (
    AgentConfig,
    ChatAgent,
    SearchAgent,
    ImageAgent,
    AgentResponse,
    ImageGenerationResult
)

config = AgentConfig()
chat_agent = ChatAgent(config)
search_agent = SearchAgent(config)
image_agent = ImageAgent(config)
```

### Design Principles

- **SOLID**: Single responsibility, open for extension
- **DRY**: Minimal code duplication, reusable patterns
- **Composability**: Mix and match agents for complex workflows

## Environment Setup

### Required Environment Variables

Create `backend/.env` with the following:

```bash
# LiteLLM proxy endpoint
LITELLM_BASE_URL="https://litellm-docker-545630944929.us-central1.run.app"
LITELLM_AUTH_TOKEN="your_litellm_token_here"

# CodexHub MCP authentication
CODEXHUB_MCP_AUTH_TOKEN="your_codexhub_token_here"

# Model selection
AI_MODEL_NAME=gemini-2.5-pro

# Provider compatibility (optional)
OPENAI_API_BASE="${LITELLM_BASE_URL}"
OPENAI_API_KEY="${LITELLM_AUTH_TOKEN}"
ANTHROPIC_API_BASE="${LITELLM_BASE_URL}"
ANTHROPIC_API_KEY="${LITELLM_AUTH_TOKEN}"
GEMINI_API_BASE="${LITELLM_BASE_URL}"
GEMINI_API_KEY="${LITELLM_AUTH_TOKEN}"
```

### Python Dependencies

Install all required packages:

```bash
cd backend
pip install -r requirements.txt
```

Key dependencies:
- `langchain-core>=0.3.0` - LangChain core functionality
- `langchain-openai>=0.2.0` - OpenAI/LiteLLM integration
- `langchain-mcp-adapters>=0.1.0` - MCP integration
- `langgraph>=0.6.7` - Agent orchestration (ReAct agents)
- `openai>=1.50.0` - OpenAI client

## Supported Models

### Gemini Models

- **`gemini-2.5-pro`** - Best for complex tasks, reasoning, and long-form generation
- **`gemini-2.5-flash`** - Balanced performance for general use cases
- **`gemini-2.5-flash-lite`** - Fast responses for simple tasks

### Claude Models

- **`claude-sonnet-4-5`** - Best state of the art model, only use if explicitly requested

### Configuration

Set your preferred model in `.env`:

```bash
AI_MODEL_NAME=gemini-2.5-pro  # or any supported model
```

**AgentConfig Properties:**
- `api_base_url` - LiteLLM endpoint URL
- `model_name` - Model identifier to use  
- `api_key` - Authentication token

## Agent Types

### 1. ChatAgent - Conversational Assistant

Basic chat completions for general queries.

**System Prompt:** "Friendly conversational AI. Natural conversations, explanations, analysis. Helpful, harmless, honest."

```python
from ai_agents import ChatAgent, AgentConfig

config = AgentConfig()
agent = ChatAgent(config)
response = await agent.execute("Explain quantum computing in simple terms")

if response.success:
    print(response.content)
```

### 2. SearchAgent - Web Search

Real-time web search and crawling via MCP:

```python
from ai_agents import SearchAgent, AgentConfig

config = AgentConfig()
agent = SearchAgent(config)
response = await agent.execute(
    "Latest AI regulations in the EU",
    use_tools=True  # Enables web MCP tools
)

if response.success:
    print(response.content)
    print(f"Tools used: {response.metadata.get('tools_used')}")
```

### 3. ImageAgent - Image Generation

LangGraph-powered agent with guaranteed MCP tool execution:

```python
import asyncio
from ai_agents import ImageAgent, AgentConfig

async def generate_image():
    config = AgentConfig()
    image_agent = ImageAgent(config)
    
    response = await image_agent.execute(
        "Generate a photorealistic image of a sunset over mountains",
        use_tools=True
    )
    
    if response.success:
        print(response.content)
        print(f"Tools used: {response.metadata.get('tools_used')}")
    else:
        print(f"Error: {response.error}")

asyncio.run(generate_image())
```

### 4. Custom Agent - Build Your Own

Extend BaseAgent for specialized use cases:

```python
from ai_agents import BaseAgent, AgentConfig

class ProductDescriptionAgent(BaseAgent):
    def __init__(self, config):
        system_prompt = """You are an expert copywriter specializing in 
        e-commerce product descriptions. Write compelling, SEO-optimized 
        descriptions that convert browsers into buyers."""
        super().__init__(config, system_prompt)

agent = ProductDescriptionAgent(AgentConfig())
response = await agent.execute("Write description for wireless headphones")
```

## MCP Integration

Agents integrate with CodexHub MCP services for extended capabilities. All MCPs use `CODEXHUB_MCP_AUTH_TOKEN` for authentication.

### Web MCP - Search & Crawling

**Endpoint:** `https://mcp.codexhub.ai/web/mcp`

**Configuration:**
```json
{
  "web": {
    "url": "https://mcp.codexhub.ai/web/mcp",
    "headers": {
      "x-team-key": "$CODEXHUB_MCP_AUTH_TOKEN"
    }
  }
}
```

**Capabilities:**
- Web search for real-time information
- Website crawling and content extraction
- Automatic integration in SearchAgent

**Usage Example:**
```python
# SearchAgent automatically uses web MCP
agent = SearchAgent(config)
response = await agent.execute("Latest tech trends", use_tools=True)
```

### Image MCP - Image Generation

**Endpoint:** `https://mcp.codexhub.ai/image/mcp`

**Configuration:**
```json
{
  "image": {
    "url": "https://mcp.codexhub.ai/image/mcp",
    "headers": {
      "x-team-key": "$CODEXHUB_MCP_AUTH_TOKEN"
    }
  }
}
```

**Capabilities:**
- Generate images from text prompts
- Professional quality output
- Persistent Google Cloud Storage URLs

**Usage Example:**
```python
# Any agent can access image generation
agent = ChatAgent(config)
response = await agent.execute(
    "Generate an image of a modern office space",
    use_tools=True
)
```

### Custom MCP Setup

Add your own MCP servers:

```python
server_configs = {
    "my-custom-mcp": {
        "transport": "streamable_http",
        "url": "https://your-mcp.com/mcp",
        "headers": {"x-api-key": "your_token"}
    }
}
await agent.setup_mcp(server_configs)
```

## Image Generation

### Overview

The ImageAgent uses **LangGraph's `create_react_agent`** to guarantee MCP tool execution, preventing LLMs from fabricating image URLs.

### Basic Usage

```python
import asyncio
from ai_agents import ImageAgent, AgentConfig

async def generate_product_image():
    config = AgentConfig()
    image_agent = ImageAgent(config)
    
    response = await image_agent.execute(
        "Generate a professional product photo of sneakers on white background",
        use_tools=True
    )
    
    if response.success:
        print(response.content)
    else:
        print(f"Error: {response.error}")

asyncio.run(generate_product_image())
```

### Response Format

Images are returned in markdown format with persistent Google Cloud Storage URLs:

```markdown
![Image of a sunset over mountains](https://storage.googleapis.com/fenado-ai-farm-public/generated/2c385c97-c286-4f12-94cb-0cb305efec87.webp)
```

Or just the URL:
```
https://storage.googleapis.com/fenado-ai-farm-public/generated/fccb3f99-7763-4f09-87ff-ca01f6b72cc8.webp
```

**Important URL Properties:**
- ✅ **Persistent** - URLs do NOT expire
- ✅ **Hosted on Google Cloud Storage** - Reliable, global CDN
- ✅ **Unique UUIDs** - Each image has a unique identifier
- ✅ **Safe for databases** - Store URLs long-term without worrying about expiration
- ✅ **Format:** `https://storage.googleapis.com/fenado-ai-farm-public/generated/{uuid}.webp`

### Extracting Image URLs

```python
import re

response = await image_agent.execute(
    "Generate an image of a cat",
    use_tools=True
)

if response.success:
    # Extract all URLs from the response
    urls = re.findall(r'https?://[^\s\)]+', response.content)
    
    print(f"Generated {len(urls)} images:")
    for i, url in enumerate(urls, 1):
        print(f"{i}. {url}")
```

**Example Output:**
```
Generated 1 image:
1. https://storage.googleapis.com/fenado-ai-farm-public/generated/fccb3f99-7763-4f09-87ff-ca01f6b72cc8.webp
```

### Downloading Images

```python
import requests
from pathlib import Path
import re

async def download_image(response, output_path):
    if response.success:
        urls = re.findall(r'https?://[^\s\)]+', response.content)
        if urls:
            img_response = requests.get(urls[0])
            Path(output_path).write_bytes(img_response.content)
            print(f"Image saved to {output_path}")
```

### Architecture Deep Dive

**Class Structure:**

```python
class ImageAgent(BaseAgent):
    """Image generation agent with LangGraph and MCP support"""
    
    def __init__(self, config: AgentConfig):
        # Specialized system prompt for image generation
        system_prompt = """You are an AI assistant specialized in generating images.
        You MUST use the available image generation tools to create images.
        NEVER fabricate or make up image URLs."""
        super().__init__(config, system_prompt)
        
        # Setup image MCP
        self._mcp_setup_done = False
    
    async def setup_image_mcp(self):
        # Connect to CodexHub Image MCP endpoint asynchronously
        server_configs = {
            "image-generation": {
                "transport": "streamable_http",  # Critical: Use streamable_http
                "url": "https://mcp.codexhub.ai/image/mcp",
                "headers": {"x-team-key": mcp_token}
            }
        }
        await self.setup_mcp(server_configs)
```

**LangGraph Integration:**

```python
from langgraph.prebuilt import create_react_agent

# Create agent with MCP tools
agent = create_react_agent(llm, mcp_tools)

# Execute with messages
result = await agent.ainvoke({
    "messages": [
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt)
    ]
})

# Verify tools were actually used
tools_called = any(
    hasattr(msg, "tool_calls") and msg.tool_calls 
    for msg in result["messages"]
)
```

**Key Implementation Details:**
- **Transport:** Must use `streamable_http` (NOT "http" or "type")
- **Authentication:** Uses `x-team-key` header with `CODEXHUB_MCP_AUTH_TOKEN`
- **Tool Loading:** Async `await client.get_tools()`
- **Verification:** LangGraph guarantees tool execution, metadata confirms usage

### Extending ImageAgent

Create specialized image agents:

```python
class ProductImageAgent(ImageAgent):
    """Specialized agent for generating product images"""
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        # Override system prompt for product-specific generation
        self.system_prompt = """Generate professional product images optimized 
        for e-commerce. Use clean backgrounds, proper lighting, and commercial 
        photography best practices."""
    
    async def generate_product_image(self, product_name: str, style: str):
        prompt = f"Generate a professional product image of {product_name} in {style} style"
        return await self.execute(prompt, use_tools=True)

# Usage
product_agent = ProductImageAgent(config)
response = await product_agent.generate_product_image("wireless headphones", "minimalist")
```

## API Endpoints

The backend server exposes these endpoints:

- **`POST /api/chat`** - Chat with conversational agent
  ```bash
  curl -X POST http://localhost:8001/api/chat \
    -H "Content-Type: application/json" \
    -d '{"message": "Hello, how are you?"}'
  ```

- **`POST /api/search`** - Web search with AI agent
  ```bash
  curl -X POST http://localhost:8001/api/search \
    -H "Content-Type: application/json" \
    -d '{"query": "Latest AI news"}'
  ```

- **`GET /api/agents/capabilities`** - List agent capabilities
  ```bash
  curl http://localhost:8001/api/agents/capabilities
  ```

## Troubleshooting

### Tools Not Being Used (`tools_used: False`)

**Symptoms:**
```
❌ CRITICAL FAILURE: MCP tools were NOT used!
❌ The LLM is fabricating image URLs instead of calling MCP!
```

**Solutions:**
1. Verify MCP client initialized: `await client.__aenter__()`
2. Check tools loaded: `len(await client.get_tools()) > 0`
3. Ensure transport is correct: Use `"streamable_http"` not `"http"`
4. Verify LangGraph agent created: `create_react_agent(llm, tools)`
5. Check system prompt instructs tool usage
6. Set `use_tools=True` in execute call

### No MCP Tools Loaded (`tools_available: 0`)

**Symptoms:**
```
❌ No tools available
```

**Solutions:**
- Check `CODEXHUB_MCP_AUTH_TOKEN` is set in `.env` file
- Verify server config uses dict format: `{"server-name": {...}}`
- Ensure `"transport": "streamable_http"` is set (not "http" or "type")
- Call async setup: `await agent.setup_image_mcp()`
- Verify MCP endpoint is accessible

### Connection Errors

**Symptoms:**
```
Failed to connect to MCP server
Connection timeout
```

**Solutions:**
- Verify internet connectivity
- Check MCP service status at `https://mcp.codexhub.ai`
- Validate `CODEXHUB_MCP_AUTH_TOKEN` is correct
- Try with a longer timeout
- Check firewall/proxy settings

### HTTP Verification Fails

**Symptoms:**
```
❌ Image URL returned HTTP 403/404
```

**Solutions:**
- LLM may have fabricated the URL - check `tools_used` metadata
- Verify the URL format: `https://storage.googleapis.com/fenado-ai-farm-public/...`
- If the image was just generated, wait a few seconds for upload
- Google Cloud Storage URLs should NOT expire - if 404 persists, regenerate
- Check network connectivity to Google Cloud Storage

### Model Not Found

**Symptoms:**
```
Model 'xyz' not found
```

**Solutions:**
- Verify model name in `.env` matches supported models
- Check LiteLLM proxy has access to the model
- Use a different model from the supported list
- Verify `LITELLM_AUTH_TOKEN` is valid

### Import Errors

**Symptoms:**
```
ModuleNotFoundError: No module named 'langgraph'
```

**Solutions:**
```bash
cd backend
pip install -r requirements.txt

# Or install specific packages
pip install langgraph>=0.6.7
pip install langchain-mcp-adapters>=0.1.9
```

## API Reference

### AgentResponse

```python
class AgentResponse:
    success: bool              # Whether execution succeeded
    content: str               # Agent's response content
    metadata: Dict[str, Any]   # Execution metadata
    error: Optional[str]       # Error message if failed
```

**Metadata Fields:**
- `model`: Model name used
- `tools_available`: Number of MCP tools loaded
- `tools_used`: Boolean indicating if tools were called
- `message_count`: Number of messages in conversation
- `execution_time`: Time taken for execution (if available)

### ImageGenerationResult

```python
class ImageGenerationResult(BaseModel):
    image_url: str              # The URL of the generated image
    description: str            # Description of the generated image
    source: str                 # Source of the image generation service
    success: bool               # Whether image generation was successful
```

**Usage:**
```python
from ai_agents import ImageAgent, AgentConfig

agent = ImageAgent(AgentConfig())
result = await agent.generate_image_structured(
    "A professional product photo of wireless headphones"
)

if result.success:
    print(f"✅ Generated: {result.image_url}")
    print(f"📝 Description: {result.description}")
    print(f"🔧 Source: {result.source}")
else:
    print(f"❌ Failed: {result.description}")
```

**Features:**
- Validates Google Cloud Storage URLs
- Checks that MCP tools were actually invoked
- Returns structured data instead of raw text
- Includes error details in `description` field when `success=False`

### BaseAgent

**Methods:**

- `__init__(config: AgentConfig, system_prompt: str = "")`: Initialize agent
- `execute(prompt: str, use_tools: bool = False) -> AgentResponse`: Execute prompt
- `setup_mcp(server_configs: Dict) -> None`: Configure MCP servers
- `get_capabilities() -> List[str]`: Get agent capabilities

### ChatAgent

**Inherits:** BaseAgent

**Purpose:** General conversational assistant

**Example:**
```python
agent = ChatAgent(config)
response = await agent.execute("Tell me about quantum computing")
```

### SearchAgent

**Inherits:** BaseAgent

**Purpose:** Web search with AI

**Example:**
```python
agent = SearchAgent(config)
response = await agent.execute("Latest AI news", use_tools=True)
```

### ImageAgent

**Inherits:** BaseAgent

**Purpose:** Image generation with guaranteed tool execution

**Methods:**

- `__init__(config: AgentConfig)`: Initialize image agent
- `execute(prompt: str, use_tools: bool = True) -> AgentResponse`: Generate images
- `generate_image_structured(prompt: str) -> ImageGenerationResult`: Generate with structured JSON output
- `setup_image_mcp() -> None`: Configure image MCP connection
- `get_capabilities() -> List[str]`: Get agent capabilities

**Example:**
```python
agent = ImageAgent(config)
response = await agent.execute("Generate a sunset", use_tools=True)

# Or use structured output
result = await agent.generate_image_structured("Generate a sunset")
if result.success:
    print(f"Image URL: {result.image_url}")
    print(f"Source: {result.source}")
```

### AgentConfig

**Properties:**

- `api_base_url: str` - LiteLLM endpoint URL (default: from `LITELLM_BASE_URL` or `https://litellm-docker-545630944929.us-central1.run.app`)
- `model_name: str` - Model identifier (default: from `AI_MODEL_NAME` or `gemini-2.5-pro`)
- `api_key: str` - Authentication token (default: from `LITELLM_AUTH_TOKEN` or `dummy-key`)

**Example:**
```python
# Load from environment
config = AgentConfig()
print(config.model_name)  # From AI_MODEL_NAME env var
print(config.api_base_url)  # From LITELLM_BASE_URL env var

# Or provide values directly
config = AgentConfig(
    api_base_url="https://custom-url.com",
    model_name="claude-sonnet-4-5",
    api_key="your-key"
)
```

## Related Documentation

- [Backend Server](../backend/server.py)
- [AI Agents Implementation](../backend/ai_agents/)
- [LangGraph MCP Integration](../backend/LANGGRAPH_MCP_INTEGRATION.md)
- [Tech Stack](./techstack.md)
- [Testing Guide](../HOW_TO_TEST.md)

---

**License:** Part of the FenadoV2 Template project.