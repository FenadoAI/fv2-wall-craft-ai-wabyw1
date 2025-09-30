# LangGraph + MCP Integration Guide

## Overview

This document explains how the **ImageAgent** and **SearchAgent** use **LangGraph** with **Model Context Protocol (MCP)** to:
- Generate real images (not fabricated URLs)
- Perform real web searches (not from training data)
- Verify tool usage and response authenticity

## Architecture

### LangGraph Integration

```python
from langgraph.prebuilt import create_react_agent

# Create agent with MCP tools
agent = create_react_agent(
    llm,  # Your LLM (ChatOpenAI, ChatAnthropic, etc.)
    mcp_tools  # Tools from MultiServerMCPClient
)

# Execute with messages
result = await agent.ainvoke({
    "messages": [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
})
```

### MCP Setup

```python
from langchain_mcp_adapters.client import MultiServerMCPClient

# Configure MCP server
client = MultiServerMCPClient({
    "image-generation": {
        "transport": "streamable_http",  # NOT "http" or "type"
        "url": "https://mcp.codexhub.ai/image/mcp",
        "headers": {"x-team-key": mcp_auth_token}
    }
})

# Initialize connection (async)
await client.__aenter__()

# Get tools (async)
tools = await client.get_tools()
```

## Test Results

### Search Agent Test Output

```
✅ Success: True
ℹ️  Metadata: {'model': 'gemini-2.5-pro', 'tools_available': 2, 'tools_used': True, 'message_count': 5}
ℹ️  MCP Tools Available: 2
ℹ️  MCP Tools Actually Used: True

📝 Full Response:
The current weather in Tokyo, Japan is mostly cloudy with a temperature of around 71-72°F (22°C)...

🎯 CORRECT: MCP tools were used and weather information found!
```

**Key Points:**
- `tools_available: 2` - Web search tools loaded
- `tools_used: True` - Tools were actually invoked
- Response contains current, real-time information (not from training data)

### Image Agent Test Output

```
✅ Success: True
ℹ️  Metadata: {'model': 'gemini-2.5-pro', 'tools_available': 1, 'tools_used': True, 'message_count': 5}
ℹ️  MCP Tools Available: 1
ℹ️  MCP Tools Actually Used: True

📝 Full Response:
https://storage.googleapis.com/fenado-ai-farm-public/generated/fccb3f99-7763-4f09-87ff-ca01f6b72cc8.webp

🔍 Verifying image URL accessibility...
   HTTP Status: 200
   ✅ Image URL is accessible (HTTP 200)

🎯 CORRECT: MCP tools were used and real image generated!
```

**Key Points:**
- `tools_available: 1` - Image generation tool loaded
- `tools_used: True` - Tool was actually invoked  
- HTTP 200 - Image URL is accessible and real

### HTTP Verification

The tests now include HTTP verification to ensure images are real:

```python
import requests

# Verify image URL
http_response = requests.head(image_url, timeout=10)

if http_response.status_code == 200:
    print("✅ Image URL is accessible")
else:
    print(f"❌ Failed with HTTP {http_response.status_code}")
```

## Key Learnings

### 1. MCP Configuration Issues

❌ **Wrong:**
```python
{
    "type": "http",  # Wrong key!
    "url": "..."
}
```

✅ **Correct:**
```python
{
    "transport": "streamable_http",  # Correct key
    "url": "..."
}
```

### 2. Server Config Format

❌ **Wrong (List):**
```python
server_configs = [{...}]  # List format doesn't work
```

✅ **Correct (Dict):**
```python
server_configs = {
    "server-name": {...}  # Dict with server name as key
}
```

### 3. Async Tool Loading

❌ **Wrong:**
```python
tools = client.get_tools()  # Returns coroutine, not tools!
```

✅ **Correct:**
```python
tools = await client.get_tools()  # Must await
```

### 4. LangGraph Parameters

The `create_react_agent` function has a simple signature:

```python
# Basic usage (no checkpointer)
agent = create_react_agent(llm, tools)

# With checkpointer (requires config)
agent = create_react_agent(llm, tools, checkpointer=MemorySaver())
result = await agent.ainvoke(
    {"messages": [...]},
    config={"configurable": {"thread_id": "1"}}
)
```

## Verification Strategy

### 1. Tool Usage Detection

```python
# Check if tools were actually called
tools_called = any(
    hasattr(msg, "tool_calls") and msg.tool_calls 
    for msg in response_messages
)
```

### 2. URL Validation

```python
# Check for Google Cloud Storage URLs
if 'storage.googleapis.com' in url:
    # Real MCP-generated image
else:
    # Potentially fabricated
```

### 3. HTTP Accessibility

```python
# Verify the image is actually accessible
response = requests.head(url, timeout=10)
assert response.status_code == 200
```

## Structured Output

The `ImageAgent` supports structured JSON output:

```python
class ImageGenerationResult(BaseModel):
    image_url: str
    description: str
    source: str
    success: bool

result = await image_agent.generate_image_structured(prompt)
# Returns validated Pydantic model
```

## Running Tests

```bash
cd backend
python tests/test_agents.py
```

### Expected Output

```
📊 Test Summary:
  Search Agent:           ✅ PASSED
  Image Agent:            ✅ PASSED
  Structured Output:      ✅ PASSED

🎉 All AI Agents tests passed!
✅ MCP tools are properly invoked (not fabricated)
✅ Structured JSON output working correctly
✅ Images verified from Google Cloud Storage
```

## Common Issues & Solutions

### Issue: "tools_used: False" (Agent not calling tools)

**Problem:** LLM is fabricating responses instead of calling MCP tools

**Causes:**
1. **For SearchAgent**: Query can be answered from training data (e.g., "What is the capital of France?")
2. **For ImageAgent**: System prompt not strong enough to force tool usage
3. MCP tools not properly loaded

**Solution:**
1. **SearchAgent**: Ask questions requiring current information:
   ```python
   # ❌ Bad: "What is the capital of France?" (LLM knows this)
   # ✅ Good: "Use web search to find today's weather in Tokyo"
   ```

2. **ImageAgent**: Ensure strong system prompt:
   ```python
   system_prompt = """You MUST use the available image generation tools.
   NEVER fabricate or make up image URLs."""
   ```

3. **Both**: Verify MCP setup:
   - MCP client initialized: `await client.__aenter__()`
   - Tools loaded: `len(await client.get_tools()) > 0`
   - LangGraph agent created: `create_react_agent(llm, tools)`

### Issue: "'list' object has no attribute 'values'"

**Problem:** Server config is a list instead of dict

**Solution:**
```python
# Change from list
[{"url": "..."}]

# To dict
{"server-name": {"url": "..."}}
```

### Issue: "Missing 'transport' key"

**Problem:** Using wrong key name

**Solution:**
```python
# Change "type" to "transport"
{"transport": "streamable_http"}
```

## Best Practices

1. **Always verify tools are loaded** before agent execution
2. **Use HTTP verification** to ensure images are real
3. **Check `tools_used` metadata** to confirm MCP invocation  
4. **Validate URL format** (storage.googleapis.com)
5. **Handle async properly** with await for MCP operations

## References

- [LangGraph MCP Agents](https://github.com/teddynote-lab/langgraph-mcp-agents)
- [LangChain MCP Adapters](https://python.langchain.com/docs/integrations/mcp/)
- [Model Context Protocol](https://modelcontextprotocol.io/)
