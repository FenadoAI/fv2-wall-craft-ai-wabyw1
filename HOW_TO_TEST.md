# How to Test

This guide shows you how to write simple, effective tests for your application. **Important: Write real tests that can fail - never write fake tests that always pass.**

## 🎯 Testing Philosophy

- **Real Tests Only**: Tests must be able to fail when something is actually broken
- **No Fake Passes**: Never write tests that always return success
- **Test Real Functionality**: Test actual behavior, not mocked responses
- **Simple and Clear**: Each test should test one specific thing

## 📋 Types of Tests

### 1. API Integration Test

Test external APIs you need to integrate with:

```python
# test_external_api.py
import requests

def test_external_api():
    """Test real API integration - this can fail if API is down"""
    try:
        response = requests.get("https://api.example.com/data")
        
        # Real test - this will fail if API returns error
        assert response.status_code == 200, f"API returned {response.status_code}"
        
        data = response.json()
        assert "expected_field" in data, "Missing required field in response"
        
        print("✅ API integration test passed")
        return True
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False

if __name__ == "__main__":
    test_external_api()
```

### 2. Business Logic Test

Test your application logic:

```python
# test_logic.py
def calculate_discount(price, user_type):
    """Business logic to test"""
    if user_type == "premium":
        return price * 0.8  # 20% discount
    elif user_type == "regular":
        return price * 0.9  # 10% discount
    else:
        return price  # No discount

def test_discount_logic():
    """Test business logic - can fail if logic is wrong"""
    # Test premium user
    result = calculate_discount(100, "premium")
    assert result == 80, f"Expected 80, got {result}"
    
    # Test regular user
    result = calculate_discount(100, "regular")
    assert result == 90, f"Expected 90, got {result}"
    
    # Test unknown user
    result = calculate_discount(100, "unknown")
    assert result == 100, f"Expected 100, got {result}"
    
    print("✅ Business logic tests passed")
    return True

if __name__ == "__main__":
    test_discount_logic()
```

### 3. AI Agent Test

Test AI functionality with real responses:

```python
# test_ai_agent.py
import asyncio
from ai_agents import SearchAgent, AgentConfig

async def test_ai_search():
    """Test AI agent - will fail if agent doesn't work"""
    try:
        config = AgentConfig()
        agent = SearchAgent(config)
        
        # Real test - this will fail if token is invalid
        response = await agent.execute("What is 2+2?", use_tools=True)
        
        # Check if response makes sense
        assert response.success, f"AI agent failed: {response.error}"
        assert "4" in response.content, f"AI gave wrong answer: {response.content}"
        
        print("✅ AI agent test passed")
        return True
    except Exception as e:
        print(f"❌ AI agent test failed: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(test_ai_search())
```

### 4. Database Test

Test database operations:

```python
# test_database.py
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
import os
from dotenv import load_dotenv

load_dotenv()

async def test_database_operations():
    """Test database - will fail if DB is down or operations fail"""
    try:
        client = AsyncIOMotorClient(os.environ['MONGO_URL'])
        db = client[os.environ['DB_NAME']]
        
        # Test write
        test_doc = {"name": "test", "value": 123}
        result = await db.test_collection.insert_one(test_doc)
        assert result.inserted_id, "Failed to insert document"
        
        # Test read
        found_doc = await db.test_collection.find_one({"name": "test"})
        assert found_doc, "Failed to find inserted document"
        assert found_doc["value"] == 123, f"Wrong value: {found_doc['value']}"
        
        # Test delete
        delete_result = await db.test_collection.delete_one({"name": "test"})
        assert delete_result.deleted_count == 1, "Failed to delete document"
        
        print("✅ Database test passed")
        return True
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(test_database_operations())
```

### 5. FastAPI Endpoint Test

Test your API endpoints:

```python
# test_endpoints.py
import requests
import json

BASE_URL = "http://localhost:8001/api"

def test_chat_endpoint():
    """Test chat endpoint - will fail if endpoint is broken"""
    try:
        payload = {
            "message": "Hello",
            "agent_type": "chat"
        }
        
        response = requests.post(f"{BASE_URL}/chat", json=payload)
        
        # Real test - will fail if endpoint returns error
        assert response.status_code == 200, f"Endpoint returned {response.status_code}"
        
        data = response.json()
        assert data.get("success"), f"Endpoint failed: {data.get('error')}"
        assert data.get("response"), "No response content"
        
        print("✅ Chat endpoint test passed")
        return True
    except Exception as e:
        print(f"❌ Chat endpoint test failed: {e}")
        return False

def test_search_endpoint():
    """Test search endpoint - will fail if search doesn't work"""
    try:
        payload = {
            "query": "capital of France",
            "max_results": 3
        }
        
        response = requests.post(f"{BASE_URL}/search", json=payload)
        assert response.status_code == 200, f"Search returned {response.status_code}"
        
        data = response.json()
        assert data.get("success"), f"Search failed: {data.get('error')}"
        assert "Paris" in data.get("summary", ""), "Search didn't find Paris"
        
        print("✅ Search endpoint test passed")
        return True
    except Exception as e:
        print(f"❌ Search endpoint test failed: {e}")
        return False

if __name__ == "__main__":
    test_chat_endpoint()
    test_search_endpoint()
```

## 🚫 What NOT to Do

### ❌ Bad Test - Always Passes
```python
def bad_test():
    """This is a fake test - it always passes"""
    print("✅ Test passed")  # This is meaningless
    return True
```

### ❌ Bad Test - Doesn't Test Real Functionality
```python
def bad_api_test():
    """This doesn't test the real API"""
    mock_response = {"status": "ok"}  # Fake response
    assert mock_response["status"] == "ok"  # Will always pass
    return True
```

### ❌ Bad Test - No Assertions
```python
def bad_test():
    """This doesn't verify anything"""
    result = some_function()
    print(f"Result: {result}")  # Just prints, doesn't test
    return True  # Always returns True
```

## ✅ What TO Do

### ✅ Good Test - Can Actually Fail
```python
def good_test():
    """This test can fail if the function is broken"""
    result = calculate_tax(100, 0.1)
    assert result == 10, f"Expected 10, got {result}"  # Will fail if wrong
    return True
```

### ✅ Good Test - Tests Real Behavior
```python
def good_api_test():
    """This tests real API behavior"""
    response = requests.get("https://real-api.com/data")
    assert response.status_code == 200  # Will fail if API is down
    assert response.json()["data"]  # Will fail if no data
    return True
```

## 🔧 Test Patterns from Tech Stack

Based on the tech stack patterns, here are common test scenarios:

### FastAPI Test Pattern
```python
from fastapi.testclient import TestClient
from server import app

client = TestClient(app)

def test_api_endpoint():
    response = client.post("/api/items", json={"name": "test"})
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "test"
```

### MongoDB Test Pattern
```python
async def test_mongodb():
    # Insert test data
    result = await db.items.insert_one({"name": "test"})
    assert result.inserted_id
    
    # Query test data
    item = await db.items.find_one({"name": "test"})
    assert item["name"] == "test"
    
    # Clean up
    await db.items.delete_one({"_id": result.inserted_id})
```


## 🎯 Test Writing Checklist

Before writing a test, ask:

1. **Can this test fail?** If not, it's not a real test
2. **Does it test real functionality?** No mocks for external dependencies you want to verify
3. **Is it specific?** Test one thing at a time
4. **Will it catch bugs?** If the code breaks, will this test fail?

## 🔧 Environment Setup

### Required .env File
Before running tests, make sure you have a `.env` file in the backend directory with all required environment variables:

```bash
# Copy the template and fill in your values
cp backend/.env.example backend/.env
```

Required variables for testing:
- `MONGO_URL` - For database tests  
- `DB_NAME` - Database name for tests
- `LITELLM_AUTH_TOKEN` - For AI model authentication (⚠️ must start with 'sk-')
- `CODEXHUB_MCP_AUTH_TOKEN` - For CodexHub web search only
- `AI_MODEL_NAME` - AI model to use (optional, has default)

**⚠️ Common Issue:** If you see `401 Unauthorized` errors, your `LITELLM_AUTH_TOKEN` is invalid or set to `"dummy-key"`. The token must start with `'sk-'` to be accepted by LiteLLM.

Without proper `.env` setup, tests will fail with missing environment variables.

## 🚀 Running Tests

### Quick Test
```bash
# Test specific functionality
python test_ai_agent.py
python test_database.py  
python test_endpoints.py
```

### Current Tests

#### AI Agents Test (No Server Required)
```bash
# Test the AI agents library (SearchAgent, ImageAgent, etc.)
cd backend && python tests/test_agents.py
```
**What it tests:** Real web search, image generation, MCP integration, tool verification

#### API Integration Test (Requires Running Server)
```bash
# Terminal 1: Start the server
cd backend && uvicorn server:app --reload --port 8001

# Terminal 2: Run API tests
cd backend && python -m pytest tests/test_api.py -v
```
**What it tests:** FastAPI endpoints (`/api/`, `/api/chat`, `/api/search`, `/api/agents/capabilities`)

**Important:** `test_api.py` will FAIL with `Connection refused` if the server is not running. This is correct behavior - it's a real integration test!

Remember: **A test that never fails is not a test - it's a lie.**

## 🔍 Troubleshooting

### Common Test Failures

#### 1. Authentication Error (401 Unauthorized)
```
Error code: 401 - Authentication Error, LiteLLM Virtual Key expected. 
Received=dummy-key, expected to start with 'sk-'.
```

**Root Cause:** The `LITELLM_AUTH_TOKEN` environment variable is missing or set to the default `"dummy-key"` value.

**Solution:** 
1. Create or update `backend/.env` file
2. Set a valid token: `LITELLM_AUTH_TOKEN=sk-your-actual-token-here`
3. Token must start with `'sk-'` to be accepted by LiteLLM

**Verify:**
```bash
cd backend
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print(f'Token set: {bool(os.getenv(\"LITELLM_AUTH_TOKEN\"))}')"
```

#### 2. Test Assertion Logic Error
```
AssertionError: assert 'Tokyo' in 'i am sorry...however, i can tell you that the capital of japan is **tokyo**.'
```

**Root Cause:** Looking for uppercase text in a lowercase string.

**Solution:** When using `.lower()`, compare with lowercase text:
```python
# ❌ WRONG - "Tokyo" will never be in a lowercase string
assert "Tokyo" in data["summary"].lower()

# ✅ CORRECT - "tokyo" matches lowercase string
assert "tokyo" in data["summary"].lower()
```

**Fixed in:** `backend/tests/test_api.py` line 38

#### 3. MCP Tools Not Being Used

**Symptom:** `tools_used: False` in metadata, response says "I am unable to access the web search tools"

**This is EXPECTED BEHAVIOR:** The AI model is intelligent and only uses tools when necessary:

✅ **Tools ARE used for:**
- Current weather, news, events
- Real-time data (stock prices, sports scores)
- Recent information not in training data
- Location-specific current information

❌ **Tools NOT used for:**
- Historical facts (capitals, dates)
- General knowledge questions
- Mathematical calculations
- Basic definitions

**Example:**
```python
# Won't use web search (general knowledge)
await agent.execute("What is the capital of France?")
# Result: tools_used = False

# WILL use web search (current info)
await agent.execute("What is the current weather in Tokyo today?")
# Result: tools_used = True
```

**To verify MCP is working:**
```bash
cd backend
curl -X POST http://localhost:8001/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "current weather in Tokyo", "max_results": 3}'
```
Should return `"tools_used": true` and real search results with sources.

#### 4. Connection Refused Error

```
requests.exceptions.ConnectionError: Connection refused
```

**Root Cause:** Backend server is not running.

**Solution:**
```bash
# Terminal 1: Start the server FIRST
cd backend && uvicorn server:app --reload --port 8001

# Terminal 2: Then run tests
cd backend && python -m pytest tests/test_api.py -v
```

#### 5. Intermittent MCP Connection Errors

**Symptom:** `httpcore.ConnectError` in logs during MCP tool execution

**This is normal behavior:** 
- Network issues can cause temporary MCP server connection failures
- Agent gracefully falls back to training data
- Test still passes with `success: True`
- No action required - this is graceful degradation working as designed

### Environment Verification

Check your environment setup:
```bash
cd backend
python -c "
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path('.') / '.env')

print('Environment Check:')
print(f'  LITELLM_AUTH_TOKEN: {\"✅\" if os.getenv(\"LITELLM_AUTH_TOKEN\") else \"❌\"}')
print(f'  CODEXHUB_MCP_AUTH_TOKEN: {\"✅\" if os.getenv(\"CODEXHUB_MCP_AUTH_TOKEN\") else \"❌\"}')
print(f'  MONGO_URL: {\"✅\" if os.getenv(\"MONGO_URL\") else \"❌\"}')
print(f'  DB_NAME: {\"✅\" if os.getenv(\"DB_NAME\") else \"❌\"}')
"
```

### Debug Logging

To see detailed tool usage:
```bash
cd backend
# Set logging to DEBUG level
export PYTHONPATH=$PWD
python -c "
import logging
logging.basicConfig(level=logging.DEBUG)

import asyncio
from ai_agents import SearchAgent, AgentConfig

async def test():
    agent = SearchAgent(AgentConfig())
    result = await agent.execute('current weather in Tokyo', use_tools=True)
    print(f'Tools used: {result.metadata.get(\"tools_used\")}')
    print(f'Tool calls: {result.metadata.get(\"tool_call_count\")}')

asyncio.run(test())
"
```

## 📚 Additional Resources

- [How to Add AI Functionality](docs/how-to-add-ai-functionality.md) - AI agents implementation guide
- [Tech Stack](docs/techstack.md) - Technology overview
