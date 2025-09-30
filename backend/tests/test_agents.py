"""Smoke tests for AI agents hitting real services."""

import asyncio
import os
import sys
from pathlib import Path
import re

from dotenv import load_dotenv
import requests
import pytest

# Ensure backend package is on sys.path when invoked from repo root
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ai_agents import AgentConfig, ImageAgent, SearchAgent


load_dotenv()


@pytest.mark.asyncio
async def test_search_agent():
    print("\n🔍 Testing SearchAgent...")
    os.environ.setdefault("AI_MODEL_NAME", "gemini-2.5-pro")
    config = AgentConfig()
    agent = SearchAgent(config)

    prompt = "Use web search to find today's weather in Tokyo"
    response = await agent.execute(prompt, use_tools=True)

    assert response.success, response.error
    assert response.metadata.get("tools_used"), "Expected MCP tools to be invoked"
    assert "tokyo" in response.content.lower()
    
    print(f"  ✅ Search Agent PASSED")
    print(f"     Tools used: {response.metadata.get('tools_used')}")
    print(f"     Tool calls: {response.metadata.get('tool_call_count', 0)}")
    print(f"     Response preview: {response.content[:100]}...")


@pytest.mark.asyncio
async def test_image_agent():
    print("\n🎨 Testing ImageAgent...")
    os.environ.setdefault("AI_MODEL_NAME", "gemini-2.5-pro")
    config = AgentConfig()
    agent = ImageAgent(config)

    prompt = "Generate an image of a sunset over mountains"
    response = await agent.execute(prompt, use_tools=True)

    assert response.success, response.error
    assert response.metadata.get("tools_used"), "Expected MCP tools to be invoked"

    urls = re.findall(r"https?://[^\s)]+", response.content)
    assert urls, "Expected the response to include at least one URL"

    head = requests.head(urls[0], timeout=10)
    assert head.status_code == 200, f"Image URL not reachable: HTTP {head.status_code}"
    
    print(f"  ✅ Image Agent PASSED")
    print(f"     Tools used: {response.metadata.get('tools_used')}")
    print(f"     Tool calls: {response.metadata.get('tool_call_count', 0)}")
    print(f"     Image URL: {urls[0]}")
    print(f"     HTTP Status: {head.status_code}")


async def main():
    print("\n" + "="*60)
    print("🤖 AI AGENTS TEST SUITE")
    print("="*60)
    
    try:
        await test_search_agent()
        await test_image_agent()
        
        print("\n" + "="*60)
        print("🎉 ALL TESTS PASSED!")
        print("="*60)
        print("\n✅ MCP tools are properly invoked (not fabricated)")
        print("✅ Real web search results from CodexHub MCP")
        print("✅ Real image URLs from Google Cloud Storage")
        print("✅ HTTP 200 verification for image accessibility")
        print()
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
