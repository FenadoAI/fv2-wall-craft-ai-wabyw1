# Extensible AI agents with LangChain and MCP support

from typing import Dict, Any, Optional, List
import os
import logging
from dataclasses import dataclass
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    # AI agent configuration
    api_base_url: str = None
    model_name: str = None
    api_key: str = None
    
    def __post_init__(self):
        # Load from env if not provided
        if self.api_base_url is None:
            self.api_base_url = os.getenv("LITELLM_BASE_URL", "https://litellm-docker-545630944929.us-central1.run.app")
        if self.model_name is None:
            self.model_name = os.getenv("AI_MODEL_NAME", "gemini-2.5-pro")
        if self.api_key is None:
            # LITELLM_AUTH_TOKEN for AI API
            self.api_key = os.getenv("LITELLM_AUTH_TOKEN", "dummy-key")


class AgentResponse(BaseModel):
    # Standard response format
    success: bool
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


class ImageGenerationResult(BaseModel):
    # Structured output for image generation
    image_url: str = Field(description="The URL of the generated image")
    description: str = Field(description="Description of the generated image")
    source: str = Field(description="Source of the image generation service")
    success: bool = Field(description="Whether image generation was successful")


class BaseAgent:
    # Base AI agent with LangChain and MCP support
    
    def __init__(self, config: AgentConfig, system_prompt: str = "You are a helpful AI assistant."):
        self.config = config
        self.system_prompt = system_prompt
        
        # LangChain ChatOpenAI setup
        self.llm = ChatOpenAI(
            base_url=config.api_base_url,
            api_key=config.api_key,
            model=config.model_name
        )
        
        # MCP client lazy init
        self.mcp_client: Optional[MultiServerMCPClient] = None
        self.mcp_tools = []
        
        logger.info(f"Initialized {self.__class__.__name__} with model {config.model_name}")
    
    async def setup_mcp(self, server_configs: Dict[str, Dict[str, Any]]):
        # Setup MCP servers and load tools
        try:
            # Initialize MCP client with server configs (dict of server name -> config)
            logger.debug("Setting up MCP with configs: %s", server_configs)
            self.mcp_client = MultiServerMCPClient(server_configs)

            # Get tools as LangChain tools (this is async!)
            if self.mcp_client:
                tools_result = await self.mcp_client.get_tools()
                logger.info(f"Tools result type: {type(tools_result)}")
                
                # Convert to list if it's not already
                if isinstance(tools_result, list):
                    self.mcp_tools = tools_result
                elif hasattr(tools_result, 'values'):
                    self.mcp_tools = list(tools_result.values())
                else:
                    self.mcp_tools = list(tools_result) if tools_result else []
                    
                logger.info("MCP setup complete with %s tools", len(self.mcp_tools))
                if self.mcp_tools:
                    logger.debug(
                        "Tool names: %s",
                        [getattr(tool, "name", "unknown") for tool in self.mcp_tools],
                    )
            else:
                self.mcp_tools = []
        except Exception as e:
            logger.error(f"Failed to setup MCP: {e}")
            import traceback
            traceback.print_exc()
            self.mcp_client = None
            self.mcp_tools = []
    
    async def execute(self, prompt: str, use_tools: bool = True) -> AgentResponse:
        # Execute agent with LangGraph
        try:
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=prompt)
            ]
            
            # Use MCP tools with LangGraph if available
            if use_tools and self.mcp_client and self.mcp_tools:
                # Use LangGraph's create_react_agent (simple form)
                from langgraph.prebuilt import create_react_agent
                
                logger.info(f"Creating agent with {len(self.mcp_tools)} tools")
                
                # Create LangGraph agent with tools (no checkpointer for simplicity)
                agent = create_react_agent(
                    self.llm,
                    self.mcp_tools
                )
                
                # Execute the agent with system prompt + user message
                result = await agent.ainvoke({
                    "messages": [
                        SystemMessage(content=self.system_prompt),
                        HumanMessage(content=prompt)
                    ]
                })
                
                # Extract the final response
                response_messages = result.get("messages", [])
                response_content = response_messages[-1].content if response_messages else ""
                
                # Check if tools were actually called
                tools_called = any(
                    hasattr(msg, "tool_calls") and msg.tool_calls 
                    for msg in response_messages
                )
                
                # Count tool invocations
                tool_call_count = sum(
                    len(msg.tool_calls) if hasattr(msg, "tool_calls") and msg.tool_calls else 0
                    for msg in response_messages
                )
                
                logger.info(f"Agent executed. Tools called: {tools_called}, Tool call count: {tool_call_count}")
                logger.debug(f"Response messages: {len(response_messages)}")
                
                # Log message types for debugging
                for i, msg in enumerate(response_messages):
                    logger.debug(f"Message {i}: {type(msg).__name__}, has tool_calls: {hasattr(msg, 'tool_calls')}")
                
                return AgentResponse(
                    success=True,
                    content=response_content,
                    metadata={
                        "model": self.config.model_name,
                        "tools_available": len(self.mcp_tools),
                        "tools_used": tools_called,
                        "tool_call_count": tool_call_count,
                        "message_count": len(response_messages)
                    }
                )
            else:
                # LLM without tools
                logger.debug(
                    "No tools available. MCP client: %s, Tools: %s",
                    self.mcp_client is not None,
                    len(self.mcp_tools),
                )
                response = await self.llm.ainvoke(messages)
                return AgentResponse(
                    success=True,
                    content=response.content,
                    metadata={
                        "model": self.config.model_name,
                        "tools_available": 0,
                        "tools_used": False
                    }
                )
            
        except Exception as e:
            logger.error(f"Error executing agent: {e}")
            import traceback
            traceback.print_exc()
            return AgentResponse(
                success=False,
                content="",
                error=str(e)
            )
    
    def get_capabilities(self) -> List[str]:
        # Get agent capabilities
        capabilities = ["text_generation", "conversation"]
        if self.mcp_client:
            capabilities.append("mcp_enabled")
        return capabilities


class SearchAgent(BaseAgent):
    # Web search and research agent
    
    def __init__(self, config: AgentConfig):
        system_prompt = """You are a research assistant with web search capabilities.
You MUST use the available web search tools to find current and accurate information.
NEVER rely on your training data for current events or real-time information.
ALWAYS use web search tools when asked about current information, weather, news, or recent events.
Cite sources from your search results."""
        
        super().__init__(config, system_prompt)
        
        # Store setup flag
        self._mcp_setup_done = False
    
    async def setup_web_search_mcp(self):
        # Setup web search MCP with auth token
        if self._mcp_setup_done:
            return
            
        mcp_token = os.getenv("CODEXHUB_MCP_AUTH_TOKEN")
        if mcp_token and mcp_token != "dummy-key":
            server_configs = {
                "web-search": {
                    "transport": "streamable_http",
                    "url": "https://mcp.codexhub.ai/web/mcp",
                    "headers": {"x-team-key": mcp_token}
                }
            }
            await self.setup_mcp(server_configs)
            self._mcp_setup_done = True
            logger.info("Web search MCP configured")
        else:
            logger.warning("CODEXHUB_MCP_AUTH_TOKEN not found, web search disabled")
    
    async def execute(self, prompt: str, use_tools: bool = True) -> AgentResponse:
        # Ensure MCP is setup before execution
        await self.setup_web_search_mcp()
        return await super().execute(prompt, use_tools)


class ChatAgent(BaseAgent):
    # General chat and assistance agent
    
    def __init__(self, config: AgentConfig):
        system_prompt = "Friendly conversational AI. Natural conversations, explanations, analysis. Helpful, harmless, honest."
        
        super().__init__(config, system_prompt)


class ImageAgent(BaseAgent):
    # Image generation agent with MCP support
    
    def __init__(self, config: AgentConfig):
        system_prompt = """You are an AI assistant specialized in generating images from text prompts. 
You MUST use the available image generation tools to create images. 
NEVER fabricate or make up image URLs.
ONLY return image URLs that you receive from the image generation tool.
Provide responses in a clear, structured format."""
        
        super().__init__(config, system_prompt)
        
        # Store setup flag
        self._mcp_setup_done = False
    
    async def setup_image_mcp(self):
        # Setup image generation MCP with auth token
        if self._mcp_setup_done:
            return
            
        mcp_token = os.getenv("CODEXHUB_MCP_AUTH_TOKEN")
        if mcp_token and mcp_token != "dummy-key":
            server_configs = {
                "image-generation": {
                    "transport": "streamable_http",
                    "url": "https://mcp.codexhub.ai/image/mcp",
                    "headers": {"x-team-key": mcp_token}
                }
            }
            await self.setup_mcp(server_configs)
            self._mcp_setup_done = True
            logger.info("Image generation MCP configured")
        else:
            logger.warning("CODEXHUB_MCP_AUTH_TOKEN not found, image generation disabled")
    
    async def execute(self, prompt: str, use_tools: bool = True) -> AgentResponse:
        # Ensure MCP is setup before execution
        await self.setup_image_mcp()
        return await super().execute(prompt, use_tools)
    
    async def generate_image_structured(self, prompt: str) -> ImageGenerationResult:
        # Generate image with structured output
        await self.setup_image_mcp()
        
        if not self.mcp_tools:
            return ImageGenerationResult(
                image_url="",
                description="Image generation failed: No tools available",
                source="none",
                success=False
            )
        
        # Use structured output with Pydantic
        llm_with_structure = self.llm.with_structured_output(ImageGenerationResult)
        
        response = await self.execute(prompt, use_tools=True)
        
        # Verify tools were actually used
        tools_used = response.metadata.get("tools_used", False)
        
        if response.success and tools_used:
            # Parse the response to extract structured data
            import re
            urls = re.findall(r'https?://[^\s\)]+', response.content)
            
            # Validate that it's a real Google Cloud Storage URL, not fabricated
            if urls and 'storage.googleapis.com' in urls[0]:
                # Extract description from markdown alt text
                description = prompt
                if '![' in response.content and '](' in response.content:
                    try:
                        description = response.content.split('![')[1].split(']')[0]
                    except:
                        pass
                
                return ImageGenerationResult(
                    image_url=urls[0],
                    description=description,
                    source="CodexHub Image MCP (Google Cloud Storage)",
                    success=True
                )
        
        # If tools weren't used, this is a fabricated response
        return ImageGenerationResult(
            image_url="",
            description=f"Image generation failed. MCP tools were not invoked. Tools available: {len(self.mcp_tools)}, Tools used: {tools_used}",
            source="none",
            success=False
        )
