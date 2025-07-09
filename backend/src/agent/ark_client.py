import os
import logging
import json
from typing import List, Dict, Any, Optional, Tuple
from volcenginesdkarkruntime import AsyncArk
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class AsyncArkLLMClient:
    """Async Ark LLM client to replace Gemini LLM functionality"""
    
    def __init__(self, api_key: str, base_url: str, model_id: str):
        self.api_key = api_key
        self.base_url = base_url
        self.model_id = model_id
        self.conversation_history = []
        self.tools = []
        
        self.client = AsyncArk(
            api_key=api_key,
            base_url=base_url,
            timeout=1800,  # 30 minutes timeout
        )
    
    def add_message(self, role: str, content: str):
        """Add a message to the conversation history"""
        self.conversation_history.append({
            "role": role,
            "content": content
        })
    
    def set_tools(self, tools: List[Dict[str, Any]]):
        """Set tools for the LLM"""
        self.tools = tools
    
    async def ainvoke(self, messages: List[Dict[str, Any]], temperature: float = 0.1) -> str:
        """
        Async invoke the LLM with messages and return the response
        
        Args:
            messages: List of conversation messages
            temperature: Temperature for response generation
            
        Returns:
            The LLM response content
        """
        try:
            response = await self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                temperature=temperature,
                tools=self.tools if self.tools else None,
                tool_choice="auto" if self.tools else None,
                thinking={
                    "type": "disabled",  # 不使用深度思考能力
                    # "type": "enabled",  # 使用深度思考能力
                    # "type": "auto",  # 模型自行判断是否使用深度思考能力
                },
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error invoking Ark LLM: {str(e)}")
            raise
    
    # Keep sync version for backward compatibility
    def invoke(self, prompt: str, temperature: float = 0.1) -> str:
        """
        Synchronous invoke (kept for backward compatibility)
        """
        import asyncio
        messages = [{"role": "user", "content": prompt}]
        return asyncio.run(self.ainvoke(messages, temperature))
    
    async def ainvoke_with_tools(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]], temperature: float = 0.1) -> Tuple[str, Optional[List[Dict[str, Any]]]]:
        """
        Async invoke the LLM with conversation messages and tools, return content and tool calls
        
        Args:
            messages: List of conversation messages
            tools: List of available tools
            temperature: Temperature for response generation
            
        Returns:
            Tuple of (content, serialized_tool_calls)
        """
        import time
        
        try:
            logger.debug(f"ArkLLMClient: Starting ainvoke_with_tools with {len(messages)} messages and {len(tools)} tools")
            logger.debug(f"ArkLLMClient: Using model: {self.model_id}")
            logger.debug(f"ArkLLMClient: Temperature: {temperature}")
            
            start_time = time.time()
            logger.debug("ArkLLMClient: About to call client.chat.completions.create...")
            
            response = await self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                temperature=temperature,
                tools=tools,
                tool_choice="auto",
                thinking={"type": "disabled"},
            )
            
            elapsed_time = time.time() - start_time
            logger.debug(f"ArkLLMClient: API call completed successfully in {elapsed_time:.2f} seconds")
            
            assistant_message = response.choices[0].message
            content = assistant_message.content or ""
            logger.debug(f"ArkLLMClient: Response content length: {len(content)}")
            
            # Serialize tool calls if present
            serialized_tool_calls = None
            if hasattr(assistant_message, 'tool_calls') and assistant_message.tool_calls:
                logger.debug(f"ArkLLMClient: Found {len(assistant_message.tool_calls)} tool calls")
                serialized_tool_calls = self.serialize_tool_calls(assistant_message.tool_calls)
                logger.debug(f"ArkLLMClient: Serialized tool calls successfully")
            else:
                logger.debug("ArkLLMClient: No tool calls found in response")
            
            logger.debug("ArkLLMClient: ainvoke_with_tools completed successfully")
            return content, serialized_tool_calls
            
        except Exception as e:
            logger.error(f"ArkLLMClient: Error invoking Ark LLM with tools: {str(e)}")
            logger.error(f"ArkLLMClient: Error type: {type(e).__name__}")
            import traceback
            logger.error(f"ArkLLMClient: Full traceback: {traceback.format_exc()}")
            raise
    
    # Keep sync version for backward compatibility
    def invoke_with_tools(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]], temperature: float = 0.1) -> Tuple[str, Optional[List[Dict[str, Any]]]]:
        """
        Synchronous invoke_with_tools (kept for backward compatibility)
        """
        import asyncio
        return asyncio.run(self.ainvoke_with_tools(messages, tools, temperature))
    
    @staticmethod
    def serialize_tool_calls(tool_calls) -> List[Dict[str, Any]]:
        """
        Central function to serialize tool calls from Ark response
        
        Args:
            tool_calls: Raw tool calls from Ark response
            
        Returns:
            List of serialized tool call dictionaries
        """
        serialized_calls = []
        for tc in tool_calls:
            serialized_calls.append({
                "id": tc.id,
                "type": tc.type,
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments
                }
            })
        return serialized_calls
    
    @staticmethod
    def parse_tool_call_arguments(tool_call) -> Dict[str, Any]:
        """
        Central function to parse tool call arguments
        
        Args:
            tool_call: Single tool call object
            
        Returns:
            Parsed arguments dictionary
        """
        try:
            return json.loads(tool_call.function.arguments)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse tool call arguments: {tool_call.function.arguments}")
            logger.error(f"Parse error: {str(e)}")
            return {}
    
    def with_structured_output(self, output_schema: BaseModel):
        """
        Return a version of this client that enforces structured output
        This mimics the LangChain interface
        """
        return StructuredArkClient(self, output_schema)


class StructuredArkClient:
    """Wrapper for structured output from Ark LLM"""
    
    def __init__(self, ark_client: AsyncArkLLMClient, output_schema: BaseModel):
        self.ark_client = ark_client
        self.output_schema = output_schema
    
    async def ainvoke(self, messages: List[Dict[str, Any]], temperature: float = 1.0):
        """
        Async invoke with structured output
        """
        # Add instructions for JSON output to the last message
        if messages and messages[-1].get("role") == "user":
            # Add JSON instructions to user message
            original_content = messages[-1]["content"]
            structured_content = f"""{original_content}

Please respond with a valid JSON object that matches this schema:
{self.output_schema.model_json_schema()}

Make sure your response is valid JSON and follows the exact structure specified."""
            
            # Create new messages with structured prompt
            structured_messages = messages[:-1] + [{
                "role": "user",
                "content": structured_content
            }]
        else:
            structured_messages = messages
        
        response_text = await self.ark_client.ainvoke(structured_messages, temperature)
        
        # Try to parse the JSON response
        import json
        try:
            # Extract JSON from response if it's wrapped in other text
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx]
            else:
                json_str = response_text
            
            response_data = json.loads(json_str)
            return self.output_schema(**response_data)
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse structured output: {response_text}")
            logger.error(f"Parse error: {str(e)}")
            # Return a default instance
            return self.output_schema()
    
    def invoke(self, prompt: str, temperature: float = 1.0):
        """
        Synchronous invoke with structured output (kept for backward compatibility)
        """
        import asyncio
        messages = [{"role": "user", "content": prompt}]
        return asyncio.run(self.ainvoke(messages, temperature))


class ArkMessage:
    """Simple message wrapper to mimic LangChain AIMessage interface"""
    
    def __init__(self, content: str):
        self.content = content 