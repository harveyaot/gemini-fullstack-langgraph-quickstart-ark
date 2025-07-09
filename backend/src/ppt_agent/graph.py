import os
import asyncio
import json
import logging
from typing import Dict, Any, List, cast

from dotenv import load_dotenv
from langgraph.types import Send
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig

from ppt_agent.state import (
    PPTOverallState,
    CoordinatorInputState,
    WebSearchInputState,
    OutlineInputState,
    HtmlInputState,
    ModifyInputState,
    FinalizeInputState
)
from ppt_agent.configuration import PPTConfiguration
from ppt_agent.prompts import (
    get_current_date,
    brief_outline_prompt,
    ppt_coordinator_prompt,
)
from ppt_agent.tools_and_schemas import SearchQueryList, PPTLLMTools
from ppt_agent.utils import (
    get_user_request,
    extract_ref_documents,
    parse_json_response,
    extract_ppt_outline_from_messages,
)

# Import shared components from agent module
from agent.ark_client import AsyncArkLLMClient
from agent.web_search_client import CustomWebSearchClient, WebSearchRequest

load_dotenv()

# Initialize logger
logger = logging.getLogger(__name__)

# Environment validation
if os.getenv("ARK_API_KEY") is None:
    raise ValueError("ARK_API_KEY is not set")

if os.getenv("ARK_BASE_URL") is None:
    raise ValueError("ARK_BASE_URL is not set")

if os.getenv("WEB_SEARCH_BASE_URL") is None:
    raise ValueError("WEB_SEARCH_BASE_URL is not set")

# Initialize web search client
web_search_client = CustomWebSearchClient(base_url=os.getenv("WEB_SEARCH_BASE_URL"))


async def coordinator_node(state: CoordinatorInputState, config: RunnableConfig) -> PPTOverallState:
    """
    Coordinator node that analyzes messages and decides next tool to use.
    Uses LLM to determine workflow and tool selection based on conversation context.
    """
    configurable = PPTConfiguration.from_runnable_config(config)
    logger.info("Executing coordinator node...")
    logger.info(f"State keys: {list(state.keys())}")
    logger.info(f"Number of messages in state: {len(state.get('messages', []))}")
    
    # Log the last few messages for context
    messages_in_state = state.get("messages", [])
    if messages_in_state:
        logger.info("Last few messages:")
        for i, msg in enumerate(messages_in_state[-3:], 1):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            content_preview = content[:100] + "..." if len(content) > 100 else content
            logger.info(f"  Message {i}: {role} - {content_preview}")
    
    try:
        logger.info("Step 1: Initializing LLM client...")
        # Initialize LLM client
        llm_client = AsyncArkLLMClient(
            api_key=os.getenv("ARK_API_KEY"),
            base_url=os.getenv("ARK_BASE_URL"),
            model_id=configurable.main_model,
        )
        logger.info(f"Step 1: LLM client initialized with model: {configurable.main_model}")

        logger.info("Step 2: Getting available tools...")
        # Get available tools
        tools = PPTLLMTools.get_tool_schemas()
        logger.info(f"Step 2: Found {len(tools)} tools")
        
        logger.info("Step 3: Preparing messages...")
        # Create messages array with system prompt first
        messages = [
            {
                "role": "system",
                "content": ppt_coordinator_prompt.format(current_date=get_current_date())
            }
        ]
        # Add conversation history (already dict messages)
        messages.extend(state["messages"])
        logger.info(f"Step 3: Prepared {len(messages)} messages for LLM")
        logger.info(f"Step 4: About to call ainvoke_with_tools with {len(tools)} tools")
        
        # Call LLM with messages and tools - using ainvoke_with_tools
        content, tool_calls = await llm_client.ainvoke_with_tools(
            messages=messages,
            tools=tools,
        )
        
        logger.info("Step 5: LLM call completed successfully")
        logger.info(f"Step 5: Content length: {len(content) if content else 0}")
        logger.info(f"Step 5: Tool calls count: {len(tool_calls) if tool_calls else 0}")
        
        if tool_calls:
            logger.info("Step 6: Processing tool calls...")
            # Get first tool call
            tool_call = tool_calls[0]
            logger.info(f"Step 6: First tool call: {tool_call.get('function', {}).get('name', 'unknown')}")
            
            tool_args = json.loads(tool_call["function"]["arguments"])
            tool_args["tool_call_id"] = tool_call["id"]
            
            logger.info(f"Step 6: Tool args parsed successfully")
            
            return {
                # need to add the tool_calls to the messages
                "messages": [
                    {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": tool_calls
                    }
                ],
                "next_tool": tool_call["function"]["name"],
                "tool_args": tool_args,
                "research_round_count": state.get("research_round_count", 0),
                "max_research_rounds": configurable.max_research_rounds,
            }
        else:
            logger.info("Step 6: No tool calls, proceeding to finalization...")
            # No tool call, proceed to finalization
            return {
                "messages": [{
                    "role": "assistant",
                    "content": content
                }],
                "next_tool": "finalize",
                "tool_args": None,
                "research_round_count": 0,
                "max_research_rounds": configurable.max_research_rounds,
            }
            
    except Exception as e:
        logger.error(f"Error in coordinator node at step unknown: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return {
            "messages": state["messages"] + [{
                "role": "assistant", 
                "content": f"处理请求时出错：{str(e)}"
            }],
            "next_tool": "finalize",
            "tool_args": None,
            "research_round_count": 0,
            "max_research_rounds": configurable.max_research_rounds,
        }


def route_to_tool_node(state: PPTOverallState) -> Send:
    """
    Router that converts general state into specific state for each node.
    This is where the magic happens - each node gets exactly what it needs.
    """
    next_tool = state.get("next_tool", "finalize")
    tool_args = state.get("tool_args", {})
    messages = state["messages"]
    
    # Handle case where tool_args might be None
    if tool_args is None:
        tool_args = {}
    
    # Get tool_call_id from tool_args
    tool_call_id = tool_args.get("tool_call_id")
    
    # Handle both old and new tool naming conventions
    if next_tool in ["web_search", "WebSearch"]:
        # Web search node only needs queries
        queries = tool_args.get("queries", [])
        specific_state = WebSearchInputState(
            queries=queries,
            tool_call_id=tool_call_id
        )
        return Send("web_search_node", specific_state)
        
    elif next_tool in ["gen_ppt_outline", "GenPptOutline"]:
        # Outline node needs user request and ref documents
        user_request = tool_args.get("user_request", get_user_request(messages))
        ref_documents = extract_ref_documents(messages)
        specific_state = OutlineInputState(
            user_request=user_request,
            ref_documents=ref_documents,
            tool_call_id=tool_call_id
        )
        return Send("gen_outline_node", specific_state)
        
    elif next_tool in ["gen_ppt_html", "GenPptHtml"]:
        # HTML node needs user request, outline, theme, and total pages
        user_request = tool_args.get("user_request", get_user_request(messages))
        ppt_outline = state.get("ppt_outline", {})
        theme = tool_args.get("theme", "professional")
        total_pages = tool_args.get("total_pages", 10)
        
        specific_state = HtmlInputState(
            user_request=user_request,
            ppt_outline=ppt_outline,
            theme=theme,
            total_pages=total_pages,
            tool_call_id=tool_call_id
        )
        return Send("gen_html_node", specific_state)
        
    elif next_tool in ["modify_ppt_html", "ModifyPptHtml"]:
        # Modify node needs page number and modification suggestions
        page_number = tool_args.get("page_number", 1)
        modification_suggestions = tool_args.get("modification_suggestions", "")
        
        specific_state = ModifyInputState(
            page_number=page_number,
            modification_suggestions=modification_suggestions,
            tool_call_id=tool_call_id
        )
        return Send("modify_html_node", specific_state)
        
    else:
        # Default to finalize
        specific_state = FinalizeInputState(messages=messages)
        return Send("finalize_response_node", specific_state)


async def web_search_node(state: WebSearchInputState, config: RunnableConfig) -> PPTOverallState:
    """
    Web search node that executes searches using the endpoint's native batch query support.
    """
    configurable = PPTConfiguration.from_runnable_config(config)
    logger.info("Starting web search...")
    
    # Extract search queries from state
    search_queries = state["queries"]  # Correct field from WebSearchInputState
    tool_call_id = state["tool_call_id"]  # Get tool_call_id from state
    
    try:
        # Use the endpoint's native batch query support
        search_request = WebSearchRequest(queries=search_queries)
        result = await web_search_client.web_search(search_request)
        
        # Format successful results
        content = {
            "code": 0,
            "message": f"Web search completed successfully with {len(result.data.items)} results",
            "data": {
                "sources": [
                    {
                        "value": item.url,
                        "title": item.title,
                        "content": item.content[:500] + "..." if len(item.content) > 500 else item.content,
                    }
                    for item in result.data.items
                ]
            }
        }
        tool_message = {
            "role": "tool",
            "tool_call_id": tool_call_id,  # Use tool_call_id from state
            "name": "WebSearch",
            "content": json.dumps(content)
        }
    except Exception as e:
        logger.error(f"Error in web search: {str(e)}")
        tool_message = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": "WebSearch",
            "content": f"搜索过程出错：{str(e)}"
        }
    finally:
        return {
            "messages": [tool_message],  # Return as list of dict messages
            "next_tool": "coordinator",
            "tool_args": None,
            "research_round_count": 1,  # Increment research round (even on error)
            "max_research_rounds": configurable.max_research_rounds,
        }


async def gen_outline_node(state: OutlineInputState, config: RunnableConfig) -> PPTOverallState:
    """Generate PPT outline based on user request and search results"""
    configurable = PPTConfiguration.from_runnable_config(config)
    logger.info("Generating PPT outline...")
    
    tool_message = None
    error_message = None
    outline_data = None
    try:
        # Initialize LLM client
        llm_client = AsyncArkLLMClient(
            api_key=os.getenv("ARK_API_KEY"),
            base_url=os.getenv("ARK_BASE_URL"),
            model_id=configurable.flash_model,
        )

        # Get input fields directly from state
        user_request = state["user_request"]
        ref_docs = state["ref_documents"]
        tool_call_id = state["tool_call_id"]  # Get tool_call_id from state

        # Prepare prompt (using the correct placeholder names)
        prompt = brief_outline_prompt.format(
            user_request=user_request,
            reference_material=ref_docs
        )

        logger.info(f"Calling LLM with prompt length: {len(prompt)} characters")

        # Get outline from LLM - using ainvoke with messages format
        prompt_messages = [{"role": "user", "content": prompt}]
        outline_response = await llm_client.ainvoke(prompt_messages)
        
        logger.info(f"LLM response received, length: {len(outline_response)} characters")
        logger.info(f"First 200 chars of response: {outline_response[:200]}...")
        
        # Parse the JSON response
        outline_data = parse_json_response(outline_response)
        
        logger.info(f"Parsed outline data keys: {list(outline_data.keys()) if outline_data else 'None'}")

        # Create tool message with outline
        tool_message = {
            "role": "tool",
            "tool_call_id": tool_call_id,  # Use tool_call_id from state
            "name": "GenPptOutline",
            "content": json.dumps({
                "code": 0,
                "message": "Successfully generated outline",
                "data": outline_data
            })
        }
        
        logger.info("Successfully created outline tool message")
        
    except Exception as e:
        logger.error(f"Error generating outline: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        
        error_message = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": "GenPptOutline",
            "content": f"生成大纲时出错：{str(e)}"
        }
    finally:
        if tool_message:
            return {
                "messages": [tool_message],  # Return as list of dict messages
                "next_tool": "coordinator",
                "tool_args": None,
                "research_round_count": 0,  # Reset research count for new tool
                "max_research_rounds": configurable.max_research_rounds,
                "ppt_outline": outline_data,  # Store outline in state for HTML generation
            }
        else:
            return {
                "messages": [error_message],  # Return as list of dict messages
                "next_tool": "coordinator",
                "tool_args": None,
                "research_round_count": 0,  # Reset research count even on error
                "max_research_rounds": configurable.max_research_rounds,
            }


async def gen_html_node(state: HtmlInputState, config: RunnableConfig) -> PPTOverallState:
    """Generate PPT HTML based on outline"""
    configurable = PPTConfiguration.from_runnable_config(config)
    logger.info("Generating PPT HTML...")
    
    tool_message = None
    error_message = None
    html_data = None

    # Get input fields directly from state
    user_request = state["user_request"]
    ppt_outline = state["ppt_outline"]
    theme = state.get("theme", "professional")  # Optional with default
    total_pages = state.get("total_pages", 10)  # Optional with default
    tool_call_id = state["tool_call_id"]  # Get tool_call_id from state
    
    # Format HTML response
    html_response = {
        "code": 0,
        "message": "Successfully generated PPT HTML on remote server",
        "data": {
            "html_remote_address": "https://ppt.remotesave.com/ppt/1234567890",  # Placeholder
            "theme": theme,
            "total_pages": total_pages
        }
    }
    
    # Create tool message
    tool_message = {
        "role": "tool",
        "tool_call_id": tool_call_id,  # Use tool_call_id from state
        "name": "GenPptHtml",
        "content": json.dumps(html_response)
    }
    
    return {
        "messages": [tool_message],  # Return as list of dict messages
        "next_tool": "coordinator",
        "tool_args": None,
        "research_round_count": 0,  # Reset research count for new tool
        "max_research_rounds": configurable.max_research_rounds,
    }


async def modify_html_node(state: ModifyInputState, config: RunnableConfig) -> PPTOverallState:
    """Modify PPT HTML based on suggestions"""
    configurable = PPTConfiguration.from_runnable_config(config)
    logger.info("Modifying PPT HTML...")
    
    # Get input fields directly from state
    page_number = state["page_number"]
    modification_suggestions = state["modification_suggestions"]
    tool_call_id = state["tool_call_id"]  # Get tool_call_id from state
    
    # Format modification response
    modify_response = {
        "code": 0,
        "message": "Successfully modified PPT HTML",
        "data": {
            "page_number": page_number,
            "modifications": modification_suggestions,
            "html": "<div>Modified PPT HTML content here</div>"  # Placeholder
        }
    }
    
    # Create tool message
    tool_message = {
        "role": "tool",
        "tool_call_id": tool_call_id,  # Use tool_call_id from state
        "name": "ModifyPptHtml",
        "content": json.dumps(modify_response)
    }
    
    return {
        "messages": [tool_message],  # Return as list of dict messages
        "next_tool": "coordinator",
        "tool_args": None,
        "research_round_count": 0,  # Reset research count for new tool
        "max_research_rounds": configurable.max_research_rounds,
    }
        

async def finalize_response_node(state: FinalizeInputState, config: RunnableConfig) -> PPTOverallState:
    """
    Async node that reads the conversation messages and provides a comprehensive summary.
    """
    logger.info("Executing response finalization node")
    
    # Build comprehensive response message based on conversation messages
    response_parts = []
    
    # Extract information from tool messages in the conversation
    outline_data = None
    html_result = None
    modification_result = None
    sources = []
    
    for msg in state["messages"]:
        if msg.get("role") == "tool":
            try:
                content = json.loads(msg["content"])
                
                if msg["name"] == "GenPptOutline" and content.get("code") == 0:
                    outline_data = content.get("data", {})
                elif msg["name"] == "GenPptHtml" and content.get("code") == 0:
                    html_result = content.get("data", {})
                elif msg["name"] == "ModifyPptHtml" and content.get("code") == 0:
                    modification_result = content.get("data", {})
                elif msg["name"] == "WebSearch" and content.get("code") == 0:
                    sources.extend(content.get("data", {}).get("sources", []))
            except (json.JSONDecodeError, AttributeError):
                continue
    
    # Build response based on found information
    if outline_data:
        response_parts.append(f"## PPT大纲：{outline_data.get('ppt_title', 'Generated PPT')}")
        
        if "slides" in outline_data:
            total_pages = sum(slide.get('current_slide_page_counts', 1) for slide in outline_data['slides'])
            response_parts.append(f"总页数：{total_pages}")
            
            response_parts.append("\n### 幻灯片结构：")
            for i, slide in enumerate(outline_data["slides"], 1):
                response_parts.append(f"{i}. {slide.get('title', 'Untitled')} ({slide.get('current_slide_page_counts', 1)}页)")
                response_parts.append(f"   内容：{slide.get('content', 'No content')}")
    
    # Add generation result
    if html_result:
        response_parts.append(f"\n## PPT生成成功！")
        response_parts.append(f"访问地址：{html_result.get('remote_address', 'No URL')}")
        response_parts.append(f"主题：{html_result.get('theme', 'professional')}")
        response_parts.append(f"页数：{html_result.get('pages_generated', 'Unknown')}")
    
    # Add modification results
    if modification_result:
        response_parts.append(f"\n## 修改完成：{modification_result.get('success_message', '修改成功')}")
    
    # Add research sources if available
    if sources:
        response_parts.append("\n## 参考资料：")
        for source in sources[:5]:  # Limit to first 5 sources
            response_parts.append(f"- [{source['title']}]({source['value']})")
    
    final_response = "\n".join(response_parts) if response_parts else "PPT制作流程已启动，正在处理您的需求..."
    
    # Create final AI message as dict
    final_message = {
        "role": "assistant",
        "content": final_response
    }
    
    return {"messages": [final_message]}  # Return as list of dict messages

# Create the PPT Agent Graph
def create_ppt_graph():
    """Create PPT graph with very specific input states, simplified output states"""
    from langgraph.graph import StateGraph, START, END
    
    # Create state graph with plain dict messages
    builder = StateGraph(
        PPTOverallState, 
        config_schema=PPTConfiguration
    )
    
    # Add all nodes
    builder.add_node("coordinator_node", coordinator_node)
    builder.add_node("web_search_node", web_search_node)
    builder.add_node("gen_outline_node", gen_outline_node)
    builder.add_node("gen_html_node", gen_html_node)
    builder.add_node("modify_html_node", modify_html_node)
    builder.add_node("finalize_response_node", finalize_response_node)
    
    # Set entry point - coordinator analyzes incoming messages
    builder.add_edge(START, "coordinator_node")
    
    # Route from coordinator to appropriate tool nodes using specific states
    builder.add_conditional_edges(
        "coordinator_node",
        route_to_tool_node,
        ["web_search_node", "gen_outline_node", "gen_html_node", "modify_html_node", "finalize_response_node"]
    )
    
    # All tool nodes return to coordinator for next step decision
    builder.add_edge("web_search_node", "coordinator_node")
    builder.add_edge("gen_outline_node", "coordinator_node")
    builder.add_edge("gen_html_node", "coordinator_node")
    builder.add_edge("modify_html_node", "coordinator_node")
    
    # Only finalize connects to end
    builder.add_edge("finalize_response_node", END)
    
    return builder.compile(name="ppt-optimized-state-graph")

# Export the compiled graph
ppt_graph = create_ppt_graph()