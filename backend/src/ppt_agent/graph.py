import os
import asyncio
import json
import logging
from typing import Dict, Any, List, cast, TypedDict, Optional, Annotated, operator

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
    ThemeAndPagesAckInputState,
    FinalizeInputState,
    ExecutionRecord,
)
from ppt_agent.configuration import PPTConfiguration
from ppt_agent.prompts import (
    get_current_date,
    brief_outline_prompt,
    ppt_coordinator_prompt,
    web_results_summary_prompt,
    research_reflection_prompt,
)
from ppt_agent.tools_and_schemas import SearchQueryList, PPTLLMTools
from ppt_agent.utils import (
    get_user_request,
    extract_ref_documents,
    parse_json_response,
    get_consecutive_search_count,
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


async def coordinator_node(
    state: CoordinatorInputState, config: RunnableConfig
) -> PPTOverallState:
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
        logger.info(
            f"Step 1: LLM client initialized with model: {configurable.main_model}"
        )

        logger.info("Step 2: Getting available tools...")
        # Get available tools
        tools = PPTLLMTools.get_tool_schemas()
        logger.info(f"Step 2: Found {len(tools)} tools")

        logger.info("Step 3: Preparing messages...")
        # Create messages array with system prompt first
        messages = [
            {
                "role": "system",
                "content": ppt_coordinator_prompt.format(
                    current_date=get_current_date()
                ),
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
            logger.info(
                f"Step 6: First tool call: {tool_call.get('function', {}).get('name', 'unknown')}"
            )

            tool_args = json.loads(tool_call["function"]["arguments"])
            tool_args["tool_call_id"] = tool_call["id"]

            logger.info(f"Step 6: Tool args parsed successfully")

            return {
                # need to add the tool_calls to the messages
                "messages": [
                    {"role": "assistant", "content": content, "tool_calls": tool_calls}
                ],
                "next_tool": tool_call["function"]["name"],
                "tool_args": tool_args,
            }
        else:
            logger.info("Step 6: No tool calls, proceeding to finalization...")
            # No tool call, proceed to finalization
            return {
                "messages": [{"role": "assistant", "content": content}],
                "next_tool": "finalize",
                "tool_args": None,
            }

    except Exception as e:
        logger.error(f"Error in coordinator node at step unknown: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback

        logger.error(f"Full traceback: {traceback.format_exc()}")
        return {
            "messages": state["messages"]
            + [{"role": "assistant", "content": f"处理请求时出错：{str(e)}"}],
            "next_tool": "finalize",
            "tool_args": None,
        }


async def web_search_node(
    state: WebSearchInputState, config: RunnableConfig
) -> PPTOverallState:
    """
    Web search node that executes searches using the endpoint's native batch query support.
    """
    configurable = PPTConfiguration.from_runnable_config(config)
    logger.info("Starting web search...")

    # Extract search queries from state
    search_queries = state["queries"]  # Correct field from WebSearchInputState
    tool_call_id = state["tool_call_id"]  # Get tool_call_id from state
    tool_execution_history = state["tool_execution_history"]  # Get execution history

    # Check consecutive web search limit
    max_count = configurable.max_consecutive_search_count
    consecutive_count = get_consecutive_search_count(tool_execution_history)
    is_success = False

    if consecutive_count >= max_count:
        # Return error message if limit exceeded
        error_message = f"工具调用失败，连续搜索次数已达到上限 {max_count} 次"
        tool_message = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": "WebSearch",
            "content": error_message,
        }
        # Add this execution to history
        execution_record = ExecutionRecord(
            tool_name="web_search",
            status="failed",
            result={"error": error_message, "reason": "consecutive_limit_exceeded"},
        )
        return {
            "messages": [tool_message],
            "next_tool": "coordinator",
            "tool_args": None,
            "tool_execution_history": [execution_record.model_dump()],
        }

    # Initialize sources_gathered outside try block
    sources_gathered = []
    web_results_summary = ""

    try:
        # Use the endpoint's native batch query support
        search_request = WebSearchRequest(queries=search_queries)
        result = await web_search_client.web_search(search_request)

        # Collect sources in the new format
        sources_gathered = [
            {
                "url": item.url,
                "title": item.title,
                "content": (
                    item.content[:1500] + "..."
                    if len(item.content) > 1500
                    else item.content
                ),
            }
            for idx, item in enumerate(result.data.items)
        ]

        source_groups = [
            sources_gathered[i : i + 4] for i in range(0, len(sources_gathered), 4)
        ]

        # Helper function to summarize a single group
        async def summarize_group(llm_client, group, group_idx):
            """Summarize a single group of sources"""
            # Format the source group for the prompt
            web_results = ""
            for idx, source in enumerate(group, 1):
                web_results += f"\n[{idx}]. Title: {source['title']}\n   URL: {source['url']}\n   Content: {source['content']}\n"

            # Generate summary for this group
            summary_prompt = web_results_summary_prompt.format(
                web_results=web_results,
                research_topics="##".join(search_queries),
                current_date=get_current_date(),
            )

            prompt_messages = [{"role": "user", "content": summary_prompt}]

            try:
                summary = await llm_client.ainvoke(prompt_messages)
                # Traverse the group and replace any short_url in the summary with the url
                for idx, source in enumerate(group, 1):

                    if f"[{idx}]" in summary:
                        summary = summary.replace(
                            f"[{idx}]", f"[{source['title']}]({source['url']})"
                        )
                return {
                    "summary": summary,
                    "source_count": len(group),
                }
            except Exception as e:
                logger.error(f"Error summarizing source group {group_idx}: {str(e)}")
                # Fallback: use raw content
                raw_content = "\n".join(
                    [
                        f"- {source['title']} ({source['url']}): {source['content'][:500]}"
                        for source in group
                    ]
                )
                return {
                    "summary": raw_content,
                    "source_count": len(group),
                }

        # Process all groups through summarization in parallel
        summarized_chunks = []
        if source_groups:
            # Initialize LLM client for summarization
            llm_client = AsyncArkLLMClient(
                api_key=os.getenv("ARK_API_KEY"),
                base_url=os.getenv("ARK_BASE_URL"),
                model_id=configurable.flash_model,
            )

            # Create tasks for all groups and run them in parallel
            tasks = [
                summarize_group(llm_client, group, group_idx)
                for group_idx, group in enumerate(source_groups)
            ]
            logger.info(f"Step 7: Created {len(tasks)} groups for summarization")
            # Execute all summarization tasks in parallel
            summarized_chunks = await asyncio.gather(*tasks)

        # Format successful results with summarized content
        web_results_summary = "\n\n---\n\n".join(
            [chunk["summary"] for chunk in summarized_chunks]
        )
        content = {
            "message": f"Web search completed successfully with {len(sources_gathered)} sources in {len(summarized_chunks)} 10 summarys",
            "data": {
                "summary": web_results_summary,
                "total_sources": len(sources_gathered),
            },
        }

        tool_message = {
            "role": "tool",
            "tool_call_id": tool_call_id,  # Use tool_call_id from state
            "name": "WebSearch",
            "content": json.dumps(content),
        }
        is_success = True
    except Exception as e:
        logger.error(f"Error in web search: {str(e)}")
        tool_message = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": "WebSearch",
            "content": f"搜索过程出错：{str(e)}",
        }
    finally:
        # Add execution record to history
        execution_record = ExecutionRecord(
            tool_name="web_search",
            status="succeeded" if is_success else "failed",
            result={"web_results_summary": web_results_summary},
        )

        return {
            "messages": [tool_message],  # Return as list of dict messages
            "next_tool": "coordinator",
            "tool_args": None,
            "tool_execution_history": [execution_record.model_dump()],
            "sources_gathered": sources_gathered,
            "web_results_summary": [web_results_summary],
        }


async def gen_outline_node(
    state: OutlineInputState, config: RunnableConfig
) -> PPTOverallState:
    """Generate PPT outline based on user request and search results"""
    configurable = PPTConfiguration.from_runnable_config(config)
    logger.info("Generating PPT outline...")

    tool_message = None
    error_message = None
    outline_data = None
    error_info = None
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
        total_pages = state["total_pages"]
        tool_call_id = state["tool_call_id"]  # Get tool_call_id from state

        # Prepare prompt (using the correct placeholder names)
        prompt = brief_outline_prompt.format(
            user_request=user_request,
            reference_material=ref_docs,
            total_pages=total_pages,
        )

        logger.info(f"Calling LLM with prompt length: {len(prompt)} characters")

        # Get outline from LLM - using ainvoke with messages format
        prompt_messages = [{"role": "user", "content": prompt}]
        outline_response = await llm_client.ainvoke(prompt_messages)

        logger.info(
            f"LLM response received, length: {len(outline_response)} characters"
        )
        logger.info(f"First 200 chars of response: {outline_response[:200]}...")

        # Parse the JSON response
        outline_data = parse_json_response(outline_response)

        logger.info(
            f"Parsed outline data keys: {list(outline_data.keys()) if outline_data else 'None'}"
        )

        # Create tool message with outline
        tool_message = {
            "role": "tool",
            "tool_call_id": tool_call_id,  # Use tool_call_id from state
            "name": "GenPptOutline",
            "content": json.dumps(
                {
                    "code": 0,
                    "message": "Successfully generated outline",
                    "data": outline_data,
                }
            ),
        }

        logger.info("Successfully created outline tool message")

    except Exception as e:
        error_info = str(e)
        logger.error(f"Error generating outline: {error_info}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback

        logger.error(f"Full traceback: {traceback.format_exc()}")

        error_message = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": "GenPptOutline",
            "content": f"生成大纲时出错：{error_info}",
        }
    finally:
        if tool_message:
            # Add execution record to history
            execution_record = ExecutionRecord(
                tool_name="gen_ppt_outline",
                status="succeeded",
                result={
                    "outline_generated": True,
                    "outline_keys": list(outline_data.keys()) if outline_data else [],
                },
            )
            return {
                "messages": [tool_message],  # Return as list of dict messages
                "next_tool": "coordinator",
                "tool_args": None,
                "ppt_outline": outline_data,  # Store outline in state for HTML generation
                "tool_execution_history": [
                    execution_record.model_dump()
                ],  # New execution to add
            }
        else:
            # Add execution record to history
            execution_record = ExecutionRecord(
                tool_name="gen_ppt_outline",
                status="failed",
                result={
                    "outline_generated": False,
                    "error": error_info or "Unknown error",
                },
            )
            return {
                "messages": [error_message],  # Return as list of dict messages
                "next_tool": "coordinator",
                "tool_args": None,
                "tool_execution_history": [
                    execution_record.model_dump()
                ],  # New execution to add
            }


async def gen_html_node(
    state: HtmlInputState, config: RunnableConfig
) -> PPTOverallState:
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
    tool_call_id = state["tool_call_id"]  # Get tool_call_id from state

    # Format HTML response
    html_response = {
        "code": 0,
        "message": "Successfully generated PPT HTML on remote server",
        "data": {
            "html_remote_address": "https://ppt.remotesave.com/ppt/1234567890",  # Placeholder
            "theme": theme,
        },
    }

    # Create tool message
    tool_message = {
        "role": "tool",
        "tool_call_id": tool_call_id,  # Use tool_call_id from state
        "name": "GenPptHtml",
        "content": json.dumps(html_response),
    }

    # Add execution record to history
    execution_record = ExecutionRecord(
        tool_name="gen_ppt_html",
        status="succeeded",
        result={"theme": theme, "html_generated": True},
    )

    return {
        "messages": [tool_message],  # Return as list of dict messages
        "next_tool": "coordinator",
        "tool_args": None,
        "tool_execution_history": [
            execution_record.model_dump()
        ],  # New execution to add
    }


async def modify_html_node(
    state: ModifyInputState, config: RunnableConfig
) -> PPTOverallState:
    """Modify PPT HTML based on suggestions"""
    configurable = PPTConfiguration.from_runnable_config(config)
    logger.info("Modifying PPT HTML...")

    # Get input fields directly from state
    page_number = state["page_number"]
    modification_suggestions = state["modification_suggestions"]
    tool_call_id = state["tool_call_id"]  # Get tool_call_id from state

    # Format modification response
    modify_response = {
        "message": "Successfully modified PPT HTML",
        "data": {
            "page_number": page_number,
            "modifications": modification_suggestions,
            "html": "<div>Modified PPT HTML content here</div>",  # Placeholder
        },
    }

    # Create tool message
    tool_message = {
        "role": "tool",
        "tool_call_id": tool_call_id,  # Use tool_call_id from state
        "name": "ModifyPptHtml",
        "content": json.dumps(modify_response),
    }

    # Add execution record to history
    execution_record = ExecutionRecord(
        tool_name="modify_ppt_html",
        status="succeeded",
        result={
            "page_number": page_number,
            "modifications": modification_suggestions,
            "html_modified": True,
        },
    )

    return {
        "messages": [tool_message],  # Return as list of dict messages
        "next_tool": "coordinator",
        "tool_args": None,
        "tool_execution_history": [
            execution_record.model_dump()
        ],  # New execution to add
    }


async def theme_and_pages_ack_node(
    state: ThemeAndPagesAckInputState, config: RunnableConfig
) -> PPTOverallState:
    """Simulate external tool that presents theme and pages suggestions to user for confirmation"""
    configurable = PPTConfiguration.from_runnable_config(config)
    logger.info("Presenting PPT theme and pages suggestions to user...")

    # Get input fields directly from state (from coordinator's tool_args)
    suggested_theme = state["theme"]
    suggested_pages = state["total_pages"]
    tool_call_id = state["tool_call_id"]  # Get tool_call_id from state

    logger.info(f"Suggesting theme: {suggested_theme}, pages: {suggested_pages}")

    # Simulate external tool response that presents suggestions to user
    theme_pages_response = {
        "message": "The user has confirmed the theme and total_pages",
        "data": {
            "confirmed_theme": suggested_theme,
            "confirmed_total_pages": suggested_pages,
        },
    }

    # Create tool message
    tool_message = {
        "role": "tool",
        "tool_call_id": tool_call_id,  # Use tool_call_id from state
        "name": "ThemeAndPagesAck",
        "content": json.dumps(theme_pages_response),
    }

    # Add execution record to history
    execution_record = ExecutionRecord(
        tool_name="ppt_theme_and_pages_ack",
        status="succeeded",
        result={
            "suggested_theme": suggested_theme,
            "suggested_total_pages": suggested_pages,
            "user_confirmation_pending": True,
        },
    )

    return {
        "messages": [tool_message],  # Return as list of dict messages
        "next_tool": "coordinator",
        "tool_args": None,
        "tool_execution_history": [
            execution_record.model_dump()
        ],  # New execution to add
        "theme": suggested_theme,
        "total_pages": suggested_pages,
    }


async def finalize_response_node(
    state: FinalizeInputState, config: RunnableConfig
) -> PPTOverallState:
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
        response_parts.append(
            f"## PPT大纲：{outline_data.get('ppt_title', 'Generated PPT')}"
        )

        if "slides" in outline_data:
            total_pages = sum(
                slide.get("current_slide_page_counts", 1)
                for slide in outline_data["slides"]
            )
            response_parts.append(f"总页数：{total_pages}")

            response_parts.append("\n### 幻灯片结构：")
            for i, slide in enumerate(outline_data["slides"], 1):
                response_parts.append(
                    f"{i}. {slide.get('title', 'Untitled')} ({slide.get('current_slide_page_counts', 1)}页)"
                )
                response_parts.append(f"   内容：{slide.get('content', 'No content')}")

    # Add generation result
    if html_result:
        response_parts.append(f"\n## PPT生成成功！")
        response_parts.append(
            f"访问地址：{html_result.get('remote_address', 'No URL')}"
        )
        response_parts.append(f"主题：{html_result.get('theme', 'professional')}")
        response_parts.append(f"页数：{html_result.get('pages_generated', 'Unknown')}")

    # Add modification results
    if modification_result:
        response_parts.append(
            f"\n## 修改完成：{modification_result.get('success_message', '修改成功')}"
        )

    # Add research sources if available
    if sources:
        response_parts.append("\n## 参考资料：")
        for source in sources[:5]:  # Limit to first 5 sources
            response_parts.append(f"- [{source['title']}]({source['value']})")

    final_response = (
        "\n".join(response_parts)
        if response_parts
        else "PPT制作流程已启动，正在处理您的需求..."
    )

    # Create final AI message as dict
    final_message = {"role": "assistant", "content": final_response}

    # Add execution record to history
    execution_record = ExecutionRecord(
        tool_name="finalize",
        status="succeeded",
        result={"response_length": len(final_response), "finalization_completed": True},
    )

    return {
        "messages": [final_message],  # Return as list of dict messages
        "tool_execution_history": [
            execution_record.model_dump()
        ],  # New execution to add
    }


async def research_reflection_node(
    state: PPTOverallState, config: RunnableConfig
) -> PPTOverallState:  # Return overall state with reflection results
    """
    Research reflection node that analyzes current search results and determines if more searching is needed.
    This is a transparent node that adds reflection results to state for routing decisions.
    """
    configurable = PPTConfiguration.from_runnable_config(config)
    logger.info("Starting research reflection...")

    # Check consecutive search count limit before executing reflection
    tool_execution_history = state.get("tool_execution_history", [])
    consecutive_count = get_consecutive_search_count(tool_execution_history)

    if consecutive_count >= configurable.max_consecutive_search_count:
        logger.info(
            f"Search count limit reached ({consecutive_count}/{configurable.max_consecutive_search_count}), "
            "skipping reflection and proceeding to coordinator"
        )
        return {
            "reflection_should_continue": False,
            "reflection_suggested_queries": [],
            "reflection_rationale": f"Search limit reached ({consecutive_count}/{configurable.max_consecutive_search_count} searches)",
        }

    # Extract user request from messages
    user_request = get_user_request(state["messages"])
    web_results_summary = "\n\n---\n\n".join(state.get("web_results_summary", []))

    try:
        # Initialize LLM client
        llm_client = AsyncArkLLMClient(
            api_key=os.getenv("ARK_API_KEY"),
            base_url=os.getenv("ARK_BASE_URL"),
            model_id=configurable.flash_model,
        )

        # Prepare prompt
        prompt = research_reflection_prompt.format(
            user_request=user_request,
            web_results_summary=web_results_summary,
            current_date=get_current_date(),
        )

        # Get reflection response from LLM
        prompt_messages = [{"role": "user", "content": prompt}]
        reflection_response = await llm_client.ainvoke(prompt_messages)

        logger.info(f"Reflection response received: {reflection_response[:200]}...")

        # Parse the JSON response
        reflection_data = parse_json_response(reflection_response)

        if reflection_data:
            should_continue = reflection_data.get("should_continue", False)
            sug_queries = reflection_data.get("sug_queries", [])
            rationale = reflection_data.get("rationale", "No rationale provided")

            logger.info(f"Reflection decision: should_continue={should_continue}")
            logger.info(f"Suggested queries: {sug_queries}")

            # Store reflection results in overall state for routing
            return {
                "reflection_should_continue": should_continue,
                "reflection_suggested_queries": sug_queries,
                "reflection_rationale": rationale,
            }
        else:
            # Failed to parse response - proceed to coordinator
            logger.warning(
                "Failed to parse reflection response, proceeding to coordinator"
            )
            return {
                "reflection_should_continue": False,
                "reflection_suggested_queries": [],
                "reflection_rationale": "Failed to parse reflection response",
            }

    except Exception as e:
        logger.error(f"Error in research reflection node: {str(e)}")
        # On error, proceed to coordinator
        return {
            "reflection_should_continue": False,
            "reflection_suggested_queries": [],
            "reflection_rationale": f"Reflection error: {str(e)}",
        }


def route_after_reflection(state: PPTOverallState) -> Send:
    """
    Route after reflection based on the reflection results.
    Directly sends to appropriate node with proper input state.
    """
    should_continue = state.get("reflection_should_continue", False)
    suggested_queries = state.get("reflection_suggested_queries", [])

    if should_continue and suggested_queries:
        logger.info(
            f"Reflection suggests continuing search with {len(suggested_queries)} queries"
        )

        # Create WebSearchInputState for the new search
        search_state = WebSearchInputState(
            queries=suggested_queries,
            tool_call_id=f"reflect_{abs(hash(str(suggested_queries))) % 100000}",  # Fake tool_call_id for reflection-triggered searches
            tool_execution_history=state.get("tool_execution_history", []),
        )
        return Send("web_search_node", search_state)
    else:
        logger.info("Reflection suggests proceeding to coordinator")
        return "coordinator_node"


# Create the PPT Agent Graph
def create_ppt_graph():
    """Create PPT graph with very specific input states, simplified output states"""
    from langgraph.graph import StateGraph, START, END

    # Create state graph with plain dict messages
    builder = StateGraph(PPTOverallState, config_schema=PPTConfiguration)

    # Add all nodes
    builder.add_node("coordinator_node", coordinator_node)
    builder.add_node("web_search_node", web_search_node)
    builder.add_node(
        "research_reflection_node", research_reflection_node
    )  # Add new node
    builder.add_node("theme_and_pages_ack_node", theme_and_pages_ack_node)
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
        [
            "web_search_node",
            # Remove "research_reflection_node" from coordinator routes since it only comes from web_search
            "theme_and_pages_ack_node",
            "gen_outline_node",
            "gen_html_node",
            "modify_html_node",
            "finalize_response_node",
        ],
    )

    # Web search goes to research reflection instead of coordinator
    builder.add_edge("web_search_node", "research_reflection_node")

    # Research reflection uses route_after_reflection to directly send with proper input states
    builder.add_conditional_edges(
        "research_reflection_node",
        route_after_reflection,
        ["web_search_node", "coordinator_node"],
    )

    # Other tool nodes return to coordinator for next step decision
    builder.add_edge("theme_and_pages_ack_node", "coordinator_node")
    builder.add_edge("gen_outline_node", "coordinator_node")
    builder.add_edge("gen_html_node", "coordinator_node")
    builder.add_edge("modify_html_node", "coordinator_node")

    # Only finalize connects to end
    builder.add_edge("finalize_response_node", END)

    return builder.compile(name="ppt-optimized-state-graph")


# Remove the route_from_research_reflection function since we're using route_after_reflection now


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

    # Handle web search
    if next_tool == "web_search":
        # Use tool_args queries (reflection routing is handled separately)
        queries = tool_args.get("queries", [])
        tool_execution_history = state.get("tool_execution_history", [])
        specific_state = WebSearchInputState(
            queries=queries,
            tool_call_id=tool_call_id,
            tool_execution_history=tool_execution_history,
        )
        return Send("web_search_node", specific_state)

    # Remove the research_reflection routing since it only comes from web_search now

    elif next_tool == "ppt_theme_and_pages_ack":
        # Theme and pages acknowledgment node only needs theme and total_pages from tool_args
        theme = tool_args.get("theme", "professional")
        total_pages = tool_args.get("total_pages", 10)
        specific_state = ThemeAndPagesAckInputState(
            theme=theme,
            total_pages=total_pages,
            tool_call_id=tool_call_id,
        )
        return Send("theme_and_pages_ack_node", specific_state)

    elif next_tool == "generate_ppt_outline":
        # Outline node needs user request, ref documents, and total pages
        user_request = tool_args.get("user_request", get_user_request(messages))
        ref_documents = "\n\n---\n\n".join(state.get("web_results_summary", []))
        total_pages = tool_args.get("total_pages", state.get("total_pages", 10))
        specific_state = OutlineInputState(
            user_request=user_request,
            ref_documents=ref_documents,
            total_pages=total_pages,
            tool_call_id=tool_call_id,
        )
        return Send("gen_outline_node", specific_state)

    elif next_tool == "generate_ppt_html":
        # HTML node needs user request, outline, and theme
        user_request = tool_args.get("user_request", get_user_request(messages))
        ppt_outline = state.get("ppt_outline", {})
        theme = tool_args.get("theme", state.get("theme", "professional"))

        specific_state = HtmlInputState(
            user_request=user_request,
            ppt_outline=ppt_outline,
            theme=theme,
            tool_call_id=tool_call_id,
        )
        return Send("gen_html_node", specific_state)

    elif next_tool == "modify_ppt_html":
        # Modify node needs page number and modification suggestions
        page_number = tool_args.get("page_number", 1)
        modification_suggestions = tool_args.get("modification_suggestions", "")

        specific_state = ModifyInputState(
            page_number=page_number,
            modification_suggestions=modification_suggestions,
            tool_call_id=tool_call_id,
        )
        return Send("modify_html_node", specific_state)

    else:
        # Default to finalize
        specific_state = FinalizeInputState(messages=messages)
        return Send("finalize_response_node", specific_state)


# Export the compiled graph
ppt_graph = create_ppt_graph()
