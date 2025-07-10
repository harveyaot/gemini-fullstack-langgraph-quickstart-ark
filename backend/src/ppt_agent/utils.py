from typing import Any, Dict, List, Optional
import json
import logging
import re

logger = logging.getLogger(__name__)


def get_user_request(messages: List[Dict[str, Any]]) -> str:
    """
    Get the user request from the messages.

    Args:
        messages: List of message dictionaries with 'role' and 'content' keys

    Returns:
        Combined user request string
    """
    if len(messages) == 1:
        return messages[-1].get("content", "")
    else:
        # Combine messages to understand the full context
        request = ""
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            if role == "user":
                request += f"User: {content}\n"
            elif role == "assistant":
                request += f"Assistant: {content}\n"
        return request


def extract_ref_documents(
    messages: List[Dict[str, Any]], top_n_items=15, top_n_chars=1000
) -> str:
    """
    Extract reference materials from tool messages in the conversation history.

    Args:
        messages: List of message dictionaries with 'role', 'content', and optional 'name' keys
        top_n_items: Maximum number of items to include
        top_n_chars: Maximum characters per item

    Returns:
        Formatted string containing reference materials
    """
    reference_parts = []

    # Process searched documents from tool messages
    searched_docs = []
    conversation_history = []

    for message in messages:
        # Only process tool messages for searched documents
        if message.get("role") == "tool":
            try:
                # Parse the content as JSON
                content = message.get("content", "")
                if not content:
                    continue

                tool_data = json.loads(content)

                # Handle web search results
                if isinstance(tool_data, dict) and "data" in tool_data:
                    web_data = tool_data["data"]
                    if "sources" in web_data and isinstance(web_data["sources"], list):
                        # Process up to top_n_items results
                        for item in web_data["sources"][:top_n_items]:
                            if isinstance(item, dict):
                                title = item.get("title", "")
                                content = item.get("content", "")
                                url = item.get("value", "")  # URL is in 'value' field

                                if title and content:
                                    # Format and append the searched document
                                    doc_reference = f"标题: {title}\n内容: {content[:top_n_chars]}...\n链接: {url}\n"
                                    searched_docs.append(doc_reference)
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parsing error in tool message: {str(e)}")
                continue
            except Exception as e:
                logger.warning(f"Error processing tool message: {str(e)}")
                continue
        elif message.get("role") in ["user", "assistant"]:
            # Extract user and assistant conversation content
            content = message.get("content", "")
            if content:
                role = message.get("role", "")
                conv_reference = f"{role.capitalize()}: {content[:top_n_chars]}...\n"
                conversation_history.append(conv_reference)

    # Combine references with headers
    if searched_docs:
        reference_parts.append("### 搜索文档")
        reference_parts.extend(searched_docs)

    if conversation_history:
        reference_parts.append("### 对话历史")
        reference_parts.extend(conversation_history)

    logger.info(f"Reference parts: {len(reference_parts)}")
    return "\n".join(reference_parts)


def parse_json_response(
    content: str, fallback_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Parse JSON response content with fallback handling

    Args:
        content: JSON string content (may contain extra text)
        fallback_data: Fallback data if parsing fails

    Returns:
        Parsed data or fallback
    """
    if not content or not isinstance(content, str):
        logger.warning(f"Invalid content for JSON parsing: {type(content)}")
        return fallback_data or {}

    # Try direct parsing first
    try:
        return json.loads(content.strip())
    except json.JSONDecodeError:
        pass

    # Try to find JSON within ```json blocks
    import re

    json_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
    if json_block_match:
        try:
            return json.loads(json_block_match.group(1))
        except json.JSONDecodeError:
            pass

    # Find JSON object by looking for { and } braces
    start_idx = content.find("{")
    if start_idx != -1:
        brace_count = 0
        end_idx = start_idx

        for i in range(start_idx, len(content)):
            if content[i] == "{":
                brace_count += 1
            elif content[i] == "}":
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i + 1
                    break

        if brace_count == 0:  # Found complete JSON object
            json_str = content[start_idx:end_idx]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass

    # Log the parsing failure
    logger.error(f"Failed to parse JSON response: {content[:200]}...")

    if fallback_data:
        return fallback_data

    # Return a basic structure as last resort
    return {
        "ppt_title": "Generated PPT",
        "slides": [
            {
                "title": "Introduction",
                "content": "PPT content could not be parsed from LLM response",
                "current_slide_page_counts": 1,
                "picture_advise": [],
            }
        ],
    }


def get_consecutive_search_count(tool_execution_history: List[Dict[str, Any]]) -> int:
    """
    Count consecutive web search calls from the end of execution history.

    Args:
        tool_execution_history: List of tool execution records

    Returns:
        Number of consecutive web search calls from the end
    """
    if not tool_execution_history:
        return 0

    # Count consecutive web search calls from the end (most recent)
    consecutive_count = 0
    for record in reversed(tool_execution_history):
        tool_name = record.get("tool_name", "")
        if tool_name == "web_search":
            consecutive_count += 1
        else:
            # Break on first non-web-search tool
            break

    return consecutive_count
