from __future__ import annotations

from typing_extensions import (
    TypedDict,
    Annotated,
)  # Use TypedDict from typing_extensions for Python < 3.12
from typing import Optional, Dict, Any, List
from pydantic import BaseModel

import operator


class ExecutionRecord(BaseModel):
    """Model for tool execution records"""

    tool_name: str
    status: str  # "succeeded" or "failed"
    result: Dict[str, Any]


class PPTOverallState(TypedDict):
    """Main state for PPT agent workflow - used for overall state management and all outputs"""

    messages: Annotated[
        List[Dict[str, Any]], operator.add
    ]  # Plain dict messages with custom reducer
    # Routing and coordination fields
    next_tool: Optional[str]
    tool_args: Optional[Dict[str, Any]]

    sources_gathered: Annotated[
        List[Dict[str, Any]], operator.add
    ]  # Search results from web searches
    images_gathered: Annotated[
        List[Dict[str, Any]], operator.add
    ]  # Search results from image searches
    # Tool execution tracking
    web_queries: Annotated[List[str], operator.add]  # Web search queries
    image_queries: Annotated[List[str], operator.add]  # Image search queries
    tool_execution_history: Annotated[
        List[Dict[str, Any]], operator.add
    ]  # Track each node execution and results
    web_results_summary: Annotated[
        List[str], operator.add
    ]  # Summary of web search results
    total_pages: Optional[int]  # Total pages of the PPT
    theme: Optional[str]  # PPT theme
    scenario: Optional[str]  # Sentiment of the PPT

    # New node outputs
    brief_outline: Optional[Dict[str, Any]]  # Generated by brief outline node
    brief_image_reference: Optional[
        List[Dict[str, Any]]
    ]  # Image search results based on brief outline
    raw_detailed_outline: Optional[str]  # Raw detailed outline from LLM
    detailed_outline: Optional[
        List[Dict[str, Any]]
    ]  # Generated by detailed outline node
    style_layout: Optional[str]  # Generated by style layout node
    template_pages: Optional[Dict[str, str]]  # Generated by template node
    all_slides_html: Optional[List[str]]  # Generated HTML for all slides
    gen_html_tool_call_id: Optional[str]  # Tool call ID for HTML generation workflow

    # Session management
    thread_id: Optional[str]  # Thread ID for organizing resources

    # Research reflection fields (temporary, for routing decisions)
    reflection_should_continue: Optional[bool]
    reflection_suggested_queries: Optional[List[str]]
    reflection_rationale: Optional[str]
