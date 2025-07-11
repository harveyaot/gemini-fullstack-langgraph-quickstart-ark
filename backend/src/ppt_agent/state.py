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
    current_tool_call: Optional[Any]
    # Shared data across nodes
    ppt_outline: Optional[
        Dict[str, Any]
    ]  # Generated by outline node, used by HTML node
    sources_gathered: Annotated[
        List[Dict[str, Any]], operator.add
    ]  # Search results from web searches
    images_gathered: Annotated[
        List[Dict[str, Any]], operator.add
    ]  # Search results from image searches
    # Tool execution tracking
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


# Entry Point Input States - Only for coordinator-visible nodes
class CoordinatorInputState(TypedDict):
    """Input state for coordinator node - analyzes full conversation"""

    messages: List[Dict[str, Any]]  # Plain dict messages


class WebSearchInputState(TypedDict):
    """Input state for web search node - only needs search queries"""

    queries: List[str]  # From PPTLLMTools.WebSearch.queries
    tool_call_id: Optional[str]  # Tool call ID for response
    tool_execution_history: Annotated[
        List[Dict[str, Any]], operator.add
    ]  # Tool execution history for consecutive check


class OutlineInputState(TypedDict):
    """Input state for outline generation group - entry point for outline workflow"""

    user_request: str  # From PPTLLMTools.GenPptOutline.user_request
    ref_documents: str  # Extracted from conversation tool messages
    total_pages: int  # From PPTLLMTools.GenPptOutline.total_pages
    tool_call_id: Optional[str]  # Tool call ID for response


class ThemeAndPagesAckInputState(TypedDict):
    """Input state for theme and pages acknowledgment node"""

    theme: str  # Suggested theme from coordinator's tool args
    total_pages: int  # Suggested total pages from coordinator's tool args
    tool_call_id: Optional[str]  # Tool call ID for response


class ModifyInputState(TypedDict):
    """Input state for HTML modification node"""

    page_number: int  # From PPTLLMTools.ModifyPptHtml.page_number
    modification_suggestions: (
        str  # From PPTLLMTools.ModifyPptHtml.modification_suggestions
    )
    tool_call_id: Optional[str]  # Tool call ID for response


class FinalizeInputState(TypedDict):
    """Input state for finalize response node - needs full conversation"""

    messages: List[Dict[str, Any]]  # Plain dict messages


class HtmlInputState(TypedDict):
    """Input state for HTML generation group - entry point for the HTML generation workflow"""

    user_request: str  # From PPTLLMTools.GenPptHtml.user_request
    ppt_outline: Dict[str, Any]  # From outline generation (brief)
    detailed_outline: List[Dict[str, Any]]  # From detailed outline generation
    theme: str  # From PPTLLMTools.GenPptHtml.theme
    tool_call_id: Optional[str]  # Tool call ID for response
