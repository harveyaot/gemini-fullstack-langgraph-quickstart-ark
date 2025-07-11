import os
import asyncio
import json
import logging
import re
import requests
from typing import Dict, Any, List

from langchain_core.runnables import RunnableConfig

from ppt_agent.state import (
    PPTOverallState,
    ExecutionRecord,
)
from ppt_agent.configuration import PPTConfiguration
from ppt_agent.prompts import (
    get_current_date,
    get_detailed_outline_prompt,
    get_style_layout_prompt,
    get_template_prompt,
    get_html_code_prompt,
)
from ppt_agent.utils import get_user_request, parse_json_response

# Import shared components from agent module
from agent.ark_client import AsyncArkLLMClient

# Use new web search client for image search
from agent.web_search_client import CustomWebSearchClient, ImageSearchRequest

# Initialize logger
logger = logging.getLogger(__name__)


async def search_brief_outline_images(
    brief_outline: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Utility function to search for images based on brief outline's picture_advise using the new web_search_client interface.
    Args:
        brief_outline: Brief outline containing slides with picture_advise
    Returns:
        List of processed image references
    """
    try:
        logger.info(
            "[BriefImageSearch] Starting image search based on brief outline (new interface)"
        )
        # Extract picture_advise from all slides
        queries = []
        slides = brief_outline.get("slides", [])
        for slide in slides:
            picture_advise = slide.get("picture_advise", [])[
                :3
            ]  # Max 3 queries per slide
            queries.extend(picture_advise)
        # Remove duplicates while preserving order
        unique_queries = list(dict.fromkeys(queries))
        if not unique_queries:
            logger.info("[BriefImageSearch] No picture queries found in brief outline")
            return []
        logger.info(
            f"[BriefImageSearch] Found {len(unique_queries)} unique image queries: {unique_queries}"
        )
        # Use the new CustomWebSearchClient
        base_url = os.getenv("WEB_SEARCH_BASE_URL", "http://localhost:8080")
        client = CustomWebSearchClient(base_url)
        # Split queries into groups of 10
        query_groups = [
            unique_queries[i : i + 10] for i in range(0, len(unique_queries), 10)
        ]

        async def search_group(group):
            req = ImageSearchRequest(queries=group, count=3)
            try:
                response = await client.image_search(req)
                group_image_reference = []
                if response and response.data and response.data.items:
                    for query, images in response.data.items.items():
                        for img in images:
                            processed_img = {
                                "query": query,
                                "desc": img.desc,
                                "features": img.features,
                                "format": img.format,
                                "width": img.width,
                                "height": img.height,
                                "image_urls": img.image_urls,
                                "source_webpage": img.source_webpage,
                            }
                            group_image_reference.append(processed_img)
                return group_image_reference
            except Exception as e:
                logger.error(
                    f"[BriefImageSearch] Error in image search group: {str(e)}"
                )
                return []

        # Run all groups in parallel
        all_results = await asyncio.gather(
            *(search_group(group) for group in query_groups)
        )
        # Flatten the results
        brief_image_reference = [img for group in all_results for img in group]
        logger.info(
            f"[BriefImageSearch] Processed {len(brief_image_reference)} image references (new interface)"
        )
        return brief_image_reference
    except Exception as e:
        logger.error(
            f"[BriefImageSearch] Error in image search utility (new interface): {str(e)}"
        )
        return []


async def image_search_node(
    state: PPTOverallState, config: RunnableConfig
) -> PPTOverallState:
    """
    Standalone image search node that searches for images based on brief outline.
    Sits between gen_outline_node and gen_detailed_outline_node.
    """
    configurable = PPTConfiguration.from_runnable_config(config)
    logger.info("Starting image search based on brief outline...")

    try:
        # Extract brief outline from state
        brief_outline = state.get("ppt_outline", {})
        tool_execution_history = state.get("tool_execution_history", [])

        # Use the existing utility function to search for images
        logger.info(
            "Searching for images based on brief outline picture recommendations..."
        )
        images_gathered = await search_brief_outline_images(brief_outline)

        logger.info(f"Image search completed with {len(images_gathered)} images found")

        # Record execution
        execution_record = ExecutionRecord(
            tool_name="image_search",
            status="succeeded",
            result={
                "images_found": len(images_gathered),
                "queries_processed": len(brief_outline.get("slides", [])),
            },
        )

        # Return updated state with image data
        return {
            "images_gathered": images_gathered,
            "tool_execution_history": [execution_record.model_dump()],
        }

    except Exception as e:
        logger.error(f"Error in image search node: {str(e)}")
        import traceback

        logger.error(f"Full traceback: {traceback.format_exc()}")

        # Record failed execution
        execution_record = ExecutionRecord(
            tool_name="image_search",
            status="failed",
            result={"error": str(e)},
        )

        return {
            "images_gathered": [],
            "tool_execution_history": [execution_record.model_dump()],
        }


async def gen_detailed_outline_node(
    state: PPTOverallState, config: RunnableConfig
) -> PPTOverallState:
    """Generate detailed PPT outline based on brief outline and research materials - Internal step in outline workflow"""
    configurable = PPTConfiguration.from_runnable_config(config)
    logger.info("Generating detailed PPT outline...")

    try:
        # Initialize LLM client
        llm_client = AsyncArkLLMClient(
            api_key=os.getenv("ARK_API_KEY"),
            base_url=os.getenv("ARK_BASE_URL"),
            model_id=configurable.flash_model,
        )

        # Extract data from overall state (passed from image_search_node)
        user_request = get_user_request(state.get("messages", []))
        brief_outline = state.get("brief_outline", {})
        ref_documents = "\n\n---\n\n".join(state.get("web_results_summary", []))

        # Get image references from state (already searched by image_search_node)
        images_gathered = state.get("images_gathered", [])
        logger.info(
            f"Using {len(images_gathered)} images from state (searched by image_search_node)"
        )

        # Combine text and image references
        if images_gathered:
            image_reference_text = "可参考的图片资料如下：\n" + str(images_gathered)
            reference_material = f"{ref_documents}\n\n{image_reference_text}"
        else:
            reference_material = ref_documents

        # Prepare prompt
        prompt = await get_detailed_outline_prompt(
            user_request=user_request,
            brief_outline=brief_outline,
            reference_material=reference_material,
        )

        logger.info(f"Calling LLM with prompt length: {len(prompt)} characters")

        # Get detailed outline from LLM
        prompt_messages = [{"role": "user", "content": prompt}]
        outline_response = await llm_client.ainvoke(prompt_messages)

        logger.info(
            f"LLM response received, length: {len(outline_response)} characters"
        )

        # Parse the response into structured format
        raw_detailed_outline = outline_response
        detailed_outline = _parse_detailed_outline_from_content(outline_response)
        # Validate that each slide has a slide_number, add if missing
        for idx, slide in enumerate(detailed_outline):
            if "slide_number" not in slide:
                slide["slide_number"] = idx + 1

        logger.info("Successfully generated detailed outline")

        # Return updated state with detailed outline
        execution_record = ExecutionRecord(
            tool_name="gen_detailed_outline",
            status="succeeded",
            result={
                "detailed_outline_generated": True,
                "slides_count": len(detailed_outline) if detailed_outline else 0,
                "image_references_used": len(images_gathered),
            },
        )
        return {
            "detailed_outline": detailed_outline,
            "raw_detailed_outline": raw_detailed_outline,
            "tool_execution_history": [execution_record.model_dump()],
        }

    except Exception as e:
        logger.error(f"Error generating detailed outline: {str(e)}")
        import traceback

        logger.error(f"Full traceback: {traceback.format_exc()}")

        execution_record = ExecutionRecord(
            tool_name="gen_detailed_outline",
            status="failed",
            result={
                "detailed_outline_generated": False,
                "error": str(e),
            },
        )
        return {
            "tool_execution_history": [execution_record.model_dump()],
        }


async def gen_style_layout_node(
    state: PPTOverallState, config: RunnableConfig
) -> PPTOverallState:
    """Generate style and layout guidelines - Entry point for HTML generation workflow"""
    configurable = PPTConfiguration.from_runnable_config(config)
    logger.info("Generating style and layout guidelines...")

    try:
        # Initialize LLM client
        llm_client = AsyncArkLLMClient(
            api_key=os.getenv("ARK_API_KEY"),
            base_url=os.getenv("ARK_BASE_URL"),
            model_id=configurable.flash_model,
        )

        # Extract data from overall state (entry point receives HtmlInputState data)
        user_request = get_user_request(state.get("messages", []))
        detailed_outline = state.get(
            "detailed_outline", []
        )  # Use detailed outline if available

        # Convert detailed outline list to string for the prompt
        ppt_outline_str = ""
        if detailed_outline:
            try:
                ppt_outline_str = json.dumps(
                    detailed_outline, ensure_ascii=False, indent=2
                )
            except Exception as e:
                logger.warning(f"Failed to convert detailed_outline to JSON: {str(e)}")
                ppt_outline_str = str(detailed_outline)

        # Prepare prompt
        prompt = await get_style_layout_prompt(
            user_request=user_request,
            ppt_outline=ppt_outline_str,
        )

        logger.info(f"Calling LLM with prompt length: {len(prompt)} characters")

        # Get style layout from LLM
        prompt_messages = [{"role": "user", "content": prompt}]
        style_response = await llm_client.ainvoke(prompt_messages)

        logger.info(f"LLM response received, length: {len(style_response)} characters")

        # Parse and clean the response
        style_layout = _parse_style_layout_from_content(style_response)

        logger.info("Successfully generated style layout")

        # Return updated state with style layout
        execution_record = ExecutionRecord(
            tool_name="gen_style_layout",
            status="succeeded",
            result={
                "style_layout_generated": True,
                "style_length": len(style_layout) if style_layout else 0,
            },
        )
        return {
            "style_layout": style_layout,
            "tool_execution_history": [execution_record.model_dump()],
        }

    except Exception as e:
        logger.error(f"Error generating style layout: {str(e)}")

        execution_record = ExecutionRecord(
            tool_name="gen_style_layout",
            status="failed",
            result={
                "style_layout_generated": False,
                "error": str(e),
            },
        )
        return {
            "tool_execution_history": [execution_record.model_dump()],
        }


async def gen_template_node(
    state: PPTOverallState, config: RunnableConfig
) -> PPTOverallState:
    """Generate template pages - Internal step in HTML generation workflow"""
    configurable = PPTConfiguration.from_runnable_config(config)
    logger.info("Generating template pages...")

    try:
        # Initialize LLM client
        llm_client = AsyncArkLLMClient(
            api_key=os.getenv("ARK_API_KEY"),
            base_url=os.getenv("ARK_BASE_URL"),
            model_id=configurable.flash_model,
        )

        # Extract data from overall state (passed from gen_style_layout_node)
        style_layout = state.get("style_layout", "")

        # Prepare prompt
        prompt = await get_template_prompt(style_layout=style_layout)

        logger.info(f"Calling LLM with prompt length: {len(prompt)} characters")

        # Get template pages from LLM
        prompt_messages = [{"role": "user", "content": prompt}]
        template_response = await llm_client.ainvoke(prompt_messages)

        logger.info(
            f"LLM response received, length: {len(template_response)} characters"
        )

        # Parse the response into structured format
        template_pages = _parse_template_pages_from_content(template_response)

        logger.info("Successfully generated template pages")

        # Return updated state with template pages
        execution_record = ExecutionRecord(
            tool_name="gen_template",
            status="succeeded",
            result={
                "template_pages_generated": True,
                "templates_count": len(template_pages) if template_pages else 0,
            },
        )
        return {
            "template_pages": template_pages,
            "tool_execution_history": [execution_record.model_dump()],
        }

    except Exception as e:
        logger.error(f"Error generating template pages: {str(e)}")

        execution_record = ExecutionRecord(
            tool_name="gen_template",
            status="failed",
            result={
                "template_pages_generated": False,
                "error": str(e),
            },
        )
        return {
            "tool_execution_history": [execution_record.model_dump()],
        }


async def gen_html_code_node(
    state: PPTOverallState, config: RunnableConfig
) -> PPTOverallState:
    """Generate HTML code for slides - Final step in HTML generation workflow, returns to coordinator"""
    configurable = PPTConfiguration.from_runnable_config(config)
    logger.info("Generating HTML code for slides...")

    # Extract tool_call_id from state (saved in gen_style_layout_node)
    tool_call_id = state.get("gen_html_tool_call_id")

    try:
        # Initialize LLM client
        llm_client = AsyncArkLLMClient(
            api_key=os.getenv("ARK_API_KEY"),
            base_url=os.getenv("ARK_BASE_URL"),
            model_id=configurable.flash_model,
        )

        # Extract data from overall state (accumulated through the workflow)
        style_layout = state.get("style_layout", "")
        template_pages = state.get("template_pages", {})
        ppt_outline = state.get("detailed_outline", [])

        # Generate HTML for all slides in parallel
        async def generate_slide_html(slide_number: int) -> str:
            """Generate HTML for a single slide"""
            prompt = await get_html_code_prompt(
                slide_number=slide_number,
                ppt_input=style_layout,  # Use style_layout like in deprecated_graph.py
                template_html=template_pages.get("content", ""),
            )

            logger.info(f"Generating HTML for slide {slide_number}")

            # Get HTML code from LLM
            prompt_messages = [{"role": "user", "content": prompt}]
            html_response = await llm_client.ainvoke(prompt_messages)

            # Parse and clean the HTML
            slide_html = _parse_html_code_from_content(html_response)
            return slide_html

        # Execute all slide generations in parallel
        slide_tasks = [
            generate_slide_html(i)
            for i, _ in enumerate(
                ppt_outline, 1
            )  # Using _ since slide_content is not used
        ]
        all_slides_html = await asyncio.gather(*slide_tasks)

        logger.info(f"Successfully generated HTML for {len(all_slides_html)} slides")

        # Create tool message for coordinator (without heavy HTML content)
        html_result = {
            "message": "Successfully generated PPT HTML",
            "data": {
                "total_slides": len(all_slides_html),
                "theme": state.get("theme", "professional"),
                "remote_address": "http://example.com/generated-ppt.html",  # Placeholder
            },
        }

        tool_message = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": "GenPptHtml",
            "content": json.dumps(html_result),
        }

        # Return updated state with tool message and slides saved separately
        execution_record = ExecutionRecord(
            tool_name="gen_html_code",
            status="succeeded",
            result={
                "html_generated": True,
                "slides_count": len(all_slides_html),
            },
        )
        return {
            "messages": [tool_message],
            "next_tool": "coordinator",
            "tool_args": None,
            "all_slides_html": all_slides_html,
            "tool_execution_history": [execution_record.model_dump()],
        }

    except Exception as e:
        logger.error(f"Error generating HTML code: {str(e)}")

        error_message = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": "GenPptHtml",
            "content": f"生成HTML代码时出错：{str(e)}",
        }

        execution_record = ExecutionRecord(
            tool_name="gen_html_code",
            status="failed",
            result={
                "html_generated": False,
                "error": str(e),
            },
        )
        return {
            "messages": [error_message],
            "next_tool": "coordinator",
            "tool_args": None,
            "tool_execution_history": [execution_record.model_dump()],
        }


# Helper functions


def _parse_detailed_outline_from_content(content: str) -> list:
    """Parse detailed outline from LLM response content"""
    slides = []
    lines = content.split("\n")
    current_slide = None
    for line in lines:
        line = line.strip()

        # 匹配 # Slide X: 标题 格式
        if line.startswith("# Slide ") and ":" in line:
            if current_slide:
                slides.append(current_slide)

            # 提取页码和标题
            parts = line.split(":", 1)
            slide_part = parts[0].replace("# Slide ", "")
            title = parts[1].strip() if len(parts) > 1 else f"幻灯片 {slide_part}"

            try:
                slide_number = int(slide_part)
            except:
                slide_number = len(slides) + 1

            current_slide = {
                "slide_number": slide_number,
                "title": title,
                "content": "",
                "layout_type": "content_list",
                "images": [],
                "tables": [],
                "visual_focus": "",
                "logic": "",
            }
        elif current_slide:
            current_slide["content"] += line + "\n"
    # 添加最后一个slide
    if current_slide:
        slides.append(current_slide)
    return slides


def _parse_template_pages_from_content(content: str) -> dict:
    """Parse template pages from LLM response content. Accepts both English and Chinese section names."""
    template_pages = {}
    # Accept both English and Chinese keys for compatibility
    section_map = {
        "cover": ["cover", "封面页完整HTML代码"],
        "content": ["content", "内容页完整HTML代码"],
        "ending": ["ending", "结束页完整HTML代码"],
    }
    for template_type, keys in section_map.items():
        found = False
        for key in keys:
            pattern = rf"<!--\s*{key}\s*-->.*?<!-- start -->(.*?)<!-- end -->"
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if match:
                template_pages[template_type] = match.group(1).strip()
                found = True
                break
        if not found:
            template_pages[template_type] = (
                f"<!-- {template_type} template not found -->"
            )
    return template_pages


def _parse_style_layout_from_content(content: str) -> str:
    """Parse and clean style layout from LLM response content"""
    # Remove any code block markers if present
    content = re.sub(r"```[a-zA-Z]*\n?", "", content)
    content = re.sub(r"\n```", "", content)

    # Remove any HTML comment markers
    content = re.sub(r"<!--.*?-->", "", content, flags=re.DOTALL)

    return content.strip()


def _parse_html_code_from_content(content: str) -> str:
    """Parse and extract HTML code from LLM response content"""
    return _extract_html_from_content(content)


def _extract_html_from_content(content: str) -> str:
    """Extract HTML code from LLM response content"""
    # Method 1: Try to extract between <!-- start --> and <!-- end --> markers
    start_end_match = re.search(r"<!-- start -->(.*?)<!-- end -->", content, re.DOTALL)
    if start_end_match:
        return start_end_match.group(1).strip()

    # Method 2: Try to extract ```html code blocks
    html_code_match = re.search(r"```html\s*\n(.*?)\n```", content, re.DOTALL)
    if html_code_match:
        return html_code_match.group(1).strip()

    # Method 3: Try to find DOCTYPE declaration
    doctype_match = re.search(
        r"(<!DOCTYPE html.*?</html>)", content, re.DOTALL | re.IGNORECASE
    )
    if doctype_match:
        return doctype_match.group(1).strip()

    # Method 4: Try to find <html> tag
    html_tag_match = re.search(r"(<html.*?</html>)", content, re.DOTALL | re.IGNORECASE)
    if html_tag_match:
        return html_tag_match.group(1).strip()

    logger.warning("Could not extract HTML from content")
    return content.strip()  # Return original content as fallback
