import os
import aiohttp
import logging
from typing import List, Dict
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class WebSearchItem(BaseModel):
    url: str
    title: str
    content: str


class WebSearchData(BaseModel):
    items: List[WebSearchItem]
    count: int


class WebSearchResponse(BaseModel):
    code: int
    message: str
    data: WebSearchData

    def get_stats(self) -> str:
        """Get formatted statistics about the search results"""
        total_results = len(self.data.items)
        if total_results == 0:
            return "found 0 results"

        # Calculate average content length
        total_length = sum(len(item.content) for item in self.data.items)
        avg_length = total_length / total_results if total_results > 0 else 0

        return f"found {total_results} results (avg content length: {avg_length:.0f} chars)"

    def get_top_urls(self, limit: int = 3) -> str:
        """Get formatted string of top URLs"""
        urls = [item.url for item in self.data.items[:limit]]
        return (
            "First {} result URLs:{}".format(
                limit, "".join(f"\n- {url}" for url in urls)
            )
            if urls
            else ""
        )


class ImageItem(BaseModel):
    desc: str
    features: str
    format: str
    height: int
    width: int
    image_urls: str
    source_webpage: str


class ImageSearchData(BaseModel):
    items: Dict[str, List[ImageItem]]


class ImageSearchResponse(BaseModel):
    code: int
    message: str
    data: ImageSearchData

    def get_stats(self) -> str:
        """Get formatted statistics about the search results"""
        total_images = sum(len(images) for images in self.data.items.values())
        return f"found {total_images} images across {len(self.data.items)} queries"


class WebSearchRequest(BaseModel):
    """Base model for web search with all validation logic"""

    queries: List[str] = Field(
        ...,
        description="List of search terms to find web pages for. Each query should be specific and focused. Examples: ['人工智能 2025', '机器学习 应用案例', '深度学习技术 进展', 'AI 医疗领域 应用']",
        min_items=1,
        max_items=10,
    )


class ImageSearchRequest(BaseModel):
    """Base model for image search with all validation logic"""

    queries: List[str] = Field(
        ...,
        description="List of search terms to find images for. Each query should be descriptive and specific. Examples: ['现代办公大楼', '玻璃幕墙设计', '绿色建筑外观', '智能建筑系统']",
        min_items=1,
        max_items=10,
    )
    background: str = Field(
        default="",
        description="Optional background context to provide additional search context. Helps refine search results. Examples: '用于商业演示', '建筑设计项目', '可持续发展报告'",
    )
    count: int = Field(
        default=3,
        description="Number of images to return per query. For a presentation, 4-6 images is recommended. For a mood board, 8-10 images might be better.",
        ge=1,
        le=5,
    )


class CustomWebSearchClient:
    """Custom web search client for replacing Google Search API"""

    def __init__(self, base_url: str):
        self.base_url = base_url
        self.headers = {"Content-Type": "application/json"}

    async def web_search(self, request: WebSearchRequest) -> WebSearchResponse:
        """
        Perform web search using the provided queries.

        Args:
            request: WebSearchRequest containing queries

        Returns:
            WebSearchResponse with search results
        """
        url = f"{self.base_url}/web"
        payload = request.model_dump()
        logger.debug(f"Making web search request to {url} with payload: {payload}")

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, json=payload, headers=self.headers
            ) as response:
                response_status = response.status
                logger.debug(f"Web search API response status: {response_status}")

                response_data = await response.json()
                logger.debug(f"Web search API response data: {response_data}")

                try:
                    return WebSearchResponse(**response_data)
                except Exception as e:
                    logger.error(
                        f"Failed to parse web search response: {response_data}"
                    )
                    logger.error(f"Validation error: {str(e)}")
                    return WebSearchResponse(
                        code=response_status,
                        message=f"Failed to parse response: {str(e)}",
                        data=WebSearchData(items=[], count=0),
                    )

    async def image_search(self, request: ImageSearchRequest) -> ImageSearchResponse:
        """
        Perform image search using the provided queries.

        Args:
            request: ImageSearchRequest containing queries, background, and count

        Returns:
            ImageSearchResponse with search results
        """
        url = f"{self.base_url}/image"
        payload = request.model_dump()
        logger.debug(f"Making image search request to {url} with payload: {payload}")

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, json=payload, headers=self.headers
            ) as response:
                response_status = response.status
                logger.debug(f"Image search API response status: {response_status}")

                response_data = await response.json()
                logger.debug(f"Image search API response data: {response_data}")

                try:
                    return ImageSearchResponse(**response_data)
                except Exception as e:
                    logger.error(
                        f"Failed to parse image search response: {response_data}"
                    )
                    logger.error(f"Validation error: {str(e)}")
                    return ImageSearchResponse(
                        code=response_status,
                        message=f"Failed to parse response: {str(e)}",
                        data=ImageSearchData(items={}),
                    )
