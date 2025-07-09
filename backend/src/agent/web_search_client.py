import os
import aiohttp
import logging
from typing import List
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
        return "First {} result URLs:{}".format(
            limit, 
            "".join(f"\n- {url}" for url in urls)
        ) if urls else ""


class WebSearchRequest(BaseModel):
    """Base model for web search with all validation logic"""
    queries: List[str] = Field(
        ...,
        description="List of search terms to find web pages for. Each query should be specific and focused. Examples: ['人工智能 2025', '机器学习 应用案例', '深度学习技术 进展', 'AI 医疗领域 应用']",
        min_items=1,
        max_items=10
    )


class CustomWebSearchClient:
    """Custom web search client for replacing Google Search API"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.headers = {
            "Content-Type": "application/json"
        }
    
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
            async with session.post(url, json=payload, headers=self.headers) as response:
                response_status = response.status
                logger.debug(f"Web search API response status: {response_status}")
                
                response_data = await response.json()
                logger.debug(f"Web search API response data: {response_data}")
                
                try:
                    return WebSearchResponse(**response_data)
                except Exception as e:
                    logger.error(f"Failed to parse web search response: {response_data}")
                    logger.error(f"Validation error: {str(e)}")
                    return WebSearchResponse(
                        code=response_status,
                        message=f"Failed to parse response: {str(e)}",
                        data=WebSearchData(items=[], count=0)
                    ) 