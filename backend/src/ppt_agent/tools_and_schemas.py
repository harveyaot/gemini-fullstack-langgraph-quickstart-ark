from typing import List, Dict, Any, Optional, Type, ClassVar
from pydantic import BaseModel, Field


class ArkSchemaTool:
    """Interface for tools that can be converted to Ark schema format"""

    @classmethod
    def to_ark_schema(cls) -> Dict[str, Any]:
        """Convert tool to Ark schema format"""
        schema = cls.model_json_schema()
        return {
            "type": "function",
            "function": {
                "name": cls.tool_name,
                "description": cls.__doc__,
                "parameters": {
                    "type": "object",
                    "properties": schema.get("properties", {}),
                    "required": schema.get("required", []),
                },
            },
        }


# Base Schema Models
class SearchQueryList(BaseModel):
    """Schema for search query generation"""

    query: List[str] = Field(
        description="A list of search queries to be used for web research."
    )
    rationale: str = Field(
        description="A brief explanation of why these queries are relevant to the research topic."
    )


class PPTOutline(BaseModel):
    """Schema for PPT outline structure"""

    ppt_title: str = Field(description="The main title of the PPT")
    slides: List[Dict[str, Any]] = Field(
        description="List of slides with their content structure"
    )


# Web Search Models
class WebSearchRequest(BaseModel):
    """Base model for web search with all validation logic"""

    queries: List[str] = Field(
        ...,
        description="List of search terms to find web pages for. Each query should be specific and focused. Examples: ['人工智能 2025', '机器学习 应用案例', '深度学习技术 进展', 'AI 医疗领域 应用']",
        min_items=1,
        max_items=10,
    )


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


# PPT Tool Models
class PPTLLMTools:
    """Collection of PPT-specific tool wrappers for LLM function calling"""

    class WebSearch(WebSearchRequest, ArkSchemaTool):
        """Search the web for information using multiple queries to gather relevant materials for PPT creation. Use this tool when you need current information, statistics, case studies, or recent developments related to the PPT topic."""

        tool_name: ClassVar[str] = "web_search"
        pass

    class ThemeAndPagesAck(BaseModel, ArkSchemaTool):
        """
        Suggest a recommended theme and total page count for the PPT before generating the outline.
        This tool analyzes the user's request and web search results to recommend a suitable number of pages (default/recommendation: 10) and a single theme.
        The user should confirm or adjust these suggestions before proceeding to outline or HTML generation.

        Fields:
        - total_pages (int): Recommended total number of PPT pages/slides, based on user needs and research (default: 10).
        - theme (str): Suggested theme for the PPT. Only one theme is suggested from the following common options:
            # Common PPT themes (for reference, only one is suggested at a time):
            # 1. Professional
            # 2. Modern
            # 3. Creative
            # 4. Minimal
            # 5. Classic
            # 6. Elegant
            # 7. Corporate
            # 8. Educational
            # 9. Technology
            # 10. Artistic

        Returns:
        - theme (str): The theme confirmed by the user for the PPT.
        - total_pages (int): The total number of pages confirmed by the user for the PPT.
        """

        tool_name: ClassVar[str] = "ppt_theme_and_pages_ack"
        theme: str = Field(
            ...,
            description="The theme for the PPT. Available options: 'professional' (business-focused, clean design), 'modern' (contemporary styling with bold colors), 'creative' (artistic and dynamic layouts), 'minimal' (clean and simple design). Default is 'professional'.",
        )
        total_pages: int = Field(
            default=10,
            description="Recommended total number of PPT pages/slides, based on user needs and research (default: 10).",
        )

    class GenPptOutline(BaseModel, ArkSchemaTool):
        """Generate a structured PPT outline based on conversation history and research materials. This tool analyzes user requirements and available research data to create a comprehensive PPT structure with slides, content descriptions, and page counts. Automatically syncs with conversation context."""

        tool_name: ClassVar[str] = "generate_ppt_outline"
        user_request: str = Field(
            ...,
            description="The user's detailed request for the PPT creation, including topic, purpose, target audience, and any specific requirements.",
        )
        total_pages: int = Field(
            ...,
            description="The total number of pages/slides to generate for the PPT outline. Should be a positive integer greater than 0.",
        )

    class GenPptHtml(BaseModel, ArkSchemaTool):
        """Generate HTML code for PPT prototype based on confirmed outline. Creates responsive layout with dynamic charts and multimedia placeholders. This tool produces a complete, interactive PPT that can be viewed in browsers and adapted for different devices. Note: This tool returns the number of generated PPT pages and the remote address of the PPT HTML code, not the source code itself."""

        tool_name: ClassVar[str] = "generate_ppt_html"
        user_request: str = Field(
            ...,
            description="The user's original request for the PPT, used to ensure the generated HTML aligns with the intended purpose and content.",
        )
        theme: str = Field(
            default="professional",
            description="Visual theme for the PPT presentation. Available options: 'professional' (business-focused, clean design), 'modern' (contemporary styling with bold colors), 'creative' (artistic and dynamic layouts), 'minimal' (clean and simple design). Default is 'professional'.",
        )

    class ModifyPptHtml(BaseModel, ArkSchemaTool):
        """Modify specific page/slide in existing PPT HTML based on user feedback. This tool is used for iterative improvements and refinements to specific slides. Only use when user explicitly requests modification of a particular page with specific changes. Automatically syncs with conversation context for current HTML state."""

        tool_name: ClassVar[str] = "modify_ppt_html"
        page_number: int = Field(
            ...,
            description="The specific page/slide number to modify (starting from 1). For example: 3 for the third slide, 1 for the title slide. Must be within the range of existing slides.",
        )
        modification_suggestions: str = Field(
            ...,
            description="Detailed description of the specific modifications to make to the page. Be specific and actionable. Examples: '替换当前的柱状图为折线图显示趋势', '更新标题为最新的项目名称', '添加三个要点的项目符号列表', '调整配色方案为蓝色主题', '增加公司logo到右上角'.",
        )

    @staticmethod
    def get_all_tools() -> List[Type[BaseModel]]:
        """Get all available PPT LLM tools"""
        return [
            PPTLLMTools.WebSearch,
            PPTLLMTools.ThemeAndPagesAck,
            PPTLLMTools.GenPptOutline,
            PPTLLMTools.GenPptHtml,
            PPTLLMTools.ModifyPptHtml,
        ]

    @staticmethod
    def get_tool_schemas() -> List[Dict[str, Any]]:
        """Get all tool schemas in Ark format for LLM integration"""
        return [tool_cls.to_ark_schema() for tool_cls in PPTLLMTools.get_all_tools()]


# Legacy models for backward compatibility
class PPTGenerationRequest(BaseModel):
    """Schema for PPT generation request (legacy)"""

    user_request: str = Field(description="The user's request for PPT creation")
    theme: str = Field(default="professional", description="Theme for the PPT")
    total_pages: int = Field(
        default=10, description="Total number of pages for the PPT"
    )


class PPTModificationRequest(BaseModel):
    """Schema for PPT modification request (legacy)"""

    page_number: int = Field(description="The page number to modify")
    modification_suggestions: str = Field(
        description="Suggestions for modifying the specified page"
    )
