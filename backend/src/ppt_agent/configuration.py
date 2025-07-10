import os
from pydantic import BaseModel, Field
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig


class PPTConfiguration(BaseModel):
    """The configuration for the PPT agent."""

    main_model: str = Field(
        default="ep-20250611103625-7trbw",
        metadata={"description": "The main language model to use for PPT generation."},
    )

    flash_model: str = Field(
        default="ep-20250619204324-ml2lb",
        metadata={
            "description": "The language model to use for PPT outline generation."
        },
    )

    outline_model: str = Field(
        default="ep-20250619204324-ml2lb",
        metadata={
            "description": "The language model to use for PPT outline generation."
        },
    )

    search_model: str = Field(
        default="ep-20250619204324-ml2lb",
        metadata={
            "description": "The language model to use for web search processing."
        },
    )

    default_theme: str = Field(
        default="professional",
        metadata={"description": "Default theme for PPT generation."},
    )

    default_total_pages: int = Field(
        default=10,
        metadata={"description": "Default number of pages for PPT generation."},
    )

    max_consecutive_search_count: int = Field(
        default=2,
        metadata={
            "description": "Maximum number of consecutive web search calls before blocking further searches."
        },
    )

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "PPTConfiguration":
        """Create a PPTConfiguration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )

        # Get raw values from environment or config
        raw_values: dict[str, Any] = {
            name: os.environ.get(name.upper(), configurable.get(name))
            for name in cls.model_fields.keys()
        }

        # Filter out None values
        values = {k: v for k, v in raw_values.items() if v is not None}

        return cls(**values)
