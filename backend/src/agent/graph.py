import os
import asyncio

from agent.tools_and_schemas import SearchQueryList, Reflection
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langgraph.types import Send
from langgraph.graph import StateGraph
from langgraph.graph import START, END
from langchain_core.runnables import RunnableConfig

from agent.state import (
    OverallState,
    QueryGenerationState,
    ReflectionState,
    WebSearchState,
)
from agent.configuration import Configuration
from agent.prompts import (
    get_current_date,
    query_writer_instructions,
    web_searcher_instructions,
    reflection_instructions,
    answer_instructions,
)
from agent.ark_client import AsyncArkLLMClient, ArkMessage
from agent.web_search_client import CustomWebSearchClient, WebSearchRequest
from agent.utils import (
    get_research_topic,
)

load_dotenv()

if os.getenv("ARK_API_KEY") is None:
    raise ValueError("ARK_API_KEY is not set")

if os.getenv("ARK_BASE_URL") is None:
    raise ValueError("ARK_BASE_URL is not set")

if os.getenv("WEB_SEARCH_BASE_URL") is None:
    raise ValueError("WEB_SEARCH_BASE_URL is not set")

# Initialize custom web search client
web_search_client = CustomWebSearchClient(base_url=os.getenv("WEB_SEARCH_BASE_URL"))


# Nodes
def generate_query(state: OverallState, config: RunnableConfig) -> QueryGenerationState:
    """LangGraph node that generates search queries based on the User's question.

    Uses Ark LLM to create an optimized search queries for web research based on
    the User's question.

    Args:
        state: Current graph state containing the User's question
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_query key containing the generated queries
    """
    configurable = Configuration.from_runnable_config(config)

    # check for custom initial search query count
    if state.get("initial_search_query_count") is None:
        state["initial_search_query_count"] = configurable.number_of_initial_queries

    # init Ark LLM
    llm = AsyncArkLLMClient(
        api_key=os.getenv("ARK_API_KEY"),
        base_url=os.getenv("ARK_BASE_URL"),
        model_id=configurable.flash_model,
    )
    structured_llm = llm.with_structured_output(SearchQueryList)

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = query_writer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        number_queries=state["initial_search_query_count"],
    )
    # Generate the search queries
    result = structured_llm.invoke(formatted_prompt, temperature=1.0)
    return {"search_query": result.query}


def continue_to_web_research(state: QueryGenerationState):
    """LangGraph node that sends the search queries to the web research node.

    This is used to spawn n number of web research nodes, one for each search query.
    """
    return [
        Send("web_research", {"search_query": search_query, "id": int(idx)})
        for idx, search_query in enumerate(state["search_query"])
    ]


def web_research(state: WebSearchState, config: RunnableConfig) -> OverallState:
    """LangGraph node that performs web research using the custom web search API.

    Executes a web search using the custom web search API and then uses Ark LLM to synthesize the results.

    Args:
        state: Current graph state containing the search query and research loop count
        config: Configuration for the runnable, including search API settings

    Returns:
        Dictionary with state update, including sources_gathered, research_loop_count, and web_research_results
    """
    # Configure
    configurable = Configuration.from_runnable_config(config)

    # Create search request
    search_request = WebSearchRequest(queries=[state["search_query"]])

    # Perform web search
    async def do_search():
        return await web_search_client.web_search(search_request)

    # Run the async search
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        search_response = loop.run_until_complete(do_search())
    finally:
        loop.close()

    # Format search results for LLM
    search_results_text = ""
    sources_gathered = []

    for idx, item in enumerate(search_response.data.items):
        short_url = f"[{idx+1}]"
        search_results_text += (
            f"\n\n{short_url} {item.title}\nURL: {item.url}\nContent: {item.content}\n"
        )

        # Extract domain name for label (similar to original implementation)
        try:
            from urllib.parse import urlparse

            domain = urlparse(item.url).netloc
            # Remove www. prefix and split by . to get main domain name
            label = (
                domain.replace("www.", "").split(".")[0] if domain else item.title[:20]
            )
        except:
            # Fallback to first few words of title
            label = (
                " ".join(item.title.split()[:3]) if item.title else f"source_{idx+1}"
            )

        sources_gathered.append(
            {
                "short_url": short_url,
                "value": item.url,
                "title": item.title,
                "content": (
                    item.content[:1000] + "..."
                    if len(item.content) > 1000
                    else item.content
                ),
                "label": label,  # This is what the frontend needs for "Related to:"
            }
        )

    # Use Ark LLM to synthesize the search results
    formatted_prompt = (
        web_searcher_instructions.format(
            current_date=get_current_date(),
            research_topic=state["search_query"],
        )
        + f"\n\nSearch Results:{search_results_text}"
    )

    llm = AsyncArkLLMClient(
        api_key=os.getenv("ARK_API_KEY"),
        base_url=os.getenv("ARK_BASE_URL"),
        model_id=configurable.flash_model,
    )

    response_text = llm.invoke(formatted_prompt, temperature=0)

    # Add citation markers to the response
    for i, source in enumerate(sources_gathered):
        citation_marker = f"[{i+1}]"
        response_text = response_text.replace(
            citation_marker, f"[{source['title']}]({source['value']})"
        )

    return {
        "sources_gathered": sources_gathered,
        "search_query": [state["search_query"]],
        "web_research_result": [response_text],
    }


def reflection(state: OverallState, config: RunnableConfig) -> ReflectionState:
    """LangGraph node that identifies knowledge gaps and generates potential follow-up queries.

    Analyzes the current summary to identify areas for further research and generates
    potential follow-up queries. Uses structured output to extract
    the follow-up query in JSON format.

    Args:
        state: Current graph state containing the running summary and research topic
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_query key containing the generated follow-up query
    """
    configurable = Configuration.from_runnable_config(config)
    # Increment the research loop count and get the reasoning model
    state["research_loop_count"] = state.get("research_loop_count", 0) + 1
    reasoning_model = state.get("reasoning_model", configurable.reflection_model)

    # Fallback to default if old Gemini model names are used
    if reasoning_model and "gemini" in reasoning_model.lower():
        reasoning_model = configurable.reflection_model

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = reflection_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n\n---\n\n".join(state["web_research_result"]),
    )
    # init Ark LLM
    llm = AsyncArkLLMClient(
        api_key=os.getenv("ARK_API_KEY"),
        base_url=os.getenv("ARK_BASE_URL"),
        model_id=reasoning_model,
    )
    result = llm.with_structured_output(Reflection).invoke(
        formatted_prompt, temperature=1.0
    )

    return {
        "is_sufficient": result.is_sufficient,
        "knowledge_gap": result.knowledge_gap,
        "follow_up_queries": result.follow_up_queries,
        "research_loop_count": state["research_loop_count"],
        "number_of_ran_queries": len(state["search_query"]),
    }


def evaluate_research(
    state: ReflectionState,
    config: RunnableConfig,
) -> OverallState:
    """LangGraph routing function that determines the next step in the research flow.

    Controls the research loop by deciding whether to continue gathering information
    or to finalize the summary based on the configured maximum number of research loops.

    Args:
        state: Current graph state containing the research loop count
        config: Configuration for the runnable, including max_research_loops setting

    Returns:
        String literal indicating the next node to visit ("web_research" or "finalize_summary")
    """
    configurable = Configuration.from_runnable_config(config)
    max_research_loops = (
        state.get("max_research_loops")
        if state.get("max_research_loops") is not None
        else configurable.max_research_loops
    )
    if state["is_sufficient"] or state["research_loop_count"] >= max_research_loops:
        return "finalize_answer"
    else:
        return [
            Send(
                "web_research",
                {
                    "search_query": follow_up_query,
                    "id": state["number_of_ran_queries"] + int(idx),
                },
            )
            for idx, follow_up_query in enumerate(state["follow_up_queries"])
        ]


def finalize_answer(state: OverallState, config: RunnableConfig):
    """LangGraph node that finalizes the research summary.

    Prepares the final output by deduplicating and formatting sources, then
    combining them with the running summary to create a well-structured
    research report with proper citations.

    Args:
        state: Current graph state containing the running summary and sources gathered

    Returns:
        Dictionary with state update, including running_summary key containing the formatted final summary with sources
    """
    configurable = Configuration.from_runnable_config(config)
    reasoning_model = state.get("reasoning_model") or configurable.answer_model

    # Fallback to default if old Gemini model names are used
    if reasoning_model and "gemini" in reasoning_model.lower():
        reasoning_model = configurable.answer_model

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = answer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n---\n\n".join(state["web_research_result"]),
    )

    # init Ark LLM
    llm = AsyncArkLLMClient(
        api_key=os.getenv("ARK_API_KEY"),
        base_url=os.getenv("ARK_BASE_URL"),
        model_id=reasoning_model,
    )
    response_content = llm.invoke(formatted_prompt, temperature=0)

    # Replace the short urls with the original urls and add all used urls to the sources_gathered
    unique_sources = []
    for source in state["sources_gathered"]:
        if source["short_url"] in response_content:
            response_content = response_content.replace(
                source["short_url"], source["value"]
            )
            unique_sources.append(source)

    return {
        "messages": [AIMessage(content=response_content)],
        "sources_gathered": unique_sources,
    }


# Create our Agent Graph
builder = StateGraph(OverallState, config_schema=Configuration)

# Define the nodes we will cycle between
builder.add_node("generate_query", generate_query)
builder.add_node("web_research", web_research)
builder.add_node("reflection", reflection)
builder.add_node("finalize_answer", finalize_answer)

# Set the entrypoint as `generate_query`
# This means that this node is the first one called
builder.add_edge(START, "generate_query")
# Add conditional edge to continue with search queries in a parallel branch
builder.add_conditional_edges(
    "generate_query", continue_to_web_research, ["web_research"]
)
# Reflect on the web research
builder.add_edge("web_research", "reflection")
# Evaluate the research
builder.add_conditional_edges(
    "reflection", evaluate_research, ["web_research", "finalize_answer"]
)
# Finalize the answer
builder.add_edge("finalize_answer", END)

graph = builder.compile(name="pro-search-agent")
