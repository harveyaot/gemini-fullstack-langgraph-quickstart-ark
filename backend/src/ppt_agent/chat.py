import logging
from typing import Dict, Any, List
from langchain_core.messages import HumanMessage, AIMessage

from ppt_agent.state import PPTOverallState
from ppt_agent.configuration import PPTConfiguration
from ppt_agent.graph import create_ppt_graph

logger = logging.getLogger(__name__)


async def process_message(message: str, current_state: PPTOverallState = None, config: PPTConfiguration = None) -> PPTOverallState:
    """
    Process a new message in the conversation
    
    Args:
        message: The user's message
        current_state: Optional current state from previous conversation
        config: PPTConfiguration instance
        
    Returns:
        Updated PPTOverallState after processing
    """
    logger.info("Processing new message...")
    
    if config is None:
        config = PPTConfiguration()  # Use defaults
    
    graph = create_ppt_graph()
    
    # Initialize or update state
    if current_state is None:
        # First message - create fresh state
        new_state = {
            "messages": [HumanMessage(content=message)],
            "research_round_count": 0,
            "max_research_rounds": config.max_research_rounds,
        }
    else:
        # Continuing conversation - append to existing state
        # But reset research round count for new user query
        new_state = {
            "messages": current_state["messages"] + [HumanMessage(content=message)],
            "research_round_count": 0,  # Reset for new query
            "max_research_rounds": config.max_research_rounds,
        }
    
    # Process through graph
    try:
        final_state = await graph.ainvoke(
            new_state,
            {"configurable": config}
        )
        return final_state
        
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        error_state = new_state.copy()
        error_state["messages"].append(
            AIMessage(content=f"处理消息时出错：{str(e)}")
        )
        return error_state 