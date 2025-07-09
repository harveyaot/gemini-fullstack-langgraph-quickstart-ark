import asyncio
import logging
from typing import List, Dict, Any
from langchain_core.messages import HumanMessage

from ppt_agent.graph import ppt_graph
from ppt_agent.state import PPTOverallState
from ppt_agent.configuration import PPTConfiguration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def run_ppt_agent(user_message: str, config: Dict[str, Any] = None) -> str:
    """
    Run the PPT agent with a user message and return the response.
    
    Args:
        user_message: The user's PPT request
        config: Optional configuration overrides
        
    Returns:
        The agent's response as a string
    """
    logger.info(f"Starting PPT agent with request: {user_message[:100]}...")
    
    # Prepare initial state
    initial_state: PPTOverallState = {
        "messages": [HumanMessage(content=user_message)],
        "user_request": "",
        "needs_web_search": False,
        "search_query": [],
        "web_research_result": [],
        "sources_gathered": [],
        "ppt_outline": None,
        "ppt_html_result": None,
        "modification_result": None,
        "theme": "professional",
        "total_pages": 10,
        "page_to_modify": None,
        "modification_suggestions": None,
        "research_loop_count": 0,
    }
    
    # Prepare configuration
    runnable_config = {"configurable": config} if config else None
    
    try:
        # Run the graph
        result = await ppt_graph.ainvoke(initial_state, config=runnable_config)
        
        # Extract the final response
        if result.get("messages"):
            final_message = result["messages"][-1]
            return final_message.content
        else:
            return "PPT generation completed, but no response was generated."
            
    except Exception as e:
        logger.error(f"Error running PPT agent: {str(e)}")
        return f"Error: {str(e)}"


async def main():
    """Example usage of the PPT agent"""
    
    # Example 1: Simple PPT request without web search
    logger.info("Example 1: Simple PPT request")
    response1 = await run_ppt_agent(
        "请帮我制作一个关于Python编程基础的PPT，包括变量、函数、类等内容"
    )
    print("Response 1:")
    print(response1)
    print("\n" + "="*50 + "\n")
    
    # Example 2: PPT request that should trigger web search
    logger.info("Example 2: PPT request with web search")
    response2 = await run_ppt_agent(
        "请制作一个关于2024年最新AI技术发展趋势的PPT，包括最新的技术突破和市场应用"
    )
    print("Response 2:")
    print(response2)
    print("\n" + "="*50 + "\n")
    
    # Example 3: PPT request with custom configuration
    logger.info("Example 3: PPT request with custom configuration")
    custom_config = {
        "default_theme": "modern",
        "default_total_pages": 15,
        "main_model": "ep-20250611103625-7trbw"
    }
    response3 = await run_ppt_agent(
        "创建一个公司季度业绩汇报PPT",
        config=custom_config
    )
    print("Response 3:")
    print(response3)


def run_interactive_mode():
    """Run the PPT agent in interactive mode"""
    print("PPT Agent (LangGraph) - Interactive Mode")
    print("Type 'quit' to exit")
    print("=" * 50)
    
    while True:
        try:
            user_input = input("\n请输入您的PPT需求: ").strip()
            
            if user_input.lower() in ['quit', 'exit', '退出']:
                print("再见！")
                break
                
            if not user_input:
                continue
                
            print("\n正在处理您的请求...")
            response = asyncio.run(run_ppt_agent(user_input))
            print(f"\n回复:\n{response}")
            
        except KeyboardInterrupt:
            print("\n\n再见！")
            break
        except Exception as e:
            print(f"\n错误: {str(e)}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        # Run in interactive mode
        run_interactive_mode()
    else:
        # Run examples
        asyncio.run(main()) 