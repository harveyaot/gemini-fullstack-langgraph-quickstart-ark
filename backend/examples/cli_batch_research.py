"""
Batch Research CLI Script for LangGraph Research Agent

This script processes multiple research queries from a TSV file in parallel using the LangGraph research agent.
It now follows the same configuration patterns as the PPT agent with proper thread ID management and 
LangGraph configuration handling.

Key Features:
- Thread ID and configuration handled through LangGraph's configurable system
- Support for different models for different tasks (reasoning, flash, query generation, etc.)
- Proper grap configuration (no longer hardcoded in state)
- Parallel processing with controlled concurrency
- Comprehensive error handling and reporting

Usage Examples:

# Basic usage with default settings
python examples/cli_batch_research.py queries.tsv

# Advanced usage with custom configuration
python examples/cli_batch_research.py queries.tsv \
    --query-column "research_question" \
    --output "my_results.tsv" \
    --max-concurrent 8 \
    --initial-queries 5 \
    --max-loops 3 \
    --reasoning-model "ep-20250611103625-7trbw" \
    --flash-model "ep-20250619204324-ml2lb"

TSV File Format:
The input TSV file should have a header row with at least one column containing queries.
Example:
    query
    What is the impact of AI on healthcare?
    How does quantum computing affect cryptography?
    What are the latest renewable energy technologies?
"""

import argparse
import asyncio
import csv
import uuid
from pathlib import Path
import time
from typing import List, Dict, Any
import dotenv

dotenv.load_dotenv()

from ppt_agent.graph import ppt_graph


async def run_single_query(
    query: str, thread_id: str, config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Run a single research query with the given thread ID.

    Args:
        query: The research question to process
        thread_id: Unique thread identifier for this query (included in state like PPT agent)
        config: Configuration dictionary containing model settings and research parameters

    Returns:
        Dictionary containing query results, timing info, and success/error status

    Note:
        Now follows LangGraph configuration patterns - thread_id and all configuration
        parameters are passed through the configurable system, not hardcoded in state.
    """
    state = {
        "messages": [
            {"role": "user", "content": query},
        ],
        "thread_id": thread_id,
        # Note: thread_id is handled through configuration, not state for research agent
        # Don't set initial_search_query_count and max_research_loops in state
        # These should be handled through configuration
    }

    # Use LangGraph configuration pattern for all settings
    run_config = {
        "configurable": {
            "max_research_loops": config.get("max_loops", 1),
        }
    }

    try:
        start_time = time.time()
        result = await ppt_graph.ainvoke(state, config=run_config)
        end_time = time.time()

        # Extract the final answer
        messages = result.get("messages", [])
        answer = messages[-1].content if messages else "No answer generated"
        sources = result.get("sources_gathered", [])

        return {
            "thread_id": thread_id,
            "query": query,
            "answer": answer,
            "sources_count": len(sources),
            "sources": sources,
            "duration": end_time - start_time,
            "success": True,
            "error": None,
        }
    except Exception as e:
        return {
            "thread_id": thread_id,
            "query": query,
            "answer": None,
            "sources_count": 0,
            "sources": [],
            "duration": 0,
            "success": False,
            "error": str(e),
        }


async def run_batch_queries(
    queries: List[str], max_concurrent: int, config: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Run queries in batches with controlled concurrency."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def run_with_semaphore(query: str) -> Dict[str, Any]:
        async with semaphore:
            thread_id = str(uuid.uuid4())
            return await run_single_query(query, thread_id, config)

    print(
        f"Starting batch processing of {len(queries)} queries with max {max_concurrent} concurrent runs..."
    )
    start_time = time.time()

    tasks = [run_with_semaphore(query) for query in queries]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    end_time = time.time()
    print(f"Batch processing completed in {end_time - start_time:.2f} seconds")

    # Handle any exceptions in results
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            processed_results.append(
                {
                    "thread_id": f"error_{i}",
                    "query": queries[i] if i < len(queries) else "Unknown",
                    "answer": None,
                    "sources_count": 0,
                    "sources": [],
                    "duration": 0,
                    "success": False,
                    "error": str(result),
                }
            )
        else:
            processed_results.append(result)

    return processed_results


def read_tsv_queries(file_path: str, query_column: str = "query") -> List[str]:
    """Read queries from a TSV file."""
    queries = []

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            reader = csv.DictReader(file, delimiter="\t")

            if query_column not in reader.fieldnames:
                available_columns = ", ".join(reader.fieldnames)
                raise ValueError(
                    f"Column '{query_column}' not found. Available columns: {available_columns}"
                )

            for row in reader:
                query = row[query_column].strip()
                if query:  # Skip empty queries
                    queries.append(query)

        print(f"Successfully loaded {len(queries)} queries from {file_path}")
        return queries

    except FileNotFoundError:
        raise FileNotFoundError(f"TSV file not found: {file_path}")
    except Exception as e:
        raise Exception(f"Error reading TSV file: {e}")


def save_results(results: List[Dict[str, Any]], output_file: str):
    """Save results to a TSV file."""
    if not results:
        print("No results to save")
        return

    with open(output_file, "w", encoding="utf-8", newline="") as file:
        fieldnames = [
            "thread_id",
            "query",
            "answer",
            "sources_count",
            "duration",
            "success",
            "error",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter="\t")

        writer.writeheader()
        for result in results:
            # Create a copy without the full sources for the main output
            row = {k: v for k, v in result.items() if k != "sources"}
            writer.writerow(row)

    print(f"Results saved to {output_file}")


async def main():
    """Main function to run the batch research agent."""
    parser = argparse.ArgumentParser(
        description="Run the LangGraph research agent in batch mode"
    )
    parser.add_argument("tsv_file", help="Path to TSV file containing queries")
    parser.add_argument(
        "--query-column",
        default="query",
        help="Name of the column containing queries (default: 'query')",
    )
    parser.add_argument(
        "--output",
        default="batch_results.tsv",
        help="Output file for results (default: 'batch_results.tsv')",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=5,
        help="Maximum number of concurrent queries (default: 5)",
    )
    parser.add_argument(
        "--initial-queries",
        type=int,
        default=3,
        help="Number of initial search queries per research",
    )
    parser.add_argument(
        "--max-loops",
        type=int,
        default=1,
        help="Maximum number of research loops per query (default: 1)",
    )

    args = parser.parse_args()

    # Validate TSV file exists
    if not Path(args.tsv_file).exists():
        print(f"Error: TSV file '{args.tsv_file}' not found")
        return

    try:
        # Read queries from TSV file
        queries = read_tsv_queries(args.tsv_file, args.query_column)

        if not queries:
            print("No queries found in the TSV file")
            return

        # Prepare configuration
        config = {
            "max_loops": args.max_loops,
        }

        # Run batch processing
        results = await run_batch_queries(queries, args.max_concurrent, config)

        # Print summary
        successful = sum(1 for r in results if r["success"])
        failed = len(results) - successful
        avg_duration = sum(r["duration"] for r in results if r["success"]) / max(
            successful, 1
        )

        print(f"\n=== Batch Processing Summary ===")
        print(f"Total queries: {len(queries)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Average duration per query: {avg_duration:.2f} seconds")

        # Save results
        save_results(results, args.output)

        # Print first few results as examples
        print(f"\n=== Sample Results ===")
        for i, result in enumerate(results[:3]):
            print(f"\nQuery {i+1}: {result['query'][:100]}...")
            if result["success"]:
                print(f"Answer: {result['answer'][:200]}...")
                print(f"Sources: {result['sources_count']}")
                print(f"Duration: {result['duration']:.2f}s")
            else:
                print(f"Error: {result['error']}")

        if failed > 0:
            print(f"\n=== Failed Queries ===")
            for result in results:
                if not result["success"]:
                    print(f"Query: {result['query'][:100]}...")
                    print(f"Error: {result['error']}")
                    print()

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
