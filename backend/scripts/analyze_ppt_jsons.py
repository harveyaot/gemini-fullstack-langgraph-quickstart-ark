#!/usr/bin/env python3
"""
Analysis script for PPT JSON files.
Analyzes image URL usage from image_search_results in detailed_outline.
"""

import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple, Set
import argparse


def extract_urls_from_text(text: str) -> Set[str]:
    """Extract URLs from text content."""
    # Pattern to match URLs (http/https)
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+[^\s<>"{}|\\^`\[\].,;!?]'
    urls = set(re.findall(url_pattern, text, re.IGNORECASE))
    return urls


def extract_image_urls_from_html(html: str) -> Set[str]:
    """Extract image URLs from HTML content."""
    urls = set()

    # Extract URLs from src attributes
    src_pattern = r'src=["\']([^"\']+)["\']'
    src_urls = re.findall(src_pattern, html, re.IGNORECASE)
    urls.update(src_urls)

    # Extract URLs from style attributes (background-image)
    style_pattern = r'background-image:\s*url\(["\']?([^"\'()]+)["\']?\)'
    style_urls = re.findall(style_pattern, html, re.IGNORECASE)
    urls.update(style_urls)

    # Extract general URLs from HTML content
    general_urls = extract_urls_from_text(html)
    urls.update(general_urls)

    # Filter to keep only image-like URLs
    image_urls = set()
    for url in urls:
        if any(
            domain in url.lower()
            for domain in ["byteimg.com", "imgur.com", "flickr.com", "images.", "img."]
        ) or any(
            ext in url.lower()
            for ext in [".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg"]
        ):
            image_urls.add(url)

    return image_urls


def analyze_single_json(filepath: str) -> Dict:
    """Analyze a single JSON file and return statistics."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

    stats = {
        "file": os.path.basename(filepath),
        "brief_outline_picture_advise_count": 0,
        "brief_outline_queries": [],
        "image_search_results_count": 0,
        "image_search_unique_queries": set(),
        "image_search_urls": set(),
        "detailed_outline_urls": set(),
        "detailed_outline_image_urls": set(),
        "all_slides_html_image_urls": set(),
        "coverage_percentage": 0.0,
        "image_url_usage_count": 0,
        "image_url_usage_percentage": 0.0,
        "final_html_usage_count": 0,
        "final_html_usage_percentage": 0.0,
        "funnel_search_to_outline": 0.0,
        "funnel_outline_to_html": 0.0,
        "funnel_search_to_html": 0.0,
    }

    # 1. Analyze brief_outline picture_advise (keep for coverage analysis)
    if "brief_outline" in data and "slides" in data["brief_outline"]:
        for slide in data["brief_outline"]["slides"]:
            if "picture_advise" in slide and slide["picture_advise"]:
                stats["brief_outline_picture_advise_count"] += len(
                    slide["picture_advise"]
                )
                stats["brief_outline_queries"].extend(slide["picture_advise"])

    # 2. Analyze image_search_results and extract image URLs
    if "image_search_results" in data:
        stats["image_search_results_count"] = len(data["image_search_results"])
        for result in data["image_search_results"]:
            if "query" in result:
                stats["image_search_unique_queries"].add(result["query"])
            if "image_urls" in result and result["image_urls"]:
                stats["image_search_urls"].add(result["image_urls"])

    # 3. Calculate coverage (how many picture_advise queries are covered by image_search_results)
    if stats["brief_outline_queries"]:
        covered_queries = 0
        image_queries_set = stats["image_search_unique_queries"]

        for advised_query in stats["brief_outline_queries"]:
            # Check if any image search query is similar or contains the advised query
            for image_query in image_queries_set:
                # Simple similarity check - if advised query keywords are in image query
                if any(
                    word.lower() in image_query.lower()
                    for word in advised_query.split()
                    if len(word) > 2
                ):
                    covered_queries += 1
                    break

        stats["coverage_percentage"] = (
            covered_queries / len(stats["brief_outline_queries"])
        ) * 100

    # 4. Analyze detailed_outline for URLs and specifically image URLs
    if "detailed_outline" in data:
        for slide in data["detailed_outline"]:
            if "content" in slide:
                urls = extract_urls_from_text(slide["content"])
                stats["detailed_outline_urls"].update(urls)
                # Filter for image URLs (those that might be from image_search_results)
                for url in urls:
                    if any(
                        domain in url.lower()
                        for domain in ["byteimg.com", "imgur.com", "flickr.com"]
                    ) or any(
                        ext in url.lower()
                        for ext in [".jpg", ".jpeg", ".png", ".gif", ".webp"]
                    ):
                        stats["detailed_outline_image_urls"].add(url)

            # Also check in images array if it exists
            if "images" in slide:
                for image in slide["images"]:
                    if isinstance(image, dict) and "image_urls" in image:
                        stats["detailed_outline_urls"].add(image["image_urls"])
                        stats["detailed_outline_image_urls"].add(image["image_urls"])
                    elif isinstance(image, str):
                        if image.startswith("http"):
                            stats["detailed_outline_urls"].add(image)
                            stats["detailed_outline_image_urls"].add(image)

    # 5. Analyze all_slides_html for image URLs
    if "all_slides_html" in data and data["all_slides_html"]:
        for html_slide in data["all_slides_html"]:
            if html_slide:
                image_urls = extract_image_urls_from_html(html_slide)
                stats["all_slides_html_image_urls"].update(image_urls)

    # 6. Calculate image URL usage and funnel analysis
    if stats["image_search_urls"]:
        # Usage in detailed_outline
        used_in_outline = stats["image_search_urls"].intersection(
            stats["detailed_outline_image_urls"]
        )
        stats["image_url_usage_count"] = len(used_in_outline)
        stats["image_url_usage_percentage"] = (
            len(used_in_outline) / len(stats["image_search_urls"])
        ) * 100

        # Usage in final HTML
        used_in_html = stats["image_search_urls"].intersection(
            stats["all_slides_html_image_urls"]
        )
        stats["final_html_usage_count"] = len(used_in_html)
        stats["final_html_usage_percentage"] = (
            len(used_in_html) / len(stats["image_search_urls"])
        ) * 100

        # Funnel analysis percentages
        total_search_urls = len(stats["image_search_urls"])
        if total_search_urls > 0:
            stats["funnel_search_to_outline"] = (
                len(used_in_outline) / total_search_urls
            ) * 100
            stats["funnel_search_to_html"] = (
                len(used_in_html) / total_search_urls
            ) * 100

        # Outline to HTML conversion rate
        if len(stats["detailed_outline_image_urls"]) > 0:
            outline_to_html = len(
                stats["detailed_outline_image_urls"].intersection(
                    stats["all_slides_html_image_urls"]
                )
            )
            stats["funnel_outline_to_html"] = (
                outline_to_html / len(stats["detailed_outline_image_urls"])
            ) * 100

    # Convert sets to counts for easier processing
    stats["image_search_unique_queries_count"] = len(
        stats["image_search_unique_queries"]
    )
    stats["image_search_urls_count"] = len(stats["image_search_urls"])
    stats["detailed_outline_urls_count"] = len(stats["detailed_outline_urls"])
    stats["detailed_outline_image_urls_count"] = len(
        stats["detailed_outline_image_urls"]
    )
    stats["all_slides_html_image_urls_count"] = len(stats["all_slides_html_image_urls"])

    return stats


def analyze_folder(folder_path: str) -> List[Dict]:
    """Analyze all JSON files in a folder."""
    results = []

    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} does not exist!")
        return results

    json_files = [f for f in os.listdir(folder_path) if f.endswith(".json")]

    if not json_files:
        print(f"No JSON files found in {folder_path}")
        return results

    print(f"Found {len(json_files)} JSON files to analyze...")

    for filename in json_files:
        filepath = os.path.join(folder_path, filename)
        stats = analyze_single_json(filepath)
        if stats:
            results.append(stats)
            print(f"Analyzed: {filename}")

    return results


def calculate_averages(results: List[Dict]) -> Dict:
    """Calculate average statistics across all files."""
    if not results:
        return {}

    totals = {
        "brief_outline_picture_advise_count": 0,
        "image_search_results_count": 0,
        "image_search_unique_queries_count": 0,
        "image_search_urls_count": 0,
        "detailed_outline_urls_count": 0,
        "detailed_outline_image_urls_count": 0,
        "all_slides_html_image_urls_count": 0,
        "coverage_percentage": 0.0,
        "image_url_usage_count": 0,
        "image_url_usage_percentage": 0.0,
        "final_html_usage_count": 0,
        "final_html_usage_percentage": 0.0,
        "funnel_search_to_outline": 0.0,
        "funnel_outline_to_html": 0.0,
        "funnel_search_to_html": 0.0,
    }

    for result in results:
        totals["brief_outline_picture_advise_count"] += result[
            "brief_outline_picture_advise_count"
        ]
        totals["image_search_results_count"] += result["image_search_results_count"]
        totals["image_search_unique_queries_count"] += result[
            "image_search_unique_queries_count"
        ]
        totals["image_search_urls_count"] += result["image_search_urls_count"]
        totals["detailed_outline_urls_count"] += result["detailed_outline_urls_count"]
        totals["detailed_outline_image_urls_count"] += result[
            "detailed_outline_image_urls_count"
        ]
        totals["all_slides_html_image_urls_count"] += result[
            "all_slides_html_image_urls_count"
        ]
        totals["coverage_percentage"] += result["coverage_percentage"]
        totals["image_url_usage_count"] += result["image_url_usage_count"]
        totals["image_url_usage_percentage"] += result["image_url_usage_percentage"]
        totals["final_html_usage_count"] += result["final_html_usage_count"]
        totals["final_html_usage_percentage"] += result["final_html_usage_percentage"]
        totals["funnel_search_to_outline"] += result["funnel_search_to_outline"]
        totals["funnel_outline_to_html"] += result["funnel_outline_to_html"]
        totals["funnel_search_to_html"] += result["funnel_search_to_html"]

    num_files = len(results)
    averages = {
        "total_files_analyzed": num_files,
        "avg_picture_advise_per_file": totals["brief_outline_picture_advise_count"]
        / num_files,
        "avg_image_search_results_per_file": totals["image_search_results_count"]
        / num_files,
        "avg_unique_queries_per_file": totals["image_search_unique_queries_count"]
        / num_files,
        "avg_image_search_urls_per_file": totals["image_search_urls_count"] / num_files,
        "avg_urls_in_detailed_outline_per_file": totals["detailed_outline_urls_count"]
        / num_files,
        "avg_image_urls_in_detailed_outline_per_file": totals[
            "detailed_outline_image_urls_count"
        ]
        / num_files,
        "avg_html_image_urls_per_file": totals["all_slides_html_image_urls_count"]
        / num_files,
        "avg_coverage_percentage": totals["coverage_percentage"] / num_files,
        "avg_image_url_usage_count_per_file": totals["image_url_usage_count"]
        / num_files,
        "avg_image_url_usage_percentage": totals["image_url_usage_percentage"]
        / num_files,
        "avg_final_html_usage_count_per_file": totals["final_html_usage_count"]
        / num_files,
        "avg_final_html_usage_percentage": totals["final_html_usage_percentage"]
        / num_files,
        "avg_funnel_search_to_outline": totals["funnel_search_to_outline"] / num_files,
        "avg_funnel_outline_to_html": totals["funnel_outline_to_html"] / num_files,
        "avg_funnel_search_to_html": totals["funnel_search_to_html"] / num_files,
        "total_picture_advise": totals["brief_outline_picture_advise_count"],
        "total_image_search_results": totals["image_search_results_count"],
        "total_unique_queries": totals["image_search_unique_queries_count"],
        "total_image_search_urls": totals["image_search_urls_count"],
        "total_urls_mentioned": totals["detailed_outline_urls_count"],
        "total_image_urls_mentioned": totals["detailed_outline_image_urls_count"],
        "total_html_image_urls": totals["all_slides_html_image_urls_count"],
        "total_image_url_usage": totals["image_url_usage_count"],
        "total_final_html_usage": totals["final_html_usage_count"],
    }

    return averages


def print_detailed_report(results: List[Dict]):
    """Print detailed report for each file."""
    print("\n" + "=" * 80)
    print("DETAILED REPORT BY FILE")
    print("=" * 80)

    for result in results:
        print(f"\nFile: {result['file']}")
        print(f"  Picture Advice Count: {result['brief_outline_picture_advise_count']}")
        print(f"  Image Search Results: {result['image_search_results_count']}")
        print(
            f"  Unique Queries in Image Search: {result['image_search_unique_queries_count']}"
        )
        print(f"  Image URLs from Search Results: {result['image_search_urls_count']}")
        print(
            f"  Image URLs Used in Detailed Outline: {result['image_url_usage_count']}"
        )
        print(f"  Image URLs Used in Final HTML: {result['final_html_usage_count']}")
        print(
            f"  Total URLs in Detailed Outline: {result['detailed_outline_urls_count']}"
        )
        print(
            f"  Image URLs in Detailed Outline: {result['detailed_outline_image_urls_count']}"
        )
        print(
            f"  Image URLs in Final HTML: {result['all_slides_html_image_urls_count']}"
        )
        print(f"  Query Coverage Percentage: {result['coverage_percentage']:.1f}%")
        print(f"  FUNNEL ANALYSIS:")
        print(f"    Search → Outline: {result['funnel_search_to_outline']:.1f}%")
        print(f"    Outline → HTML: {result['funnel_outline_to_html']:.1f}%")
        print(f"    Search → HTML: {result['funnel_search_to_html']:.1f}%")

        if result["brief_outline_queries"]:
            print(f"  Picture Advice Queries: {result['brief_outline_queries']}")


def print_summary_report(averages: Dict):
    """Print summary statistics."""
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    print(f"Total Files Analyzed: {averages['total_files_analyzed']}")
    print(f"\nIMAGE URL FUNNEL ANALYSIS:")
    print(
        f"  Avg Image URLs from Search Results per file: {averages['avg_image_search_urls_per_file']:.2f}"
    )
    print(
        f"  Avg Image URLs Used in Detailed Outline per file: {averages['avg_image_url_usage_count_per_file']:.2f}"
    )
    print(
        f"  Avg Image URLs Used in Final HTML per file: {averages['avg_final_html_usage_count_per_file']:.2f}"
    )
    print(f"\nFUNNEL CONVERSION RATES:")
    print(
        f"  Search Results → Detailed Outline: {averages['avg_funnel_search_to_outline']:.2f}%"
    )
    print(
        f"  Detailed Outline → Final HTML: {averages['avg_funnel_outline_to_html']:.2f}%"
    )
    print(
        f"  Search Results → Final HTML: {averages['avg_funnel_search_to_html']:.2f}%"
    )

    print(f"\nAVERAGE STATISTICS PER FILE:")
    print(f"  Picture Advice per file: {averages['avg_picture_advise_per_file']:.2f}")
    print(
        f"  Image Search Results per file: {averages['avg_image_search_results_per_file']:.2f}"
    )
    print(f"  Unique Queries per file: {averages['avg_unique_queries_per_file']:.2f}")
    print(
        f"  Total URLs in Detailed Outline per file: {averages['avg_urls_in_detailed_outline_per_file']:.2f}"
    )
    print(
        f"  Image URLs in Detailed Outline per file: {averages['avg_image_urls_in_detailed_outline_per_file']:.2f}"
    )
    print(
        f"  Image URLs in Final HTML per file: {averages['avg_html_image_urls_per_file']:.2f}"
    )
    print(f"  Query Coverage Percentage: {averages['avg_coverage_percentage']:.2f}%")

    print(f"\nTOTAL ACROSS ALL FILES:")
    print(f"  Total Picture Advice: {averages['total_picture_advise']}")
    print(f"  Total Image Search Results: {averages['total_image_search_results']}")
    print(f"  Total Unique Queries: {averages['total_unique_queries']}")
    print(
        f"  Total Image URLs from Search Results: {averages['total_image_search_urls']}"
    )
    print(
        f"  Total Image URLs Used in Detailed Outline: {averages['total_image_url_usage']}"
    )
    print(
        f"  Total Image URLs Used in Final HTML: {averages['total_final_html_usage']}"
    )
    print(f"  Total URLs Mentioned: {averages['total_urls_mentioned']}")
    print(
        f"  Total Image URLs in Detailed Outline: {averages['total_image_urls_mentioned']}"
    )
    print(f"  Total Image URLs in Final HTML: {averages['total_html_image_urls']}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze PPT JSON files for image URL usage and search statistics"
    )
    parser.add_argument("folder_path", help="Path to folder containing JSON files")
    parser.add_argument(
        "--detailed",
        "-d",
        action="store_true",
        help="Show detailed report for each file",
    )

    args = parser.parse_args()

    # Analyze all JSON files in the folder
    results = analyze_folder(args.folder_path)

    if not results:
        print("No valid JSON files found or analyzed.")
        return

    # Calculate averages
    averages = calculate_averages(results)

    # Print reports
    if args.detailed:
        print_detailed_report(results)

    print_summary_report(averages)


if __name__ == "__main__":
    main()
