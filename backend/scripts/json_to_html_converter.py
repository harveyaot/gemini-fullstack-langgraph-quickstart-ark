#!/usr/bin/env python3
"""
JSON to HTML Converter Script

This script converts a folder of JSON files (containing all_slides_html data)
into separate HTML files with an index page for navigation.

Usage:
    python json_to_html_converter.py <input_folder_path>
"""

import json
import os
import sys
from pathlib import Path
import re
from datetime import datetime


def sanitize_filename(filename):
    """Sanitize filename to be safe for filesystem and URL-friendly."""
    # Remove or replace invalid characters for filesystem
    filename = re.sub(r'[<>:"/\\|?*]', "_", filename)

    # Replace spaces with underscores
    filename = re.sub(r"\s+", "_", filename.strip())

    # Replace other potentially problematic characters for URLs
    filename = re.sub(r"[%#&+=]", "_", filename)

    # Remove multiple consecutive underscores
    filename = re.sub(r"_{2,}", "_", filename)

    # Remove leading/trailing underscores
    filename = filename.strip("_")

    # Limit length and ensure we don't end with a dot
    filename = filename[:100].rstrip(".")

    # If filename is empty after sanitization, provide a default
    if not filename:
        filename = "untitled"

    return filename


def extract_title_from_html(html_content):
    """Extract title from HTML content."""
    # Try to get title from <title> tag
    title_match = re.search(r"<title[^>]*>([^<]+)</title>", html_content, re.IGNORECASE)
    if title_match:
        return title_match.group(1).strip()

    # Try to get from h1 tag
    h1_match = re.search(r"<h1[^>]*>([^<]+)</h1>", html_content, re.IGNORECASE)
    if h1_match:
        return h1_match.group(1).strip()

    # Try to get from h2 tag
    h2_match = re.search(r"<h2[^>]*>([^<]+)</h2>", html_content, re.IGNORECASE)
    if h2_match:
        return h2_match.group(1).strip()

    return "Untitled Slide"


def process_json_file(json_path, output_dir, presentation_index):
    """Process a single JSON file and extract slides."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Extract presentation info
        thread_id = data.get("thread_id", f"presentation_{presentation_index}")
        ppt_title = None

        # Try to get PPT title from brief_outline
        if "brief_outline" in data and "ppt_title" in data["brief_outline"]:
            ppt_title = data["brief_outline"]["ppt_title"]

        # Get all_slides_html
        all_slides_html = data.get("all_slides_html", [])

        if not all_slides_html:
            print(f"Warning: No slides found in {json_path}")
            return []

        presentation_info = {
            "thread_id": thread_id,
            "title": ppt_title or f"Presentation {presentation_index}",
            "source_file": json_path.name,
            "slides": [],
        }

        # Create presentation directory
        pres_dir = output_dir / sanitize_filename(
            f"{presentation_index:03d}_{thread_id}"
        )
        pres_dir.mkdir(exist_ok=True)

        # Process each slide
        for slide_index, slide_html in enumerate(all_slides_html, 1):
            # Extract title from slide content
            slide_title = extract_title_from_html(slide_html)

            # Create filename
            slide_filename = (
                f"slide_{slide_index:02d}_{sanitize_filename(slide_title)}.html"
            )
            slide_path = pres_dir / slide_filename

            # Add navigation to slide
            enhanced_html = add_navigation_to_slide(
                slide_html,
                slide_index,
                len(all_slides_html),
                presentation_index,
                thread_id,
            )

            # Write slide file
            with open(slide_path, "w", encoding="utf-8") as f:
                f.write(enhanced_html)

            presentation_info["slides"].append(
                {
                    "index": slide_index,
                    "title": slide_title,
                    "filename": slide_filename,
                    "path": str(slide_path.relative_to(output_dir)),
                }
            )

            print(f"Created: {slide_path}")

        return presentation_info

    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {json_path}: {e}")
        return None
    except Exception as e:
        print(f"Error processing {json_path}: {e}")
        return None


def add_navigation_to_slide(
    html_content, current_slide, total_slides, presentation_index, thread_id
):
    """Add navigation controls to slide HTML."""

    navigation_css = """
    <style>
        .navigation-bar {
            position: fixed;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(0, 0, 0, 0.8);
            padding: 10px 20px;
            border-radius: 25px;
            display: flex;
            gap: 15px;
            align-items: center;
            z-index: 1000;
        }
        .nav-btn {
            background: #4A90E2;
            color: white;
            border: none;
            padding: 8px 15px;
            border-radius: 20px;
            cursor: pointer;
            text-decoration: none;
            font-size: 14px;
            transition: background-color 0.3s;
        }
        .nav-btn:hover {
            background: #357ABD;
        }
        .nav-btn:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
        .slide-counter {
            color: white;
            font-size: 14px;
            font-weight: bold;
        }
        .home-btn {
            background: #28a745;
        }
        .home-btn:hover {
            background: #1e7e34;
        }
    </style>
    """

    # Create navigation buttons
    prev_disabled = "disabled" if current_slide == 1 else ""
    next_disabled = "disabled" if current_slide == total_slides else ""

    prev_slide = max(1, current_slide - 1)
    next_slide = min(total_slides, current_slide + 1)

    navigation_html = f"""
    <div class="navigation-bar">
        <a href="../../index.html" class="nav-btn home-btn">🏠 Home</a>
        <button class="nav-btn" onclick="navigateToSlide({prev_slide})" {prev_disabled}>← Previous</button>
        <span class="slide-counter">{current_slide} / {total_slides}</span>
        <button class="nav-btn" onclick="navigateToSlide({next_slide})" {next_disabled}>Next →</button>
    </div>
    
    <script>
        function navigateToSlide(slideNumber) {{
            try {{
                // Get current path and decode it to handle non-ASCII characters
                const currentPath = decodeURIComponent(window.location.pathname);
                const pathParts = currentPath.split('/');
                const filename = pathParts[pathParts.length - 1];
                
                // Extract the base filename after the slide number and underscore
                const slidePattern = /^slide_\\d{{2}}_(.+)\\.html$/;
                const match = filename.match(slidePattern);
                
                if (match) {{
                    const baseName = match[1];
                    const newFilename = `slide_${{slideNumber.toString().padStart(2, '0')}}_${{baseName}}.html`;
                    
                    // Use relative navigation to avoid encoding issues
                    window.location.href = newFilename;
                }} else {{
                    console.error('Could not parse filename:', filename);
                    // Fallback: reload current page
                    window.location.reload();
                }}
            }} catch (error) {{
                console.error('Navigation error:', error);
                // Fallback: reload current page
                window.location.reload();
            }}
        }}
        
        // Keyboard navigation
        document.addEventListener('keydown', function(e) {{
            if (e.key === 'ArrowLeft' && {current_slide} > 1) {{
                navigateToSlide({prev_slide});
            }} else if (e.key === 'ArrowRight' && {current_slide} < {total_slides}) {{
                navigateToSlide({next_slide});
            }} else if (e.key === 'Home') {{
                window.location.href = '../../index.html';
            }}
        }});
        
        // Debug function to check current filename
        console.log('Current slide:', {current_slide}, 'Filename:', decodeURIComponent(window.location.pathname));
    </script>
    """

    # Insert navigation into HTML
    # Try to insert before closing body tag
    if "</body>" in html_content:
        html_content = html_content.replace("</body>", f"{navigation_html}</body>")
    else:
        # If no body tag, append to end
        html_content += navigation_html

    # Insert CSS into head
    if "</head>" in html_content:
        html_content = html_content.replace("</head>", f"{navigation_css}</head>")
    else:
        # If no head tag, prepend CSS
        html_content = navigation_css + html_content

    return html_content


def create_index_html(presentations, output_dir):
    """Create index.html with navigation to all presentations and slides."""

    index_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Presentation Gallery</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        }}
        
        h1 {{
            text-align: center;
            color: #2c3e50;
            margin-bottom: 10px;
            font-size: 2.5em;
        }}
        
        .subtitle {{
            text-align: center;
            color: #7f8c8d;
            margin-bottom: 40px;
            font-size: 1.1em;
        }}
        
        .presentation-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
            gap: 25px;
            margin-bottom: 40px;
        }}
        
        .presentation-card {{
            background: #f8f9fa;
            border-radius: 12px;
            padding: 25px;
            border-left: 5px solid #3498db;
            transition: transform 0.3s, box-shadow 0.3s;
        }}
        
        .presentation-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 15px 30px rgba(0,0,0,0.15);
        }}
        
        .presentation-title {{
            font-size: 1.4em;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 10px;
        }}
        
        .presentation-meta {{
            color: #7f8c8d;
            font-size: 0.9em;
            margin-bottom: 15px;
        }}
        
        .slides-list {{
            list-style: none;
            padding: 0;
            margin: 0;
        }}
        
        .slide-item {{
            margin: 8px 0;
        }}
        
        .slide-link {{
            display: inline-block;
            padding: 8px 15px;
            background: #3498db;
            color: white;
            text-decoration: none;
            border-radius: 20px;
            font-size: 0.9em;
            transition: background-color 0.3s;
            width: 100%;
            box-sizing: border-box;
        }}
        
        .slide-link:hover {{
            background: #2980b9;
        }}
        
        .stats {{
            text-align: center;
            margin-top: 40px;
            padding: 20px;
            background: #ecf0f1;
            border-radius: 10px;
        }}
        
        .stats-item {{
            display: inline-block;
            margin: 0 20px;
            text-align: center;
        }}
        
        .stats-number {{
            font-size: 2em;
            font-weight: bold;
            color: #3498db;
            display: block;
        }}
        
        .stats-label {{
            color: #7f8c8d;
            font-size: 0.9em;
        }}
        
        .generation-time {{
            text-align: center;
            color: #95a5a6;
            font-size: 0.8em;
            margin-top: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 Presentation Gallery</h1>
        <p class="subtitle">Generated HTML presentations from JSON data</p>
        
        <div class="presentation-grid">
"""

    total_slides = 0

    for presentation in presentations:
        if presentation is None:
            continue

        total_slides += len(presentation["slides"])

        index_html += f"""
            <div class="presentation-card">
                <div class="presentation-title">{presentation['title']}</div>
                <div class="presentation-meta">
                    📁 Source: {presentation['source_file']}<br>
                    🎬 Slides: {len(presentation['slides'])}<br>
                    🆔 ID: {presentation['thread_id'][:20]}...
                </div>
                <ul class="slides-list">
        """

        for slide in presentation["slides"]:
            index_html += f"""
                    <li class="slide-item">
                        <a href="{slide['path']}" class="slide-link">
                            {slide['index']:02d}. {slide['title'][:50]}{'...' if len(slide['title']) > 50 else ''}
                        </a>
                    </li>
            """

        index_html += """
                </ul>
            </div>
        """

    index_html += f"""
        </div>
        
        <div class="stats">
            <div class="stats-item">
                <span class="stats-number">{len([p for p in presentations if p is not None])}</span>
                <span class="stats-label">Presentations</span>
            </div>
            <div class="stats-item">
                <span class="stats-number">{total_slides}</span>
                <span class="stats-label">Total Slides</span>
            </div>
        </div>
        
        <div class="generation-time">
            Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </div>
</body>
</html>"""

    # Write index file
    index_path = output_dir / "index.html"
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(index_html)

    print(f"Created index: {index_path}")


def main():
    """Main function to process folder of JSON files."""
    if len(sys.argv) != 2:
        print("Usage: python json_to_html_converter.py <input_folder_path>")
        sys.exit(1)

    input_folder = Path(sys.argv[1])

    if not input_folder.exists():
        print(f"Error: Input folder '{input_folder}' does not exist.")
        sys.exit(1)

    if not input_folder.is_dir():
        print(f"Error: '{input_folder}' is not a directory.")
        sys.exit(1)

    # Create output directory
    output_folder = Path(f"{input_folder}_htmls")
    output_folder.mkdir(exist_ok=True)

    print(f"Converting JSON files from: {input_folder}")
    print(f"Output directory: {output_folder}")
    print("-" * 50)

    # Find all JSON files
    json_files = list(input_folder.glob("*.json"))

    if not json_files:
        print(f"No JSON files found in {input_folder}")
        sys.exit(1)

    print(f"Found {len(json_files)} JSON files")

    # Process each JSON file
    presentations = []
    for i, json_file in enumerate(json_files, 1):
        print(f"Processing {json_file.name}...")
        presentation_info = process_json_file(json_file, output_folder, i)
        presentations.append(presentation_info)

    # Create index.html
    print("\nCreating index.html...")
    create_index_html(presentations, output_folder)

    print(f"\n✅ Conversion complete!")
    print(f"📁 Output folder: {output_folder}")
    print(f"🌐 Open {output_folder / 'index.html'} to start browsing")


if __name__ == "__main__":
    main()
