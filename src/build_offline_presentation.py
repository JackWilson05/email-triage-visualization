"""Create a self-contained presentation copy after D3 and website data exist.

The GitHub Pages site can load D3 from its fallback CDN. This script makes a
network-free copy for classroom presentation by embedding the local D3 vendor
file and the chart data directly into the page.
"""

from __future__ import annotations

from pathlib import Path
import re


PROJECT = Path(__file__).resolve().parents[1]
DOCS = PROJECT / "docs"
OUTPUT = PROJECT / "submission" / "presentation_offline.html"


def main() -> None:
    d3_path = DOCS / "vendor" / "d3.v5.min.js"
    data_path = DOCS / "data" / "website_data.json"
    required = [d3_path, data_path, DOCS / "index.html", DOCS / "styles.css", DOCS / "app.js"]
    missing = [str(path.relative_to(PROJECT)) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required build files: " + ", ".join(missing))

    html = (DOCS / "index.html").read_text(encoding="utf-8")
    css = (DOCS / "styles.css").read_text(encoding="utf-8")
    app = (DOCS / "app.js").read_text(encoding="utf-8")
    d3 = d3_path.read_text(encoding="utf-8")
    payload = data_path.read_text(encoding="utf-8")
    html = html.replace('<link rel="stylesheet" href="styles.css" />', f"<style>{css}</style>")
    # Replace the local D3 tag and its CDN fallback as one unit. A regex keeps
    # this resilient to harmless whitespace changes in index.html.
    d3_tags = r'\s*<script src="vendor/d3\.v5\.min\.js"></script>\s*<script>.*?</script>'
    html, replaced = re.subn(d3_tags, lambda _: f"\n    <script>{d3}</script>", html, count=1, flags=re.DOTALL)
    if replaced != 1:
        raise ValueError("Could not embed the D3 script block from index.html.")
    # The app reads this global first, so it never tries to fetch a JSON file
    # when the self-contained presentation is opened from disk.
    data_tag = '<script src="data/website_data.js"></script>'
    if data_tag not in html:
        raise ValueError("Could not embed website_data.js from index.html.")
    html = html.replace(data_tag, f"<script>window.WEBSITE_DATA = {payload};</script>")
    html = html.replace('<script src="app.js"></script>', f"<script>{app}</script>")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(html, encoding="utf-8")
    print(f"Built {OUTPUT}")


if __name__ == "__main__":
    main()
