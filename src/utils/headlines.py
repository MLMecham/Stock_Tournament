import re
import xml.etree.ElementTree as ET

import requests

_RSS_URL = "https://finance.yahoo.com/rss/topstories"


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text).strip()


def get_headlines(max_words: int = 500) -> str:
    """Fetch today's Yahoo Finance top stories. Returns a plain-text summary."""
    resp = requests.get(_RSS_URL, timeout=10, headers={"User-Agent": "stock-tournament/1.0"})
    resp.raise_for_status()

    root = ET.fromstring(resp.content)
    parts = []
    for item in root.findall(".//item"):
        title = item.findtext("title", "").strip()
        desc = _strip_html(item.findtext("description", ""))
        parts.append(f"{title}. {desc}" if desc and desc != title else title)

    words = " ".join(parts).split()[:max_words]
    return " ".join(words)
