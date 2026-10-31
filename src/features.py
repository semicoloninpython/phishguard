"""Turn a URL string into the UCI features that can be computed from text alone.

Only 9 of the 30 UCI features are derivable from the URL text itself; the other 21 need
the page HTML, WHOIS/DNS or web-ranking services. Encoding: -1 phishing-like, 0 suspicious,
1 legitimate-like (same convention as the dataset).
"""
from urllib.parse import urlsplit
import ipaddress
import re

SHORTENERS = {"bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly",
              "cutt.ly", "rb.gy", "shorturl.at", "tiny.cc", "tr.im", "lnkd.in"}


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return bool(re.fullmatch(r"0x[0-9a-f]+", host))


def url_to_uci_features(raw_url: str) -> dict:
    url = raw_url.strip()
    candidate = url if "://" in url else "https://" + url
    p = urlsplit(candidate)
    host = (p.hostname or "").lower()
    try:
        port = p.port
    except ValueError:
        port = None
    dots = host.removeprefix("www.").count(".")
    n = len(url)
    return {
        "having_ip_address": -1 if _is_ip(host) else 1,
        "url_length": 1 if n < 54 else (0 if n <= 75 else -1),
        "shortining_service": -1 if host in SHORTENERS else 1,
        "having_at_symbol": -1 if "@" in url else 1,
        "double_slash_redirecting": -1 if candidate.rfind("//") > 7 else 1,
        "prefix_suffix": -1 if "-" in host else 1,
        "having_sub_domain": 1 if dots <= 1 else (0 if dots == 2 else -1),
        "port": -1 if port not in (None, 80, 443) else 1,
        "https_token": -1 if "https" in host else 1,
    }
