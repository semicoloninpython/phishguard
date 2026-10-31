"""Shared helpers: paths, data loading and human-readable feature metadata."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "phishing.csv"
TARGET = "Result"
CLASS_LABELS = {0: "Phishing", 1: "Legitimate"}  # encoded labels used by every model
ORIGINAL_TO_ENCODED = {-1: 0, 1: 1}              # UCI uses -1 = phishing, 1 = legitimate

# name -> (friendly title, group, where the value comes from, short description)
FEATURE_INFO = {
 "having_ip_address": ("IP address in URL", "Address bar", "URL text", "Raw IP (e.g. 192.168.1.5) used instead of a domain name."),
 "url_length": ("URL length", "Address bar", "URL text", "Long URLs (>75 chars) can hide the real destination."),
 "shortining_service": ("URL shortener", "Address bar", "URL text", "bit.ly-style services hide the final site."),
 "having_at_symbol": ("'@' symbol", "Address bar", "URL text", "Browsers ignore everything before '@', which can disguise the target."),
 "double_slash_redirecting": ("'//' redirect", "Address bar", "URL text", "A second '//' inside the URL can redirect the visitor."),
 "prefix_suffix": ("Hyphen in domain", "Address bar", "URL text", "Fake brands often use hyphens, e.g. paypal-secure.com."),
 "having_sub_domain": ("Sub-domain depth", "Address bar", "URL text", "Many dots/sub-domains imitate trusted names."),
 "sslfinal_state": ("SSL certificate state", "Address bar", "Certificate lookup", "HTTPS with a trusted, mature certificate issuer."),
 "domain_registration_length": ("Domain registration length", "Address bar", "WHOIS", "Phishing domains are usually registered for ~1 year only."),
 "favicon": ("Favicon source", "Address bar", "Page HTML", "Icon loaded from a different domain than the page."),
 "port": ("Non-standard port", "Address bar", "URL text", "Services on unusual ports are suspicious."),
 "https_token": ("'https' inside domain", "Address bar", "URL text", "Domains like https-paypal.com try to look secure."),
 "request_url": ("External objects", "Abnormal", "Page HTML", "Share of images/videos loaded from other domains."),
 "url_of_anchor": ("Anchor links", "Abnormal", "Page HTML", "Share of <a> tags that point elsewhere or to '#'."),
 "links_in_tags": ("Links in meta/script tags", "Abnormal", "Page HTML", "Meta/script/link tags pointing to other domains."),
 "sfh": ("Server form handler", "Abnormal", "Page HTML", "Forms that submit to blank or foreign pages."),
 "submitting_to_email": ("Form submits to e-mail", "Abnormal", "Page HTML", "Forms using mailto: to steal data."),
 "abnormal_url": ("Abnormal URL", "Abnormal", "WHOIS", "Host name missing from the WHOIS record."),
 "redirect": ("Redirect count", "HTML & JavaScript", "HTTP fetch", "Number of redirects before the page loads."),
 "on_mouseover": ("Status-bar tampering", "HTML & JavaScript", "Page JavaScript", "onMouseOver changes the visible link in the status bar."),
 "rightclick": ("Right-click disabled", "HTML & JavaScript", "Page JavaScript", "Blocking right-click hides the page source."),
 "popupwindow": ("Pop-up with text fields", "HTML & JavaScript", "Page JavaScript", "Pop-ups asking for credentials."),
 "iframe": ("Invisible iFrame", "HTML & JavaScript", "Page HTML", "Hidden frames that embed another page."),
 "age_of_domain": ("Domain age", "Domain based", "WHOIS", "Young domains (<6 months) are riskier."),
 "dnsrecord": ("DNS record exists", "Domain based", "DNS", "Missing DNS record suggests a throw-away domain."),
 "web_traffic": ("Web traffic rank", "Domain based", "Alexa-style service", "Popular sites rank high; phishing sites rarely do."),
 "page_rank": ("PageRank", "Domain based", "Ranking service", "Importance of the page on the web."),
 "google_index": ("Indexed by Google", "Domain based", "Search engine", "Phishing pages are short-lived and often not indexed."),
 "links_pointing_to_page": ("Inbound links", "Domain based", "Search engine", "Legitimate pages are linked from other sites."),
 "statistical_report": ("Statistical report", "Domain based", "Phishing blacklists", "Host/IP appears in public phishing statistics."),
}
FEATURES = list(FEATURE_INFO)
URL_ONLY_FEATURES = [f for f, v in FEATURE_INFO.items() if v[2] == "URL text"]


def title_of(feature: str) -> str:
    return FEATURE_INFO.get(feature, (feature.replace("_", " ").title(),))[0]


def load_data(deduplicate: bool = False) -> pd.DataFrame:
    """Load data/phishing.csv (optionally dropping exact duplicate rows)."""
    if not DATA_FILE.exists():
        raise FileNotFoundError("data/phishing.csv is missing. Run: python fetch_data.py")
    df = pd.read_csv(DATA_FILE)
    return df.drop_duplicates().reset_index(drop=True) if deduplicate else df


def get_split(seed: int = 42, test_size: float = 0.2):
    """Re-create the exact train/test split used by src/train.py (same seed, same dedupe)."""
    from sklearn.model_selection import train_test_split
    df = load_data(deduplicate=True)
    X, y = df[FEATURES], (df[TARGET] == 1).astype(int)  # 0 = phishing, 1 = legitimate
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
