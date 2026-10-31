from urllib.parse import urlsplit, unquote
import ipaddress
import re

SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd",
    "buff.ly", "cutt.ly", "rb.gy", "shorturl.at", "tiny.cc"
}
SUSPICIOUS_TLDS = {
    "zip", "mov", "top", "xyz", "click", "work", "country", "gq",
    "tk", "ml", "cf", "ga"
}
BRAND_BAIT = {
    "login", "verify", "verification", "account", "secure", "update",
    "wallet", "banking", "password", "signin", "confirm", "payment"
}

def inspect_url(raw_url):
    """Inspect URL text only. Does not visit the site or contact its server."""
    raw_url = (raw_url or "").strip()
    if not raw_url:
        return {"valid": False, "error": "Enter a website URL to scan."}

    candidate = raw_url if "://" in raw_url else "https://" + raw_url
    try:
        parsed = urlsplit(candidate)
        host = (parsed.hostname or "").lower().rstrip(".")
        if parsed.scheme not in ("http", "https") or not host or "." not in host:
            raise ValueError
        # Accessing .port validates malformed port values.
        port = parsed.port
    except Exception:
        return {"valid": False, "error": "That does not look like a valid website URL. Try a domain such as example.com."}

    findings = []
    score = 0

    def flag(points, title, detail, severity="warning"):
        nonlocal score
        score += points
        findings.append({"title": title, "detail": detail, "severity": severity, "points": points})

    try:
        ipaddress.ip_address(host)
        flag(25, "Uses an IP address", "The address points directly to an IP instead of using a normal domain.")
    except ValueError:
        pass

    if len(raw_url) > 100:
        flag(10, "Unusually long URL", f"The URL contains {len(raw_url)} characters. Long URLs can hide the real destination.")
    if "@" in parsed.netloc:
        flag(25, "Contains an @ symbol", "The @ character can make the actual destination harder to notice.")
    if host.startswith("xn--") or ".xn--" in host:
        flag(20, "Internationalized/punycode domain", "The domain uses encoded characters. Check carefully for look-alike letters.")
    if host in SHORTENERS:
        flag(15, "URL shortener detected", "Shortened links hide the final destination until they are expanded.")
    labels = host.split(".")
    if len(labels) >= 5:
        flag(12, "Many subdomains", f"The host contains {len(labels)-2} subdomain levels.")
    if "-" in labels[-2] if len(labels) >= 2 else False:
        flag(8, "Hyphen in registered domain", "A hyphen in the main domain is not proof of phishing, but deserves a closer look.")
    tld = labels[-1] if labels else ""
    if tld in SUSPICIOUS_TLDS:
        flag(10, "Higher-risk top-level domain", f".{tld} is sometimes seen in abuse reports. A TLD alone cannot establish whether a site is malicious.")
    if parsed.scheme != "https":
        flag(10, "No HTTPS", "The URL uses HTTP rather than HTTPS. HTTPS does not guarantee legitimacy, but it protects the connection.")
    if port and port not in (80, 443):
        flag(12, "Unusual port", f"The URL uses port {port}, which is uncommon for ordinary public websites.")
    if "%" in parsed.path or "%" in parsed.query:
        decoded = unquote(parsed.path + parsed.query)
        if decoded != parsed.path + parsed.query:
            flag(8, "Encoded URL characters", "The path or query contains percent-encoded characters. This can be normal, but may obscure a link.")
    url_lower = (host + parsed.path + parsed.query).lower()
    bait_hits = sorted(word for word in BRAND_BAIT if word in url_lower)
    if len(bait_hits) >= 2:
        flag(12, "Multiple account-action keywords", "The URL includes several terms often used in login or account-bait links: " + ", ".join(bait_hits[:5]) + ".")
    elif len(bait_hits) == 1:
        flag(5, "Account-action keyword", f"The URL contains '{bait_hits[0]}'. This is common on legitimate sites too, so it is only a weak signal.")

    score = min(score, 100)
    if score >= 40:
        verdict = "Potentially suspicious"
        explanation = "Several URL-level warning signs were found. Avoid entering passwords or payment details until you verify the destination independently."
        status = "danger"
    elif score >= 18:
        verdict = "Use caution"
        explanation = "Some URL-level warning signs were found. Check the domain carefully and verify it through an official source."
        status = "warning"
    else:
        verdict = "No obvious URL warning signs"
        explanation = "This quick URL-only check did not find many common warning signs. It does not prove that the website is safe."
        status = "safe"

    if not findings:
        findings.append({
            "title": "No strong URL indicators found",
            "detail": "The URL structure looks ordinary under these basic checks. Website content, reputation and redirects were not inspected.",
            "severity": "safe",
            "points": 0
        })

    return {
        "valid": True,
        "url": candidate,
        "host": host,
        "score": score,
        "verdict": verdict,
        "explanation": explanation,
        "status": status,
        "findings": findings
    }
