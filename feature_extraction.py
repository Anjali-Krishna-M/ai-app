from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, Iterable
from urllib.parse import urlparse

SUSPICIOUS_WORDS = ("login", "verify", "secure", "update", "bank", "account", "signin")
SHORTENERS = (
    "bit.ly",
    "tinyurl.com",
    "goo.gl",
    "ow.ly",
    "t.co",
    "is.gd",
    "buff.ly",
    "adf.ly",
)
TOP_DOMAINS = {
    "google.com",
    "youtube.com",
    "facebook.com",
    "amazon.com",
    "wikipedia.org",
    "twitter.com",
    "instagram.com",
    "linkedin.com",
    "apple.com",
    "microsoft.com",
    "netflix.com",
    "paypal.com",
    "github.com",
}


@dataclass
class URLFeatures:
    url_length: int
    dot_count: int
    special_char_count: int
    has_ip: int
    has_https: int
    domain_length: int
    suspicious_word_count: int
    subdomain_count: int
    is_shortened: int
    directory_count: int
    query_length: int
    digit_count: int
    hyphen_count: int
    max_popular_domain_similarity: float
    typosquat_flag: int

    def to_dict(self) -> Dict[str, float | int]:
        return self.__dict__.copy()


def normalize_url(url: str) -> str:
    url = (url or "").strip().lower()
    if not url:
        return ""
    if not re.match(r"^[a-z]+://", url):
        url = f"http://{url}"
    return url


def _extract_domain(normalized_url: str) -> str:
    parsed = urlparse(normalized_url)
    return parsed.netloc.split(":")[0]


def _is_ip(domain: str) -> bool:
    try:
        ipaddress.ip_address(domain)
        return True
    except ValueError:
        return False


def _domain_similarity(domain: str, reference_domains: Iterable[str] = TOP_DOMAINS) -> float:
    if not domain:
        return 0.0
    return max(SequenceMatcher(None, domain, ref).ratio() for ref in reference_domains)


def extract_url_features(url: str) -> URLFeatures:
    normalized = normalize_url(url)
    parsed = urlparse(normalized)
    domain = _extract_domain(normalized)
    path = parsed.path or ""
    query = parsed.query or ""

    suspicious_word_count = sum(word in normalized for word in SUSPICIOUS_WORDS)
    special_char_count = sum(normalized.count(ch) for ch in ("@", "-", "_", "?", "="))
    similarity = _domain_similarity(domain)

    return URLFeatures(
        url_length=len(normalized),
        dot_count=normalized.count("."),
        special_char_count=special_char_count,
        has_ip=int(_is_ip(domain)),
        has_https=int(parsed.scheme == "https"),
        domain_length=len(domain),
        suspicious_word_count=suspicious_word_count,
        subdomain_count=max(domain.count(".") - 1, 0),
        is_shortened=int(any(shortener in domain for shortener in SHORTENERS)),
        directory_count=path.count("/"),
        query_length=len(query),
        digit_count=sum(ch.isdigit() for ch in normalized),
        hyphen_count=normalized.count("-"),
        max_popular_domain_similarity=similarity,
        typosquat_flag=int(0.75 <= similarity < 1.0),
    )
