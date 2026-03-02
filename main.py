"""Streamlit Pakistan news signal tool using Google Trends and GDELT."""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse
from pathlib import Path
import shutil
import subprocess
import tempfile

import pandas as pd
import requests
import streamlit as st

try:
    from pytrends.request import TrendReq

    PYTRENDS_AVAILABLE = True
except Exception:
    PYTRENDS_AVAILABLE = False

try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

try:
    import yt_dlp

    YTDLP_AVAILABLE = True
except Exception:
    YTDLP_AVAILABLE = False

GDELT_ENDPOINT = "https://api.gdeltproject.org/api/v2/doc/doc"
ELEVEN_BASE = "https://api.elevenlabs.io/v1"

CATEGORY_CODES = {
    "All": 0,
    "News": 16,
    "Politics": 396,
    "Business": 12,
    "Sports": 20,
}

PROVINCES = [
    "Punjab",
    "Sindh",
    "Khyber Pakhtunkhwa",
    "Balochistan",
]

PROVINCE_ALIASES = {
    "punjab": "Punjab",
    "sindh": "Sindh",
    "khyber pakhtunkhwa": "Khyber Pakhtunkhwa",
    "khyber pakhtunkhwa province": "Khyber Pakhtunkhwa",
    "kpk": "Khyber Pakhtunkhwa",
    "balochistan": "Balochistan",
    "baluchistan": "Balochistan",
}

WEBSEARCH_CATEGORY_HINTS = {
    "All": "latest news",
    "News": "latest news",
    "Politics": "politics and government",
    "Business": "business, economy, and markets",
    "Sports": "sports",
}


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def safe_float(value: object) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


from datetime import datetime, timezone
from typing import Optional

def parse_seendate(value: str) -> Optional[datetime]:
    if not value:
        return None

    value = value.strip()

    # Common DOC API format: 20260107T153000Z
    try:
        return datetime.strptime(value, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    except Exception:
        pass

    # Older/other format you originally assumed: 20260107153000
    try:
        return datetime.strptime(value, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    except Exception:
        return None



def domain_from_url(url: str) -> str:
    try:
        netloc = urlparse(url).netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc
    except Exception:
        return ""


@st.cache_data(ttl=900, show_spinner=False)
def fetch_trends(
    keyword: str, category_code: int
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict, Dict]:
    cleaned = keyword.strip()
    if not cleaned:
        return pd.DataFrame(), pd.DataFrame(), {}, {"error": "Keyword is empty."}

    pytrends = TrendReq(hl="en-US", tz=300, timeout=(10, 25))

    payloads = [
        {"cat": category_code, "timeframe": "now 14-d"},
        {"cat": 0, "timeframe": "now 14-d"},
        {"cat": category_code, "timeframe": "now 7-d"},
        {"cat": 0, "timeframe": "now 7-d"},
        {"cat": category_code, "timeframe": "today 1-m"},
        {"cat": 0, "timeframe": "today 1-m"},
    ]

    last_error: Optional[Exception] = None
    used_payload: Optional[Dict] = None

    for payload in payloads:
        try:
            for attempt in range(2):
                try:
                    pytrends.build_payload(
                        [cleaned],
                        cat=payload["cat"],
                        timeframe=payload["timeframe"],
                        geo="PK",
                        gprop="",
                    )
                    used_payload = payload
                    break
                except Exception as exc:
                    last_error = exc
                    if "429" in str(exc):
                        time.sleep(2 + attempt * 3)
                        continue
                    raise
            if used_payload is not None:
                break
        except Exception as exc:
            last_error = exc
            continue

    if used_payload is None:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            {},
            {"error": f"Google Trends request failed: {last_error}"},
        )

    interest = pytrends.interest_over_time()
    interest = interest.drop(columns=["isPartial"], errors="ignore")
    try:
        region = pytrends.interest_by_region(
            resolution="REGION", inc_low_vol=True, inc_geo_code=False
        )
    except Exception:
        region = pd.DataFrame()
    try:
        related = pytrends.related_queries()
    except Exception:
        related = {}
    meta = {
        "warning": None,
        "payload": used_payload,
    }
    if used_payload["cat"] != category_code or used_payload["timeframe"] != "now 14-d":
        meta["warning"] = (
            "Google Trends fell back to a broader query. "
            "Try a different keyword or category."
        )
    return interest, region, related, meta


def compute_interest_metrics(interest: pd.DataFrame, keyword: str) -> Dict:
    if interest.empty or keyword not in interest.columns:
        return {
            "current_interest": None,
            "week_over_week": None,
            "trend_spike": 0.0,
        }

    series = interest[keyword].dropna()
    if series.empty:
        return {
            "current_interest": None,
            "week_over_week": None,
            "trend_spike": 0.0,
        }

    current_interest = float(series.iloc[-1])
    now_ts = series.index.max()
    last_24h = series[series.index >= now_ts - timedelta(hours=24)]
    prev_week_24h = series[
        (series.index >= now_ts - timedelta(days=7, hours=24))
        & (series.index < now_ts - timedelta(days=7))
    ]

    if last_24h.empty or prev_week_24h.empty:
        week_over_week = None
        trend_spike = 0.0
    else:
        last_avg = float(last_24h.mean())
        prev_avg = float(prev_week_24h.mean())
        if prev_avg <= 0:
            week_over_week = None
            trend_spike = 0.0
        else:
            week_over_week = ((last_avg - prev_avg) / prev_avg) * 100.0
            trend_spike = max(0.0, week_over_week)

    return {
        "current_interest": current_interest,
        "week_over_week": week_over_week,
        "trend_spike": trend_spike,
    }


def compute_breakout_flag(related: Dict, keyword: str) -> Dict:
    rising = None
    if related and keyword in related:
        rising = related[keyword].get("rising")
    if rising is None or getattr(rising, "empty", True):
        return {"breakout": False, "max_rise": None, "rising": pd.DataFrame()}

    breakout = False
    max_rise: Optional[float] = None
    for value in rising["value"].tolist():
        if isinstance(value, str) and value.strip().lower() == "breakout":
            breakout = True
            max_rise = None
            break
        numeric = safe_float(value)
        if numeric is not None:
            max_rise = numeric if max_rise is None else max(max_rise, numeric)
            if numeric >= 5000:
                breakout = True

    return {"breakout": breakout, "max_rise": max_rise, "rising": rising}


def compute_region_metrics(region: pd.DataFrame, keyword: str, threshold: float) -> Dict:
    if region.empty or keyword not in region.columns:
        return {
            "province_values": {},
            "province_count": 0,
            "concentration_ratio": None,
        }

    values: Dict[str, float] = {}
    for idx, row in region.iterrows():
        key = PROVINCE_ALIASES.get(str(idx).strip().lower())
        if key in PROVINCES:
            values[key] = float(row[keyword])

    province_count = sum(1 for v in values.values() if v >= threshold)
    total = sum(values.values())
    concentration_ratio = None
    if total > 0:
        concentration_ratio = max(values.values(), default=0.0) / total

    return {
        "province_values": values,
        "province_count": province_count,
        "concentration_ratio": concentration_ratio,
    }


import random
import re
import time
import requests

def build_gdelt_query(keyword: str, pakistan_only: bool) -> str:
    """
    Build a GDELT DOC API query string that avoids invalid parentheses.
    - If the keyword looks like a boolean query (OR/AND/NOT), leave it mostly as-is.
    - If it's a plain term/phrase, DO NOT wrap in parentheses.
    - Quote phrases with spaces.
    """
    q = (keyword or "").strip()
    if not q:
        return "sourceCountry:PK" if pakistan_only else ""

    # If user already wrote a boolean query, don't try to "help" too much.
    looks_boolean = bool(re.search(r"\b(OR|AND|NOT)\b", q, flags=re.IGNORECASE))

    # Remove outer parentheses that cause GDELT error when there's no OR inside
    # e.g. "(basant)" -> "basant"
    if q.startswith("(") and q.endswith(")") and not looks_boolean:
        q = q[1:-1].strip()

    # For simple phrases, quote them (GDELT accepts quoted phrases)
    if not looks_boolean and re.search(r"\s", q):
        q = f"\"{q}\""

    # Apply Pakistan filter
    if pakistan_only:
        q = f"{q} sourceCountry:PK"

    return q


def gdelt_query(
    query: str,
    start_dt: datetime,
    end_dt: datetime,
    pakistan_only: bool,
    maxrecords: int = 50,
    session: Optional[requests.Session] = None,
    max_retries: int = 5,
):
    q = build_gdelt_query(query, pakistan_only)

    params = {
        "query": q,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": maxrecords,
        "startdatetime": start_dt.strftime("%Y%m%d%H%M%S"),
        "enddatetime": end_dt.strftime("%Y%m%d%H%M%S"),
        "sort": "DateDesc",
    }

    sess = session or requests.Session()
    headers = {
        # Helps some CDNs / APIs treat you as a "real" client
        "User-Agent": "pk-news-signal/1.0 (+streamlit; contact: you@example.com)",
        "Accept": "application/json",
    }

    last_exc = None
    for attempt in range(max_retries):
        try:
            r = sess.get(GDELT_ENDPOINT, params=params, headers=headers, timeout=30)

            # Handle rate limiting explicitly
            if r.status_code == 429:
                retry_after = r.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    sleep_s = int(retry_after)
                else:
                    # exponential backoff + jitter
                    sleep_s = min(60, (2 ** attempt)) + random.uniform(0, 1.5)
                time.sleep(sleep_s)
                continue

            r.raise_for_status()
            data = r.json()

            # Surface API-level query errors (400s sometimes still return JSON)
            if isinstance(data, dict) and data.get("status") == "error":
                raise RuntimeError(f"GDELT error: {data.get('message')} (query={q})")

            return data.get("articles", [])

        except Exception as exc:
            last_exc = exc
            # Backoff on transient failures too
            time.sleep(min(10, (2 ** attempt)) + random.uniform(0, 0.5))

    raise RuntimeError(f"GDELT request failed after retries: {last_exc} (query={q})")




@st.cache_data(ttl=900, show_spinner=False)
def fetch_gdelt_articles(keyword: str, hours: int, pakistan_only: bool) -> Tuple[List[Dict], Optional[str]]:
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(hours=hours)
    try:
        return gdelt_query(keyword, start_dt, end_dt, pakistan_only), None
    except Exception as e:
        return [], repr(e)


def filter_articles_by_hours(articles: List[Dict], hours: int) -> List[Dict]:
    if not articles:
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    filtered = []
    for article in articles:
        seen = parse_seendate(article.get("seendate", ""))
        if seen and seen >= cutoff:
            filtered.append(article)
    return filtered



def compute_gdelt_metrics(articles_24h: List[Dict], articles_6h: List[Dict]) -> Dict:
    count_24h = len(articles_24h)
    count_6h = len(articles_6h)
    velocity = count_6h / 6.0

    domains = []
    tones = []
    seen_times = []
    pk_domains = 0

    for article in articles_24h:
        domain = article.get("domain") or domain_from_url(article.get("url", ""))
        if domain:
            domains.append(domain)
            if domain.endswith(".pk"):
                pk_domains += 1
        tone = safe_float(article.get("tone"))
        if tone is not None:
            tones.append(tone)
        seen = parse_seendate(article.get("seendate", ""))
        if seen:
            seen_times.append(seen)

    unique_domains = len(set(domains))
    avg_tone = float(sum(tones) / len(tones)) if tones else None

    earliest = min(seen_times) if seen_times else None
    now = datetime.now(timezone.utc)
    hours_since_first = None
    if earliest:
        hours_since_first = (now - earliest).total_seconds() / 3600.0

    local_domain_share = None
    if domains:
        local_domain_share = pk_domains / len(domains)

    return {
        "count_24h": count_24h,
        "count_6h": count_6h,
        "velocity": velocity,
        "unique_domains": unique_domains,
        "avg_tone": avg_tone,
        "earliest": earliest,
        "hours_since_first": hours_since_first,
        "local_domain_share": local_domain_share,
    }


def compute_score(
    trend_spike: float,
    velocity: float,
    unique_domains: int,
    province_count: int,
    hours_since_first: Optional[float],
    trend_cap: float,
    velocity_cap: float,
    diversity_cap: float,
    freshness_hours: float,
) -> Dict:
    trend_norm = clamp01(trend_spike / trend_cap) if trend_cap > 0 else 0.0
    velocity_norm = clamp01(velocity / velocity_cap) if velocity_cap > 0 else 0.0
    diversity_norm = clamp01(unique_domains / diversity_cap) if diversity_cap > 0 else 0.0
    province_norm = clamp01(province_count / 4.0)
    if hours_since_first is None or freshness_hours <= 0:
        freshness_norm = 0.0
    else:
        freshness_norm = clamp01(1.0 - (hours_since_first / freshness_hours))

    score = (
        0.35 * trend_norm
        + 0.25 * velocity_norm
        + 0.15 * diversity_norm
        + 0.15 * province_norm
        + 0.10 * freshness_norm
    )

    return {
        "score": score,
        "trend_norm": trend_norm,
        "velocity_norm": velocity_norm,
        "diversity_norm": diversity_norm,
        "province_norm": province_norm,
        "freshness_norm": freshness_norm,
    }


def rank_articles(
    articles: List[Dict],
    score_metrics: Dict[str, float],
    freshness_hours: float,
) -> List[Dict]:
    if not articles:
        return []

    domains = []
    for article in articles:
        domain = article.get("domain") or domain_from_url(article.get("url", ""))
        if domain:
            domains.append(domain)
    domain_counts = Counter(domains)

    now = datetime.now(timezone.utc)
    ranked = []
    for article in articles:
        domain = article.get("domain") or domain_from_url(article.get("url", ""))
        seen = parse_seendate(article.get("seendate", ""))
        hours_since = None
        if seen:
            hours_since = (now - seen).total_seconds() / 3600.0

        if freshness_hours > 0 and hours_since is not None:
            freshness_norm = clamp01(1.0 - (hours_since / freshness_hours))
        else:
            freshness_norm = 0.0

        if domain and domain in domain_counts and domain_counts[domain] > 0:
            diversity_component = clamp01(1.0 / domain_counts[domain])
        else:
            diversity_component = 0.0

        score = (
            0.35 * score_metrics.get("trend_norm", 0.0)
            + 0.25 * score_metrics.get("velocity_norm", 0.0)
            + 0.15 * diversity_component
            + 0.15 * score_metrics.get("province_norm", 0.0)
            + 0.10 * freshness_norm
        )

        ranked.append(
            {
                "score": score,
                "title": article.get("title", ""),
                "url": article.get("url", ""),
                "domain": domain,
                "seen": seen,
                "tone": article.get("tone"),
                "freshness_norm": freshness_norm,
                "diversity_component": diversity_component,
            }
        )

    ranked.sort(key=lambda item: (item["score"], item["seen"] or datetime.min), reverse=True)
    return ranked


def run_pipeline(
    keyword: str,
    category_code: int,
    pakistan_only: bool,
    province_threshold: float,
    trend_cap: float,
    velocity_cap: float,
    diversity_cap: float,
    freshness_hours: float,
) -> Dict:
    interest, region, related, trends_meta = fetch_trends(keyword, category_code)
    interest_metrics = compute_interest_metrics(interest, keyword)
    breakout_metrics = compute_breakout_flag(related, keyword)
    region_metrics = compute_region_metrics(region, keyword, province_threshold)

    articles_24h, gdelt_error = fetch_gdelt_articles(keyword, 24, pakistan_only)
    articles_6h = filter_articles_by_hours(articles_24h, 6)
    gdelt_metrics = compute_gdelt_metrics(articles_24h, articles_6h)

    score_metrics = compute_score(
        interest_metrics["trend_spike"] or 0.0,
        gdelt_metrics["velocity"],
        gdelt_metrics["unique_domains"],
        region_metrics["province_count"],
        gdelt_metrics["hours_since_first"],
        trend_cap,
        velocity_cap,
        diversity_cap,
        freshness_hours,
    )

    ranked_articles = rank_articles(articles_24h, score_metrics, freshness_hours)

    return {
        "keyword": keyword,
        "trends_meta": trends_meta,
        "interest_metrics": interest_metrics,
        "breakout_metrics": breakout_metrics,
        "region_metrics": region_metrics,
        "articles_24h": articles_24h,
        "articles_6h": articles_6h,
        "gdelt_metrics": gdelt_metrics,
        "gdelt_error": gdelt_error,
        "score_metrics": score_metrics,
        "ranked_articles": ranked_articles,
    }


def build_websearch_prompt(
    keyword: str,
    category_label: str,
    max_items: int,
    pakistan_only: bool,
    include_x: bool,
) -> str:
    category_hint = WEBSEARCH_CATEGORY_HINTS.get(category_label, "latest news")
    source_hint = ""
    if pakistan_only:
        source_hint = "Prefer Pakistan-based outlets and .pk domains. "
    x_hint = ""
    if include_x:
        x_hint = (
            "Also search X (x.com) for relevant posts from credible accounts "
            "(e.g., major journalists, official government accounts, regulators). "
            "Include up to 2 x.com links if available. "
        )
    return (
        "You are a news researcher. Use live web search.\n"
        f"Find up to {max_items} recent Pakistan-focused {category_hint} articles about "
        f"\"{keyword}\".\n"
        f"{source_hint}{x_hint}"
        "Return ONLY a JSON array. Each item must include:\n"
        "- title\n"
        "- source (publisher)\n"
        "- date (ISO yyyy-mm-dd if available)\n"
        "- url\n"
        "- summary (1 sentence)\n"
        "Only include items that have a valid url."
    )



def parse_articles_from_text(text: str) -> Tuple[List[Dict], Optional[str]]:
    if not text:
        return [], "Empty response from web search."
    payload = None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            try:
                payload = json.loads(text[start : end + 1])
            except json.JSONDecodeError as exc:
                return [], f"Could not parse JSON: {exc}"
        else:
            return [], "Could not find JSON array in response."

    if isinstance(payload, dict) and "articles" in payload:
        payload = payload["articles"]

    if not isinstance(payload, list):
        return [], "Web search response is not a list."

    articles = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url", "")).strip()
        title = str(item.get("title", "")).strip()
        if not url or not title:
            continue
        articles.append(
            {
                "title": title,
                "source": str(item.get("source", "")).strip() or "Unknown",
                "date": str(item.get("date", "")).strip(),
                "url": url,
                "summary": str(item.get("summary", "")).strip(),
            }
        )
    return articles, None


def resolve_api_key(user_key: str) -> Optional[str]:
    if user_key:
        return user_key.strip()
    try:
        # Streamlit secrets support multiple shapes; accept common variants.
        for key_name in ("OPENAI_API_KEY", "openai_api_key"):
            secret_key = st.secrets.get(key_name)
            if secret_key:
                return str(secret_key).strip()
        openai_block = st.secrets.get("openai")
        if isinstance(openai_block, dict):
            secret_key = openai_block.get("api_key") or openai_block.get("OPENAI_API_KEY")
            if secret_key:
                return str(secret_key).strip()
    except Exception:
        pass
    return None


def web_cache_key(
    keyword: str,
    category_label: str,
    pakistan_only: bool,
    model_name: str,
    include_x: bool,
    max_items: int,
) -> str:
    return (
        f"{keyword.strip().lower()}|{category_label}|{int(pakistan_only)}|"
        f"{model_name.strip()}|{int(include_x)}|{int(max_items)}"
    )


# -------------------------------
# ElevenLabs TTS helpers
# -------------------------------
def resolve_elevenlabs_api_key(user_key: str) -> Optional[str]:
    if user_key:
        return user_key.strip()
    try:
        secret_key = st.secrets.get("ELEVENLABS_API_KEY")
        if secret_key:
            return str(secret_key).strip()
    except Exception:
        pass
    env_key = os.getenv("ELEVENLABS_API_KEY")
    if env_key:
        return env_key.strip()
    return None


@st.cache_data(show_spinner=False)
def eleven_list_voices(api_key: str) -> List[Dict[str, Any]]:
    if not api_key:
        return []
    headers = {"xi-api-key": api_key}
    try:
        r = requests.get(f"{ELEVEN_BASE}/voices", headers=headers, timeout=20)
        r.raise_for_status()
        data = r.json() or {}
        voices = data.get("voices", []) or []
        return [
            {
                "id": v.get("voice_id"),
                "name": v.get("name"),
                "category": v.get("category"),
                "labels": v.get("labels") or {},
                "description": v.get("description") or "",
            }
            for v in voices
            if v.get("voice_id") and v.get("name")
        ]
    except Exception as e:
        st.warning(f"Could not load ElevenLabs voices: {e}")
        return []


def eleven_tts(
    api_key: str,
    voice_id: str,
    text: str,
    *,
    model_id: str = "eleven_multilingual_v2",
    stability: float | None = None,
    similarity_boost: float | None = None,
    style: float | None = None,
    use_speaker_boost: bool | None = None,
) -> bytes:
    if not api_key:
        raise RuntimeError("ELEVENLABS_API_KEY not set.")
    if not voice_id:
        raise RuntimeError("Please select a voice.")

    headers = {
        "xi-api-key": api_key,
        "accept": "audio/mpeg",
        "content-type": "application/json",
    }

    payload: Dict[str, Any] = {
        "text": text,
        "model_id": model_id,
    }
    voice_settings: Dict[str, Any] = {}
    if stability is not None:
        voice_settings["stability"] = float(stability)
    if similarity_boost is not None:
        voice_settings["similarity_boost"] = float(similarity_boost)
    if style is not None:
        voice_settings["style"] = float(style)
    if use_speaker_boost is not None:
        voice_settings["use_speaker_boost"] = bool(use_speaker_boost)
    if voice_settings:
        payload["voice_settings"] = voice_settings

    url = f"{ELEVEN_BASE}/text-to-speech/{voice_id}"
    r = requests.post(url, headers=headers, json=payload, timeout=120)
    r.raise_for_status()
    return r.content


def extract_response_text(response: object) -> str:
    text = getattr(response, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text
    try:
        chunks = []
        for item in getattr(response, "output", []) or []:
            for content in getattr(item, "content", []) or []:
                if getattr(content, "type", None) == "output_text":
                    chunk = getattr(content, "text", "")
                    if chunk:
                        chunks.append(chunk)
        return "\n".join(chunks)
    except Exception:
        return ""


def fetch_websearch_articles(
    keyword: str,
    category_label: str,
    model_name: str,
    api_key: str,
    max_items: int,
    pakistan_only: bool,
    include_x: bool,
) -> Tuple[List[Dict], Optional[str]]:
    if not OPENAI_AVAILABLE:
        return [], "openai is not installed. Run: pip install openai"
    resolved_key = resolve_api_key(api_key)
    if not resolved_key:
        return [], "OpenAI API key not found. Add OPENAI_API_KEY to Streamlit secrets (.streamlit/secrets.toml)."

    client = OpenAI(api_key=resolved_key)
    prompt = build_websearch_prompt(
        keyword, category_label, max_items, pakistan_only, include_x
    )
    models_to_try = [model_name, "gpt-5"]
    last_error = None

    for model in models_to_try:
        if not model:
            continue
        try:
            response = client.responses.create(
                model=model,
                tools=[{"type": "web_search"}],
                input=prompt,
            )
            text = extract_response_text(response)
            articles, parse_error = parse_articles_from_text(text)
            if parse_error:
                return [], parse_error
            return articles, None
        except Exception as exc:
            last_error = str(exc)
            continue

    return [], last_error or "Unknown OpenAI error."


NEWS_INTERPRETER_PROMPT = (
    "You are a news interpreter, not a news reader.\n"
    "You will be given:\n"
    "1. A one-shot example script covering 3 news items.\n"
    "2. A new list of topics to cover this week.\n"
    "Your task is to write a new 3-item news interpreter script that strictly follows "
    "the tone, structure, pacing, and analytical style of the example.\n\n"
    "Core Editorial Philosophy\n"
    "Do NOT summarize headlines.\n"
    "Do NOT sound like a bulletin reader.\n"
    "Do NOT give opinions.\n"
    "You are interpreting momentum.\n"
    "You are answering one core question repeatedly:\n"
    "\"What actually moved the Pakistani economy this week -- and why should I care?\"\n"
    "Every story must:\n"
    "* Reflect real traction (being discussed, reacted to, referenced)\n"
    "* Appear across credible sources (e.g., Business Recorder, major journalists, policy announcements)\n"
    "* Have economic spillover effects\n"
    "This is momentum-led curation, not commentary.\n\n"
    "Required Structure (Must Match Example)\n"
    "For each of the 3 stories:\n"
    "1. Clear Opening Hook\n"
    "2. What Happened (Tight Context)\n"
    "3. Why It Matters (Economic Spillover)\n"
    "4. Momentum Framing\n"
    "5. Concise Closing Line\n"
    
    

    "Length should approximately match the example script.\n"
)

MOMENTUM_FILTER_PROMPT = (
    "You are a momentum filter for the Pakistani economy.\n\n"
    "Your job is to surface only news that reflects real economic movement, not general headlines.\n"
    "Get 10 news items.\n"
    "You are identifying developments that answer:\n"
    "\"What actually moved the Pakistani economy this week?\"\n\n"
    "Visible Traction\n"
    "The development must:\n"
    "- Be discussed or referenced repeatedly\n"
    "- Trigger market movement\n"
    "- Cause analyst reactions\n"
    "- Influence pricing, yields, currency, or equity sentiment\n"
    "- Lead to follow-up statements or responses\n"
    "- Avoid isolated announcements with no reaction\n\n"
    "Economic Spillover\n"
    "The news must affect real economic flows, such as:\n"
    "- Power tariffs -> industrial costs\n"
    "- Circular debt -> energy stability\n"
    "- Interest rates -> borrowing, consumption, investment\n"
    "- Oil prices -> inflation, current account\n"
    "- IMF developments -> FX reserves, policy tightening\n"
    "- PSX moves -> liquidity and sentiment\n"
    "- Auto sales -> consumer demand\n"
    "- Export or import shifts -> trade balance\n"
    "- Commodity prices -> input costs\n"
    "- Tax or fiscal changes -> corporate or household cash flow\n"
    "If there is no second-order impact, ignore it.\n\n"
    "Prioritize These Buckets\n"
    "- Power & Energy\n"
    "- Interest Rates / Monetary Policy\n"
    "- Oil & Commodities\n"
    "- Markets / PSX / Listed Companies\n"
    "- Industry & Mobility (Auto, Transport, Exports)\n"
    "- Macro / IMF / World Bank (only when economically material)\n\n"
    "Explicitly Ignore\n"
    "- Political statements without economic action\n"
    "- Opinion columns\n"
    "- One-off corporate PR\n"
    "- Minor company earnings without broader implications\n"
    "- Social, crime, or diplomatic news\n"
    "- Speculation without policy or market movement\n\n"
    "Return ONLY a JSON array of exactly 10 items. Each item must include:\n"
    "- title\n"
    "- source (publisher)\n"
    "- date (ISO yyyy-mm-dd if available)\n"
    "- url\n"
    "- summary (1 sentence explaining the economic spillover)\n"
    "Only include items with a valid url.\n"
)


def generate_interpreter_script(
    *,
    example_script: str,
    topics: List[str],
    sources_by_topic: Dict[str, List[Dict]],
    extra_context: str = "",
    expert_context: str = "",
    model_name: str,
    api_key: str,
) -> Tuple[str, Optional[str]]:
    if not OPENAI_AVAILABLE:
        return "", "openai is not installed. Run: pip install openai"
    resolved_key = resolve_api_key(api_key)
    if not resolved_key:
        return "", "OpenAI API key not found. Add OPENAI_API_KEY to Streamlit secrets (.streamlit/secrets.toml)."
    if not example_script.strip():
        return "", "Example script is empty."
    if len(topics) != 3:
        return "", "Please provide exactly 3 topics."

    client = OpenAI(api_key=resolved_key)

    # Keep evidence compact: a few links per topic.
    evidence_lines = []
    for idx, topic in enumerate(topics, start=1):
        evidence_lines.append(f"TOPIC {idx}: {topic}")
        items = sources_by_topic.get(topic, [])[:6]
        if not items:
            evidence_lines.append("Sources: (none)")
        else:
            evidence_lines.append("Sources:")
            for j, item in enumerate(items, start=1):
                title = str(item.get("title", "")).strip()
                source = str(item.get("source", "")).strip()
                date = str(item.get("date", "")).strip()
                url = str(item.get("url", "")).strip()
                summary = str(item.get("summary", "")).strip()
                evidence_lines.append(
                    f"{j}. {title} | {source} | {date} | {url} | {summary}"
                )
        evidence_lines.append("")

    prompt = (
        f"{NEWS_INTERPRETER_PROMPT}\n\n"
        "ONE-SHOT EXAMPLE SCRIPT:\n"
        f"{example_script.strip()}\n\n"
        "NEW TOPICS (write exactly 3 stories in this order):\n"
        f"1) {topics[0]}\n2) {topics[1]}\n3) {topics[2]}\n\n"
        "EVIDENCE (use this to stay anchored in real traction; do not cite URLs aloud):\n"
        + "\n".join(evidence_lines).strip()
    )
    if extra_context.strip():
        prompt += (
            "\n\nADDITIONAL CONTEXT (from provided video links; use if relevant; do not cite URLs aloud):\n"
            + extra_context.strip()
        )
    if expert_context.strip():
        prompt += (
            "\n\nEXPERT INTERVIEWS (use relevant quotes sparingly; weave into the script naturally):\n"
            + expert_context.strip()
        )

    try:
        response = client.responses.create(model=model_name, input=prompt)
        text = extract_response_text(response).strip()
        if not text:
            return "", "Empty response from model."
        return text, None
    except Exception as exc:
        return "", str(exc)


def download_video_audio(
    url: str,
    workdir: Path,
    progress_cb: Optional[callable] = None,
) -> Tuple[Optional[Path], Optional[str]]:
    if not YTDLP_AVAILABLE:
        return None, "yt-dlp is not installed. Run: pip install yt-dlp"

    outtmpl = str(workdir / "download.%(ext)s")

    def hook(d):
        if progress_cb is None:
            return
        if d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            downloaded = d.get("downloaded_bytes")
            if total and downloaded:
                progress_cb(min(downloaded / total, 1.0))
        elif d.get("status") == "finished":
            progress_cb(1.0)

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "progress_hooks": [hook],
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
        return Path(filename), None
    except Exception as exc:
        return None, str(exc)


def convert_to_mp3(input_path: Path, output_path: Path) -> Tuple[Optional[Path], Optional[str]]:
    if not input_path.exists():
        return None, "Downloaded file not found."

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        try:
            import imageio_ffmpeg  # type: ignore

            ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg_path = None
    if not ffmpeg_path:
        return None, "ffmpeg is not available. Install ffmpeg to convert audio."

    cmd = [
        ffmpeg_path,
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-acodec",
        "libmp3lame",
        "-ar",
        "44100",
        "-ac",
        "1",
        "-q:a",
        "2",
        str(output_path),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return output_path, None
    except Exception as exc:
        return None, str(exc)


def transcribe_audio(
    mp3_path: Path,
    api_key: str,
    model_name: str = "gpt-4o-transcribe",
) -> Tuple[Optional[str], Optional[str]]:
    if not OPENAI_AVAILABLE:
        return None, "openai is not installed. Run: pip install openai"
    resolved_key = resolve_api_key(api_key)
    if not resolved_key:
        return None, "OpenAI API key not found. Add OPENAI_API_KEY to Streamlit secrets (.streamlit/secrets.toml)."
    if not mp3_path.exists():
        return None, "MP3 file not found."

    size_mb = mp3_path.stat().st_size / (1024 * 1024)
    if size_mb > 25:
        return None, "Audio file exceeds 25MB. Please provide a shorter clip."

    client = OpenAI(api_key=resolved_key)
    try:
        with mp3_path.open("rb") as f:
            resp = client.audio.transcriptions.create(model=model_name, file=f)
        text = getattr(resp, "text", None)
        if not text:
            text = getattr(resp, "transcript", None)
        if not text:
            return None, "Empty transcription."
        return str(text).strip(), None
    except Exception as exc:
        return None, str(exc)


def fetch_momentum_news(
    *,
    model_name: str,
    api_key: str,
) -> Tuple[List[Dict], Optional[str]]:
    if not OPENAI_AVAILABLE:
        return [], "openai is not installed. Run: pip install openai"
    resolved_key = resolve_api_key(api_key)
    if not resolved_key:
        return [], "OpenAI API key not found. Add OPENAI_API_KEY to Streamlit secrets (.streamlit/secrets.toml)."

    client = OpenAI(api_key=resolved_key)
    models_to_try = [model_name, "gpt-5"]
    last_error = None
    for model in models_to_try:
        if not model:
            continue
        try:
            response = client.responses.create(
                model=model,
                tools=[{"type": "web_search"}],
                input=MOMENTUM_FILTER_PROMPT,
            )
            text = extract_response_text(response)
            items, parse_error = parse_articles_from_text(text)
            if parse_error:
                return [], parse_error
            return items[:10], None
        except Exception as exc:
            last_error = str(exc)
            continue
    return [], last_error or "Unknown OpenAI error."


SCRIPT_EDITOR_PROMPT = (
    "You are editing an existing 3-item Pakistan economy news interpreter script.\n"
    "You are NOT writing from scratch unless asked.\n"
    "Preserve the original tone, structure, pacing, and analytical style.\n"
    "Do NOT add bullet points, headings, or section labels.\n"
    "Maintain clear separation between the 3 stories.\n"
    "Do NOT add opinions. Stay momentum-led and spillover-focused.\n"
    "If the request is ambiguous, ask one short clarifying question, otherwise apply the change.\n"
    "Return ONLY JSON with keys:\n"
    "- assistant_message: string (brief, conversational)\n"
    "- revised_script: string (the full updated script)\n"
)


def parse_json_object(text: str) -> Tuple[Optional[Dict], Optional[str]]:
    if not text:
        return None, "Empty response."
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj, None
        return None, "Response is not a JSON object."
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                obj = json.loads(text[start : end + 1])
                if isinstance(obj, dict):
                    return obj, None
                return None, "Response is not a JSON object."
            except json.JSONDecodeError as exc:
                return None, f"Could not parse JSON: {exc}"
        return None, "Could not find JSON object in response."


def revise_interpreter_script(
    *,
    current_script: str,
    user_request: str,
    api_key: str,
    model_name: str = "gpt-5.2",
) -> Tuple[Optional[Dict], Optional[str]]:
    if not OPENAI_AVAILABLE:
        return None, "openai is not installed. Run: pip install openai"
    resolved_key = resolve_api_key(api_key)
    if not resolved_key:
        return None, "OpenAI API key not found. Add OPENAI_API_KEY to Streamlit secrets (.streamlit/secrets.toml)."
    if not current_script.strip():
        return None, "Current script is empty."
    if not user_request.strip():
        return None, "Change request is empty."

    client = OpenAI(api_key=resolved_key)
    prompt = (
        f"{SCRIPT_EDITOR_PROMPT}\n\n"
        "CURRENT SCRIPT:\n"
        f"{current_script.strip()}\n\n"
        "USER REQUEST:\n"
        f"{user_request.strip()}\n"
    )

    try:
        response = client.responses.create(model=model_name, input=prompt)
        text = extract_response_text(response)
        obj, err = parse_json_object(text)
        if err:
            return None, err
        if "assistant_message" not in obj or "revised_script" not in obj:
            return None, "Missing keys in JSON response."
        return obj, None
    except Exception as exc:
        return None, str(exc)


st.set_page_config(page_title="Pakistan News Trends", layout="wide")

st.title("Pakistan News Signal Tracker")

if "web_articles_cache" not in st.session_state:
    st.session_state["web_articles_cache"] = {}
if "web_error_cache" not in st.session_state:
    st.session_state["web_error_cache"] = {}
if "signal_result" not in st.session_state:
    st.session_state["signal_result"] = None
if "script_sources" not in st.session_state:
    st.session_state["script_sources"] = {}
if "script_source_errors" not in st.session_state:
    st.session_state["script_source_errors"] = {}
if "generated_script" not in st.session_state:
    st.session_state["generated_script"] = ""
if "momentum_news" not in st.session_state:
    st.session_state["momentum_news"] = []
if "momentum_news_error" not in st.session_state:
    st.session_state["momentum_news_error"] = None
if "script_draft" not in st.session_state:
    st.session_state["script_draft"] = ""
if "script_versions" not in st.session_state:
    st.session_state["script_versions"] = []
if "script_chat" not in st.session_state:
    st.session_state["script_chat"] = []
if "script_edit_error" not in st.session_state:
    st.session_state["script_edit_error"] = None
if "voiceover_audio" not in st.session_state:
    st.session_state["voiceover_audio"] = None
if "voiceover_error" not in st.session_state:
    st.session_state["voiceover_error"] = None
if "video_items" not in st.session_state:
    st.session_state["video_items"] = []
if "video_errors" not in st.session_state:
    st.session_state["video_errors"] = {}
if "video_context_text" not in st.session_state:
    st.session_state["video_context_text"] = ""
if "video_transcripts" not in st.session_state:
    st.session_state["video_transcripts"] = []
if "expert_context_text" not in st.session_state:
    st.session_state["expert_context_text"] = ""

if not PYTRENDS_AVAILABLE:
    st.error("pytrends is not installed. Run: pip install pytrends")
    st.stop()
if not OPENAI_AVAILABLE:
    st.warning("openai is not installed. Web search will be disabled.")

with st.sidebar:
    st.header("Inputs")
    keyword = st.text_input("Keyword", value="elections")
    category_label = st.selectbox("Category", list(CATEGORY_CODES.keys()), index=1)
    pakistan_only = st.toggle("Pakistan sources only", value=True)
    run = st.button("Run", type="primary")

    st.divider()
    st.subheader("Scoring controls")
    province_threshold = st.slider("Province threshold", 0, 100, 20, 5)
    trend_cap = st.number_input("Trend growth cap (%)", value=200.0, min_value=10.0)
    velocity_cap = st.number_input("Velocity cap (articles/hr)", value=20.0, min_value=1.0)
    diversity_cap = st.number_input("Diversity cap (unique domains)", value=15.0, min_value=1.0)
    freshness_hours = st.number_input("Freshness window (hours)", value=12.0, min_value=1.0)

if run:
    category_code = CATEGORY_CODES.get(category_label, 0)
    keyword_cleaned = keyword.strip()
    if not keyword_cleaned:
        st.error("Please enter a keyword.")
        st.stop()
    with st.spinner("Running signal pipeline..."):
        result = run_pipeline(
            keyword_cleaned,
            category_code,
            pakistan_only,
            province_threshold,
            trend_cap,
            velocity_cap,
            diversity_cap,
            freshness_hours,
        )
    st.session_state["signal_result"] = result

result = st.session_state.get("signal_result")

if result is None:
    st.info("Enter a keyword and click Run.")
else:
    keyword_cleaned = result["keyword"]
    trends_meta = result["trends_meta"]
    if trends_meta.get("error"):
        st.error(trends_meta["error"])
        if "429" in str(trends_meta["error"]):
            st.info("Google Trends rate-limited this request. Try again in a few minutes.")
    elif trends_meta.get("warning"):
        st.warning(trends_meta["warning"])
    if result["gdelt_error"]:
        st.error(f"GDELT error (24h): {result['gdelt_error']}")

    interest_metrics = result["interest_metrics"]
    breakout_metrics = result["breakout_metrics"]
    region_metrics = result["region_metrics"]
    gdelt_metrics = result["gdelt_metrics"]
    score_metrics = result["score_metrics"]
    articles_24h = result["articles_24h"]
    ranked_articles = result["ranked_articles"]

    st.subheader("Overall score")
    score_col, detail_col = st.columns([1, 3])
    with score_col:
        st.metric("Score", f"{score_metrics['score'] * 100:.1f}")
        if breakout_metrics["breakout"]:
            st.warning("Breakout rising queries detected")
    with detail_col:
        st.caption("Component normalization (0 to 1)")
        st.dataframe(
            pd.DataFrame(
                {
                    "TrendSpike": [score_metrics["trend_norm"]],
                    "ArticleVelocity": [score_metrics["velocity_norm"]],
                    "SourceDiversity": [score_metrics["diversity_norm"]],
                    "ProvinceSpread": [score_metrics["province_norm"]],
                    "Freshness": [score_metrics["freshness_norm"]],
                }
            )
        )

    st.subheader("Google Trends metrics")
    trend_cols = st.columns(4)
    trend_cols[0].metric(
        "Current interest",
        "N/A" if interest_metrics["current_interest"] is None else f"{interest_metrics['current_interest']:.0f}",
    )
    trend_cols[1].metric(
        "Week-over-week growth",
        "N/A"
        if interest_metrics["week_over_week"] is None
        else f"{interest_metrics['week_over_week']:.1f}%",
    )
    trend_cols[2].metric("Breakout flag", "Yes" if breakout_metrics["breakout"] else "No")
    trend_cols[3].metric("Province spread", f"{region_metrics['province_count']} / 4")

    if region_metrics["province_values"]:
        region_df = pd.DataFrame(
            {
                "Province": list(region_metrics["province_values"].keys()),
                "Interest": list(region_metrics["province_values"].values()),
            }
        )
        st.dataframe(region_df, hide_index=True)
        if region_metrics["concentration_ratio"] is not None:
            st.caption(
                f"Concentration ratio (top province / total): {region_metrics['concentration_ratio']:.2f}"
            )

    if breakout_metrics["rising"] is not None and not breakout_metrics["rising"].empty:
        st.subheader("Related rising queries")
        st.dataframe(breakout_metrics["rising"].head(10), hide_index=True)

    st.subheader("GDELT metrics")
    gdelt_cols = st.columns(4)
    gdelt_cols[0].metric("Articles (24h)", gdelt_metrics["count_24h"])
    gdelt_cols[1].metric("Velocity (per hour)", f"{gdelt_metrics['velocity']:.1f}")
    gdelt_cols[2].metric("Source diversity", gdelt_metrics["unique_domains"])
    tone_value = gdelt_metrics["avg_tone"]
    gdelt_cols[3].metric("Avg tone", "N/A" if tone_value is None else f"{tone_value:.2f}")

    extra_cols = st.columns(3)
    if gdelt_metrics["earliest"]:
        extra_cols[0].metric(
            "First appearance",
            gdelt_metrics["earliest"].strftime("%Y-%m-%d %H:%M UTC"),
        )
    if gdelt_metrics["hours_since_first"] is not None:
        extra_cols[1].metric(
            "Hours since first",
            f"{gdelt_metrics['hours_since_first']:.1f}",
        )
    if gdelt_metrics["local_domain_share"] is not None:
        extra_cols[2].metric(
            "Local domain share",
            f"{gdelt_metrics['local_domain_share'] * 100:.0f}%",
        )

    if articles_24h:
        st.subheader("Recent articles (24h)")
        article_rows = []
        for article in articles_24h[:50]:
            article_rows.append(
                {
                    "Seen": article.get("seendate"),
                    "Title": article.get("title"),
                    "Domain": article.get("domain") or domain_from_url(article.get("url", "")),
                    "Source": article.get("sourcecountry"),
                    "Tone": article.get("tone"),
                    "URL": article.get("url"),
                }
            )
        st.dataframe(pd.DataFrame(article_rows), hide_index=True)
    if ranked_articles:
        st.subheader("Top articles by score")
        st.caption(
            "Ranking uses global trend/velocity/province scores plus per-article freshness and domain uniqueness."
        )
        top_rows = []
        for article in ranked_articles[:10]:
            title = article["title"] or "Untitled"
            url = article["url"]
            domain = article["domain"] or "unknown"
            seen = article["seen"]
            seen_label = seen.strftime("%Y-%m-%d %H:%M UTC") if seen else "unknown"
            score_label = f"{article['score'] * 100:.1f}"
            top_rows.append(
                {
                    "Title": title,
                    "URL": url,
                    "Score": score_label,
                    "Domain": domain,
                    "Seen": seen_label,
                }
            )
        st.dataframe(
            pd.DataFrame(top_rows),
            hide_index=True,
            column_config={
                "URL": st.column_config.LinkColumn("Link"),
            },
        )

st.divider()
st.header("Scripting")
st.caption(
    "Separate from the signal metrics: provide 3 topics, fetch recent sources via web search (incl. X), "
    "then generate a 3-item interpreter script using your one-shot example."
)

st.subheader("Web search setup")
ws_col_a, ws_col_b = st.columns([2, 1])
with ws_col_a:
    st.write(
        "Uses OpenAI + `web_search` to collect sources (news sites and optional x.com posts)."
    )
with ws_col_b:
    script_model_name = st.text_input("Model", value="gpt-5.2", key="script_model")
    script_api_key = ""
    if resolve_api_key(""):
        st.success("OpenAI API key loaded from Streamlit secrets.")
    else:
        st.warning(
            "Missing OpenAI API key. Add OPENAI_API_KEY to Streamlit secrets (.streamlit/secrets.toml)."
        )

st.subheader("Find News (Momentum Filter)")
st.caption("Fetch 10 Pakistan macro-moving items for the week (web search; links included).")

find_col_a, find_col_b = st.columns([1, 2])
with find_col_a:
    find_news_clicked = st.button(
        "Find news",
        type="primary",
        disabled=(not OPENAI_AVAILABLE),
        key="find_momentum_news",
    )
with find_col_b:
    st.write("Uses the model above and the OpenAI API key from Streamlit secrets.")

if find_news_clicked:
    with st.spinner("Finding momentum news..."):
        items, err = fetch_momentum_news(
            model_name=script_model_name,
            api_key=script_api_key,
        )
    st.session_state["momentum_news"] = items
    st.session_state["momentum_news_error"] = err

if st.session_state.get("momentum_news_error"):
    st.error(st.session_state["momentum_news_error"])
elif st.session_state.get("momentum_news"):
    with st.container(border=True):
        for idx, item in enumerate(st.session_state["momentum_news"], start=1):
            title = item.get("title", "Untitled")
            url = item.get("url", "")
            source = item.get("source", "")
            date = item.get("date", "")
            summary = item.get("summary", "")
            if url:
                st.markdown(f"{idx}. **[{title}]({url})**")
            else:
                st.markdown(f"{idx}. **{title}**")
            meta = " · ".join([x for x in [source, date] if x])
            if meta:
                st.caption(meta)
            if summary:
                st.write(summary)
            if idx != len(st.session_state["momentum_news"]):
                st.divider()

st.divider()

script_col_a, script_col_b = st.columns([2, 1])
with script_col_a:
    topics_text = st.text_area(
        "Topics (exactly 3, one per line)",
        value="Power tariffs\nPolicy rate / SBP\nOil prices / inflation",
        height=110,
        key="script_topics",
    )
with script_col_b:
    script_category_label = st.selectbox(
        "Category hint",
        list(WEBSEARCH_CATEGORY_HINTS.keys()),
        index=list(WEBSEARCH_CATEGORY_HINTS.keys()).index("Business")
        if "Business" in WEBSEARCH_CATEGORY_HINTS
        else 0,
        key="script_category",
    )
    script_pakistan_only = st.toggle(
        "Prefer Pakistan sources",
        value=True,
        key="script_pk_only",
    )
    script_include_x = st.toggle(
        "Include X (x.com)",
        value=True,
        key="script_include_x",
    )
    script_max_items = st.slider(
        "Sources per topic",
        3,
        12,
        6,
        1,
        key="script_sources_per_topic",
    )

st.subheader("Video links (optional)")
st.caption(
    "Paste YouTube/Instagram links (one per line). We'll download, convert to MP3, transcribe, and include the text as extra context."
)
video_links_text = st.text_area(
    "Video URLs",
    value="",
    height=110,
    key="video_links",
)
video_limit = st.slider("Max videos to process", 1, 20, 5, 1, key="video_limit")
process_videos = st.button(
    "Process video links",
    disabled=(not OPENAI_AVAILABLE),
    key="process_videos",
)
if process_videos:
    urls = [line.strip() for line in video_links_text.splitlines() if line.strip()]
    urls = urls[:video_limit]
    st.session_state["video_items"] = []
    st.session_state["video_errors"] = {}
    st.session_state["video_transcripts"] = []
    for u in urls:
        with st.spinner(f"Downloading/transcribing: {u}"):
            progress = st.progress(0.0)
            status = st.empty()
            try:
                with tempfile.TemporaryDirectory() as tmp:
                    tmp_path = Path(tmp)
                    status.write("Downloading audio...")
                    audio_path, err = download_video_audio(
                        u, tmp_path, progress_cb=progress.progress
                    )
                    if err:
                        st.session_state["video_errors"][u] = err
                        progress.empty()
                        status.empty()
                        continue

                    status.write("Converting to MP3...")
                    mp3_path, err = convert_to_mp3(audio_path, tmp_path / "audio.mp3")
                    if err:
                        st.session_state["video_errors"][u] = err
                        progress.empty()
                        status.empty()
                        continue

                    status.write("Transcribing...")
                    transcript, err = transcribe_audio(
                        mp3_path,
                        script_api_key,
                        model_name="gpt-4o-transcribe",
                    )
                    if err:
                        st.session_state["video_errors"][u] = err
                        progress.empty()
                        status.empty()
                        continue

                    st.session_state["video_transcripts"].append(
                        {"url": u, "transcript": transcript}
                    )
                    progress.progress(1.0)
            finally:
                progress.empty()
                status.empty()

    # Build a single context blob for prompting.
    lines: List[str] = []
    for item in st.session_state["video_transcripts"]:
        url = item.get("url", "")
        transcript = item.get("transcript", "")
        lines.append(f"URL: {url}")
        lines.append("Transcript:")
        lines.append(transcript)
        lines.append("")
    st.session_state["video_context_text"] = "\n".join(lines).strip()

if st.session_state.get("video_errors"):
    with st.expander("Video processing errors", expanded=False):
        for u, e in st.session_state["video_errors"].items():
            st.error(f"{u}: {e}")

if st.session_state.get("video_transcripts"):
    st.subheader("Video transcripts")
    dfv = pd.DataFrame(
        [
            {
                "url": x.get("url", ""),
                "transcript": x.get("transcript", ""),
            }
            for x in st.session_state["video_transcripts"]
        ]
    )
    st.dataframe(
        dfv,
        hide_index=True,
        column_config={"url": st.column_config.LinkColumn("url")},
    )
    st.text_area(
        "Transcript text (used as additional context)",
        value=st.session_state.get("video_context_text", ""),
        height=220,
        key="video_context_preview",
    )
    st.download_button(
        "Download transcript context",
        data=(st.session_state.get("video_context_text", "") or "").encode("utf-8"),
        file_name="video_transcripts.txt",
        mime="text/plain",
        key="download_video_context",
    )

st.subheader("Expert interviews (.txt)")
st.caption("Upload one or more .txt files. The quotes will be used as added context for the final script.")
expert_files = st.file_uploader(
    "Expert interview files",
    type=["txt"],
    accept_multiple_files=True,
    key="expert_files",
)
expert_texts: List[str] = []
if expert_files:
    for f in expert_files:
        try:
            content = f.getvalue().decode("utf-8", errors="replace").strip()
        except Exception:
            content = ""
        if content:
            expert_texts.append(f"# {f.name}\n{content}")
st.session_state["expert_context_text"] = "\n\n".join(expert_texts).strip()

if st.session_state.get("expert_context_text"):
    st.text_area(
        "Expert interview context (used in script)",
        value=st.session_state.get("expert_context_text", ""),
        height=220,
        key="expert_context_preview",
    )
    st.download_button(
        "Download expert context",
        data=(st.session_state.get("expert_context_text", "") or "").encode("utf-8"),
        file_name="expert_interviews.txt",
        mime="text/plain",
        key="download_expert_context",
    )

example_file = st.file_uploader(
    "One-shot example script (.txt)",
    type=["txt"],
    key="example_script_file",
)
example_script_text = ""
if example_file is not None:
    try:
        example_script_text = example_file.getvalue().decode("utf-8", errors="replace")
    except Exception:
        example_script_text = ""

topics = [line.strip() for line in topics_text.splitlines() if line.strip()]
topics_ok = len(topics) == 3
if not topics_ok:
    st.warning("Enter exactly 3 topics (one per line).")

fetch_sources_clicked = st.button(
    "Fetch sources for 3 topics",
    type="primary",
    disabled=(not topics_ok or not OPENAI_AVAILABLE),
    key="fetch_script_sources",
)
if fetch_sources_clicked:
    st.session_state["script_sources"] = {}
    st.session_state["script_source_errors"] = {}
    for topic in topics:
        cache_key = web_cache_key(
            topic,
            script_category_label,
            script_pakistan_only,
            script_model_name,
            script_include_x,
            script_max_items,
        )
        with st.spinner(f"Searching sources: {topic}"):
            articles, err = fetch_websearch_articles(
                topic,
                script_category_label,
                script_model_name,
                script_api_key,
                script_max_items,
                script_pakistan_only,
                script_include_x,
            )
        st.session_state["web_articles_cache"][cache_key] = articles
        st.session_state["web_error_cache"][cache_key] = err
        st.session_state["script_sources"][topic] = articles
        st.session_state["script_source_errors"][topic] = err

st.subheader("Sources")
for topic in topics if topics_ok else []:
    err = st.session_state.get("script_source_errors", {}).get(topic)
    items = st.session_state.get("script_sources", {}).get(topic, [])
    with st.expander(topic, expanded=True):
        if err:
            st.error(err)
        elif not items:
            st.info("No sources fetched yet. Click \"Fetch sources for 3 topics\".")
        else:
            df = pd.DataFrame(items)
            # Ensure stable column order when present.
            cols = [c for c in ["title", "source", "date", "url", "summary"] if c in df.columns]
            if cols:
                df = df[cols]
            st.dataframe(
                df,
                hide_index=True,
                column_config={
                    "url": st.column_config.LinkColumn("url"),
                },
            )

generate_clicked = st.button(
    "Generate interpreter script",
    disabled=(not topics_ok or not OPENAI_AVAILABLE),
    key="generate_script",
)
if generate_clicked:
    if not example_script_text.strip():
        st.error("Upload the one-shot example script (.txt) first.")
    else:
        missing = [t for t in topics if not st.session_state.get("script_sources", {}).get(t)]
        if missing:
            st.error("Fetch sources first for: " + ", ".join(missing))
        else:
            with st.spinner("Generating script..."):
                script, err = generate_interpreter_script(
                    example_script=example_script_text,
                    topics=topics,
                    sources_by_topic=st.session_state.get("script_sources", {}),
                    extra_context=st.session_state.get("video_context_text", ""),
                    expert_context=st.session_state.get("expert_context_text", ""),
                    model_name=script_model_name,
                    api_key=script_api_key,
                )
            if err:
                st.error(err)
            else:
                st.session_state["generated_script"] = script
                st.session_state["script_draft"] = script
                # The text_area is keyed, so update its session value explicitly.
                st.session_state["script_draft_area"] = script
                st.session_state["script_versions"] = [script]
                st.session_state["script_chat"] = []
                st.session_state["script_edit_error"] = None

if st.session_state.get("generated_script"):
    st.subheader("Generated script")
    st.text_area(
        "Script",
        value=st.session_state.get("script_draft", ""),
        height=520,
        key="script_draft_area",
    )

    st.subheader("Script changes (chat)")
    st.caption("Describe changes; the editor uses gpt-5.2. The full script is updated on each turn.")

    chat_box = st.container(border=True)
    with chat_box:
        for msg in st.session_state.get("script_chat", []):
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "user":
                st.markdown(f"**You:** {content}")
            else:
                st.markdown(f"**Editor:** {content}")

    edit_request = st.text_area("Change request", height=110, key="script_edit_request")
    edit_cols = st.columns([1, 1, 2])
    with edit_cols[0]:
        apply_edit = st.button("Apply changes", type="primary", key="apply_script_edit")
    with edit_cols[1]:
        undo_edit = st.button("Undo last", key="undo_script_edit")
    with edit_cols[2]:
        clear_chat = st.button("Clear chat", key="clear_script_chat")

    if clear_chat:
        st.session_state["script_chat"] = []
        st.session_state["script_edit_error"] = None

    if undo_edit:
        versions = st.session_state.get("script_versions", [])
        if len(versions) >= 2:
            versions.pop()
            st.session_state["script_versions"] = versions
            st.session_state["script_draft"] = versions[-1]
            st.session_state["script_draft_area"] = versions[-1]
        else:
            st.info("No previous version to undo to.")

    if apply_edit:
        st.session_state["script_edit_error"] = None
        current = st.session_state.get("script_draft_area", "").strip()
        req = st.session_state.get("script_edit_request", "").strip()
        if not req:
            st.warning("Enter a change request.")
        else:
            st.session_state["script_chat"].append({"role": "user", "content": req})
            with st.spinner("Applying changes..."):
                obj, err = revise_interpreter_script(
                    current_script=current,
                    user_request=req,
                    api_key=script_api_key,
                    model_name="gpt-5.2",
                )
            if err:
                st.session_state["script_edit_error"] = err
                st.session_state["script_chat"].append({"role": "assistant", "content": f"Error: {err}"})
            else:
                assistant_msg = str(obj.get("assistant_message", "")).strip()
                revised = str(obj.get("revised_script", "")).strip()
                if assistant_msg:
                    st.session_state["script_chat"].append({"role": "assistant", "content": assistant_msg})
                if revised:
                    st.session_state["script_draft"] = revised
                    st.session_state["script_draft_area"] = revised
                    st.session_state["generated_script"] = revised
                    st.session_state["script_versions"] = st.session_state.get("script_versions", []) + [revised]

    if st.session_state.get("script_edit_error"):
        st.error(st.session_state["script_edit_error"])

    st.subheader("Voiceover (ElevenLabs)")
    st.caption("Generate a voiceover from the current script draft.")

    vo_cols = st.columns([2, 1, 1])
    with vo_cols[0]:
        # Prefer secrets/env. Still allow manual override if not present.
        resolved_eleven_key = resolve_elevenlabs_api_key("")
        if resolved_eleven_key:
            st.success("ElevenLabs API key loaded from Streamlit secrets / environment.")
            elevenlabs_key_input = ""
        else:
            elevenlabs_key_input = st.text_input(
                "ElevenLabs API key",
                type="password",
                key="elevenlabs_api_key",
                help="Prefer setting ELEVENLABS_API_KEY in environment or Streamlit secrets.",
            )
    with vo_cols[1]:
        model_id = st.text_input("Model ID", value="eleven_multilingual_v2", key="elevenlabs_model_id")
        stability = st.slider("Stability", 0.0, 1.0, 0.5, 0.05, key="eleven_stability")
        similarity_boost = st.slider("Similarity boost", 0.0, 1.0, 0.75, 0.05, key="eleven_similarity")
    with vo_cols[2]:
        style = st.slider("Style", 0.0, 1.0, 0.0, 0.05, key="eleven_style")
        use_speaker_boost = st.toggle("Speaker boost", value=True, key="eleven_speaker_boost")
        gen_voice = st.button(
            "Generate voiceover",
            type="primary",
            key="generate_voiceover",
        )
        clear_voice = st.button("Clear audio", key="clear_voiceover")

    resolved_eleven_key = resolve_elevenlabs_api_key(elevenlabs_key_input) or ""
    voices = eleven_list_voices(resolved_eleven_key) if resolved_eleven_key else []
    voice_options = [
        f"{v['name']} ({v['category']})" if v.get("category") else v["name"]
        for v in voices
    ]
    voice_idx: Optional[int] = None
    if voice_options:
        voice_idx = st.selectbox(
            "Voice",
            list(range(len(voice_options))),
            format_func=lambda i: voice_options[i],
            index=0,
            key="eleven_voice_idx",
        )
    else:
        st.info("Enter your ElevenLabs API key to load available voices.")

    st.session_state["eleven_voice_id"] = (
        voices[voice_idx]["id"] if voices and voice_idx is not None else None
    )
    st.session_state["eleven_settings"] = {
        "stability": stability if voices else None,
        "similarity_boost": similarity_boost if voices else None,
        "style": style if voices else None,
        "use_speaker_boost": use_speaker_boost if voices else None,
    }
    if clear_voice:
        st.session_state["voiceover_audio"] = None
        st.session_state["voiceover_error"] = None

    if gen_voice:
        with st.spinner("Generating voiceover..."):
            try:
                audio_bytes = eleven_tts(
                    resolved_eleven_key,
                    st.session_state.get("eleven_voice_id") or "",
                    st.session_state.get("script_draft_area", ""),
                    model_id=model_id,
                    stability=st.session_state.get("eleven_settings", {}).get("stability"),
                    similarity_boost=st.session_state.get("eleven_settings", {}).get("similarity_boost"),
                    style=st.session_state.get("eleven_settings", {}).get("style"),
                    use_speaker_boost=st.session_state.get("eleven_settings", {}).get("use_speaker_boost"),
                )
                st.session_state["voiceover_audio"] = audio_bytes
                st.session_state["voiceover_error"] = None
            except Exception as exc:
                st.session_state["voiceover_audio"] = None
                st.session_state["voiceover_error"] = str(exc)

    if st.session_state.get("voiceover_error"):
        st.error(st.session_state["voiceover_error"])
    elif st.session_state.get("voiceover_audio"):
        st.audio(st.session_state["voiceover_audio"], format="audio/mpeg")
        st.download_button(
            "Download MP3",
            data=st.session_state["voiceover_audio"],
            file_name="news_script_voiceover.mp3",
            mime="audio/mpeg",
            key="download_voiceover",
        )
