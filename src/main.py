import os
import json
import time
import math
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple, Any, Optional

import requests
import feedparser
import numpy as np
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from bs4 import BeautifulSoup

from openai import OpenAI


# ============================================================
# CONFIG
# ============================================================
TICKERS_50 = [
    "AAPL","MSFT","GOOGL","AMZN","META","NVDA","AVGO","AMD","ASML","TSM",
    "ORCL","CRM","ADBE","NFLX","COST","QCOM","INTU","IBM","TXN","AMAT",
    "SNOW","DDOG","MDB","NOW","PANW","CRWD","ZS","SHOP","UBER","ABNB",
    "MELI","COIN","SMCI","ARM","MU","KLAC","LRCX","ANET","FTNT","CDNS",
    "CAT","DE","ETN","HON","GE","RTX","LMT","ITW","EMR","PH"
]

SHEET_TAB_OUTPUT = "OUTPUT"
SHEET_TAB_LOGS = "LOGS"
SHEET_TAB_BACKTEST = "BACKTEST"

MAX_NEWS_ITEMS_PER_TICKER = 6
MAX_FILING_CHARS = 14000
DEBATE_MAX_ROUNDS = 2

# News RSS
REUTERS_MARKETS_RSS = "https://www.reuters.com/markets/rss"
YAHOO_TICKER_RSS_TEMPLATE = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"

# SEC (official)
SEC_TICKER_CIK_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_DATA_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_nodash}/{primary_doc}"

# FMP
FMP_HIST_URL = "https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}"


# ============================================================
# UTILITIES
# ============================================================
def now_buenos_aires_str() -> str:
    tz = timezone(timedelta(hours=-3))
    return datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S (America/Argentina/Buenos_Aires)")


def sec_headers() -> Dict[str, str]:
    # SEC requests a descriptive User-Agent.
    # This should be YOUR contact. Using a real email reduces risk of throttling.
    return {
        "User-Agent": "AlphaAgentsBot/1.0 (contact: danieleabruno@gmail.com)",
        "Accept-Encoding": "gzip, deflate",
        "Accept": "application/json,text/html,*/*",
    }


def http_get_json(url: str, params: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None,
                  timeout: int = 30, retries: int = 3, backoff: float = 1.2) -> Dict[str, Any]:
    last_err = None
    for i in range(retries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
            # simple handling for rate limits
            if r.status_code in (429, 503):
                time.sleep(backoff * (i + 1))
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            time.sleep(backoff * (i + 1))
    raise RuntimeError(f"GET JSON failed for {url}: {last_err}")


def http_get_text(url: str, headers: Optional[Dict[str, str]] = None, timeout: int = 30,
                  retries: int = 3, backoff: float = 1.2) -> str:
    last_err = None
    for i in range(retries):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            if r.status_code in (429, 503):
                time.sleep(backoff * (i + 1))
                continue
            r.raise_for_status()
            return r.text
        except Exception as e:
            last_err = e
            time.sleep(backoff * (i + 1))
    raise RuntimeError(f"GET text failed for {url}: {last_err}")


def _clean_cell(x: Any) -> Any:
    # Google Sheets JSON rejects NaN / Inf
    if x is None:
        return ""
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return ""
    # numpy scalar
    if isinstance(x, (np.floating,)):
        xf = float(x)
        if math.isnan(xf) or math.isinf(xf):
            return ""
        return xf
    return x


def clean_2d(rows: List[List[Any]]) -> List[List[Any]]:
    return [[_clean_cell(c) for c in row] for row in rows]


# ============================================================
# GOOGLE SHEETS
# ============================================================
def open_sheet():
    sheet_id = os.environ["SHEET_ID"].strip()
    info = json.loads(os.environ["GSPREAD_SERVICE_ACCOUNT_JSON"])
    scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    creds = Credentials.from_service_account_info(info, scopes=scopes)
    gc = gspread.authorize(creds)
    return gc.open_by_key(sheet_id)


def get_or_create_ws(sh, title: str, rows=3000, cols=40):
    try:
        return sh.worksheet(title)
    except gspread.WorksheetNotFound:
        return sh.add_worksheet(title=title, rows=rows, cols=cols)


# ============================================================
# DATA: PRICES via FMP  (Valuation Agent tool)
# ============================================================
def fetch_price_metrics_fmp(ticker: str) -> Dict[str, float]:
    api_key = os.environ["FMP_API_KEY"].strip()
    url = FMP_HIST_URL.format(ticker=ticker)
    params = {"timeseries": 260, "apikey": api_key}

    data = http_get_json(url, params=params, headers=None, timeout=30, retries=3, backoff=1.3)
    hist = data.get("historical", [])
    if not hist or len(hist) < 30:
        return {"price": np.nan, "ret_3m": np.nan, "ret_6m": np.nan, "vol_1y": np.nan, "max_dd_1y": np.nan}

    closes = []
    for h in hist:
        c = h.get("close")
        if isinstance(c, (int, float)):
            closes.append(float(c))

    # FMP usually returns newest -> oldest. We'll assume closes[0] is latest.
    if len(closes) < 30:
        return {"price": np.nan, "ret_3m": np.nan, "ret_6m": np.nan, "vol_1y": np.nan, "max_dd_1y": np.nan}

    price = closes[0]

    def pct_return(days: int) -> float:
        if len(closes) <= days:
            return np.nan
        return float((closes[0] / closes[days]) - 1.0)

    ret_3m = pct_return(63)
    ret_6m = pct_return(126)

    # oldest -> newest for time series stats
    close_series = pd.Series(list(reversed(closes)))
    rets = close_series.pct_change().dropna()
    vol_1y = float(rets.std() * math.sqrt(252)) if len(rets) > 10 else np.nan

    running_max = close_series.cummax()
    dd = (close_series / running_max) - 1.0
    max_dd_1y = float(dd.min()) if len(dd) else np.nan

    return {
        "price": float(price),
        "ret_3m": ret_3m,
        "ret_6m": ret_6m,
        "vol_1y": vol_1y,
        "max_dd_1y": max_dd_1y,
    }


# ============================================================
# DATA: NEWS (Sentiment Agent tool)
# ============================================================
def fetch_news_items(ticker: str) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []

    yahoo_url = YAHOO_TICKER_RSS_TEMPLATE.format(ticker=ticker)
    y = feedparser.parse(yahoo_url)
    for e in (y.entries or [])[:MAX_NEWS_ITEMS_PER_TICKER]:
        items.append({
            "source": "YahooRSS",
            "title": e.get("title", "") or "",
            "link": e.get("link", "") or "",
            "published": e.get("published", "") or "",
        })

    r = feedparser.parse(REUTERS_MARKETS_RSS)
    for e in (r.entries or [])[:3]:
        items.append({
            "source": "ReutersRSS",
            "title": e.get("title", "") or "",
            "link": e.get("link", "") or "",
            "published": e.get("published", "") or "",
        })

    return items[:MAX_NEWS_ITEMS_PER_TICKER + 3]


# ============================================================
# DATA: SEC FILINGS (Fundamental Agent tool)
# ============================================================
def fetch_ticker_cik_map() -> Dict[str, str]:
    data = http_get_json(SEC_TICKER_CIK_URL, headers=sec_headers(), timeout=30, retries=3, backoff=1.4)
    out: Dict[str, str] = {}
    for _, row in data.items():
        t = (row.get("ticker") or "").upper()
        cik_num = row.get("cik_str")
        if t and isinstance(cik_num, int):
            out[t] = str(cik_num).zfill(10)
    return out


def pick_latest_10q_10k(submissions_json: Dict[str, Any]) -> Tuple[str, str, str]:
    recent = submissions_json.get("filings", {}).get("recent", {})
    forms = recent.get("form", []) or []
    accessions = recent.get("accessionNumber", []) or []
    primary_docs = recent.get("primaryDocument", []) or []

    for form, acc, doc in zip(forms, accessions, primary_docs):
        if form in ("10-Q", "10-K"):
            return form, acc, doc
    return "", "", ""


def fetch_latest_filing_text(ticker: str, cik: str) -> Dict[str, str]:
    sub_url = SEC_DATA_SUBMISSIONS.format(cik=cik)
    sub = http_get_json(sub_url, headers=sec_headers(), timeout=30, retries=3, backoff=1.4)

    form, acc, doc = pick_latest_10q_10k(sub)
    if not form:
        return {"form": "", "url": "", "text": ""}

    cik_int = str(int(cik))
    acc_nodash = acc.replace("-", "")
    filing_url = SEC_ARCHIVES_BASE.format(cik_int=cik_int, accession_nodash=acc_nodash, primary_doc=doc)

    html = http_get_text(filing_url, headers=sec_headers(), timeout=30, retries=3, backoff=1.4)
    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text(separator="\n")
    text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])

    lowered = text.lower()
    anchors = ["item 1a", "risk factors", "management’s discussion", "management's discussion", "md&a", "item 7"]
    found = [lowered.find(a) for a in anchors if lowered.find(a) != -1]
    idx = min(found) if found else 0

    chunk = text[idx:idx + MAX_FILING_CHARS]
    return {"form": form, "url": filing_url, "text": chunk}


# ============================================================
# LLM AGENTS
# ============================================================
def llm_client() -> OpenAI:
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"].strip())


def call_llm(system: str, user: str, model: str = "gpt-4o-mini") -> str:
    client = llm_client()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
    return (resp.choices[0].message.content or "").strip()


def _safe_json_parse(txt: str) -> Dict[str, Any]:
    # Robust parsing: if model returns extra text, try to extract JSON object.
    txt = txt.strip()
    if txt.startswith("{") and txt.endswith("}"):
        return json.loads(txt)

    # fallback: find first {...} block
    start = txt.find("{")
    end = txt.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(txt[start:end+1])

    raise ValueError(f"Model did not return JSON. Got: {txt[:400]}")


def valuation_agent(ticker: str, m: Dict[str, float], risk_profile: str) -> Dict[str, Any]:
    system = f"""You are a valuation equity analyst. Risk profile: {risk_profile}.
Output MUST be valid JSON with keys: vote, rationale, risks, invalidation.
Allowed vote: BUY, SELL.
Use only the provided numeric metrics; do NOT invent numbers."""
    user = f"""Ticker: {ticker}
Metrics:
price={m['price']}
ret_3m={m['ret_3m']}
ret_6m={m['ret_6m']}
vol_1y={m['vol_1y']}
max_dd_1y={m['max_dd_1y']}

Decide BUY or SELL given the risk profile."""
    return _safe_json_parse(call_llm(system, user))


def sentiment_agent(ticker: str, news: List[Dict[str, str]], risk_profile: str) -> Dict[str, Any]:
    system = f"""You are a sentiment equity analyst. Risk profile: {risk_profile}.
Output MUST be valid JSON with keys: vote, rationale, risks, invalidation.
Allowed vote: BUY, SELL.
Base your analysis ONLY on the news items provided."""
    user = f"""Ticker: {ticker}
News items (may include general market context):
{json.dumps(news, indent=2)}

Decide BUY or SELL."""
    return _safe_json_parse(call_llm(system, user))


def fundamental_agent(ticker: str, filing: Dict[str, str], risk_profile: str) -> Dict[str, Any]:
    system = f"""You are a fundamental equity analyst. Risk profile: {risk_profile}.
Output MUST be valid JSON with keys: vote, rationale, risks, invalidation.
Allowed vote: BUY, SELL.
Base your analysis ONLY on the SEC filing excerpt provided. If filing is missing/empty, say so explicitly."""
    user = f"""Ticker: {ticker}
Filing form: {filing.get('form')}
Filing url: {filing.get('url')}
Filing text excerpt:
{filing.get('text')}

Decide BUY or SELL."""
    return _safe_json_parse(call_llm(system, user))


def debate_and_consensus(ticker: str, a: Dict[str, Any], b: Dict[str, Any], c: Dict[str, Any], risk_profile: str) -> Dict[str, Any]:
    system = f"""You coordinate a multi-agent debate (valuation, sentiment, fundamental). Risk profile: {risk_profile}.
You MUST output valid JSON with keys: final_vote, confidence_0_100, synthesis, key_risks.
final_vote must be BUY or SELL.
You must reconcile disagreements. Use at most {DEBATE_MAX_ROUNDS} rounds internally (simulate)."""
    user = f"""Ticker: {ticker}
Valuation agent output: {json.dumps(a)}
Sentiment agent output: {json.dumps(b)}
Fundamental agent output: {json.dumps(c)}

Return the consensus."""
    return _safe_json_parse(call_llm(system, user))


# ============================================================
# BACKTEST (disabled for now; keeps pipeline stable)
# ============================================================
def backtest_equal_weight_placeholder() -> Dict[str, Any]:
    # Keep placeholders as "" (NOT NaN) so Sheets updates never fail.
    return {
        "portfolio_cum_return": "",
        "portfolio_sharpe": "",
        "benchmark_cum_return": "",
        "benchmark_sharpe": ""
    }


# ============================================================
# PIPELINE
# ============================================================
def run_for_profile(risk_profile: str, cik_map: Dict[str, str]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    logs: List[Dict[str, Any]] = []

    for i, t in enumerate(TICKERS_50, start=1):
        try:
            # light pacing
            if i % 10 == 0:
                time.sleep(0.4)

            metrics = fetch_price_metrics_fmp(t)
            news = fetch_news_items(t)

            cik = cik_map.get(t.upper(), "")
            filing = {"form": "", "url": "", "text": ""}
            if cik:
                time.sleep(0.2)
                filing = fetch_latest_filing_text(t, cik)

            va = valuation_agent(t, metrics, risk_profile)
            sa = sentiment_agent(t, news, risk_profile)
            fa = fundamental_agent(t, filing, risk_profile)

            cons = debate_and_consensus(t, va, sa, fa, risk_profile)

            rows.append({
                "ticker": t,
                "price": metrics["price"],
                "ret_3m": metrics["ret_3m"],
                "ret_6m": metrics["ret_6m"],
                "vol_1y": metrics["vol_1y"],
                "max_dd_1y": metrics["max_dd_1y"],
                "final_vote": cons.get("final_vote", ""),
                "confidence_0_100": cons.get("confidence_0_100", ""),
                "synthesis": cons.get("synthesis", ""),
                "key_risks": cons.get("key_risks", ""),
                "sec_form": filing.get("form", ""),
                "sec_url": filing.get("url", "")
            })

            logs.append({
                "as_of": now_buenos_aires_str(),
                "risk_profile": risk_profile,
                "ticker": t,
                "valuation": va,
                "sentiment": sa,
                "fundamental": fa,
                "consensus": cons,
                "news": news,
                "sec_url": filing.get("url", "")
            })

        except Exception as e:
            # do not crash entire run
            rows.append({
                "ticker": t,
                "price": "",
                "ret_3m": "",
                "ret_6m": "",
                "vol_1y": "",
                "max_dd_1y": "",
                "final_vote": "SELL",
                "confidence_0_100": 0,
                "synthesis": f"ERROR: {e}",
                "key_risks": "N/A",
                "sec_form": "",
                "sec_url": ""
            })
            logs.append({
                "as_of": now_buenos_aires_str(),
                "risk_profile": risk_profile,
                "ticker": t,
                "error": str(e)
            })

    df = pd.DataFrame(rows)

    # sort: BUY first then higher confidence
    df["buy_flag"] = (df["final_vote"] == "BUY").astype(int)
    df["conf_num"] = pd.to_numeric(df["confidence_0_100"], errors="coerce").fillna(0)
    df = df.sort_values(["buy_flag", "conf_num"], ascending=[False, False]).drop(columns=["buy_flag", "conf_num"])

    log_df = pd.DataFrame(logs)

    # placeholder backtest block
    bt = {
        "backtest_start": (pd.Timestamp.utcnow() - pd.Timedelta(days=120)).date().isoformat(),
        "backtest_end": pd.Timestamp.utcnow().date().isoformat(),
        "picks_top15": df[df["final_vote"] == "BUY"]["ticker"].head(15).tolist(),
        **backtest_equal_weight_placeholder()
    }

    return df, log_df, bt


def write_to_sheets(df_neutral: pd.DataFrame, df_averse: pd.DataFrame, logs: pd.DataFrame,
                    bt_neutral: Dict[str, Any], bt_averse: Dict[str, Any]) -> None:
    sh = open_sheet()

    ws_out = get_or_create_ws(sh, SHEET_TAB_OUTPUT, rows=3000, cols=40)
    ws_logs = get_or_create_ws(sh, SHEET_TAB_LOGS, rows=4000, cols=25)
    ws_bt = get_or_create_ws(sh, SHEET_TAB_BACKTEST, rows=200, cols=20)

    ts = now_buenos_aires_str()

    # OUTPUT blocks
    out_rows: List[List[Any]] = []
    out_rows.append(["as_of", ts])
    out_rows.append([])
    out_rows.append(["RISK PROFILE", "risk-neutral"])
    out_rows.append(list(df_neutral.columns))
    out_rows.extend(df_neutral.fillna("").astype(str).values.tolist())
    out_rows.append([])
    out_rows.append(["RISK PROFILE", "risk-averse"])
    out_rows.append(list(df_averse.columns))
    out_rows.extend(df_averse.fillna("").astype(str).values.tolist())

    # LOGS (truncate long fields to keep Sheets manageable)
    logs_small = logs.copy()
    for col in ["news", "valuation", "sentiment", "fundamental", "consensus"]:
        if col in logs_small.columns:
            logs_small[col] = logs_small[col].astype(str).str.slice(0, 1200)

    logs_rows: List[List[Any]] = [list(logs_small.columns)] + logs_small.fillna("").astype(str).values.tolist()

    # BACKTEST summary
    bt_rows: List[List[Any]] = []
    bt_rows.append(["as_of", ts])
    bt_rows.append([])
    bt_rows.append(["risk_profile", "backtest_start", "backtest_end", "top15_picks", "port_cum", "port_sharpe", "bench_cum", "bench_sharpe"])
    bt_rows.append([
        "risk-neutral",
        bt_neutral.get("backtest_start", ""),
        bt_neutral.get("backtest_end", ""),
        ",".join(bt_neutral.get("picks_top15", []) or []),
        bt_neutral.get("portfolio_cum_return", ""),
        bt_neutral.get("portfolio_sharpe", ""),
        bt_neutral.get("benchmark_cum_return", ""),
        bt_neutral.get("benchmark_sharpe", "")
    ])
    bt_rows.append([
        "risk-averse",
        bt_averse.get("backtest_start", ""),
        bt_averse.get("backtest_end", ""),
        ",".join(bt_averse.get("picks_top15", []) or []),
        bt_averse.get("portfolio_cum_return", ""),
        bt_averse.get("portfolio_sharpe", ""),
        bt_averse.get("benchmark_cum_return", ""),
        bt_averse.get("benchmark_sharpe", "")
    ])

    # Clean to avoid NaN/Inf issues
    out_rows = clean_2d(out_rows)
    logs_rows = clean_2d(logs_rows)
    bt_rows = clean_2d(bt_rows)

    ws_out.clear()
    ws_out.update(values=out_rows, range_name="A1")

    ws_logs.clear()
    ws_logs.update(values=logs_rows, range_name="A1")

    ws_bt.clear()
    ws_bt.update(values=bt_rows, range_name="A1")


def main():
    # Validate required env upfront
    required = ["SHEET_ID", "GSPREAD_SERVICE_ACCOUNT_JSON", "OPENAI_API_KEY", "FMP_API_KEY"]
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        raise RuntimeError(f"Missing required environment variables/secrets: {missing}")

    cik_map = fetch_ticker_cik_map()

    df_neutral, logs_neutral, bt_neutral = run_for_profile("risk-neutral", cik_map)
    df_averse, logs_averse, bt_averse = run_for_profile("risk-averse", cik_map)

    logs = pd.concat([logs_neutral, logs_averse], ignore_index=True)

    write_to_sheets(df_neutral, df_averse, logs, bt_neutral, bt_averse)

    print("OK: AlphaAgents written to Google Sheet (OUTPUT/LOGS/BACKTEST).")


if __name__ == "__main__":
    main()
