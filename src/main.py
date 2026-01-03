import os
import json
import time
import math
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple

import requests
import feedparser
import numpy as np
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from bs4 import BeautifulSoup

from openai import OpenAI


# ----------------------------
# CONFIG
# ----------------------------
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
MAX_FILING_CHARS = 14000      # recorte para tokens
DEBATE_MAX_ROUNDS = 2

# Reuters RSS Markets (general) + Yahoo Finance RSS por ticker
REUTERS_MARKETS_RSS = "https://www.reuters.com/markets/rss"
YAHOO_TICKER_RSS_TEMPLATE = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"

SEC_TICKER_CIK_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_DATA_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_nodash}/{primary_doc}"


# ----------------------------
# UTILS
# ----------------------------
def now_buenos_aires_str() -> str:
    tz = timezone(timedelta(hours=-3))
    return datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S (America/Argentina/Buenos_Aires)")


def open_sheet():
    sheet_id = os.environ["SHEET_ID"].strip()
    info = json.loads(os.environ["GSPREAD_SERVICE_ACCOUNT_JSON"])
    scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    creds = Credentials.from_service_account_info(info, scopes=scopes)
    gc = gspread.authorize(creds)
    return gc.open_by_key(sheet_id)


def get_or_create_ws(sh, title: str, rows=2000, cols=40):
    try:
        return sh.worksheet(title)
    except gspread.WorksheetNotFound:
        return sh.add_worksheet(title=title, rows=rows, cols=cols)


def safe_sleep(sec: float):
    time.sleep(sec)


def sec_headers() -> Dict[str, str]:
    # SEC pide User-Agent real. Poné uno identificable.
    return {
        "User-Agent": "AlphaAgentsBot/1.0 (contact: you@example.com)",
        "Accept-Encoding": "gzip, deflate",
        "Host": "www.sec.gov"
    }


# ----------------------------
# DATA: PRICES (Valuation Agent tool)
# ----------------------------
def fetch_price_metrics(ticker: str) -> Dict[str, float]:
    api_key = os.environ["FMP_API_KEY"].strip()
    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}"
    params = {"timeseries": 260, "apikey": api_key}

    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    hist = data.get("historical", [])
    if not hist or len(hist) < 30:
        return {"price": np.nan, "ret_3m": np.nan, "ret_6m": np.nan, "vol_1y": np.nan, "max_dd_1y": np.nan}

    # FMP devuelve del más nuevo al más viejo (normalmente).
    closes = [h.get("close") for h in hist if h.get("close") is not None]
    closes = [float(x) for x in closes if isinstance(x, (int, float))]

    if len(closes) < 30:
        return {"price": np.nan, "ret_3m": np.nan, "ret_6m": np.nan, "vol_1y": np.nan, "max_dd_1y": np.nan}

    price = closes[0]

    def pct_return(days: int) -> float:
        # days ~ trading days back, index days
        if len(closes) <= days:
            return np.nan
        return float((closes[0] / closes[days]) - 1.0)

    ret_3m = pct_return(63)
    ret_6m = pct_return(126)

    # daily returns for vol / drawdown
    close_series = pd.Series(list(reversed(closes)))  # oldest -> newest
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



# ----------------------------
# DATA: NEWS (Sentiment Agent tool)
# ----------------------------
def fetch_news_items(ticker: str) -> List[Dict[str, str]]:
    items = []

    # Yahoo RSS per ticker
    yahoo_url = YAHOO_TICKER_RSS_TEMPLATE.format(ticker=ticker)
    y = feedparser.parse(yahoo_url)
    for e in (y.entries or [])[:MAX_NEWS_ITEMS_PER_TICKER]:
        items.append({
            "source": "YahooRSS",
            "title": e.get("title", ""),
            "link": e.get("link", ""),
            "published": e.get("published", "")
        })

    # Reuters markets general (no ticker-specific)
    # (se usa como contexto macro/mercado; NO siempre menciona el ticker)
    r = feedparser.parse(REUTERS_MARKETS_RSS)
    for e in (r.entries or [])[:3]:
        items.append({
            "source": "ReutersRSS",
            "title": e.get("title", ""),
            "link": e.get("link", ""),
            "published": e.get("published", "")
        })

    return items[:MAX_NEWS_ITEMS_PER_TICKER + 3]


# ----------------------------
# DATA: SEC FILINGS (Fundamental Agent tool)
# ----------------------------
def fetch_ticker_cik_map() -> Dict[str, str]:
    resp = requests.get(SEC_TICKER_CIK_URL, headers=sec_headers(), timeout=30)
    resp.raise_for_status()
    data = resp.json()
    # company_tickers.json: dict of index -> {ticker, cik_str, ...}
    out = {}
    for _, row in data.items():
        t = row.get("ticker", "")
        cik = str(row.get("cik_str", "")).zfill(10)
        if t:
            out[t.upper()] = cik
    return out


def pick_latest_10q_10k(submissions_json: Dict) -> Tuple[str, str, str]:
    """
    Returns: (form, accessionNumber, primaryDocument)
    """
    recent = submissions_json.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accessions = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])

    for form, acc, doc in zip(forms, accessions, primary_docs):
        if form in ("10-Q", "10-K"):
            return form, acc, doc
    return "", "", ""


def fetch_latest_filing_text(ticker: str, cik: str) -> Dict[str, str]:
    sub_url = SEC_DATA_SUBMISSIONS.format(cik=cik)
    resp = requests.get(sub_url, headers=sec_headers(), timeout=30)
    resp.raise_for_status()
    sub = resp.json()

    form, acc, doc = pick_latest_10q_10k(sub)
    if not form:
        return {"form": "", "url": "", "text": ""}

    cik_int = str(int(cik))  # remove leading zeros
    acc_nodash = acc.replace("-", "")
    filing_url = SEC_ARCHIVES_BASE.format(cik_int=cik_int, accession_nodash=acc_nodash, primary_doc=doc)

    html = requests.get(filing_url, headers=sec_headers(), timeout=30)
    html.raise_for_status()

    soup = BeautifulSoup(html.text, "lxml")
    text = soup.get_text(separator="\n")
    text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])

    # focus sections (rough)
    # keep only chunk around "Item 1A" / "Risk Factors" etc if present; otherwise top slice
    lowered = text.lower()
    anchors = ["item 1a", "risk factors", "management’s discussion", "md&a", "item 7"]
    idx = min([lowered.find(a) for a in anchors if lowered.find(a) != -1] + [0])
    chunk = text[idx:idx + MAX_FILING_CHARS]

    return {"form": form, "url": filing_url, "text": chunk}


# ----------------------------
# LLM AGENTS
# ----------------------------
def llm_client() -> OpenAI:
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"].strip())


def call_llm(system: str, user: str, model: str = "gpt-4o-mini") -> str:
    # Usamos un modelo económico por default; podés subirlo después.
    client = llm_client()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


def valuation_agent(ticker: str, m: Dict[str, float], risk_profile: str) -> Dict[str, str]:
    system = f"""You are a valuation equity analyst. Risk profile: {risk_profile}.
Output MUST be valid JSON with keys: vote, rationale, risks, invalidation.
Allowed vote: BUY, SELL.
Be explicit and concise. Use only the provided metrics."""
    user = f"""Ticker: {ticker}
Metrics:
price={m['price']}
ret_3m={m['ret_3m']}
ret_6m={m['ret_6m']}
vol_1y={m['vol_1y']}
max_dd_1y={m['max_dd_1y']}

Decide BUY or SELL given the risk profile."""
    txt = call_llm(system, user)
    return json.loads(txt)


def sentiment_agent(ticker: str, news: List[Dict[str, str]], risk_profile: str) -> Dict[str, str]:
    system = f"""You are a sentiment equity analyst. Risk profile: {risk_profile}.
Output MUST be valid JSON with keys: vote, rationale, risks, invalidation.
Allowed vote: BUY, SELL.
Base your analysis ONLY on the news items provided. If news is weak/irrelevant, say so and decide anyway."""
    user = f"""Ticker: {ticker}
News items (may include general market context):
{json.dumps(news, indent=2)}

Decide BUY or SELL."""
    txt = call_llm(system, user)
    return json.loads(txt)


def fundamental_agent(ticker: str, filing: Dict[str, str], risk_profile: str) -> Dict[str, str]:
    system = f"""You are a fundamental equity analyst. Risk profile: {risk_profile}.
Output MUST be valid JSON with keys: vote, rationale, risks, invalidation.
Allowed vote: BUY, SELL.
Base your analysis ONLY on the SEC filing text provided (10-K/10-Q excerpt)."""
    user = f"""Ticker: {ticker}
Filing form: {filing.get('form')}
Filing url: {filing.get('url')}
Filing text excerpt:
{filing.get('text')}

Decide BUY or SELL."""
    txt = call_llm(system, user)
    return json.loads(txt)


def debate_and_consensus(ticker: str, a: Dict[str, str], b: Dict[str, str], c: Dict[str, str], risk_profile: str) -> Dict[str, str]:
    """
    a=valuation, b=sentiment, c=fundamental
    If disagreement, run up to DEBATE_MAX_ROUNDS to converge.
    """
    system = f"""You coordinate a multi-agent debate (valuation, sentiment, fundamental). Risk profile: {risk_profile}.
You MUST output valid JSON with keys: final_vote, confidence_0_100, synthesis, key_risks.
final_vote must be BUY or SELL.
You must reconcile disagreements. Use at most {DEBATE_MAX_ROUNDS} rounds internally (simulate)."""

    user = f"""Ticker: {ticker}
Valuation agent output: {json.dumps(a)}
Sentiment agent output: {json.dumps(b)}
Fundamental agent output: {json.dumps(c)}

Return the consensus."""
    txt = call_llm(system, user)
    return json.loads(txt)


# ----------------------------
# BACKTEST (paper-style downstream metric)
# ----------------------------
def backtest_equal_weight(picks: List[str], benchmark_universe: List[str], start: str, end: str) -> Dict[str, float]:
    # TODO: reimplementar con FMP. Por ahora lo desactivamos para no romper el pipeline.
    return {
        "portfolio_cum_return": np.nan,
        "portfolio_sharpe": np.nan,
        "benchmark_cum_return": np.nan,
        "benchmark_sharpe": np.nan
    }

    """
    Basic backtest: equal-weight portfolio of picks vs equal-weight universe benchmark.
    Returns cumulative returns and sharpe (simple).
    """
    def port_return(tickers: List[str]) -> Tuple[float, float]:
        if not tickers:
            return np.nan, np.nan
        prices = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
        if isinstance(prices, pd.Series):
            prices = prices.to_frame()
        prices = prices.dropna(how="all")
        rets = prices.pct_change().dropna()
        # equal weight
        port = rets.mean(axis=1)
        cum = float((1 + port).prod() - 1)
        sharpe = float((port.mean() / (port.std() + 1e-9)) * math.sqrt(252))
        return cum, sharpe

    p_cum, p_sh = port_return(picks)
    b_cum, b_sh = port_return(benchmark_universe)

    return {
        "portfolio_cum_return": p_cum,
        "portfolio_sharpe": p_sh,
        "benchmark_cum_return": b_cum,
        "benchmark_sharpe": b_sh
    }


# ----------------------------
# PIPELINE
# ----------------------------
def run_for_profile(risk_profile: str) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    cik_map = fetch_ticker_cik_map()

    rows = []
    logs = []

    for i, t in enumerate(TICKERS_50, start=1):
        try:
            # rate limit a bit
            if i % 10 == 0:
                safe_sleep(0.6)

            metrics = fetch_price_metrics(t)
            news = fetch_news_items(t)

            cik = cik_map.get(t.upper(), "")
            filing = {"form": "", "url": "", "text": ""}
            if cik:
                # SEC rate limiting
                safe_sleep(0.25)
                filing = fetch_latest_filing_text(t, cik)

            # Agents
            va = valuation_agent(t, metrics, risk_profile)
            sa = sentiment_agent(t, news, risk_profile)
            fa = fundamental_agent(t, filing, risk_profile)

            # Debate / consensus
            cons = debate_and_consensus(t, va, sa, fa, risk_profile)

            rows.append({
                "ticker": t,
                "price": metrics["price"],
                "ret_3m": metrics["ret_3m"],
                "ret_6m": metrics["ret_6m"],
                "vol_1y": metrics["vol_1y"],
                "max_dd_1y": metrics["max_dd_1y"],
                "final_vote": cons["final_vote"],
                "confidence_0_100": cons["confidence_0_100"],
                "synthesis": cons["synthesis"],
                "key_risks": cons["key_risks"],
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
            rows.append({"ticker": t, "final_vote": "SELL", "confidence_0_100": 0, "synthesis": f"ERROR: {e}", "key_risks": "N/A"})
            logs.append({"as_of": now_buenos_aires_str(), "risk_profile": risk_profile, "ticker": t, "error": str(e)})

    df = pd.DataFrame(rows)
    # Rank: BUY first, then higher confidence
    df["buy_flag"] = (df["final_vote"] == "BUY").astype(int)
    df = df.sort_values(["buy_flag", "confidence_0_100"], ascending=[False, False]).drop(columns=["buy_flag"])

    log_df = pd.DataFrame(logs)

    # Paper-style backtest window (example): last 4 months
    # (no es “en vivo”; es una métrica downstream como en el paper)
    end = datetime.utcnow().date().isoformat()
    start = (datetime.utcnow() - pd.Timedelta(days=120)).date().isoformat()

    picks = df[df["final_vote"] == "BUY"]["ticker"].head(15).tolist()  # como el paper: pool tech=15; acá top15 BUY
    bt = backtest_equal_weight(picks, TICKERS_50, start=start, end=end)

    return df, log_df, {"backtest_start": start, "backtest_end": end, "picks_top15": picks, **bt}


def write_to_sheets(df_neutral: pd.DataFrame, df_averse: pd.DataFrame, logs: pd.DataFrame, bt_neutral: Dict, bt_averse: Dict):
    sh = open_sheet()

    ws_out = get_or_create_ws(sh, SHEET_TAB_OUTPUT, rows=2000, cols=40)
    ws_logs = get_or_create_ws(sh, SHEET_TAB_LOGS, rows=4000, cols=20)
    ws_bt = get_or_create_ws(sh, SHEET_TAB_BACKTEST, rows=200, cols=20)

    # OUTPUT: two blocks
    ts = now_buenos_aires_str()

    out_rows = []
    out_rows.append(["as_of", ts])
    out_rows.append([])
    out_rows.append(["RISK PROFILE", "risk-neutral"])
    out_rows.append(list(df_neutral.columns))
    out_rows.extend(df_neutral.fillna("").astype(str).values.tolist())
    out_rows.append([])
    out_rows.append(["RISK PROFILE", "risk-averse"])
    out_rows.append(list(df_averse.columns))
    out_rows.extend(df_averse.fillna("").astype(str).values.tolist())

    ws_out.clear()
out_rows = [[_clean_cell(c) for c in row] for row in out_rows]

    # LOGS (truncate to keep sheet manageable)
    logs_small = logs.copy()
    if "news" in logs_small.columns:
        logs_small["news"] = logs_small["news"].astype(str).str.slice(0, 800)
    for col in ["valuation", "sentiment", "fundamental", "consensus"]:
        if col in logs_small.columns:
            logs_small[col] = logs_small[col].astype(str).str.slice(0, 1200)

    logs_rows = [list(logs_small.columns)] + logs_small.fillna("").astype(str).values.tolist()
    ws_logs.clear()
logs_rows = [[_clean_cell(c) for c in row] for row in logs_rows]

    # BACKTEST summary
    bt_rows = []
    bt_rows.append(["as_of", ts])
    bt_rows.append([])
    bt_rows.append(["risk_profile", "backtest_start", "backtest_end", "top15_picks", "port_cum", "port_sharpe", "bench_cum", "bench_sharpe"])
    bt_rows.append(["risk-neutral", bt_neutral["backtest_start"], bt_neutral["backtest_end"], ",".join(bt_neutral["picks_top15"]),
                    bt_neutral["portfolio_cum_return"], bt_neutral["portfolio_sharpe"], bt_neutral["benchmark_cum_return"], bt_neutral["benchmark_sharpe"]])
    bt_rows.append(["risk-averse", bt_averse["backtest_start"], bt_averse["backtest_end"], ",".join(bt_averse["picks_top15"]),
                    bt_averse["portfolio_cum_return"], bt_averse["portfolio_sharpe"], bt_averse["benchmark_cum_return"], bt_averse["benchmark_sharpe"]])

    ws_bt.clear()
    def _clean_cell(x):
    if x is None:
        return ""
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return ""
    return x

bt_rows = [[_clean_cell(c) for c in row] for row in bt_rows]
ws_bt.update(values=bt_rows, range_name="A1")

    ws_bt.update("A1", bt_rows)


def main():
    df_neutral, logs_neutral, bt_neutral = run_for_profile("risk-neutral")
    df_averse, logs_averse, bt_averse = run_for_profile("risk-averse")

    logs = pd.concat([logs_neutral, logs_averse], ignore_index=True)
    write_to_sheets(df_neutral, df_averse, logs, bt_neutral, bt_averse)

    print("OK: AlphaAgents written to Google Sheet.")


if __name__ == "__main__":
    main()
