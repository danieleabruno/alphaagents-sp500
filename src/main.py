import os
import json
from datetime import datetime, timezone, timedelta

import gspread
from google.oauth2.service_account import Credentials

TICKERS_50 = [
    "AAPL","MSFT","GOOGL","AMZN","META","NVDA","AVGO","AMD","ASML","TSM",
    "ORCL","CRM","ADBE","NFLX","COST","QCOM","INTU","IBM","TXN","AMAT",
    "SNOW","DDOG","MDB","NOW","PANW","CRWD","ZS","SHOP","UBER","ABNB",
    "MELI","COIN","SMCI","ARM","MU","KLAC","LRCX","ANET","FTNT","CDNS",
    "CAT","DE","ETN","HON","GE","RTX","LMT","ITW","EMR","PH"
]

def now_buenos_aires():
    tz = timezone(timedelta(hours=-3))
    return datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S (Argentina)")

def open_sheet():
    sheet_id = os.environ["SHEET_ID"]
    info = json.loads(os.environ["GSPREAD_SERVICE_ACCOUNT_JSON"])

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_info(info, scopes=scopes)
    gc = gspread.authorize(creds)
    return gc.open_by_key(sheet_id)

def get_or_create_output(sh):
    try:
        return sh.worksheet("OUTPUT")
    except:
        return sh.add_worksheet(title="OUTPUT", rows=300, cols=10)

def main():
    sh = open_sheet()
    ws = get_or_create_output(sh)

    rows = []
    rows.append(["as_of", now_buenos_aires()])
    rows.append([])
    rows.append(["ticker"])

    for t in TICKERS_50:
        rows.append([t])

    ws.clear()
    ws.update("A1", rows)

    print("OK – Sheet actualizado")

if __name__ == "__main__":
    main()
