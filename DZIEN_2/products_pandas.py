#!/usr/bin/env python3
"""
Demo: Pandas + SQLAlchemy + MySQL
- generuje DataFrame (5000 rekordów)
- zapisuje do MySQL jako tabelę
- pokazuje odczyt i proste zapytania

Ustaw zmienne środowiskowe lub wypełnij stałe poniżej:
  MYSQL_USER, MYSQL_PASS, MYSQL_HOST, MYSQL_PORT, MYSQL_DB
"""

from __future__ import annotations
import os
import math
import time
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.types import (
    Integer, BigInteger, String, Float, DateTime
)

# ------------------------
# 1) KONFIGURACJA POŁĄCZENIA
# ------------------------
MYSQL_USER = os.getenv("MYSQL_USER", "user")
MYSQL_PASS = os.getenv("MYSQL_PASS", "password")
MYSQL_HOST = os.getenv("MYSQL_HOST", "127.0.0.1")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_DB   = os.getenv("MYSQL_DB",   "demo_db")

# SQLAlchemy URL (z driverem PyMySQL)
DATABASE_URL = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASS}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"

# Opcjonalnie: jeżeli baza jeszcze nie istnieje, możesz:
# 1) połączyć się bez nazwy DB i utworzyć ją:
# server_engine = create_engine(f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASS}@{MYSQL_HOST}:{MYSQL_PORT}", pool_pre_ping=True, pool_recycle=1800)
# with server_engine.begin() as conn:
#     conn.execute(text(f"CREATE DATABASE IF NOT EXISTS {MYSQL_DB} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"))

# Silnik do docelowej bazy
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,      # zdrowie połączeń
    pool_recycle=1800,       # odświeżanie dłuższych połączeń
    echo=False               # ustaw True gdy chcesz log SQL
)

# ------------------------
# 2) GENERACJA DANYCH (5000 rekordów)
# ------------------------
rng = np.random.default_rng(seed=42)
N = 5000

def random_names(n: int) -> list[str]:
    first = np.array(["Aki", "Kai", "Ren", "Mika", "Sora", "Rin", "Kenta", "Yui", "Hana", "Taro"])
    last  = np.array(["Tanaka", "Sato", "Suzuki", "Takahashi", "Kobayashi", "Watanabe", "Ito", "Yamamoto"])
    return (rng.choice(first, n) + " " + rng.choice(last, n)).tolist()

categories = ["valves", "actuators", "seals", "sensors", "controllers"]
countries  = ["PL", "DE", "CZ", "SK", "SE", "NO", "FR", "IT"]

df = pd.DataFrame({
    "product_id": np.arange(1, N + 1, dtype=np.int64),
    "product_name": random_names(N),
    "category": rng.choice(categories, N, p=[0.35, 0.15, 0.2, 0.15, 0.15]),
    "quantity": rng.integers(1, 500, size=N, dtype=np.int32),
    # ceny ~ log-normal, zaokrąglone do 2 miejsc, minimum 5.00
    "unit_price": np.round(np.maximum(5.0, rng.lognormal(mean=3.0, sigma=0.5, size=N)), 2),
    # przychód = quantity * unit_price (dla wygody analizy)
    "revenue_est": 0.0,  # uzupełnimy niżej
    # losowy timestamp z ostatnich 365 dni
    "ts": pd.to_datetime("now").normalize() - pd.to_timedelta(rng.integers(0, 365, size=N), unit="D"),
    "country": rng.choice(countries, N, p=[0.4, 0.12, 0.08, 0.07, 0.08, 0.05, 0.1, 0.1])
})
df["revenue_est"] = (df["quantity"] * df["unit_price"]).round(2)

# ------------------------
# 3) ZAPIS DO MYSQL (to_sql)
# ------------------------
table_name = "products_demo"

# Mapowanie typów kolumn -> MySQL
dtype_map = {
    "product_id": BigInteger(),
    "product_name": String(120),
    "category": String(40),
    "quantity": Integer(),
    "unit_price": Float(asdecimal=True),
    "revenue_est": Float(asdecimal=True),
    "ts": DateTime(),
    "country": String(8),
}

# Tworzenie/odświeżenie tabeli (replace) i zapis chunkami
with engine.begin() as conn:
    df.to_sql(
        table_name,
        conn,
        if_exists="replace",
        index=False,
        dtype=dtype_map,
        chunksize=1000,   # bezpieczne porcjowanie
        method="multi"    # szybszy insert
    )

    # Indeksy, które przydadzą się do analiz/filtrów
    conn.execute(text(f"ALTER TABLE {table_name} ADD PRIMARY KEY (product_id)"))
    conn.execute(text(f"CREATE INDEX ix_{table_name}_category ON {table_name}(category)"))
    conn.execute(text(f"CREATE INDEX ix_{table_name}_ts ON {table_name}(ts)"))
    conn.execute(text(f"CREATE INDEX ix_{table_name}_country ON {table_name}(country)"))

print(f"✓ Wstawiono {len(df):,} rekordów do tabeli `{table_name}` w bazie `{MYSQL_DB}`.")

# ------------------------
# 4) ODCZYT I SZYBKA ANALIZA (pandas + SQL)
# ------------------------
# a) odczyt top 5
with engine.connect() as conn:
    preview = pd.read_sql(text(f"SELECT * FROM {table_name} ORDER BY product_id ASC LIMIT 5"), conn)
print("\nPodgląd 5 pierwszych wierszy:")
print(preview)

# b) agregacja po kategorii (SQL -> pandas)
with engine.connect() as conn:
    agg = pd.read_sql(
        text(f"""
            SELECT category,
                   COUNT(*) AS cnt,
                   SUM(quantity) AS total_qty,
                   ROUND(AVG(unit_price),2) AS avg_price,
                   ROUND(SUM(revenue_est),2) AS sum_revenue
            FROM {table_name}
            GROUP BY category
            ORDER BY sum_revenue DESC
        """),
        conn
    )
print("\nAgregacja po kategorii:")
print(agg)

# c) filtr po dacie i kraju (parametryzowany SQL)
from datetime import datetime, timedelta
date_from = (pd.Timestamp.utcnow() - pd.Timedelta(days=90)).to_pydatetime()
country = "PL"

with engine.connect() as conn:
    recent_pl = pd.read_sql(
        text(f"""
            SELECT *
            FROM {table_name}
            WHERE ts >= :date_from AND country = :country
            ORDER BY ts DESC
            LIMIT 10
        """),
        conn,
        params={"date_from": date_from, "country": country}
    )

print("\nOstatnie 10 rekordów z 90 dni dla PL:")
print(recent_pl)
