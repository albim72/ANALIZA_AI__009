#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analiza danych syntetycznych klientów (CSV) z użyciem NumPy, pandas, matplotlib.

Uruchomienie (w tym samym folderze co CSV):
    python analiza_klienci.py --csv dane_syntetyczne_faker.csv --out out

Wyniki:
- Konsola: podsumowania, tabele przestawne.
- Katalog --out: wykresy PNG i pliki CSV z agregacjami.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8")
    # sanity check i typy
    expected_cols = ["idklienta", "imię", "nazwisko", "miasto", "liczba_zamówień", "kategoria"]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Brak kolumn w CSV: {missing}")
    df["idklienta"] = pd.to_numeric(df["idklienta"], errors="coerce").astype("Int64")
    df["liczba_zamówień"] = pd.to_numeric(df["liczba_zamówień"], errors="coerce")
    return df


def basic_overview(df: pd.DataFrame):
    print("\n=== PODSTAWOWY PRZEGLĄD ===")
    print("Liczba rekordów:", len(df))
    print("Unikalnych klientów (idklienta):", df["idklienta"].nunique())
    print("\nPrzykładowe wiersze:\n", df.head(5))
    print("\nBraki danych per kolumna:\n", df.isna().sum())
    print("\nStatystyki liczby zamówień:\n", df["liczba_zamówień"].describe())


def ensure_out_dir(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)


def plot_and_save(fig, out_path: Path):
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def analyze_categories(df: pd.DataFrame, out_dir: Path):
    print("\n=== KATEGORIE ===")
    cat_counts = df["kategoria"].value_counts(dropna=False).sort_index()
    cat_ratio = (cat_counts / len(df)).round(4)
    cat_summary = pd.DataFrame({"liczba": cat_counts, "udział": cat_ratio})
    print(cat_summary)

    # zapis
    cat_summary.to_csv(out_dir / "kategorie_podsumowanie.csv", encoding="utf-8")

    # wykres udziałów kategorii (bar)
    fig = plt.figure()
    cat_counts.plot(kind="bar")
    plt.title("Liczba klientów w kategoriach")
    plt.xlabel("Kategoria")
    plt.ylabel("Liczba klientów")
    plot_and_save(fig, out_dir / "kategorie_bar.png")

    # wykres udziału (pie)
    fig = plt.figure()
    plt.pie(cat_counts.values, labels=cat_counts.index, autopct="%1.1f%%", startangle=90)
    plt.title("Udział kategorii")
    plot_and_save(fig, out_dir / "kategorie_pie.png")


def analyze_orders(df: pd.DataFrame, out_dir: Path):
    print("\n=== LICZBA ZAMÓWIEŃ ===")
    orders = df["liczba_zamówień"].values
    print("Min/Max:", np.min(orders), np.max(orders))
    print("Średnia/Mediana/Std:", np.mean(orders).round(2), np.median(orders), np.std(orders).round(2))

    # histogram liczby zamówień
    fig = plt.figure()
    plt.hist(orders, bins=57)  # zakres 4-60 => około 57 wartości
    plt.title("Histogram liczby zamówień")
    plt.xlabel("Liczba zamówień")
    plt.ylabel("Liczba klientów")
    plot_and_save(fig, out_dir / "liczba_zamowien_hist.png")

    # boxplot
    fig = plt.figure()
    plt.boxplot(orders, vert=True, showmeans=True)
    plt.title("Boxplot liczby zamówień")
    plt.ylabel("Liczba zamówień")
    plot_and_save(fig, out_dir / "liczba_zamowien_box.png")


def analyze_cities(df: pd.DataFrame, out_dir: Path, top_n: int = 20):
    print("\n=== MIASTA ===")
    city_counts = df["miasto"].value_counts().head(top_n)
    print(f"TOP {top_n} miast:\n", city_counts)

    city_counts.to_csv(out_dir / "miasta_top.csv", encoding="utf-8")

    fig = plt.figure()
    city_counts.plot(kind="bar")
    plt.title(f"TOP {top_n} miast (liczba klientów)")
    plt.xlabel("Miasto")
    plt.ylabel("Liczba klientów")
    plt.xticks(rotation=45, ha="right")
    plot_and_save(fig, out_dir / "miasta_top_bar.png")


def analyze_names(df: pd.DataFrame, out_dir: Path, top_n: int = 20):
    print("\n=== IMIONA / NAZWISKA ===")
    first_names = df["imię"].value_counts().head(top_n)
    last_names = df["nazwisko"].value_counts().head(top_n)

    print(f"TOP {top_n} imion:\n", first_names)
    print(f"\nTOP {top_n} nazwisk:\n", last_names)

    first_names.to_csv(out_dir / "imiona_top.csv", encoding="utf-8")
    last_names.to_csv(out_dir / "nazwiska_top.csv", encoding="utf-8")

    fig = plt.figure()
    first_names.plot(kind="bar")
    plt.title(f"TOP {top_n} imion")
    plt.xlabel("Imię")
    plt.ylabel("Liczba klientów")
    plt.xticks(rotation=45, ha="right")
    plot_and_save(fig, out_dir / "imiona_top_bar.png")

    fig = plt.figure()
    last_names.plot(kind="bar")
    plt.title(f"TOP {top_n} nazwisk")
    plt.xlabel("Nazwisko")
    plt.ylabel("Liczba klientów")
    plt.xticks(rotation=45, ha="right")
    plot_and_save(fig, out_dir / "nazwiska_top_bar.png")


def analyze_category_vs_orders(df: pd.DataFrame, out_dir: Path):
    print("\n=== KATEGORIA vs. LICZBA ZAMÓWIEŃ ===")
    pivot = df.pivot_table(index="kategoria", values="liczba_zamówień",
                           aggfunc=["count", "mean", "median", "std"])
    pivot.columns = ["liczba_klientów", "średnia_zamówień", "mediana_zamówień", "std_zamówień"]
    print(pivot)

    pivot.to_csv(out_dir / "kategoria_vs_zamowienia.csv", encoding="utf-8")

    # średnia zamówień per kategoria
    fig = plt.figure()
    pivot["średnia_zamówień"].plot(kind="bar")
    plt.title("Średnia liczba zamówień w kategoriach")
    plt.xlabel("Kategoria")
    plt.ylabel("Średnia liczba zamówień")
    plot_and_save(fig, out_dir / "kategoria_srednia_zamowien_bar.png")


def analyze_city_vs_category(df: pd.DataFrame, out_dir: Path, top_n: int = 15):
    # bierzemy top miasta wg liczby klientów
    top_cities = df["miasto"].value_counts().head(top_n).index
    sub = df[df["miasto"].isin(top_cities)]
    crosstab = pd.crosstab(sub["miasto"], sub["kategoria"], normalize="index").round(3)
    print("\n=== STRUKTURA KATEGORII w TOP MIASTACH ===\n", crosstab)

    crosstab.to_csv(out_dir / "miasto_vs_kategoria_top.csv", encoding="utf-8")

    # wykres (każda kategoria jako słupek obok siebie per miasto)
    fig = plt.figure()
    ax = plt.gca()
    crosstab.plot(kind="bar", ax=ax)
    plt.title(f"Udział kategorii w TOP {top_n} miastach")
    plt.xlabel("Miasto")
    plt.ylabel("Udział w ramach miasta")
    plt.xticks(rotation=45, ha="right")
    plot_and_save(fig, out_dir / "miasto_vs_kategoria_top_bar.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="dane_syntetyczne_faker.csv",
                        help="Ścieżka do pliku CSV z danymi.")
    parser.add_argument("--out", type=str, default="out",
                        help="Katalog wyjściowy na wykresy i tabele.")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out)
    ensure_out_dir(out_dir)

    print("Wczytywanie danych z:", csv_path)
    df = load_data(csv_path)

    basic_overview(df)
    analyze_categories(df, out_dir)
    analyze_orders(df, out_dir)
    analyze_cities(df, out_dir, top_n=20)
    analyze_names(df, out_dir, top_n=20)
    analyze_category_vs_orders(df, out_dir)
    analyze_city_vs_category(df, out_dir, top_n=15)

    print(f"\nZapisano wyniki do katalogu: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
