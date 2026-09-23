"""
data_loader.py
--------------
Loads and preprocesses the MovieLens Latest Small dataset.

Responsibilities:
  - Read the raw CSV files (movies, ratings, tags)
  - Extract release year from title strings
  - Aggregate user-applied tags into a single string per movie
  - Compute per-movie statistics (avg rating, rating count, popularity)
  - Return clean, merged DataFrames ready for the recommendation engines
"""

import os
import re

import numpy as np
import pandas as pd
import streamlit as st


# Path to the extracted dataset folder
_BASE_DIR = os.path.dirname(os.path.dirname(__file__))
_DATASET_DIR = os.path.join(_BASE_DIR, "data", "ml-latest-small")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_year(title: str):
    """Pull the (YYYY) from the end of a MovieLens title string."""
    match = re.search(r"\((\d{4})\)\s*$", title)
    return int(match.group(1)) if match else np.nan


def _clean_title(title: str) -> str:
    """Remove the trailing (YYYY) from a title."""
    return re.sub(r"\s*\(\d{4}\)\s*$", "", title).strip()


# ---------------------------------------------------------------------------
# Public API  (all functions are cached so they only run once per session)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_movies() -> pd.DataFrame:
    """
    Load movies.csv and enrich it with parsed year and cleaned title.

    Columns returned:
        movieId, title, genres, year, clean_title
    """
    path = os.path.join(_DATASET_DIR, "movies.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"movies.csv not found at {path}. Run `python setup_data.py` first."
        )

    df = pd.read_csv(path)

    # Parse out the release year and a cleaner title
    df["year"] = df["title"].apply(_extract_year)
    df["clean_title"] = df["title"].apply(_clean_title)

    return df


@st.cache_data(show_spinner=False)
def load_ratings() -> pd.DataFrame:
    """
    Load ratings.csv and convert the Unix timestamp to a datetime.

    Columns returned:
        userId, movieId, rating, timestamp, date
    """
    path = os.path.join(_DATASET_DIR, "ratings.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"ratings.csv not found at {path}. Run `python setup_data.py` first."
        )

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["timestamp"], unit="s")
    return df


@st.cache_data(show_spinner=False)
def load_tags() -> pd.DataFrame:
    """
    Load tags.csv — user-applied free-text tags for movies.

    Columns returned:
        userId, movieId, tag, timestamp
    """
    path = os.path.join(_DATASET_DIR, "tags.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"tags.csv not found at {path}. Run `python setup_data.py` first."
        )

    df = pd.read_csv(path)
    # Some tags may be NaN — drop them
    df = df.dropna(subset=["tag"])
    df["tag"] = df["tag"].astype(str).str.lower().str.strip()
    return df


@st.cache_data(show_spinner=False)
def aggregate_tags() -> pd.DataFrame:
    """
    Combine all tags for each movie into a single space-delimited string.

    Returns a DataFrame with columns: movieId, tags_combined
    """
    tags = load_tags()
    agg = (
        tags.groupby("movieId")["tag"]
        .apply(lambda x: " ".join(x))
        .reset_index()
        .rename(columns={"tag": "tags_combined"})
    )
    return agg


@st.cache_data(show_spinner=False)
def compute_movie_stats() -> pd.DataFrame:
    """
    Compute per-movie rating statistics.

    Returns a DataFrame with columns:
        movieId, avg_rating, rating_count, popularity_score
    
    popularity_score uses a weighted-rating formula (similar to IMDB's):
        WR = (v / (v + m)) * R  +  (m / (v + m)) * C
    where:
        v = number of votes for the movie
        m = minimum votes required (25th percentile)
        R = average rating for the movie
        C = mean rating across the entire dataset
    """
    ratings = load_ratings()

    stats = ratings.groupby("movieId")["rating"].agg(
        avg_rating="mean",
        rating_count="count"
    ).reset_index()

    # Weighted-rating formula for a fairer popularity score
    C = stats["avg_rating"].mean()
    m = stats["rating_count"].quantile(0.25)

    stats["popularity_score"] = (
        (stats["rating_count"] / (stats["rating_count"] + m)) * stats["avg_rating"]
        + (m / (stats["rating_count"] + m)) * C
    )

    stats["avg_rating"] = stats["avg_rating"].round(2)
    stats["popularity_score"] = stats["popularity_score"].round(3)

    return stats


@st.cache_data(show_spinner=False)
def build_movie_dataset() -> pd.DataFrame:
    """
    Master function — merges movies + aggregated tags + rating stats
    into a single DataFrame ready for the recommendation pipeline.

    Columns returned:
        movieId, title, genres, year, clean_title,
        tags_combined, avg_rating, rating_count, popularity_score
    """
    movies = load_movies()
    tags_agg = aggregate_tags()
    stats = compute_movie_stats()

    # Left join so we keep all movies, even those with no tags or ratings
    merged = movies.merge(tags_agg, on="movieId", how="left")
    merged = merged.merge(stats, on="movieId", how="left")

    # Fill missing values sensibly
    merged["tags_combined"] = merged["tags_combined"].fillna("")
    merged["avg_rating"] = merged["avg_rating"].fillna(0.0)
    merged["rating_count"] = merged["rating_count"].fillna(0).astype(int)
    merged["popularity_score"] = merged["popularity_score"].fillna(0.0)

    return merged
