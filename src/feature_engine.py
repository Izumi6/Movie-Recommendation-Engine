"""
feature_engine.py
-----------------
Builds the TF-IDF feature matrix used by the content-based recommender.

The idea is simple: for each movie we concatenate its genres and user-applied
tags into a single "soup" string, then vectorize the whole corpus with TF-IDF.
The resulting sparse matrix lets us measure how similar any two movies are
based purely on their textual metadata.
"""

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer


@st.cache_resource(show_spinner=False)
def build_tfidf_matrix(_movies_df: pd.DataFrame):
    """
    Construct a TF-IDF matrix from the combined genres + tags of every movie.

    Parameters
    ----------
    _movies_df : pd.DataFrame
        The merged movie dataset (output of data_loader.build_movie_dataset).
        Must contain columns: 'genres', 'tags_combined'.

    Returns
    -------
    tfidf_matrix : sparse CSR matrix, shape (n_movies, n_features)
        Each row is a movie's TF-IDF vector.
    feature_names : list[str]
        The vocabulary terms corresponding to each column in the matrix.
    """
    # Build the "soup" — genres converted from pipe-delimited to spaces,
    # then concatenated with the free-text tags.  Lowercased for consistency.
    soup = (
        _movies_df["genres"]
        .str.replace("|", " ", regex=False)
        .str.lower()
        + " "
        + _movies_df["tags_combined"].fillna("").str.lower()
    )

    vectorizer = TfidfVectorizer(
        max_features=5000,       # keep the matrix manageable
        stop_words="english",   # drop common English filler words
        min_df=2,                # ignore terms that appear in < 2 movies
        ngram_range=(1, 2),      # capture bigrams like "sci fi", "dark comedy"
    )

    tfidf_matrix = vectorizer.fit_transform(soup)
    feature_names = vectorizer.get_feature_names_out().tolist()

    return tfidf_matrix, feature_names


def get_top_features(movie_idx: int, tfidf_matrix, feature_names, top_n: int = 10):
    """
    Return the highest-weighted TF-IDF terms for a single movie.

    Useful for explaining *why* two movies are considered similar.

    Parameters
    ----------
    movie_idx : int
        Row index (not movieId!) in the TF-IDF matrix.
    tfidf_matrix : sparse matrix
        The matrix from build_tfidf_matrix().
    feature_names : list[str]
        Vocabulary list from build_tfidf_matrix().
    top_n : int
        How many top terms to return.

    Returns
    -------
    list[tuple[str, float]]
        Pairs of (term, tfidf_weight), sorted descending by weight.
    """
    row = tfidf_matrix[movie_idx].toarray().flatten()
    top_indices = row.argsort()[-top_n:][::-1]
    return [(feature_names[i], round(float(row[i]), 4)) for i in top_indices if row[i] > 0]
