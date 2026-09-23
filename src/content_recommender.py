from __future__ import annotations

"""
content_recommender.py
----------------------
Content-based movie recommendation engine.

How it works:
  1. We have a TF-IDF matrix where each row represents a movie's "profile"
     built from its genres and user-applied tags.
  2. Given one or more query movies, we compute the cosine similarity between
     their TF-IDF vectors and every other movie in the catalog.
  3. The movies with the highest similarity scores are returned as
     recommendations.

This approach is especially good for the "cold start" problem — we can
recommend movies to a brand-new user as long as they tell us at least one
movie they like.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def recommend_similar(
    movie_titles: list[str],
    movies_df: pd.DataFrame,
    tfidf_matrix,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Find movies that are most similar to the given titles.

    Parameters
    ----------
    movie_titles : list[str]
        One or more movie titles (must match the 'title' column exactly).
    movies_df : pd.DataFrame
        The full merged movie dataset with an implicit integer index
        matching the rows of tfidf_matrix.
    tfidf_matrix : sparse matrix
        TF-IDF feature matrix (one row per movie, same order as movies_df).
    top_n : int
        Number of recommendations to return.

    Returns
    -------
    pd.DataFrame
        Recommended movies with a 'content_score' column (0-1),
        sorted descending by score.  Excludes the query movies themselves.
    """
    # Look up the row indices for the requested titles
    query_indices = []
    for title in movie_titles:
        mask = movies_df["title"] == title
        if mask.any():
            query_indices.append(movies_df[mask].index[0])

    if not query_indices:
        # No valid movies found — return an empty frame
        return pd.DataFrame(columns=list(movies_df.columns) + ["content_score"])

    # Average the TF-IDF vectors when multiple query movies are given.
    # This gives us a "user profile" that blends the metadata of all
    # movies the user liked.
    query_vectors = tfidf_matrix[query_indices]
    if len(query_indices) > 1:
        # .mean() on a sparse matrix returns np.matrix, which newer
        # versions of scikit-learn / NumPy reject.  Convert to array.
        combined_vector = np.asarray(query_vectors.mean(axis=0))
    else:
        combined_vector = query_vectors

    # Cosine similarity against the entire catalog
    sim_scores = cosine_similarity(combined_vector, tfidf_matrix).flatten()

    # Exclude the query movies themselves from the results
    for idx in query_indices:
        sim_scores[idx] = -1.0

    # Grab the top-N indices
    top_indices = sim_scores.argsort()[-top_n:][::-1]

    results = movies_df.iloc[top_indices].copy()
    results["content_score"] = sim_scores[top_indices]

    # Drop any results with zero similarity (shouldn't happen often)
    results = results[results["content_score"] > 0].reset_index(drop=True)

    return results


def recommend_by_genre(
    genres: list[str],
    movies_df: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Simple genre-based fallback recommender.

    Used when TF-IDF similarity isn't available (e.g. for a movie with
    no tags and very generic genre labels).

    Parameters
    ----------
    genres : list[str]
        Genres the user is interested in (e.g. ["Action", "Sci-Fi"]).
    movies_df : pd.DataFrame
        The merged movie dataset.
    top_n : int
        Number of results.

    Returns
    -------
    pd.DataFrame
        Top-rated movies that match *any* of the requested genres.
    """
    pattern = "|".join(genres)
    mask = movies_df["genres"].str.contains(pattern, case=False, na=False)
    matched = movies_df[mask].copy()

    # Sort by popularity score so we surface well-known films
    matched = matched.sort_values("popularity_score", ascending=False)
    return matched.head(top_n).reset_index(drop=True)
