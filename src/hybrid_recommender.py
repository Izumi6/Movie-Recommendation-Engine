from __future__ import annotations

"""
hybrid_recommender.py
---------------------
Weighted hybrid recommendation engine that blends content-based and
collaborative filtering scores.

The core formula:
    hybrid_score = alpha * content_score + (1 - alpha) * collab_score

alpha is a tunable parameter (0 to 1) exposed in the Streamlit sidebar:
  - alpha = 1.0  →  pure content-based (good for cold-start / new users)
  - alpha = 0.0  →  pure collaborative  (good when we have rich user history)
  - alpha = 0.5  →  balanced blend       (the default)

The module handles edge cases:
  - If a movie only has a content score (no collaborative data), it uses
    content-based alone.
  - If a movie only has a collaborative score, it uses that alone.
"""

import pandas as pd
import numpy as np

from src.content_recommender import recommend_similar
from src.collaborative_recommender import (
    recommend_for_user,
    recommend_for_new_user,
)


def hybrid_recommend(
    movie_titles: list[str],
    movies_df: pd.DataFrame,
    tfidf_matrix,
    ratings_df: pd.DataFrame,
    predicted_ratings: np.ndarray,
    user_index: dict,
    movie_index: dict,
    alpha: float = 0.5,
    user_id: int | None = None,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Generate hybrid recommendations by merging content and collaborative results.

    Parameters
    ----------
    movie_titles : list[str]
        Movies the user has selected / liked.
    movies_df : pd.DataFrame
        Full merged movie dataset.
    tfidf_matrix : sparse matrix
        TF-IDF feature matrix.
    ratings_df : pd.DataFrame
        Raw ratings dataframe.
    predicted_ratings : np.ndarray
        Reconstructed rating matrix from SVD.
    user_index, movie_index : dict
        Index mappings.
    alpha : float
        Blending weight (0 = pure collaborative, 1 = pure content).
    user_id : int or None
        If provided, use this existing user for collaborative filtering.
        If None, treat as a new user and use the liked movies to find a
        similar existing user.
    top_n : int
        Number of final recommendations.

    Returns
    -------
    pd.DataFrame
        Columns include 'content_score', 'collab_score', 'hybrid_score'.
    """
    # -- Step 1: Get content-based recommendations ----------------------------
    content_recs = recommend_similar(
        movie_titles, movies_df, tfidf_matrix, top_n=top_n * 3
    )

    # -- Step 2: Get collaborative recommendations ----------------------------
    if user_id is not None and user_id in user_index:
        collab_recs = recommend_for_user(
            user_id, ratings_df, movies_df,
            predicted_ratings, user_index, movie_index,
            top_n=top_n * 3,
        )
    else:
        # New user path — look up movieIds from the selected titles
        liked_ids = []
        for title in movie_titles:
            match = movies_df[movies_df["title"] == title]
            if not match.empty:
                liked_ids.append(int(match.iloc[0]["movieId"]))

        collab_recs = recommend_for_new_user(
            liked_ids, ratings_df, movies_df,
            predicted_ratings, user_index, movie_index,
            top_n=top_n * 3,
        )

    # -- Step 3: Merge the two result sets on movieId -------------------------
    if content_recs.empty and collab_recs.empty:
        return pd.DataFrame(
            columns=list(movies_df.columns) + [
                "content_score", "collab_score", "hybrid_score"
            ]
        )

    # Start with content results and merge in collab scores
    if not content_recs.empty and not collab_recs.empty:
        # Keep only the score columns from collab for the merge
        collab_scores = collab_recs[["movieId", "collab_score"]].copy()

        merged = content_recs.merge(collab_scores, on="movieId", how="outer")

        # Fill in missing movie metadata from either side
        # (outer join may leave some columns NaN for collab-only entries)
        for col in movies_df.columns:
            if col in merged.columns and merged[col].isna().any():
                fill_map = movies_df.set_index("movieId")[col]
                merged[col] = merged[col].fillna(merged["movieId"].map(fill_map))

    elif not content_recs.empty:
        merged = content_recs.copy()
    else:
        merged = collab_recs.copy()

    # Ensure both score columns exist
    if "content_score" not in merged.columns:
        merged["content_score"] = 0.0
    if "collab_score" not in merged.columns:
        merged["collab_score"] = 0.0

    merged["content_score"] = merged["content_score"].fillna(0.0)
    merged["collab_score"] = merged["collab_score"].fillna(0.0)

    # -- Step 4: Compute the hybrid score -------------------------------------
    merged["hybrid_score"] = (
        alpha * merged["content_score"]
        + (1 - alpha) * merged["collab_score"]
    )

    # Sort by hybrid score and return the top N
    merged = merged.sort_values("hybrid_score", ascending=False)
    merged = merged.head(top_n).reset_index(drop=True)

    # Round scores for clean display
    for col in ["content_score", "collab_score", "hybrid_score"]:
        merged[col] = merged[col].round(4)

    return merged
