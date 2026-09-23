"""
collaborative_recommender.py
-----------------------------
Collaborative filtering engine built on Truncated SVD (matrix factorization).

How it works:
  1. Build a user–item rating matrix where rows are users and columns are
     movies, with cells containing the rating (0 if unrated).
  2. Apply SVD to decompose this matrix into latent factors that capture
     hidden patterns in user preferences (e.g., "likes dark thrillers",
     "prefers animated comedies").
  3. Multiply the decomposed matrices back together to get *predicted*
     ratings for every user–movie pair — including movies the user hasn't
     seen yet.
  4. Recommend the top-N highest-predicted movies that the user hasn't
     already rated.

This approach excels when we have enough rating history.  For completely
new users, we offer a "virtual user" mechanism: the user selects a few
movies they like, we inject those ratings, project into latent space,
and generate predictions.
"""

import numpy as np
import pandas as pd
import streamlit as st
from scipy.sparse.linalg import svds


# ---------------------------------------------------------------------------
# Model training (cached so SVD only runs once)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def train_svd_model(_ratings_df: pd.DataFrame, n_factors: int = 50):
    """
    Train the SVD collaborative filtering model.

    Parameters
    ----------
    _ratings_df : pd.DataFrame
        Raw ratings with columns: userId, movieId, rating.
    n_factors : int
        Number of latent factors for SVD decomposition.

    Returns
    -------
    predicted_ratings : np.ndarray
        Full reconstructed rating matrix (users × movies).
    user_index : dict
        Maps userId → row index in the matrix.
    movie_index : dict
        Maps movieId → column index in the matrix.
    user_means : pd.Series
        Per-user mean rating (used for de-meaning / re-meaning).
    """
    # Pivot into the user-item matrix
    rating_matrix = _ratings_df.pivot_table(
        index="userId", columns="movieId", values="rating"
    ).fillna(0)

    # Build index lookups
    user_ids = rating_matrix.index.tolist()
    movie_ids = rating_matrix.columns.tolist()
    user_index = {uid: i for i, uid in enumerate(user_ids)}
    movie_index = {mid: i for i, mid in enumerate(movie_ids)}

    # Centre the matrix by subtracting each user's mean rating.
    # This helps SVD focus on *deviations* from the average, which
    # captures taste more accurately than raw scores.
    matrix = rating_matrix.values.astype(float)
    user_means = np.mean(matrix, axis=1)
    matrix_centred = matrix - user_means.reshape(-1, 1)

    # Truncated SVD — keep the top-k latent factors
    k = min(n_factors, min(matrix_centred.shape) - 1)
    U, sigma, Vt = svds(matrix_centred, k=k)

    # Reconstruct the predicted rating matrix and add means back
    sigma_diag = np.diag(sigma)
    predicted = np.dot(np.dot(U, sigma_diag), Vt) + user_means.reshape(-1, 1)

    return predicted, user_index, movie_index, pd.Series(user_means, index=user_ids)


# ---------------------------------------------------------------------------
# Recommendation functions
# ---------------------------------------------------------------------------

def recommend_for_user(
    user_id: int,
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    predicted_ratings: np.ndarray,
    user_index: dict,
    movie_index: dict,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Get top-N recommendations for an existing user.

    We rank movies by their predicted rating and exclude anything the
    user has already rated.

    Parameters
    ----------
    user_id : int
        The target user's ID.
    ratings_df : pd.DataFrame
        Raw ratings dataframe.
    movies_df : pd.DataFrame
        Merged movie dataset.
    predicted_ratings : np.ndarray
        Reconstructed user-item matrix from train_svd_model().
    user_index, movie_index : dict
        Index mappings from train_svd_model().
    top_n : int
        Number of recommendations.

    Returns
    -------
    pd.DataFrame
        Recommended movies with a 'collab_score' column.
    """
    if user_id not in user_index:
        return pd.DataFrame(columns=list(movies_df.columns) + ["collab_score"])

    u_idx = user_index[user_id]
    user_predictions = predicted_ratings[u_idx]

    # Movies this user already rated — we won't recommend these
    already_rated = set(
        ratings_df[ratings_df["userId"] == user_id]["movieId"].tolist()
    )

    # Build a list of (movieId, predicted_score) for unrated movies
    candidates = []
    for mid, m_idx in movie_index.items():
        if mid not in already_rated:
            candidates.append((mid, float(user_predictions[m_idx])))

    # Sort by predicted score descending
    candidates.sort(key=lambda x: x[1], reverse=True)
    top_candidates = candidates[:top_n]

    if not top_candidates:
        return pd.DataFrame(columns=list(movies_df.columns) + ["collab_score"])

    result_ids = [c[0] for c in top_candidates]
    result_scores = {c[0]: c[1] for c in top_candidates}

    results = movies_df[movies_df["movieId"].isin(result_ids)].copy()
    results["collab_score"] = results["movieId"].map(result_scores)

    # Normalise scores to 0-1 range for easier blending with content scores
    max_score = results["collab_score"].max()
    min_score = results["collab_score"].min()
    if max_score > min_score:
        results["collab_score"] = (
            (results["collab_score"] - min_score) / (max_score - min_score)
        )
    else:
        results["collab_score"] = 1.0

    results = results.sort_values("collab_score", ascending=False)
    return results.head(top_n).reset_index(drop=True)


def recommend_for_new_user(
    liked_movie_ids: list[int],
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    predicted_ratings: np.ndarray,
    user_index: dict,
    movie_index: dict,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Generate recommendations for a "new" user who doesn't exist in the
    training data.

    Strategy: build a binary preference vector from the liked movies,
    then find the existing user whose predicted-rating profile most
    closely matches (via dot-product similarity).  Use that user's
    predicted ratings as a proxy for the new user's tastes.
    """
    if not liked_movie_ids:
        return pd.DataFrame(columns=list(movies_df.columns) + ["collab_score"])

    # Build a binary indicator vector for the new user's liked movies
    n_movies = predicted_ratings.shape[1]
    new_user_vec = np.zeros(n_movies)
    valid_liked = []
    for mid in liked_movie_ids:
        if mid in movie_index:
            new_user_vec[movie_index[mid]] = 1.0
            valid_liked.append(mid)

    if not valid_liked:
        return pd.DataFrame(columns=list(movies_df.columns) + ["collab_score"])

    # Fast similarity: dot product of the new user's binary vector against
    # all existing users' predicted rating rows (already available).
    # This avoids the expensive rebuild of the full rating matrix.
    sim_scores = predicted_ratings.dot(new_user_vec)
    best_user_idx = int(sim_scores.argmax())

    # Use that user's predicted ratings
    best_predictions = predicted_ratings[best_user_idx]

    candidates = []
    liked_set = set(valid_liked)
    for mid, m_idx in movie_index.items():
        if mid not in liked_set:
            candidates.append((mid, float(best_predictions[m_idx])))

    candidates.sort(key=lambda x: x[1], reverse=True)
    top_candidates = candidates[:top_n]

    if not top_candidates:
        return pd.DataFrame(columns=list(movies_df.columns) + ["collab_score"])

    result_ids = [c[0] for c in top_candidates]
    result_scores = {c[0]: c[1] for c in top_candidates}

    results = movies_df[movies_df["movieId"].isin(result_ids)].copy()
    results["collab_score"] = results["movieId"].map(result_scores)

    max_s = results["collab_score"].max()
    min_s = results["collab_score"].min()
    if max_s > min_s:
        results["collab_score"] = (results["collab_score"] - min_s) / (max_s - min_s)
    else:
        results["collab_score"] = 1.0

    results = results.sort_values("collab_score", ascending=False)
    return results.head(top_n).reset_index(drop=True)

