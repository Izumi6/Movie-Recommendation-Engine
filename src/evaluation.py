from __future__ import annotations

"""
evaluation.py
-------------
Model evaluation utilities for the recommendation engine.

Metrics implemented:
  - RMSE               — accuracy of predicted ratings vs actual
  - Precision@K        — how many recommended items the user actually liked
  - Recall@K           — how many of the user's liked items we surfaced
  - Catalog Coverage   — what fraction of the catalog ever gets recommended
  - Recommendation Diversity — how different the recommendations are from each other

All metrics are computed over a held-out test set (80/20 temporal split)
so the numbers reflect real generalization performance.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def temporal_train_test_split(
    ratings_df: pd.DataFrame,
    test_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split ratings by time — the most recent ratings become the test set.

    This mirrors real-world usage where we train on past behaviour and
    predict future preferences.

    Parameters
    ----------
    ratings_df : pd.DataFrame
        Must contain columns: userId, movieId, rating, timestamp.
    test_fraction : float
        Fraction of each user's ratings to hold out for testing.

    Returns
    -------
    train_df, test_df : tuple of DataFrames
    """
    ratings_sorted = ratings_df.sort_values(["userId", "timestamp"])

    train_parts = []
    test_parts = []

    for user_id, group in ratings_sorted.groupby("userId"):
        n = len(group)
        split_idx = int(n * (1 - test_fraction))
        train_parts.append(group.iloc[:split_idx])
        test_parts.append(group.iloc[split_idx:])

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)

    return train_df, test_df


def compute_rmse(
    test_df: pd.DataFrame,
    predicted_ratings: np.ndarray,
    user_index: dict,
    movie_index: dict,
) -> float:
    """
    Root Mean Squared Error between actual test ratings and SVD predictions.

    Only evaluates user-movie pairs that exist in both the test set and
    the trained model's index.
    """
    errors = []

    for _, row in test_df.iterrows():
        uid = row["userId"]
        mid = row["movieId"]
        actual = row["rating"]

        if uid in user_index and mid in movie_index:
            predicted = predicted_ratings[user_index[uid]][movie_index[mid]]
            errors.append((actual - predicted) ** 2)

    if not errors:
        return float("nan")

    return float(np.sqrt(np.mean(errors)))


def precision_at_k(
    user_id: int,
    recommended_ids: list[int],
    test_df: pd.DataFrame,
    relevance_threshold: float = 4.0,
) -> float:
    """
    Fraction of recommended movies that the user rated ≥ threshold in the
    test set.

    Parameters
    ----------
    user_id : int
    recommended_ids : list[int]
        movieIds we recommended.
    test_df : pd.DataFrame
        Held-out test ratings.
    relevance_threshold : float
        Minimum rating to count as "relevant".
    """
    user_test = test_df[test_df["userId"] == user_id]
    relevant = set(
        user_test[user_test["rating"] >= relevance_threshold]["movieId"].tolist()
    )

    if not recommended_ids:
        return 0.0

    hits = sum(1 for mid in recommended_ids if mid in relevant)
    return hits / len(recommended_ids)


def recall_at_k(
    user_id: int,
    recommended_ids: list[int],
    test_df: pd.DataFrame,
    relevance_threshold: float = 4.0,
) -> float:
    """
    Fraction of the user's liked test movies that appear in our recommendations.
    """
    user_test = test_df[test_df["userId"] == user_id]
    relevant = set(
        user_test[user_test["rating"] >= relevance_threshold]["movieId"].tolist()
    )

    if not relevant:
        return 0.0

    hits = sum(1 for mid in recommended_ids if mid in relevant)
    return hits / len(relevant)


def catalog_coverage(
    all_recommended_ids: set[int],
    total_movies: int,
) -> float:
    """
    What percentage of the total catalog has been recommended at least once.

    Higher coverage means the system isn't just recommending the same
    popular movies over and over.
    """
    if total_movies == 0:
        return 0.0
    return len(all_recommended_ids) / total_movies


def recommendation_diversity(
    recommended_indices: list[int],
    tfidf_matrix,
) -> float:
    """
    Average pairwise cosine distance between recommended items.

    A score closer to 1.0 means the recommendations are highly diverse
    (very different from each other).  Closer to 0.0 means they're all
    similar to each other.
    """
    if len(recommended_indices) < 2:
        return 0.0

    vectors = tfidf_matrix[recommended_indices]
    sim_matrix = cosine_similarity(vectors)

    # Extract the upper triangle (excluding diagonal)
    n = sim_matrix.shape[0]
    total_sim = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total_sim += sim_matrix[i][j]
            count += 1

    avg_similarity = total_sim / count if count > 0 else 0.0

    # Convert similarity to distance
    return 1.0 - avg_similarity


def run_full_evaluation(
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    predicted_ratings: np.ndarray,
    user_index: dict,
    movie_index: dict,
    tfidf_matrix,
    top_k: int = 10,
    sample_users: int = 50,
) -> dict:
    """
    Run the complete evaluation suite and return a summary dictionary.

    To keep runtime reasonable, we evaluate on a sample of users rather
    than all 610.

    Returns
    -------
    dict with keys: rmse, avg_precision, avg_recall, coverage, diversity
    """
    train_df, test_df = temporal_train_test_split(ratings_df)

    # RMSE on the full test set
    rmse = compute_rmse(test_df, predicted_ratings, user_index, movie_index)

    # Sample users for precision/recall evaluation
    test_users = test_df["userId"].unique()
    rng = np.random.RandomState(42)
    sampled = rng.choice(
        test_users,
        size=min(sample_users, len(test_users)),
        replace=False,
    )

    precisions = []
    recalls = []
    all_recommended = set()

    for uid in sampled:
        if uid not in user_index:
            continue

        u_idx = user_index[uid]
        user_preds = predicted_ratings[u_idx]

        # Movies this user rated in *training* data
        already_rated = set(
            train_df[train_df["userId"] == uid]["movieId"].tolist()
        )

        # Rank unrated movies by predicted score
        candidates = []
        for mid, m_idx in movie_index.items():
            if mid not in already_rated:
                candidates.append((mid, float(user_preds[m_idx])))

        candidates.sort(key=lambda x: x[1], reverse=True)
        top_ids = [c[0] for c in candidates[:top_k]]

        all_recommended.update(top_ids)

        precisions.append(precision_at_k(uid, top_ids, test_df))
        recalls.append(recall_at_k(uid, top_ids, test_df))

    # Catalog coverage
    total_movies = len(movies_df)
    cov = catalog_coverage(all_recommended, total_movies)

    # Diversity — pick a random set of recommended movie indices
    rec_movie_indices = []
    for mid in list(all_recommended)[:50]:
        mask = movies_df["movieId"] == mid
        if mask.any():
            rec_movie_indices.append(movies_df[mask].index[0])

    div = recommendation_diversity(rec_movie_indices, tfidf_matrix)

    return {
        "rmse": round(rmse, 4),
        "avg_precision_at_k": round(float(np.mean(precisions)), 4) if precisions else 0.0,
        "avg_recall_at_k": round(float(np.mean(recalls)), 4) if recalls else 0.0,
        "catalog_coverage": round(cov, 4),
        "recommendation_diversity": round(div, 4),
        "num_users_evaluated": len(precisions),
        "k": top_k,
    }
