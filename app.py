"""
app.py
------
Streamlit frontend for the Movie Recommendation Engine.

Run with:
    streamlit run app.py

Pages (via sidebar navigation):
    1. Home / Explore      -- trending movies, genre browsing
    2. Find Similar Movies -- content-based search
    3. Personalized Picks  -- hybrid recommendations based on user preferences
    4. Analytics Dashboard -- dataset stats, model evaluation, visualizations
    5. About               -- how the engine works
"""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Make sure Python can find our src/ package
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader import build_movie_dataset, load_ratings
from src.feature_engine import build_tfidf_matrix, get_top_features
from src.content_recommender import recommend_similar, recommend_by_genre
from src.collaborative_recommender import train_svd_model
from src.hybrid_recommender import hybrid_recommend
from src.evaluation import run_full_evaluation
from src.utils import (
    clean_title,
    get_genre_color,
)


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CineMatch -- Movie Recommendation Engine",
    page_icon="C",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load custom CSS
css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
if os.path.isfile(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data and model loading (all cached -- runs once per session)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_all_data():
    """Load and prepare the full movie dataset."""
    movies = build_movie_dataset()
    ratings = load_ratings()
    return movies, ratings


@st.cache_resource(show_spinner=False)
def load_models(_movies_df, _ratings_df):
    """Build TF-IDF matrix and train SVD model."""
    tfidf_matrix, feature_names = build_tfidf_matrix(_movies_df)
    predicted, user_idx, movie_idx, user_means = train_svd_model(_ratings_df)
    return tfidf_matrix, feature_names, predicted, user_idx, movie_idx


def initialise():
    """Load everything and return the core objects."""
    with st.spinner("Loading movie dataset and training models..."):
        movies_df, ratings_df = load_all_data()
        tfidf_matrix, feature_names, predicted, user_idx, movie_idx = load_models(
            movies_df, ratings_df
        )
    return movies_df, ratings_df, tfidf_matrix, feature_names, predicted, user_idx, movie_idx


# ---------------------------------------------------------------------------
# UI helper functions
# ---------------------------------------------------------------------------

def render_movie_card(movie_row, score=None, score_label="Score"):
    """Render a single movie as a styled glassmorphism card."""
    title = movie_row.get("title", "Unknown")
    genres = movie_row.get("genres", "")
    avg_rating = movie_row.get("avg_rating", 0)
    rating_count = movie_row.get("rating_count", 0)
    year = movie_row.get("year", "")
    year_display = f" ({int(year)})" if pd.notna(year) and year else ""

    # Build the poster placeholder
    name = clean_title(title)
    words = name.split()
    initials = "".join(w[0].upper() for w in words[:2]) if words else "?"
    h = hash(name) % 360
    color_a = f"hsl({h}, 70%, 45%)"
    color_b = f"hsl({(h + 40) % 360}, 60%, 55%)"

    # Build genre badges inline
    badge_parts = []
    if genres and genres != "(no genres listed)":
        for g in genres.split("|"):
            g = g.strip()
            if g and g != "(no genres listed)":
                gc = get_genre_color(g)
                badge_parts.append(
                    f'<span style="background:{gc};color:#fff;padding:2px 8px;'
                    f'border-radius:20px;font-size:0.7rem;font-weight:500;'
                    f'margin-right:3px;display:inline-block;margin-bottom:3px;">{g}</span>'
                )
    badges_html = "".join(badge_parts)

    # Build score bar if applicable
    score_section = ""
    if score is not None and score > 0:
        pct = min(score * 100, 100)
        score_section = (
            f'<div style="margin-top:0.5rem;">'
            f'<span style="font-size:0.75rem;color:#94a3b8;">{score_label}</span>'
            f'<span style="font-size:0.8rem;font-weight:600;color:#14b8a6;float:right;">{score:.2f}</span>'
            f'<div style="background:rgba(255,255,255,0.06);border-radius:8px;height:8px;overflow:hidden;margin-top:4px;">'
            f'<div style="height:100%;border-radius:8px;width:{pct}%;'
            f'background:linear-gradient(135deg,#14b8a6,#8b5cf6,#ec4899);"></div>'
            f'</div></div>'
        )

    # Rating display
    rating_str = f"{avg_rating:.1f} / 5.0" if avg_rating else "N/A"

    # Assemble the complete card as a single HTML block
    html = (
        f'<div style="background:rgba(17,24,39,0.7);backdrop-filter:blur(12px);'
        f'border:1px solid rgba(255,255,255,0.08);border-radius:16px;padding:1.2rem;'
        f'box-shadow:0 4px 20px rgba(0,0,0,0.4);margin-bottom:1rem;">'
        # Poster
        f'<div style="width:100%;aspect-ratio:2/3;background:linear-gradient(135deg,{color_a},{color_b});'
        f'border-radius:12px;display:flex;align-items:center;justify-content:center;'
        f'font-size:2.4rem;font-weight:700;color:#fff;text-shadow:0 2px 8px rgba(0,0,0,0.3);'
        f'letter-spacing:2px;">{initials}</div>'
        # Title
        f'<div style="margin-top:0.8rem;font-family:Inter,sans-serif;font-weight:600;'
        f'font-size:0.95rem;color:#f1f5f9;line-height:1.3;">{name}{year_display}</div>'
        # Genre badges
        f'<div style="margin-top:0.4rem;">{badges_html}</div>'
        # Rating
        f'<div style="margin-top:0.4rem;font-size:0.85rem;font-weight:600;color:#fbbf24;">'
        f'{rating_str} '
        f'<span style="font-size:0.72rem;color:#94a3b8;font-weight:400;">({rating_count:,} ratings)</span>'
        f'</div>'
        # Score bar
        f'{score_section}'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def render_movie_grid(movies_df_subset, score_col=None, score_label="Score", cols=4):
    """Render a grid of movie cards."""
    rows_data = movies_df_subset.to_dict("records")

    for i in range(0, len(rows_data), cols):
        row_movies = rows_data[i : i + cols]
        columns = st.columns(cols)
        for j, movie in enumerate(row_movies):
            with columns[j]:
                score_val = movie.get(score_col) if score_col else None
                render_movie_card(movie, score=score_val, score_label=score_label)


def gradient_header(text):
    """Render a section header with the gradient underline."""
    st.markdown(f"### {text}")
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Page: Home / Explore
# ---------------------------------------------------------------------------

def page_home(movies_df, ratings_df):
    """Landing page -- trending and top-rated movies with genre filtering."""

    # Hero
    st.markdown("""
    <div class="hero-header">
        <h1>CineMatch</h1>
        <p>Discover your next favourite movie with AI-powered recommendations
        combining content analysis and collaborative intelligence.</p>
    </div>
    """, unsafe_allow_html=True)

    # Quick dataset stats
    total_movies = len(movies_df)
    total_ratings = len(ratings_df)
    total_users = ratings_df["userId"].nunique()
    avg_rating_global = ratings_df["rating"].mean()

    c1, c2, c3, c4 = st.columns(4)
    for col, val, label in [
        (c1, f"{total_movies:,}", "MOVIES"),
        (c2, f"{total_ratings:,}", "RATINGS"),
        (c3, f"{total_users:,}", "USERS"),
        (c4, f"{avg_rating_global:.2f}", "AVG RATING"),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("")

    # Genre filter
    all_genres = sorted(set(
        g for genres in movies_df["genres"].dropna()
        for g in genres.split("|") if g != "(no genres listed)"
    ))

    selected_genre = st.selectbox(
        "Filter by genre", ["All Genres"] + all_genres, index=0
    )

    if selected_genre != "All Genres":
        filtered = movies_df[
            movies_df["genres"].str.contains(selected_genre, case=False, na=False)
        ]
    else:
        filtered = movies_df

    # Trending movies (highest popularity score, minimum 50 ratings)
    gradient_header("Trending Movies")
    trending = (
        filtered[filtered["rating_count"] >= 50]
        .sort_values("popularity_score", ascending=False)
        .head(8)
    )
    render_movie_grid(trending)

    # Top rated (by average, minimum 100 ratings for quality)
    gradient_header("Highest Rated")
    top_rated = (
        filtered[filtered["rating_count"] >= 100]
        .sort_values("avg_rating", ascending=False)
        .head(8)
    )
    render_movie_grid(top_rated)

    # Recently released
    gradient_header("Most Recent Releases")
    recent = (
        filtered[filtered["year"].notna()]
        .sort_values("year", ascending=False)
        .head(8)
    )
    render_movie_grid(recent)


# ---------------------------------------------------------------------------
# Page: Find Similar Movies
# ---------------------------------------------------------------------------

def page_similar(movies_df, tfidf_matrix, feature_names):
    """Content-based search -- pick a movie, get similar ones."""

    gradient_header("Find Similar Movies")
    st.write("Select a movie you enjoy, and we will find titles with similar genres, "
             "themes, and tags using TF-IDF cosine similarity.")

    # Movie selector
    all_titles = movies_df["title"].sort_values().tolist()
    selected = st.selectbox(
        "Search for a movie",
        all_titles,
        index=None,
        placeholder="Start typing a movie title...",
    )

    num_results = st.slider("Number of recommendations", 4, 20, 10, step=2)

    if selected:
        # Show the selected movie's details
        sel_row = movies_df[movies_df["title"] == selected].iloc[0]
        st.markdown("---")
        st.markdown("**Your selection:**")

        c1, c2 = st.columns([1, 3])
        with c1:
            render_movie_card(sel_row.to_dict())
        with c2:
            # Show the top TF-IDF features for this movie
            idx = movies_df[movies_df["title"] == selected].index[0]
            top_feats = get_top_features(idx, tfidf_matrix, feature_names, top_n=12)
            if top_feats:
                st.markdown("**Key features (TF-IDF terms):**")
                feat_df = pd.DataFrame(top_feats, columns=["Term", "Weight"])
                fig = px.bar(
                    feat_df, x="Weight", y="Term", orientation="h",
                    color="Weight",
                    color_continuous_scale=["#14b8a6", "#8b5cf6"],
                )
                fig.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="#94a3b8",
                    height=300,
                    margin=dict(l=0, r=0, t=10, b=0),
                    showlegend=False,
                    coloraxis_showscale=False,
                    yaxis=dict(autorange="reversed"),
                )
                st.plotly_chart(fig, use_container_width=True)

        # Generate recommendations
        st.markdown("---")
        gradient_header(f"Movies similar to '{clean_title(selected)}'")

        with st.spinner("Computing similarities..."):
            recs = recommend_similar([selected], movies_df, tfidf_matrix, top_n=num_results)

        if recs.empty:
            st.info("No similar movies found. Try a different title.")
        else:
            # Similarity scores chart
            chart_data = recs.head(10).copy()
            chart_data["clean_name"] = chart_data["title"].apply(clean_title)
            fig = px.bar(
                chart_data, x="content_score", y="clean_name", orientation="h",
                color="content_score",
                color_continuous_scale=["#14b8a6", "#8b5cf6", "#ec4899"],
                labels={"content_score": "Similarity", "clean_name": ""},
            )
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#94a3b8",
                height=350,
                margin=dict(l=0, r=20, t=10, b=0),
                showlegend=False,
                coloraxis_showscale=False,
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Movie cards grid
            render_movie_grid(recs, score_col="content_score", score_label="Similarity")


# ---------------------------------------------------------------------------
# Page: Personalized Picks
# ---------------------------------------------------------------------------

def page_personalized(
    movies_df, ratings_df, tfidf_matrix, predicted, user_idx, movie_idx
):
    """Hybrid recommendations -- user selects liked movies and tunes alpha."""

    gradient_header("Personalized Picks")
    st.write("Tell us movies you love, and our hybrid engine will find your "
             "perfect next watch by blending content analysis with collaborative "
             "user patterns.")

    # Sidebar controls
    with st.sidebar:
        st.markdown("### Recommendation Settings")

        alpha = st.slider(
            "Content vs Collaborative blend",
            min_value=0.0, max_value=1.0, value=0.5, step=0.05,
            help="1.0 = pure content-based, 0.0 = pure collaborative, 0.5 = balanced"
        )

        # Visual indicator
        if alpha > 0.7:
            blend_label = "Content-heavy"
        elif alpha < 0.3:
            blend_label = "Collaborative-heavy"
        else:
            blend_label = "Balanced blend"
        st.caption(blend_label)

        num_recs = st.slider("Number of recommendations", 4, 20, 10, step=2)

    # Movie selection
    popular_titles = (
        movies_df[movies_df["rating_count"] >= 20]
        .sort_values("popularity_score", ascending=False)["title"]
        .tolist()
    )

    selected_movies = st.multiselect(
        "Select movies you enjoy (pick 2-5 for best results)",
        popular_titles,
        max_selections=10,
        placeholder="Start typing movie titles...",
    )

    # Optional: user rating for each selected movie
    user_ratings = {}
    if selected_movies:
        st.markdown("**Rate your selections** (optional -- helps refine recommendations):")
        cols = st.columns(min(len(selected_movies), 4))
        for i, title in enumerate(selected_movies):
            with cols[i % len(cols)]:
                user_ratings[title] = st.slider(
                    f"{clean_title(title)[:25]}",
                    1.0, 5.0, 4.0, 0.5,
                    key=f"rate_{i}",
                )

    # Generate button
    if st.button("Get Recommendations", type="primary", use_container_width=True):
        if not selected_movies:
            st.warning("Please select at least one movie first!")
        else:
            with st.spinner("Generating personalised recommendations..."):
                recs = hybrid_recommend(
                    movie_titles=selected_movies,
                    movies_df=movies_df,
                    tfidf_matrix=tfidf_matrix,
                    ratings_df=ratings_df,
                    predicted_ratings=predicted,
                    user_index=user_idx,
                    movie_index=movie_idx,
                    alpha=alpha,
                    top_n=num_recs,
                )

            if recs.empty:
                st.info("Could not generate recommendations. Try different movies.")
            else:
                # Score comparison chart
                if len(recs) >= 3:
                    st.markdown("---")
                    gradient_header("Score Breakdown")

                    top5 = recs.head(5).copy()
                    top5["clean_name"] = top5["title"].apply(clean_title)

                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        name="Content Score",
                        x=top5["clean_name"],
                        y=top5["content_score"],
                        marker_color="#14b8a6",
                    ))
                    fig.add_trace(go.Bar(
                        name="Collaborative Score",
                        x=top5["clean_name"],
                        y=top5["collab_score"],
                        marker_color="#8b5cf6",
                    ))
                    fig.add_trace(go.Scatter(
                        name="Hybrid Score",
                        x=top5["clean_name"],
                        y=top5["hybrid_score"],
                        mode="lines+markers",
                        marker=dict(color="#ec4899", size=10),
                        line=dict(color="#ec4899", width=2),
                    ))
                    fig.update_layout(
                        barmode="group",
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font_color="#94a3b8",
                        height=350,
                        margin=dict(l=0, r=0, t=30, b=0),
                        legend=dict(
                            orientation="h", y=1.12, x=0.5, xanchor="center",
                            font=dict(size=12),
                        ),
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Movie cards
                gradient_header("Your Recommendations")
                render_movie_grid(
                    recs, score_col="hybrid_score", score_label="Match Score"
                )


# ---------------------------------------------------------------------------
# Page: Analytics Dashboard
# ---------------------------------------------------------------------------

def page_analytics(
    movies_df, ratings_df, tfidf_matrix, predicted, user_idx, movie_idx
):
    """Dataset visualizations and model evaluation metrics."""

    gradient_header("Analytics Dashboard")

    tab1, tab2, tab3 = st.tabs([
        "Dataset Insights", "Model Evaluation", "Deep Dive"
    ])

    # -- Tab 1: Dataset Insights ---------------------------------------------
    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            # Rating distribution
            st.markdown("**Rating Distribution**")
            fig = px.histogram(
                ratings_df, x="rating", nbins=10,
                color_discrete_sequence=["#14b8a6"],
                labels={"rating": "Rating", "count": "Count"},
            )
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#94a3b8",
                height=320,
                margin=dict(l=0, r=0, t=10, b=0),
                bargap=0.08,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Genre popularity
            st.markdown("**Genre Popularity (by number of movies)**")
            genre_counts = {}
            for genres in movies_df["genres"].dropna():
                for g in genres.split("|"):
                    if g != "(no genres listed)":
                        genre_counts[g] = genre_counts.get(g, 0) + 1

            genre_df = pd.DataFrame(
                sorted(genre_counts.items(), key=lambda x: x[1], reverse=True),
                columns=["Genre", "Count"],
            )

            fig = px.bar(
                genre_df, x="Count", y="Genre", orientation="h",
                color="Genre",
                color_discrete_map={g: get_genre_color(g) for g in genre_df["Genre"]},
            )
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#94a3b8",
                height=450,
                margin=dict(l=0, r=0, t=10, b=0),
                showlegend=False,
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig, use_container_width=True)

        # Ratings over time
        st.markdown("**Ratings Activity Over Time**")
        ratings_time = ratings_df.copy()
        ratings_time["month"] = ratings_time["date"].dt.to_period("M").astype(str)
        monthly = ratings_time.groupby("month").size().reset_index(name="count")
        monthly = monthly.tail(60)  # Last 60 months

        fig = px.area(
            monthly, x="month", y="count",
            color_discrete_sequence=["#8b5cf6"],
            labels={"month": "Month", "count": "Number of Ratings"},
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#94a3b8",
            height=300,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(showgrid=False, tickangle=-45),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
        )
        fig.update_traces(
            fill="tozeroy",
            fillcolor="rgba(139,92,246,0.15)",
            line=dict(width=2),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Movies per decade
        col3, col4 = st.columns(2)

        with col3:
            st.markdown("**Movies per Decade**")
            decade_df = movies_df[movies_df["year"].notna()].copy()
            decade_df["decade"] = (decade_df["year"] // 10 * 10).astype(int).astype(str) + "s"
            decade_counts = decade_df["decade"].value_counts().sort_index().reset_index()
            decade_counts.columns = ["Decade", "Count"]

            fig = px.bar(
                decade_counts, x="Decade", y="Count",
                color="Count",
                color_continuous_scale=["#14b8a6", "#8b5cf6"],
            )
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#94a3b8",
                height=300,
                margin=dict(l=0, r=0, t=10, b=0),
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col4:
            st.markdown("**Average Rating by Genre**")
            genre_ratings = []
            for _, row in movies_df.iterrows():
                if pd.notna(row["genres"]) and row["avg_rating"] > 0:
                    for g in row["genres"].split("|"):
                        if g != "(no genres listed)":
                            genre_ratings.append({"Genre": g, "Rating": row["avg_rating"]})

            gr_df = pd.DataFrame(genre_ratings)
            avg_by_genre = gr_df.groupby("Genre")["Rating"].mean().sort_values(ascending=True).reset_index()

            fig = px.bar(
                avg_by_genre, x="Rating", y="Genre", orientation="h",
                color="Rating",
                color_continuous_scale=["#ec4899", "#14b8a6"],
            )
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#94a3b8",
                height=400,
                margin=dict(l=0, r=0, t=10, b=0),
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig, use_container_width=True)

    # -- Tab 2: Model Evaluation ---------------------------------------------
    with tab2:
        st.write("Running evaluation on a held-out test set (80/20 temporal split)...")

        if st.button("Run Evaluation", type="primary"):
            with st.spinner("Evaluating model performance (this may take a moment)..."):
                results = run_full_evaluation(
                    ratings_df, movies_df, predicted, user_idx, movie_idx,
                    tfidf_matrix, top_k=10, sample_users=50,
                )

            # Display metrics as cards
            m1, m2, m3 = st.columns(3)
            for col, val, label in [
                (m1, f"{results['rmse']:.3f}", "RMSE"),
                (m2, f"{results['avg_precision_at_k']:.3f}", f"PRECISION@{results['k']}"),
                (m3, f"{results['avg_recall_at_k']:.3f}", f"RECALL@{results['k']}"),
            ]:
                with col:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{val}</div>
                        <div class="metric-label">{label}</div>
                    </div>
                    """, unsafe_allow_html=True)

            m4, m5, m6 = st.columns(3)
            for col, val, label in [
                (m4, f"{results['catalog_coverage']*100:.1f}%", "CATALOG COVERAGE"),
                (m5, f"{results['recommendation_diversity']:.3f}", "DIVERSITY"),
                (m6, f"{results['num_users_evaluated']}", "USERS TESTED"),
            ]:
                with col:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{val}</div>
                        <div class="metric-label">{label}</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("")

            # Radar chart of metrics
            categories = ["RMSE\n(inverted)", "Precision@K", "Recall@K", "Coverage", "Diversity"]
            # Normalize RMSE inversely (lower is better)
            rmse_norm = max(0, 1 - results["rmse"] / 5)
            values = [
                rmse_norm,
                results["avg_precision_at_k"],
                results["avg_recall_at_k"],
                results["catalog_coverage"],
                results["recommendation_diversity"],
            ]
            values.append(values[0])  # close the polygon
            categories.append(categories[0])

            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill="toself",
                fillcolor="rgba(139,92,246,0.15)",
                line=dict(color="#8b5cf6", width=2),
                marker=dict(size=8, color="#ec4899"),
            ))
            fig.update_layout(
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(
                        visible=True, range=[0, 1],
                        gridcolor="rgba(255,255,255,0.08)",
                        color="#94a3b8",
                    ),
                    angularaxis=dict(
                        gridcolor="rgba(255,255,255,0.08)",
                        color="#94a3b8",
                    ),
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#94a3b8",
                height=400,
                margin=dict(l=60, r=60, t=40, b=40),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Interpretation
            st.markdown("""
            <div class="info-box">
                <h3>How to read these metrics</h3>
                <ul>
                    <li><strong>RMSE</strong> -- Root Mean Squared Error of predicted vs actual ratings.
                        Lower is better; values under 1.0 are strong for MovieLens.</li>
                    <li><strong>Precision@K</strong> -- Of the K movies we recommended, what fraction
                        did the user actually rate 4 or more stars.</li>
                    <li><strong>Recall@K</strong> -- Of all movies the user liked (4+ stars),
                        what fraction appeared in our top-K recommendations.</li>
                    <li><strong>Catalog Coverage</strong> -- Percentage of the entire movie catalog
                        that our model recommends at least once across all users. Higher means
                        less "popularity bias".</li>
                    <li><strong>Diversity</strong> -- Average pairwise cosine distance between
                        recommended items.  Higher means the recommendations span different genres
                        and themes rather than clustering around one niche.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Click the button above to run model evaluation. "
                    "This computes metrics on a held-out test set and takes around 10-20 seconds.")

    # -- Tab 3: Deep Dive ----------------------------------------------------
    with tab3:
        st.markdown("**Top-Rated Movies per Genre** (min 50 ratings)")

        # Build a cross-tab of genres vs top movies
        all_genres = sorted(set(
            g for genres in movies_df["genres"].dropna()
            for g in genres.split("|") if g != "(no genres listed)"
        ))

        selected_genres = st.multiselect(
            "Select genres to compare", all_genres,
            default=["Action", "Comedy", "Drama", "Sci-Fi"][:min(4, len(all_genres))],
        )

        if selected_genres:
            heatmap_data = {}
            for genre in selected_genres:
                top = (
                    movies_df[
                        (movies_df["genres"].str.contains(genre, na=False))
                        & (movies_df["rating_count"] >= 50)
                    ]
                    .nlargest(5, "avg_rating")
                )
                for _, r in top.iterrows():
                    heatmap_data.setdefault(clean_title(r["title"]), {})[genre] = r["avg_rating"]

            if heatmap_data:
                hm_df = pd.DataFrame(heatmap_data).T.fillna(0)

                fig = px.imshow(
                    hm_df,
                    color_continuous_scale=["#0a0e1a", "#14b8a6", "#ec4899"],
                    aspect="auto",
                    labels=dict(color="Avg Rating"),
                )
                fig.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="#94a3b8",
                    height=max(300, len(heatmap_data) * 35),
                    margin=dict(l=0, r=0, t=10, b=0),
                )
                st.plotly_chart(fig, use_container_width=True)

        # Ratings distribution by genre (box plot)
        st.markdown("**Rating Spread by Genre**")
        genre_box_data = []
        for _, row in movies_df.iterrows():
            if pd.notna(row["genres"]) and row["avg_rating"] > 0 and row["rating_count"] >= 10:
                for g in row["genres"].split("|"):
                    if g != "(no genres listed)" and g in (selected_genres if selected_genres else all_genres[:6]):
                        genre_box_data.append({"Genre": g, "Avg Rating": row["avg_rating"]})

        if genre_box_data:
            gb_df = pd.DataFrame(genre_box_data)
            fig = px.box(
                gb_df, x="Genre", y="Avg Rating",
                color="Genre",
                color_discrete_map={g: get_genre_color(g) for g in gb_df["Genre"].unique()},
            )
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#94a3b8",
                height=350,
                margin=dict(l=0, r=0, t=10, b=0),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Page: About
# ---------------------------------------------------------------------------

def page_about():
    """Explain how the recommendation engine works."""

    gradient_header("About CineMatch")

    st.markdown("""
    <div class="info-box">
        <h3>What is CineMatch?</h3>
        <p>CineMatch is an intelligent movie recommendation engine that combines two
        powerful machine-learning approaches to suggest movies you will love.  It is built
        on the <strong>MovieLens</strong> dataset -- the gold-standard academic benchmark
        for recommendation research -- and runs entirely on your local machine with no
        external API keys needed.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="info-box">
            <h3>Content-Based Filtering</h3>
            <p>Analyses the <em>attributes</em> of movies -- genres, user-applied tags,
            and textual metadata -- to find titles that share similar characteristics
            with movies you already enjoy.</p>
            <ul>
                <li>Builds a <strong>TF-IDF</strong> (Term Frequency-Inverse Document
                Frequency) vector for each movie from its genre labels and crowd-sourced
                tags.</li>
                <li>Computes <strong>cosine similarity</strong> between your selected
                movie's vector and all other movies in the catalog.</li>
                <li>Excellent for <em>cold-start</em> scenarios -- works even for
                brand-new users with no rating history.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-box">
            <h3>Collaborative Filtering</h3>
            <p>Leverages <em>collective user behaviour</em> -- the idea that people who
            agreed in the past will agree in the future.</p>
            <ul>
                <li>Constructs a <strong>user-item rating matrix</strong> from ~100K
                ratings across 610 users and 9,742 movies.</li>
                <li>Applies <strong>Truncated SVD</strong> (Singular Value Decomposition)
                to uncover latent factors -- hidden dimensions of taste like "prefers
                dark thrillers" or "enjoys animated comedies".</li>
                <li>Predicts ratings for unseen movies and recommends the highest-scoring
                ones.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        <h3>Hybrid Engine</h3>
        <p>The hybrid recommender blends both approaches with a tunable weight:</p>
        <p style="text-align:center; font-size:1.15rem; font-weight:600; color:#14b8a6;">
            hybrid_score = alpha x content_score + (1 - alpha) x collab_score
        </p>
        <p>Move the slider in the <strong>Personalized Picks</strong> page to adjust alpha.
        A balanced 50/50 blend typically gives the best results, but you can lean
        content-heavy for niche tastes or collaborative-heavy when you trust the crowd.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        <h3>Evaluation Metrics</h3>
        <ul>
            <li><strong>RMSE</strong> -- How close our predicted ratings are to actual ones</li>
            <li><strong>Precision@K</strong> -- Fraction of recommendations the user truly liked</li>
            <li><strong>Recall@K</strong> -- Fraction of liked movies that appeared in our list</li>
            <li><strong>Catalog Coverage</strong> -- Diversity of items we recommend across all users</li>
            <li><strong>Recommendation Diversity</strong> -- How different the recommended items are from each other</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        <h3>Tech Stack</h3>
        <ul>
            <li><strong>Python 3.10+</strong> -- core language</li>
            <li><strong>Streamlit</strong> -- interactive web interface</li>
            <li><strong>pandas / NumPy</strong> -- data wrangling</li>
            <li><strong>scikit-learn</strong> -- TF-IDF vectorization, cosine similarity</li>
            <li><strong>SciPy</strong> -- SVD matrix factorization</li>
            <li><strong>Plotly</strong> -- interactive visualizations</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main -- sidebar navigation and page routing
# ---------------------------------------------------------------------------

def main():
    """Entry point -- set up sidebar navigation and route to pages."""

    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding:1rem 0 0.5rem;">
            <div style="font-size:2.2rem; font-weight:800; font-family:Outfit,sans-serif;
                background:linear-gradient(135deg,#14b8a6,#8b5cf6);
                -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                background-clip:text;">CM</div>
            <h2 style="margin:0.3rem 0 0; font-size:1.4rem;">CineMatch</h2>
            <p style="font-size:0.8rem; opacity:0.7;">Movie Recommendation Engine</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        page = st.radio(
            "Navigate",
            [
                "Home / Explore",
                "Find Similar Movies",
                "Personalized Picks",
                "Analytics Dashboard",
                "About",
            ],
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.caption("Built with Streamlit | MovieLens Dataset")

    # Load data and models
    try:
        movies_df, ratings_df, tfidf_matrix, feature_names, predicted, user_idx, movie_idx = (
            initialise()
        )
    except FileNotFoundError as e:
        st.error(f"Dataset not found. Please run `python setup_data.py` first.\n\n{e}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

    # Route to the selected page
    if page == "Home / Explore":
        page_home(movies_df, ratings_df)
    elif page == "Find Similar Movies":
        page_similar(movies_df, tfidf_matrix, feature_names)
    elif page == "Personalized Picks":
        page_personalized(
            movies_df, ratings_df, tfidf_matrix, predicted, user_idx, movie_idx
        )
    elif page == "Analytics Dashboard":
        page_analytics(
            movies_df, ratings_df, tfidf_matrix, predicted, user_idx, movie_idx
        )
    elif page == "About":
        page_about()


if __name__ == "__main__":
    main()


# WSGI/ASGI compatibility stub for deployment platforms
def application(environ, start_response):
    start_response('200 OK', [('Content-Type', 'text/plain')])
    return [b"CineMatch Recommendation Engine"]

app = application
handler = application

