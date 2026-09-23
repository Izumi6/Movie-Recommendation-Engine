"""
utils.py
--------
Shared helper functions used across the recommendation engine.
Handles title parsing, genre colors, rating formatting, and
placeholder generation for the UI movie cards.
"""

import re
from typing import Optional


# -- Color palette for genre badges (hand-picked for dark backgrounds) -------

GENRE_COLORS = {
    "Action":      "#EF4444",
    "Adventure":   "#F97316",
    "Animation":   "#FBBF24",
    "Children":    "#A3E635",
    "Comedy":      "#34D399",
    "Crime":       "#14B8A6",
    "Documentary": "#22D3EE",
    "Drama":       "#60A5FA",
    "Fantasy":     "#818CF8",
    "Film-Noir":   "#6B7280",
    "Horror":      "#DC2626",
    "IMAX":        "#0EA5E9",
    "Musical":     "#E879F9",
    "Mystery":     "#A78BFA",
    "Romance":     "#FB7185",
    "Sci-Fi":      "#2DD4BF",
    "Thriller":    "#F43F5E",
    "War":         "#78716C",
    "Western":     "#D97706",
    "(no genres listed)": "#9CA3AF",
}

# Fallback colour if a genre isn't in the map
DEFAULT_GENRE_COLOR = "#8B5CF6"


def extract_year(title: str) -> Optional[int]:
    """
    Pull the release year from a MovieLens-style title string.

    Example:
        >>> extract_year("Toy Story (1995)")
        1995
        >>> extract_year("Some Movie Without Year")
        None
    """
    match = re.search(r"\((\d{4})\)\s*$", title)
    if match:
        return int(match.group(1))
    return None


def clean_title(title: str) -> str:
    """
    Strip the trailing (YYYY) from a movie title.

    Example:
        >>> clean_title("Toy Story (1995)")
        'Toy Story'
    """
    return re.sub(r"\s*\(\d{4}\)\s*$", "", title).strip()


def get_genre_color(genre: str) -> str:
    """Map a genre name to a consistent hex colour for UI badges."""
    return GENRE_COLORS.get(genre, DEFAULT_GENRE_COLOR)


def format_rating(rating: float) -> str:
    """
    Return a decorated rating string.

    Example:
        >>> format_rating(4.23)
        '4.2 / 5.0'
    """
    return f"{rating:.1f} / 5.0"


def get_poster_placeholder(title: str) -> str:
    """
    Generate an HTML snippet for a styled movie-card placeholder.
    Uses the first letters of up to two words from the cleaned title,
    with a gradient background for visual variety.
    """
    name = clean_title(title)
    words = name.split()
    initials = "".join(w[0].upper() for w in words[:2]) if words else "?"

    # Pick two gradient colours based on the title hash so each
    # movie gets a unique-looking placeholder
    h = hash(name) % 360
    color_a = f"hsl({h}, 70%, 45%)"
    color_b = f"hsl({(h + 40) % 360}, 60%, 55%)"

    return f"""
    <div style="
        width: 100%; aspect-ratio: 2/3;
        background: linear-gradient(135deg, {color_a}, {color_b});
        border-radius: 12px;
        display: flex; align-items: center; justify-content: center;
        font-size: 2.4rem; font-weight: 700; color: #fff;
        text-shadow: 0 2px 8px rgba(0,0,0,0.3);
        letter-spacing: 2px;
    ">{initials}</div>
    """


def parse_genres(genres_str: str) -> list:
    """
    Split a pipe-delimited genre string into a list.

    Example:
        >>> parse_genres("Action|Adventure|Comedy")
        ['Action', 'Adventure', 'Comedy']
    """
    if not genres_str or genres_str == "(no genres listed)":
        return []
    return [g.strip() for g in genres_str.split("|")]


def genre_badges_html(genres_str: str) -> str:
    """
    Build a row of coloured genre badges as an HTML string.
    Used inside movie cards in the Streamlit UI.
    """
    genres = parse_genres(genres_str)
    if not genres:
        return ""

    badges = []
    for g in genres:
        color = get_genre_color(g)
        badges.append(
            f'<span style="background:{color}; color:#fff; padding:2px 10px; '
            f'border-radius:20px; font-size:0.75rem; font-weight:500; '
            f'margin-right:4px; display:inline-block; margin-bottom:4px;">{g}</span>'
        )
    return " ".join(badges)
