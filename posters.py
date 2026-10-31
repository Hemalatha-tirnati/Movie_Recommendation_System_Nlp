"""
posters.py
----------
Movie poster provider and fallback generator.
Uses curated real-poster URLs for popular movies and generates an SVG
fallback for movies without a curated poster URL.
Stored completely outside the dataset folder as per project requirements.
"""

import urllib.parse
import json
import logging
from typing import Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# Standard TMDB CDN base URL
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"
LOGGER = logging.getLogger(__name__)

# Curated poster path dictionary for TMDB movies (by normalized title)
# Stored outside dataset/
POSTER_PATHS: Dict[str, str] = {
    # Reference image movies
    "inception": f"{TMDB_IMAGE_BASE}/oYuLEt3zVCKq57qu2F8dT7NIa6f.jpg",
    "interstellar": f"{TMDB_IMAGE_BASE}/gEU2QniE6E77NI6lCU6MxlNBvIx.jpg",
    "the dark knight": f"{TMDB_IMAGE_BASE}/qJ2tW6WMUDux911r6m7haRef0WH.jpg",
    "fight club": f"{TMDB_IMAGE_BASE}/pB8BM7pdSp6B6Ih7QZ4DrQ3PmJK.jpg",
    "forrest gump": f"{TMDB_IMAGE_BASE}/arw2vcBveWOVZr6pxd9XTd1TdQa.jpg",
    "the shawshank redemption": f"{TMDB_IMAGE_BASE}/9cqNxx0GxF0bflZmeSMuL5tnGzr.jpg",
    "pulp fiction": f"{TMDB_IMAGE_BASE}/d5iIlFn5s0ImszYzBPb8JPIfbXD.jpg",
    "minions": "https://upload.wikimedia.org/wikipedia/en/1/19/Minions_%282015_film%29.jpg",
    "deadpool": "https://upload.wikimedia.org/wikipedia/en/c/ca/Deadpool.png",
    "guardians of the galaxy": "https://upload.wikimedia.org/wikipedia/en/3/33/Guardians_of_the_Galaxy_%28film%29_poster.jpg",
    "jurassic world": "https://upload.wikimedia.org/wikipedia/en/6/6e/Jurassic_World_poster.jpg",
    "pirates of the caribbean: the curse of the black pearl": "https://upload.wikimedia.org/wikipedia/en/8/89/Pirates_of_the_Caribbean_-_The_Curse_of_the_Black_Pearl.png",
    "dawn of the planet of the apes": "https://upload.wikimedia.org/wikipedia/en/7/77/Dawn_of_the_Planet_of_the_Apes.jpg",
    "the hunger games: mockingjay - part 1": "https://upload.wikimedia.org/wikipedia/en/6/63/MockingjayPart1Poster3.jpg",
    "big hero 6": "https://upload.wikimedia.org/wikipedia/en/4/4b/Big_Hero_6_%28film%29_poster.jpg",
    "terminator genisys": "https://upload.wikimedia.org/wikipedia/en/b/bc/Terminator_Genisys.JPG",
    "captain america: civil war": "https://upload.wikimedia.org/wikipedia/en/5/53/Captain_America_Civil_War_poster.jpg",
    "the martian": "https://upload.wikimedia.org/wikipedia/en/c/cd/The_Martian_film_poster.jpg",
    
    # Popular TMDB classics
    "avatar": f"{TMDB_IMAGE_BASE}/kyeqWdyUXW608qlYkRqosgbbJyK.jpg",
    "the avengers": f"{TMDB_IMAGE_BASE}/RYMX2wcKCBAr24UyPD7xwmjaTn.jpg",
    "titanic": f"{TMDB_IMAGE_BASE}/9xjZS2rlVxm8SFx8kPC3aIGCOYQ.jpg",
    "the dark knight rises": f"{TMDB_IMAGE_BASE}/hr0L2aueqlP2BYUblTTjmtn0hw4.jpg",
    "iron man": f"{TMDB_IMAGE_BASE}/78lPtwv72eTNqFW9COBYI0dWDJa.jpg",
    "the matrix": f"{TMDB_IMAGE_BASE}/f89U3ADr1oiB1s9GkdPOEpXUk5H.jpg",
    "gladiator": f"{TMDB_IMAGE_BASE}/ty8TGRuvJLPUmAR1H1nRIsgwvim.jpg",
    "the godfather": f"{TMDB_IMAGE_BASE}/3bhkrj58Vtu7enYsRolD1fZdja1.jpg",
    "the lord of the rings: the return of the king": f"{TMDB_IMAGE_BASE}/rCzpDGLbOoPwLjy3OAm5NUPOTrC.jpg",
    "the lord of the rings: the two towers": f"{TMDB_IMAGE_BASE}/5VTN0pR8gcqV3EPUHHfMGnJYN9L.jpg",
    "whiplash": f"{TMDB_IMAGE_BASE}/7fn624j5lj3xTme2SgiLCeuedmO.jpg",
    "the wolf of wall street": f"{TMDB_IMAGE_BASE}/34m2tygAYBGqA9MXKhRDtzYd4MR.jpg",
    "memento": f"{TMDB_IMAGE_BASE}/yuNs09hvpHVU1cBTCAk9zxsL2oW.jpg",
    "mad max: fury road": f"{TMDB_IMAGE_BASE}/8tZYtuWezp8JbcsvHYO0O46tFbo.jpg",
    "star wars": f"{TMDB_IMAGE_BASE}/6FfCtAuVAW8XJjZ7eWeLibRLWTw.jpg",
    "wall-e": f"{TMDB_IMAGE_BASE}/hbhFnRzzg6ZDmm8YAmxBnQpQIPh.jpg"
}
POSTER_CATALOG_VERSION = tuple(POSTER_PATHS.items())

# Color palettes by genre for dynamic SVG poster generation
GENRE_GRADIENTS = {
    "action": ("#1e293b", "#0f172a", "#ef4444"),
    "adventure": ("#0f766e", "#134e4a", "#14b8a6"),
    "animation": ("#7c3aed", "#4c1d95", "#ec4899"),
    "comedy": ("#d97706", "#78350f", "#f59e0b"),
    "crime": ("#18181b", "#09090b", "#dc2626"),
    "drama": ("#1e1b4b", "#0f172a", "#6366f1"),
    "fantasy": ("#4338ca", "#312e81", "#a855f7"),
    "horror": ("#450a0a", "#1c1917", "#b91c1c"),
    "romance": ("#831843", "#500724", "#f43f5e"),
    "science fiction": ("#0369a1", "#082f49", "#06b6d4"),
    "sci-fi": ("#0369a1", "#082f49", "#06b6d4"),
    "thriller": ("#1f2937", "#111827", "#e11d48"),
    "default": ("#1e293b", "#0f172a", "#4361ee")
}


def generate_svg_poster(title: str, genre: str = "Movie", year: str = "", rating: float = 0.0) -> str:
    """
    Generates a stylish, clean, dark-themed SVG poster as a data-URI string.
    Ensures that every movie card looks polished, cinematic, and never has a broken image.
    """
    clean_genre = genre.lower().split(",")[0].strip() if genre else "default"
    c_start, c_end, accent = GENRE_GRADIENTS.get(clean_genre, GENRE_GRADIENTS["default"])

    display_title = title if len(title) <= 24 else title[:21] + "..."
    display_title = display_title.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    clean_genre_display = genre if len(genre) <= 20 else genre[:18] + ".."
    clean_genre_display = clean_genre_display.replace("&", "&amp;")

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 450" width="100%" height="100%">
  <defs>
    <linearGradient id="grad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="{c_start}"/>
      <stop offset="100%" stop-color="{c_end}"/>
    </linearGradient>
    <radialGradient id="glow" cx="50%" cy="35%" r="50%">
      <stop offset="0%" stop-color="{accent}" stop-opacity="0.3"/>
      <stop offset="100%" stop-color="{accent}" stop-opacity="0"/>
    </radialGradient>
  </defs>
  
  <rect width="300" height="450" rx="14" fill="url(#grad)"/>
  <rect width="300" height="450" rx="14" fill="url(#glow)"/>
  
  <g transform="translate(115, 120)" fill="{accent}" opacity="0.85">
    <rect x="0" y="0" width="70" height="52" rx="8" fill="none" stroke="{accent}" stroke-width="4"/>
    <circle cx="20" cy="26" r="6" fill="{accent}"/>
    <circle cx="50" cy="26" r="6" fill="{accent}"/>
    <line x1="35" y1="10" x2="35" y2="42" stroke="{accent}" stroke-width="3"/>
  </g>
  
  <rect x="30" y="220" width="40" height="4" rx="2" fill="{accent}"/>
  
  <text x="30" y="260" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-size="20" font-weight="700" fill="#ffffff">
    {display_title}
  </text>
  
  <text x="30" y="295" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-size="13" font-weight="500" fill="#94a3b8">
    {year + " • " if year else ""}{clean_genre_display}
  </text>
  
  <g transform="translate(30, 330)">
    <rect x="0" y="0" width="75" height="26" rx="13" fill="rgba(255,255,255,0.1)"/>
    <text x="12" y="18" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-size="13" font-weight="700" fill="#facc15">
      ★ {rating:.1f}
    </text>
  </g>
  
  <text x="30" y="420" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-size="10" font-weight="600" fill="#475569" letter-spacing="1.5">
    MOVIE REVIEWS NLP
  </text>
</svg>"""

    encoded_svg = urllib.parse.quote(svg)
    return f"data:image/svg+xml;utf8,{encoded_svg}"


def get_poster_url(title: str, genre: str = "Movie", year: str = "", rating: float = 0.0) -> str:
    """
    Returns the poster image URL for a movie.
    If cached TMDB CDN URL exists, returns it; otherwise returns high-res SVG poster data-URI.
    """
    clean_title = title.strip().lower()
    if clean_title in POSTER_PATHS:
        return POSTER_PATHS[clean_title]
    return generate_svg_poster(title, genre=genre, year=year, rating=rating)


def find_movie_poster_urls(movie_details: tuple[tuple[str, str], ...]) -> list[Optional[str]]:
    """Find poster thumbnails for several movie titles in one Wikimedia request."""
    candidate_movies: Dict[str, list[tuple[int, int]]] = {}
    titles = []
    for movie_index, (title, year) in enumerate(movie_details):
        candidates = []
        if year.strip().isdigit():
            candidates.append(f"{title} ({year.strip()} film)")
        candidates.extend((f"{title} (film)", title))
        for rank, candidate in enumerate(dict.fromkeys(candidates)):
            key = candidate.casefold()
            titles.append(candidate)
            candidate_movies.setdefault(key, []).append((movie_index, rank))

    if not titles:
        return []

    query_url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode({
        "action": "query",
        "prop": "pageimages",
        "piprop": "thumbnail",
        "pilicense": "any",
        "pithumbsize": "500",
        "titles": "|".join(titles),
        "format": "json"
    })
    request = Request(
        query_url,
        headers={"User-Agent": "MovieRecommendationSystem/1.0 (movie poster lookup)"}
    )
    try:
        with urlopen(request, timeout=8) as response:
            result = json.load(response)
    except HTTPError as error:
        LOGGER.warning("Batch poster lookup failed: HTTP %s", error.code)
        return [None] * len(movie_details)
    except (URLError, TimeoutError, json.JSONDecodeError) as error:
        LOGGER.warning("Batch poster lookup failed: %s", error)
        return [None] * len(movie_details)

    poster_urls: list[Optional[str]] = [None] * len(movie_details)
    poster_ranks = [len(titles)] * len(movie_details)
    pages = result.get("query", {}).get("pages", {})
    for page in pages.values():
        thumbnail = page.get("thumbnail")
        if (
            not isinstance(thumbnail, dict)
            or thumbnail.get("width", 0) >= thumbnail.get("height", 0)
        ):
            continue
        image_url = thumbnail.get("source", "")
        if not image_url.startswith("https://"):
            continue

        for movie_index, rank in candidate_movies.get(page.get("title", "").casefold(), []):
            if rank < poster_ranks[movie_index]:
                poster_urls[movie_index] = image_url
                poster_ranks[movie_index] = rank

    return poster_urls
