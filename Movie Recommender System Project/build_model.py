"""
build_model.py  –  Improved Movie Recommender System
=====================================================
Dataset : tmdb_5000_movies.csv + tmdb_5000_credits.csv (Kaggle)
Outputs : movies.pkl  (processed DataFrame)
          similarity.pkl  (cosine-similarity matrix)
          vectorizer.pkl  (fitted TfidfVectorizer – reuse at inference)

Key improvements over the original
-----------------------------------
1.  TF-IDF instead of plain CountVectorizer  → penalises overly common words
2.  Weighted tags  → genres × 3, director × 3, cast × 2, keywords × 2
3.  Low-vote filtering  → removes obscure titles with inflated ratings
4.  Popularity-adjusted scorer  → blends cosine sim with normalised vote avg
5.  Location layer  → per-country language + regional preference scores
6.  Fuzzy title matching  → forgiving at query time
7.  Proper logging  → replaces print(); configurable via LOG_LEVEL env var
8.  Config dataclass  → single place for every tunable constant
9.  Clean recommend() API  → returns a rich list of dicts, not just titles

Usage
-----
    python build_model.py                        # build & save model
    python build_model.py --country IN           # test with India locale
    python build_model.py --movie "Inception"    # quick inference smoke-test
"""

import ast
import argparse
import logging
import os
import pickle
from dataclasses import dataclass, field
from difflib import get_close_matches
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class Config:
    # Data paths
    movies_csv: str = "tmdb_5000_movies.csv"
    credits_csv: str = "tmdb_5000_credits.csv"
    output_dir: str = "."

    # Vectorizer
    max_features: int = 5000

    # Tag weights (repeat the field that many times before joining)
    weight_overview: int = 1
    weight_genres: int = 3
    weight_keywords: int = 2
    weight_cast: int = 2
    weight_crew: int = 3          # director

    # Quality filter  –  drop movies below this percentile of vote_count
    min_vote_quantile: float = 0.25

    # Scoring blend  (0.0 = pure content, 1.0 = pure location)
    location_alpha: float = 0.25
    popularity_weight: float = 0.10   # fraction of score from vote_average

    # Inference defaults
    top_n: int = 10
    default_country: str = "US"

    # Cast: how many actors to keep
    cast_limit: int = 3


cfg = Config()


# ---------------------------------------------------------------------------
# Location data
# ---------------------------------------------------------------------------
COUNTRY_LANGUAGE_MAP: dict[str, list[str]] = {
    "IN": ["hi", "ta", "te", "ml", "kn", "mr", "en"],
    "US": ["en"],
    "GB": ["en"],
    "KR": ["ko", "en"],
    "JP": ["ja", "en"],
    "CN": ["zh", "en"],
    "FR": ["fr", "en"],
    "DE": ["de", "en"],
    "ES": ["es", "en"],
    "IT": ["it", "en"],
    "BR": ["pt", "en"],
    "MX": ["es", "en"],
    "RU": ["ru", "en"],
    "TR": ["tr", "en"],
    "TH": ["th", "en"],
    "ID": ["id", "en"],
    "PK": ["ur", "hi", "en"],
    "EG": ["ar", "en"],
    "NG": ["en"],
}

REGIONAL_KEYWORDS: dict[str, list[str]] = {
    "IN": ["bollywood", "india", "hindi", "mumbai", "delhi", "pune",
           "bengaluru", "chennai", "kolkata", "rajasthan"],
    "KR": ["korea", "korean", "seoul", "busan"],
    "JP": ["japan", "japanese", "tokyo", "osaka", "kyoto", "anime"],
    "CN": ["china", "chinese", "beijing", "shanghai", "hong kong"],
    "FR": ["france", "french", "paris"],
    "DE": ["germany", "german", "berlin"],
    "ES": ["spain", "spanish", "madrid"],
    "BR": ["brazil", "brazilian", "rio"],
    "MX": ["mexico", "mexican"],
    "RU": ["russia", "russian", "moscow"],
}


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
def _parse_names(obj: str) -> list[str]:
    """Extract all 'name' values from a JSON-ish list-of-dicts string."""
    try:
        return [i["name"] for i in ast.literal_eval(obj) if "name" in i]
    except Exception:
        return []


def _parse_names_top(obj: str, limit: int = 3) -> list[str]:
    """Extract the first `limit` name values."""
    return _parse_names(obj)[:limit]


def _parse_director(obj: str) -> list[str]:
    """Return [director_name] or [] from a crew JSON string."""
    try:
        for i in ast.literal_eval(obj):
            if i.get("job") == "Director":
                return [i["name"]]
    except Exception:
        pass
    return []


def _clean_names(names: list[str]) -> list[str]:
    """Strip spaces so 'Tom Hanks' → 'TomHanks' (avoids bag-of-words split)."""
    return [n.replace(" ", "") for n in names]


# ---------------------------------------------------------------------------
# Stemmer
# ---------------------------------------------------------------------------
_ps = PorterStemmer()

def _stem(text: str) -> str:
    return " ".join(_ps.stem(w) for w in text.split())


# ---------------------------------------------------------------------------
# Tag builder  (weighted repetition)
# ---------------------------------------------------------------------------
def build_weighted_tags(row: pd.Series) -> str:
    """
    Combine overview + genres + keywords + cast + director into a single
    tag string, repeating each field according to its configured weight.
    """
    overview  = row["overview"]  * cfg.weight_overview
    genres    = row["genres"]    * cfg.weight_genres
    keywords  = row["keywords"]  * cfg.weight_keywords
    cast      = row["cast"]      * cfg.weight_cast
    crew      = row["crew"]      * cfg.weight_crew
    return " ".join(overview + genres + keywords + cast + crew).lower()


# ---------------------------------------------------------------------------
# Location Scorer
# ---------------------------------------------------------------------------
class LocationScorer:
    """
    Computes a [0, 1] location-affinity score for each movie given a
    user's country.  Combines:
      - language_score  : primary / secondary language match
      - regional_score  : tag keywords associated with the user's region
    """

    def __init__(self, df: pd.DataFrame, country: str = "US"):
        self.df = df
        self.country = country.upper()
        self.preferred_langs = COUNTRY_LANGUAGE_MAP.get(self.country, ["en"])
        self.regional_kws    = REGIONAL_KEYWORDS.get(self.country, [])

    def language_score(self, lang: str) -> float:
        if not self.preferred_langs:
            return 0.0
        if lang == self.preferred_langs[0]:   # native / primary
            return 1.0
        if lang in self.preferred_langs:      # secondary (usually 'en')
            return 0.5
        return 0.0

    def regional_score(self, tags: str) -> float:
        if not self.regional_kws:
            return 0.0
        hits = sum(1 for kw in self.regional_kws if kw in tags)
        return min(hits / len(self.regional_kws), 1.0)

    def score(self, idx: int) -> float:
        row  = self.df.iloc[idx]
        lang = row.get("original_language", "en")
        tags = row.get("tags", "")
        return 0.6 * self.language_score(lang) + 0.4 * self.regional_score(tags)


# ---------------------------------------------------------------------------
# Fuzzy title match
# ---------------------------------------------------------------------------
def find_movie_title(query: str, titles: list[str], cutoff: float = 0.55) -> str:
    """
    Return the closest matching title, case-insensitively.
    Raises ValueError if nothing is found above `cutoff`.
    """
    lower_map = {t.lower(): t for t in titles}
    matches = get_close_matches(query.lower(), lower_map.keys(), n=1, cutoff=cutoff)
    if not matches:
        raise ValueError(
            f"No match found for '{query}'. "
            f"Try: {get_close_matches(query.lower(), lower_map.keys(), n=3, cutoff=0.3)}"
        )
    return lower_map[matches[0]]


# ---------------------------------------------------------------------------
# Core recommendation function
# ---------------------------------------------------------------------------
def recommend(
    movie: str,
    df: pd.DataFrame,
    sim_matrix: np.ndarray,
    country: str = "US",
    n: int = 10,
    location_alpha: float = 0.25,
    popularity_weight: float = 0.10,
) -> list[dict]:
    """
    Return top-`n` recommendations for `movie`.

    Each result dict contains:
        title, movie_id, blended_score, cosine_score, location_score
    """
    # --- resolve title with fuzzy match ---
    canonical = find_movie_title(movie, df["title"].tolist())
    if canonical != movie:
        log.info("Matched '%s' → '%s'", movie, canonical)

    idx = df.index[df["title"] == canonical].tolist()
    if not idx:
        raise ValueError(f"'{canonical}' not found after fuzzy match.")
    idx = idx[0]

    # --- cosine similarity scores ---
    cos_scores = list(enumerate(sim_matrix[idx]))

    # --- popularity normalisation ---
    max_vote_avg = df["vote_average"].max() if "vote_average" in df.columns else 1.0

    # --- location scorer ---
    loc_scorer = LocationScorer(df, country=country)

    # --- blend scores ---
    results = []
    for i, cos in cos_scores:
        if i == idx:                          # skip the query movie itself
            continue
        loc = loc_scorer.score(i)
        pop = (df.iloc[i].get("vote_average", 0) / max_vote_avg) if max_vote_avg else 0

        # three-way blend
        content_score = cos * (1 - popularity_weight) + pop * popularity_weight
        blended       = (1 - location_alpha) * content_score + location_alpha * loc

        results.append({
            "title":         df.iloc[i]["title"],
            "movie_id":      int(df.iloc[i]["movie_id"]),
            "blended_score": round(float(blended), 4),
            "cosine_score":  round(float(cos), 4),
            "location_score": round(float(loc), 4),
        })

    results.sort(key=lambda x: x["blended_score"], reverse=True)
    return results[:n]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def build_pipeline(config: Config) -> tuple[pd.DataFrame, np.ndarray, TfidfVectorizer]:
    """
    Full data → model pipeline.
    Returns (processed_df, similarity_matrix, fitted_vectorizer).
    """

    # 1. Load ---
    log.info("Loading '%s' and '%s' …", config.movies_csv, config.credits_csv)
    movies  = pd.read_csv(config.movies_csv)
    credits = pd.read_csv(config.credits_csv)

    # 2. Merge ---
    log.info("Merging on 'title' …")
    movies = movies.merge(credits, on="title")

    required_cols = [
        "movie_id", "title", "overview", "genres", "keywords",
        "cast", "crew", "vote_average", "vote_count", "original_language",
    ]
    missing = [c for c in required_cols if c not in movies.columns]
    if missing:
        log.warning("Missing columns (will be skipped): %s", missing)

    keep = [c for c in required_cols if c in movies.columns]
    movies = movies[keep].dropna().reset_index(drop=True)
    log.info("Rows after dropna: %d", len(movies))

    # 3. Filter low-vote movies ---
    if "vote_count" in movies.columns:
        threshold = movies["vote_count"].quantile(config.min_vote_quantile)
        before    = len(movies)
        movies    = movies[movies["vote_count"] >= threshold].reset_index(drop=True)
        log.info(
            "Dropped %d low-vote movies (threshold = %.0f votes). Remaining: %d",
            before - len(movies), threshold, len(movies),
        )

    # 4. Parse JSON columns ---
    log.info("Parsing genres, keywords, cast, crew …")
    movies["genres"]   = movies["genres"].apply(_parse_names)
    movies["keywords"] = movies["keywords"].apply(_parse_names)
    movies["cast"]     = movies["cast"].apply(lambda x: _parse_names_top(x, config.cast_limit))
    movies["crew"]     = movies["crew"].apply(_parse_director)
    movies["overview"] = movies["overview"].apply(lambda x: x.split() if isinstance(x, str) else [])

    # 5. Normalise names (remove spaces) ---
    for col in ("cast", "crew", "genres", "keywords"):
        movies[col] = movies[col].apply(_clean_names)

    # 6. Build weighted tag string ---
    log.info("Building weighted tag strings …")
    movies["tags"] = movies.apply(build_weighted_tags, axis=1)

    # 7. Stem ---
    log.info("Stemming …")
    movies["tags"] = movies["tags"].apply(_stem)

    # 8. Vectorise ---
    log.info("Fitting TF-IDF vectorizer (max_features=%d) …", config.max_features)
    vectorizer = TfidfVectorizer(max_features=config.max_features, stop_words="english")
    vectors    = vectorizer.fit_transform(movies["tags"]).toarray()
    log.info("Vector matrix shape: %s", vectors.shape)

    # 9. Cosine similarity ---
    log.info("Computing cosine similarity matrix …")
    sim = cosine_similarity(vectors)
    log.info("Similarity matrix shape: %s", sim.shape)

    # 10. Slim output dataframe ---
    out_cols = ["movie_id", "title", "tags", "original_language", "vote_average"]
    out_cols = [c for c in out_cols if c in movies.columns]
    out_df   = movies[out_cols].reset_index(drop=True)

    return out_df, sim, vectorizer


def save_artifacts(
    df: pd.DataFrame,
    sim: np.ndarray,
    vec: TfidfVectorizer,
    output_dir: str = ".",
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    movies_path     = out / "movies.pkl"
    sim_path        = out / "similarity.pkl"
    vectorizer_path = out / "vectorizer.pkl"

    with open(movies_path, "wb") as f:
        pickle.dump(df, f)
    with open(sim_path, "wb") as f:
        pickle.dump(sim, f)
    with open(vectorizer_path, "wb") as f:
        pickle.dump(vec, f)

    log.info("Saved → %s", movies_path)
    log.info("Saved → %s", sim_path)
    log.info("Saved → %s", vectorizer_path)


def load_artifacts(
    output_dir: str = ".",
) -> tuple[pd.DataFrame, np.ndarray, TfidfVectorizer]:
    out = Path(output_dir)
    with open(out / "movies.pkl", "rb") as f:
        df = pickle.load(f)
    with open(out / "similarity.pkl", "rb") as f:
        sim = pickle.load(f)
    with open(out / "vectorizer.pkl", "rb") as f:
        vec = pickle.load(f)
    return df, sim, vec


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build / test the Movie Recommender model")
    p.add_argument("--movies-csv",  default=cfg.movies_csv,  help="Path to tmdb_5000_movies.csv")
    p.add_argument("--credits-csv", default=cfg.credits_csv, help="Path to tmdb_5000_credits.csv")
    p.add_argument("--output-dir",  default=cfg.output_dir,  help="Directory for .pkl outputs")
    p.add_argument("--country",     default=cfg.default_country, help="2-letter country code for location test")
    p.add_argument("--movie",       default=None,            help="Movie title to test after building")
    p.add_argument("--top-n",       type=int, default=cfg.top_n, help="Number of recommendations")
    p.add_argument("--no-build",    action="store_true",     help="Skip build; load existing .pkl files")
    p.add_argument("--alpha",       type=float, default=cfg.location_alpha,
                   help="Location blend weight (0=pure content, 1=pure location)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # --- apply CLI overrides to config ---
    cfg.movies_csv     = args.movies_csv
    cfg.credits_csv    = args.credits_csv
    cfg.output_dir     = args.output_dir
    cfg.location_alpha = args.alpha

    if args.no_build:
        log.info("--no-build: loading existing artifacts …")
        df, sim, vec = load_artifacts(cfg.output_dir)
    else:
        df, sim, vec = build_pipeline(cfg)
        save_artifacts(df, sim, vec, cfg.output_dir)

    # --- summary ---
    log.info("")
    log.info("✅  Model ready")
    log.info("    Movies    : %d", len(df))
    log.info("    Sim matrix: %s", sim.shape)
    log.info("    Vocabulary: %d terms", len(vec.vocabulary_))
    log.info("")

    # --- optional smoke-test ---
    test_movie = args.movie or "The Dark Knight"
    country    = args.country

    log.info("Smoke-test: recommend('%s', country='%s', n=%d, alpha=%.2f)",
             test_movie, country, args.top_n, cfg.location_alpha)
    try:
        recs = recommend(
            test_movie, df, sim,
            country=country,
            n=args.top_n,
            location_alpha=cfg.location_alpha,
            popularity_weight=cfg.popularity_weight,
        )
        header = f"{'#':<3}  {'Title':<40}  {'Blended':>8}  {'Cosine':>8}  {'Location':>9}"
        log.info(header)
        log.info("-" * len(header))
        for rank, r in enumerate(recs, 1):
            log.info(
                "%-3d  %-40s  %8.4f  %8.4f  %9.4f",
                rank, r["title"][:40], r["blended_score"], r["cosine_score"], r["location_score"],
            )
    except ValueError as e:
        log.error("Recommendation failed: %s", e)


if __name__ == "__main__":
    main()
