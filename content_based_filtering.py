import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def content_based_recommendation(
    data: pd.DataFrame,
    item_name: str,
    top_n: int = 10
) -> pd.DataFrame:
    """
    Recommends similar products based on product content (Tags).
    Supports partial and case-insensitive search.
    Returns full product details for UI display.
    """

    # -----------------------------
    # 1. Input validation
    # -----------------------------
    if not item_name or not isinstance(item_name, str):
        return pd.DataFrame()

    # -----------------------------
    # 2. Case-insensitive partial match
    # -----------------------------
    item_name = item_name.lower().strip()

    matches = data[
        data["Name"]
        .astype(str)
        .str.lower()
        .str.contains(item_name, na=False)
    ]

    if matches.empty:
        return pd.DataFrame()

    # Pick the first matching product
    item_index = matches.index[0]

    # -----------------------------
    # 3. TF-IDF on Tags
    # -----------------------------
    tfidf = TfidfVectorizer(stop_words="english")

    tfidf_matrix = tfidf.fit_transform(
        data["Tags"].fillna("")
    )

    # -----------------------------
    # 4. Cosine similarity
    # -----------------------------
    similarity_matrix = cosine_similarity(
        tfidf_matrix,
        tfidf_matrix
    )

    # -----------------------------
    # 5. Get top similar items
    # -----------------------------
    similarity_scores = list(
        enumerate(similarity_matrix[item_index])
    )

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    # Skip the first one (same product)
    top_similar_indices = [
        idx for idx, score in similarity_scores[1 : top_n + 1]
    ]

    # -----------------------------
    # 6. Return FULL product details
    # -----------------------------
    recommended_products = data.iloc[top_similar_indices][
        [
            "Name",
            "Brand",
            "Rating",
            "ReviewCount",
            "ImageURL"
        ]
    ]

    return recommended_products.reset_index(drop=True)


# -----------------------------
# Local testing
# -----------------------------
if __name__ == "__main__":
    from preprocess_data import process_data

    raw_data = pd.read_csv("clean_data.csv")
    data = process_data(raw_data)

    search_term = "oil"
    results = content_based_recommendation(data, search_term, top_n=5)

    print(results)
