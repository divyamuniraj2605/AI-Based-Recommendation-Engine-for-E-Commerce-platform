import streamlit as st
import pandas as pd

from preprocess_data import process_data
from rating_based_recommendation import get_top_rated_items
from content_based_filtering import content_based_recommendation
from collaborative_based_filtering import collaborative_filtering_recommendations


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(layout="wide")


# =========================================================
# LOAD DATA
# =========================================================
@st.cache_data
def load_data():
    raw = pd.read_csv("clean_data.csv")
    return process_data(raw)

data = load_data()


# =========================================================
# CATEGORY NORMALIZATION
# =========================================================
CATEGORY_MAPPING = {
    "Beauty": ["beauty", "cosmetic", "makeup", "skin", "hair", "cream", "lotion", "shampoo"],
    "Electronics": ["electronic", "mobile", "laptop", "charger", "camera", "headphone", "earphone"],
    "Fashion": ["fashion", "clothing", "dress", "shirt", "jeans", "shoes", "footwear"],
    "Home & Kitchen": ["home", "kitchen", "cookware", "utensil", "furniture", "decor"],
    "Health": ["health", "medicine", "supplement", "vitamin", "wellness", "medical"],
    "Sports & Fitness": ["sports", "fitness", "gym", "exercise", "yoga"],
    "Books & Stationery": ["book", "novel", "study", "stationery", "pen", "notebook"],
    "Others": []
}


def map_to_main_category(raw_category):
    if not isinstance(raw_category, str):
        return "Others"
    raw_category = raw_category.lower()
    for main_cat, keywords in CATEGORY_MAPPING.items():
        for kw in keywords:
            if kw in raw_category:
                return main_cat
    return "Others"


data["MainCategory"] = data["Category"].apply(map_to_main_category)


# =========================================================
# SESSION STATE
# =========================================================
if "search_history" not in st.session_state:
    st.session_state.search_history = []

if "selected_category" not in st.session_state:
    st.session_state.selected_category = "All"


# =========================================================
# HEADER
# =========================================================
st.title("🛒 AI-Enabled Recommendation Engine")
st.caption("Personalized product recommendations using AI")
st.divider()


# =========================================================
# LAYOUT
# =========================================================
left_col, right_col = st.columns([1, 4])


# =========================================================
# LEFT PANEL
# =========================================================
with left_col:
    st.subheader("👤 Account")

    user_id = st.number_input(
        "User ID (0 = New User)",
        min_value=0,
        step=1
    )

    product_search = st.text_input(
        "Search Product",
        placeholder="Enter product name"
    )

    get_rec = st.button("Get Recommendations")

    # Recent activity
    st.subheader("🕒 Recent Activity")
    if st.session_state.search_history:
        for item in reversed(st.session_state.search_history[-5:]):
            st.markdown(f"- {item}")
    else:
        st.caption("No recent searches")

    # Categories
    st.subheader("📂 Categories")
    categories = sorted(data["MainCategory"].unique().tolist())
    categories.insert(0, "All")

    selected_category = st.radio(
        "Browse by Category",
        categories,
        index=categories.index(st.session_state.selected_category)
    )

    st.session_state.selected_category = selected_category


# =========================================================
# HELPER FUNCTIONS
# =========================================================
def filter_by_category(df):
    if st.session_state.selected_category == "All":
        return df
    return df[df["MainCategory"] == st.session_state.selected_category]


def display_products(recs_with_details, products_per_row=3):
    placeholder_img = "https://via.placeholder.com/300x300.png?text=No+Image"

    for i, (_, product) in enumerate(recs_with_details.head(9).iterrows()):
        if i % products_per_row == 0:
            cols = st.columns(products_per_row)

        with cols[i % products_per_row]:
            img_url = product.get("ImageURL")
            final_img = img_url if isinstance(img_url, str) and img_url.startswith("http") else placeholder_img

            st.markdown(
                f"""
                <div style="height:260px; display:flex; align-items:center;
                            justify-content:center; background:#111;
                            border-radius:10px; overflow:hidden;">
                    <img src="{final_img}"
                         style="max-height:100%; max-width:100%; object-fit:contain;">
                </div>
                """,
                unsafe_allow_html=True
            )

            name = product["Name"]
            display_name = name[:55] + "..." if len(name) > 55 else name

            st.markdown(f"**{display_name}**")
            st.markdown(f"Brand: {product['Brand']}")
            st.markdown(f"⭐ {round(product['Rating'], 1)}")

            st.button("Add", key=f"{product['Name']}_{i}")


# =========================================================
# RIGHT PANEL – RECOMMENDATIONS
# =========================================================
with right_col:
    filtered_data = filter_by_category(data)

    # ---------- DEFAULT ----------
    if not get_rec:
        st.subheader("⭐ Top Rated Products")
        display_products(get_top_rated_items(filtered_data, top_n=9))

    else:
        if product_search.strip():
            st.session_state.search_history.append(product_search)

        # ---------- NEW USER ----------
        if user_id == 0:
            st.subheader("⭐ Recommended for You")
            display_products(get_top_rated_items(filtered_data, top_n=9))

        # ---------- EXISTING USER ----------
        else:
            st.subheader(f"🎯 Welcome Back, User {user_id}")

            # CONTENT-BASED
            st.markdown("### 🧠 Similar Products")
            base_product = (
                product_search
                if product_search.strip()
                else filtered_data[filtered_data["ID"] == user_id]["Name"].iloc[0]
            )

            content_recs = content_based_recommendation(
                filtered_data,
                base_product,
                top_n=6
            )
            display_products(content_recs)

            # COLLABORATIVE
            st.markdown("### 👥 Users Like You Also Liked")
            collab_recs = collaborative_filtering_recommendations(
                filtered_data,
                user_id,
                top_n=6
            )
            display_products(collab_recs)
