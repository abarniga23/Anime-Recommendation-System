import streamlit as st
import pandas as pd
import joblib
import os
from main import load_data, train_recommender, train_popularity_model, recommend_similar_anime, recommend_by_genre, search_titles

# Page config
st.set_page_config(
    page_title="Anime Recommender",
    page_icon="🎌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-title { font-size: 3em; color: #FF6B6B; text-align: center; margin-bottom: 20px; }
    .anime-card { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
    .info-text { font-size: 1.1em; }
    .score-badge { background: #FFD700; color: black; padding: 5px 10px; border-radius: 5px; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-title">🎌 Anime Recommendation System</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("⚙️ Settings")
st.sidebar.markdown("---")

# Load data and models (cached for performance)
@st.cache_resource
def load_models():
    df = load_data()
    
    # Check if models exist, if not train them
    if not os.path.exists("models/tfidf_vectorizer.pkl"):
        st.info("Training models for the first time... This may take a moment.")
        tfidf_matrix = train_recommender(df)
        pop_model, _ = train_popularity_model(df)
    else:
        tfidf_matrix = joblib.load("models/tfidf_matrix.pkl")
        pop_model = joblib.load("models/popularity_model.pkl")
    
    return df, tfidf_matrix, pop_model

df, tfidf_matrix, pop_model = load_models()

# Main navigation
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Similar Anime", 
    "🏆 By Genre", 
    "🔎 Search", 
    "📊 About"
])

# TAB 1: SIMILAR ANIME RECOMMENDATIONS
with tab1:
    st.subheader("Find anime similar to your favorite!")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Get list of anime titles
        anime_titles = sorted(df["title"].unique())
        selected_title = st.selectbox(
            "Select an anime:",
            anime_titles,
            key="similar_title"
        )
    
    with col2:
        age_filter = st.selectbox(
            "Age rating filter:",
            ["None", "PG-13", "PG", "R", "R+", "Rx"],
            key="age_filter"
        )
    
    if st.button("Get Recommendations", key="btn_similar"):
        with st.spinner("Finding similar anime..."):
            results = recommend_similar_anime(
                selected_title, 
                df, 
                tfidf_matrix, 
                pop_model,
                age_filter=age_filter if age_filter != "None" else None
            )
            
            if results is not None:
                st.success(f"Found {len(results)} recommendations!")
                
                for idx, (i, row) in enumerate(results.iterrows(), 1):
                    with st.container():
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.markdown(f"### {idx}. {row['title']}")
                            st.markdown(f"**Genres:** {row['Genres']}")
                            st.markdown(f"**Type:** {row['Type']} | **Rating:** {row['Rating']}")
                        
                        with col2:
                            st.markdown(f"<span class='score-badge'>⭐ {row['Score']}</span>", unsafe_allow_html=True)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Similarity", f"{row['similarity']*100:.1f}%")
                        with col2:
                            st.metric("Popularity", f"{row['popularity_prob']*100:.1f}%")
                        with col3:
                            st.metric("Members", f"{int(row['Members']):,}")
                        
                        st.markdown("---")
            else:
                st.warning("Anime not found in database.")

# TAB 2: GENRE-BASED RECOMMENDATIONS
with tab2:
    st.subheader("Top anime by genre")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        genre = st.text_input(
            "Enter a genre (e.g., Action, Romance, Drama, Fantasy):",
            placeholder="Type genre name...",
            key="genre_input"
        )
    
    with col2:
        st.write("")  # Spacing
        if st.button("Search Genre", key="btn_genre"):
            if genre:
                with st.spinner(f"Searching for {genre} anime..."):
                    results = recommend_by_genre(df, genre)
                    
                    if results is not None:
                        st.success(f"Found {len(results)} anime in {genre}!")
                        
                        for idx, (i, row) in enumerate(results.iterrows(), 1):
                            col1, col2, col3 = st.columns([2, 1, 1])
                            
                            with col1:
                                st.markdown(f"**{idx}. {row['title']}**")
                                st.markdown(f"*{row['Genres']}*")
                            
                            with col2:
                                st.metric("Score", f"{row['Score']}")
                            
                            with col3:
                                st.metric("Members", f"{int(row['Members']/1000)}K")
                            
                            st.markdown("---")
                    else:
                        st.warning(f"No anime found for genre: {genre}")
            else:
                st.warning("Please enter a genre name")

# TAB 3: SEARCH
with tab3:
    st.subheader("Search for anime by title")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        keyword = st.text_input(
            "Enter keyword:",
            placeholder="Search anime title...",
            key="search_input"
        )
    
    with col2:
        st.write("")  # Spacing
        if st.button("Search", key="btn_search"):
            if keyword:
                with st.spinner("Searching..."):
                    results = search_titles(df, keyword)
                    
                    if results is not None:
                        st.success(f"Found {len(results)} results!")
                        
                        for _, row in results.iterrows():
                            col1, col2 = st.columns([3, 1])
                            
                            with col1:
                                st.markdown(f"**{row['title']}**")
                                st.markdown(f"{row['Genres']} • {row['Type']}")
                            
                            with col2:
                                st.metric("Score", f"{row['Score']}")
                            
                            st.markdown("---")
                    else:
                        st.warning(f"No results found for '{keyword}'")
            else:
                st.warning("Please enter a search keyword")

# TAB 4: ABOUT
with tab4:
    st.markdown("## 📊 About This System")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🧠 ML Models Used")
        st.markdown("""
        - **TF-IDF Vectorizer**: Analyzes anime titles, descriptions, genres, and themes
        - **Cosine Similarity**: Finds similar anime based on content
        - **Logistic Regression**: Predicts anime popularity
        """)
    
    with col2:
        st.markdown("### 📈 Dataset Stats")
        st.metric("Total Anime", len(df))
        st.metric("Avg Score", f"{df['Score'].mean():.2f}")
        st.metric("Total Members", f"{int(df['Members'].sum()):,}")
    
    st.markdown("---")
    
    st.markdown("### 🎯 Features")
    st.markdown("""
    1. **Similar Anime**: Find recommendations based on a specific anime
    2. **Genre Browsing**: Explore top-rated anime by genre
    3. **Search**: Quick search for any anime in the database
    4. **Age Filtering**: Filter recommendations by age rating
    5. **Popularity Scoring**: ML-powered popularity predictions
    """)
    
    st.markdown("---")
    
    st.info("💡 **Tip**: Use the sidebar to switch between different recommendation modes!")
