import streamlit as st
import pickle
import pandas as pd
import requests
import time
from pathlib import Path

# Set page config
st.set_page_config(page_title="Movie Recommender System", page_icon="🎬", layout="wide")

# Cache the poster fetching to avoid repeated API calls
@st.cache_data
def fetch_poster(movie_id):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                poster_path = data.get('poster_path')
                if poster_path:
                    full_path = "https://image.tmdb.org/t/p/w500" + poster_path
                    return full_path
            
            if attempt == max_retries - 1:
                return "https://via.placeholder.com/500x750/1e1e1e/ffffff?text=No+Poster+Available"
                
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(0.5)
                continue
            return "https://via.placeholder.com/500x750/1e1e1e/ffffff?text=Poster+Unavailable"
    
    return "https://via.placeholder.com/500x750/1e1e1e/ffffff?text=No+Poster"

# Function to recommend movies using the improved model
def recommend(movie, df, sim_matrix, country="US", n=5, location_alpha=0.25, popularity_weight=0.10):
    """
    Improved recommendation function that matches the new build_model.py format.
    Returns list of dicts with title, movie_id, and scores.
    """
    try:
        # Find movie index
        movie_index = df[df['title'] == movie].index[0]
        
        # Get cosine similarity scores
        cos_scores = list(enumerate(sim_matrix[movie_index]))
        
        # Calculate popularity normalization
        max_vote_avg = df["vote_average"].max() if "vote_average" in df.columns else 1.0
        
        # Calculate location score (simplified - using language preference)
        def get_location_score(idx):
            lang = df.iloc[idx].get("original_language", "en")
            # US prefers English
            if country == "US":
                return 1.0 if lang == "en" else 0.5
            return 0.6  # Default score for other countries
        
        # Blend scores
        results = []
        for i, cos in cos_scores:
            if i == movie_index:  # Skip the query movie itself
                continue
            
            loc = get_location_score(i)
            pop = (df.iloc[i].get("vote_average", 0) / max_vote_avg) if max_vote_avg else 0
            
            # Three-way blend
            content_score = cos * (1 - popularity_weight) + pop * popularity_weight
            blended = (1 - location_alpha) * content_score + location_alpha * loc
            
            results.append({
                "title": df.iloc[i]["title"],
                "movie_id": int(df.iloc[i]["movie_id"]),
                "blended_score": round(float(blended), 4),
                "cosine_score": round(float(cos), 4),
            })
        
        results.sort(key=lambda x: x["blended_score"], reverse=True)
        return results[:n]
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return []

# Load data
try:
    movies = pickle.load(open('movies.pkl', 'rb'))
    similarity = pickle.load(open('similarity.pkl', 'rb'))
    
    # Check if vectorizer exists (new model format)
    vectorizer_path = Path('vectorizer.pkl')
    if vectorizer_path.exists():
        vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
        st.sidebar.success("✨ Using improved TF-IDF model")
    
    # Title
    st.title('🎬 MOVIE RECOMMENDATION SYSTEM')
    st.markdown("### Powered by Advanced Content-Based Filtering")
    st.markdown("---")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Country selection
        country = st.selectbox(
            "🌍 Your Country",
            ["US", "IN", "GB", "KR", "JP", "CN", "FR", "DE", "ES", "IT", "BR", "MX"],
            index=0,
            help="Recommendations will be personalized for your region"
        )
        
        # Number of recommendations
        num_recommendations = st.slider(
            "📊 Number of Recommendations",
            min_value=3,
            max_value=15,
            value=5,
            help="How many movies to recommend"
        )
        
        # Advanced settings
        with st.expander("🔧 Advanced Settings"):
            location_alpha = st.slider(
                "Location Weight",
                min_value=0.0,
                max_value=1.0,
                value=0.25,
                step=0.05,
                help="0 = Pure content similarity, 1 = Pure location preference"
            )
            popularity_weight = st.slider(
                "Popularity Weight",
                min_value=0.0,
                max_value=0.5,
                value=0.10,
                step=0.05,
                help="How much to consider movie ratings"
            )
        
        st.markdown("---")
        st.header("📊 Statistics")
        st.metric("Total Movies", len(movies))
        st.metric("Similarity Matrix", f"{similarity.shape[0]} × {similarity.shape[1]}")
        if vectorizer_path.exists():
            st.metric("TF-IDF Features", "5,000")
        
        st.markdown("---")
        st.header("ℹ️ About")
        st.info("""
        **Improved Features:**
        - 🎯 TF-IDF Vectorization
        - 🌍 Location-based filtering
        - ⭐ Popularity scoring
        - 🎬 Quality movie filtering
        - 🔍 Weighted tag analysis
        
        This system analyzes genres, keywords, cast, crew, and plot with intelligent weighting.
        """)
    
    # Movie selection
    st.subheader("Select a movie to get recommendations")
    
    # Add search functionality
    search_term = st.text_input("🔍 Search for a movie:", "")
    
    if search_term:
        filtered_movies = movies[movies['title'].str.contains(search_term, case=False, na=False)]['title'].values
        if len(filtered_movies) > 0:
            selected_movie = st.selectbox('Choose a movie:', filtered_movies, index=0)
        else:
            st.warning(f"No movies found matching '{search_term}'")
            selected_movie = st.selectbox('Choose a movie:', movies['title'].values, index=0)
    else:
        selected_movie = st.selectbox('Choose a movie:', movies['title'].values, index=0)
    
    # Show movie info
    movie_info = movies[movies['title'] == selected_movie].iloc[0]
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if 'vote_average' in movie_info:
            st.metric("⭐ Rating", f"{movie_info['vote_average']:.1f}/10")
    with col2:
        if 'original_language' in movie_info:
            st.metric("🌐 Language", movie_info['original_language'].upper())
    with col3:
        st.metric("🆔 Movie ID", movie_info['movie_id'])
    
    # Recommend button
    if st.button('🎯 Show Recommendations', type="primary", use_container_width=True):
        with st.spinner('🔍 Finding similar movies...'):
            recs = recommend(
                selected_movie, 
                movies, 
                similarity,
                country=country,
                n=num_recommendations,
                location_alpha=location_alpha,
                popularity_weight=popularity_weight
            )
        
        if recs:
            st.success(f'✨ Here are your top {len(recs)} recommendations!')
            st.markdown("---")
            
            # Display recommendations in columns (5 per row)
            for row_start in range(0, len(recs), 5):
                cols = st.columns(5)
                for idx, col in enumerate(cols):
                    if row_start + idx < len(recs):
                        rec = recs[row_start + idx]
                        with col:
                            try:
                                poster_url = fetch_poster(rec['movie_id'])
                                st.image(poster_url, use_column_width=True)
                            except:
                                st.image("https://via.placeholder.com/500x750/1e1e1e/ffffff?text=Image+Error", use_column_width=True)
                            
                            st.markdown(f"**{rec['title']}**")
                            st.caption(f"Score: {rec['blended_score']:.3f}")
                            
                            # Show score breakdown in expander
                            with st.expander("📊 Details"):
                                st.write(f"Content: {rec['cosine_score']:.3f}")
                                st.write(f"Blended: {rec['blended_score']:.3f}")
        else:
            st.warning("No recommendations found. Please try another movie.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>Built with ❤️ using Streamlit | Data from TMDB | Improved TF-IDF Model</p>
        </div>
        """,
        unsafe_allow_html=True
    )

except FileNotFoundError as e:
    st.error("⚠️ Model files not found!")
    st.info("Please run 'build_model.py' first to generate the required files.")
    st.code("python build_model.py", language="bash")
    st.error(f"Missing file: {e}")
    
except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.info("Please make sure all required files are present and try again.")
    import traceback
    st.code(traceback.format_exc())
