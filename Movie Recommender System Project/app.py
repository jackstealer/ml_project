import streamlit as st
import pickle
import pandas as pd
import requests
from functools import lru_cache
import time

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
            
            # If no poster found, return placeholder
            if attempt == max_retries - 1:
                return "https://via.placeholder.com/500x750/1e1e1e/ffffff?text=No+Poster+Available"
                
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(0.5)  # Wait before retry
                continue
            # Return placeholder on final error
            return "https://via.placeholder.com/500x750/1e1e1e/ffffff?text=Poster+Unavailable"
    
    return "https://via.placeholder.com/500x750/1e1e1e/ffffff?text=No+Poster"

# Function to recommend movies
def recommend(movie):
    try:
        movie_index = movies[movies['title'] == movie].index[0]
        distances = similarity[movie_index]
        movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
        
        recommended_movies = []
        recommended_movies_posters = []
        
        for i in movies_list:
            movie_id = movies.iloc[i[0]].movie_id
            recommended_movies.append(movies.iloc[i[0]].title)
            # Fetch poster from API
            recommended_movies_posters.append(fetch_poster(movie_id))
        
        return recommended_movies, recommended_movies_posters
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return [], []

# Load data
try:
    movies = pickle.load(open('movies.pkl', 'rb'))
    similarity = pickle.load(open('similarity.pkl', 'rb'))
    
    # Title
    st.title('🎬 MOVIE RECOMMENDATION SYSTEM')
    st.markdown("---")
    
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
    
    # Recommend button
    if st.button('Show Recommendations', type="primary"):
        with st.spinner('Finding similar movies...'):
            recommended_movie_names, recommended_movie_posters = recommend(selected_movie)
        
        if recommended_movie_names:
            st.success('Here are your recommendations!')
            st.markdown("---")
            
            # Display recommendations in columns
            cols = st.columns(5)
            for idx, col in enumerate(cols):
                with col:
                    try:
                        st.image(recommended_movie_posters[idx], use_column_width=True)
                    except:
                        st.image("https://via.placeholder.com/500x750/1e1e1e/ffffff?text=Image+Error", use_column_width=True)
                    st.markdown(f"**{recommended_movie_names[idx]}**")
        else:
            st.warning("No recommendations found. Please try another movie.")
    
    # Statistics in sidebar
    with st.sidebar:
        st.header("📊 Statistics")
        st.metric("Total Movies", len(movies))
        st.metric("Similarity Matrix Size", f"{similarity.shape[0]} x {similarity.shape[1]}")
        
        st.markdown("---")
        st.header("ℹ️ About")
        st.info("""
        This movie recommender system uses:
        - **Content-Based Filtering**
        - **Cosine Similarity**
        - **5000 Feature Vectors**
        
        It analyzes movie genres, keywords, cast, crew, and plot to find similar movies.
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>Built with ❤️ using Streamlit | Data from TMDB</p>
        </div>
        """,
        unsafe_allow_html=True
    )

except FileNotFoundError:
    st.error("⚠️ Model files not found!")
    st.info("Please run 'build_model.py' first to generate the required files.")
    st.code("python build_model.py", language="bash")
    
except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.info("Please make sure all required files are present and try again.")
