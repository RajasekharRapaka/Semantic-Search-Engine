import streamlit as st
import re
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load the pre-trained SentenceTransformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Function to tokenize and clean user query
def clean_data(data):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(data)
    # Remove punctuation, stopwords, and convert to lowercase
    cleaned_tokens = [word.lower() for word in tokens if word.isalnum() and word.lower() not in stop_words]
    return " ".join(cleaned_tokens)

# Function to format movie/series result for display
def format_movie_series(movie_series, id):
    return f"[{movie_series}](https://www.opensubtitles.org/en/subtitles/{id})"

# Function to calculate similarity and retrieve sorted indices
def get_sorted_indices(query, data):
    query_embed = model.encode(query).reshape(1, -1)
    similarities = cosine_similarity(query_embed, model.encode(data))
    return similarities.argsort(axis=1)[:, ::-1].flatten()

# Load the DataFrame outside of the search button block
df_30_percent_data = pd.read_csv("C:/Users/rjsek/Downloads/final_df.csv")  # Replace "your_data.csv" with the actual file path

# Streamlit UI
st.set_page_config(layout="wide")  # Wide layout for better display
st.title("🎥 Movies/Series Subtitle Search Engine 🔎")
search_query = st.text_input("Search here 🔬", placeholder="Enter a name of the movie or a series to search")

if "results" not in st.session_state:
    st.session_state.results = None

if st.button("Search"):
    search_query_cleaned = clean_data(search_query)
    sorted_indices = get_sorted_indices(search_query_cleaned, df_30_percent_data['Movies/Series'])
    st.session_state.results = sorted_indices

if st.session_state.results is not None:
    sorted_indices = st.session_state.results

    # Pagination
    page_size = 10
    total_pages = len(sorted_indices) // page_size + 1
    page_number = st.number_input("Page Number", min_value=1, max_value=total_pages, value=1)
    start_idx = (page_number - 1) * page_size
    end_idx = min(start_idx + page_size, len(sorted_indices))
    
    # Display results for the selected page
    for idx in sorted_indices[start_idx:end_idx]:
        movie_series = df_30_percent_data.iloc[idx]['Movies/Series']
        id = df_30_percent_data.iloc[idx]['id']
        st.markdown(format_movie_series(movie_series, id), unsafe_allow_html=True)

    # Display page information at the bottom
    st.markdown(f"Page {page_number} of {total_pages}")
