import streamlit as st
import re
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import time

# Load the pre-trained SentenceTransformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Function to clean user query
def clean_data(data):
    # Remove punctuation and convert to lowercase
    data = re.sub(r'[^\w\s]', '', data)
    data = data.lower()
    return data

# Function to perform search
def perform_search(query, database):
    # Calculate similarity scores
    scores = []
    for item in database:
        item_embedding = model.encode(item).reshape(1, -1)
        score = cosine_similarity(query, item_embedding)[0][0]
        scores.append(score)
    
    # Sort by similarity scores and return top results
    results = [(score, item) for score, item in zip(scores, database)]
    results.sort(reverse=True)
    return results

# Streamlit UI
st.title("🎥 Movies/Series Subtitle Search Engine 🔎")
search_query = st.text_input("Search here 🔬",placeholder="Enter a name of the movie or a series to search")

if st.button("Search"):
    st.subheader("Subtitle Files ⬇️")
    search_query_cleaned = clean_data(search_query)
    query_embed = model.encode(search_query_cleaned).reshape(1, -1)

    # Simulated database
    df_10_percent_data = pd.read_csv("C:/Users/rjsek/Downloads/df_10_percent_data.csv")  # Replace "your_data.csv" with the actual file path
    database = df_10_percent_data['Movies/Series'].tolist()

    # Perform search
    search_results = perform_search(query_embed, database)

    # Paginate the results
    num_results = len(search_results)
    num_pages = (num_results - 1) // 10 + 1

    page_num = st.number_input("Enter page number:", min_value=1, max_value=num_pages, value=1)

    start_idx = (page_num - 1) * 10
    end_idx = min(page_num * 10, num_results)

    paginated_results = search_results[start_idx:end_idx]

    # Display the results for the selected page
    for _, item in paginated_results:
        st.markdown(f"[{item}](https://www.opensubtitles.org/en/subtitles/{item})")


        # Add a short delay between requests
        time.sleep(0.1)  # Adjust the delay as needed to avoid triggering security measures

    # Display pagination at the bottom of the page
    st.write(f"Page {page_num} of {num_pages}")
