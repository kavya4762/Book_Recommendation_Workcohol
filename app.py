
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai

# Load data
@st.cache_data
def load_data():
    books = pd.read_csv("Books.csv")
    ratings = pd.read_csv("Ratings.csv")
    
    # Optional: drop NA and prepare text field
    books['title'] = books['title'].fillna('')
    books['author'] = books['author'].fillna('')
    books['text'] = books['title'] + " " + books['author']
    
    ratings = ratings[ratings['rating'] > 5]  # Filter for positive ratings
    return books, ratings

books, ratings = load_data()

# Compute similarity matrix
@st.cache_resource
def compute_similarity():
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(books['text'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

cosine_sim = compute_similarity()

# Recommender function
def recommend_books(book_title, num=5):
    idx = books[books['title'].str.contains(book_title, case=False, na=False)].index
    if idx.empty:
        return pd.DataFrame()
    idx = idx[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num+1]
    book_indices = [i[0] for i in sim_scores]
    return books.iloc[book_indices][['title', 'author']]

# GenAI function
def generate_summary(recommended_books):
    titles = recommended_books['title'].tolist()
    prompt = (
        f"Based on the following book recommendations: {', '.join(titles)}, "
        "write a short and friendly personalized message suggesting them to a book lover."
    )
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

# Streamlit UI
st.title("📚 Book Recommender + GenAI")
book_input = st.text_input("Enter a book you like:")

openai_api_key = st.text_input("Enter your OpenAI API key:", type="password")
openai.api_key = openai_api_key

if st.button("Recommend"):
    if book_input and openai_api_key:
        recommended = recommend_books(book_input, num=5)
        if not recommended.empty:
            st.write("### 🔍 Recommendations:")
            st.table(recommended)

            st.write("### 🤖 Personalized Message:")
            st.write(generate_summary(recommended))
        else:
            st.error("No recommendations found. Try a different book title.")
    else:
        st.warning("Please enter both a book title and your OpenAI API key.")
