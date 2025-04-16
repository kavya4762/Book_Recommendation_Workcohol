import numpy as np
import pandas as pd
import tensorflow as tf
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load models & data
ncf_model = tf.keras.models.load_model("ncf_model.h5")
df = pd.read_pickle("book_embeddings.pkl")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def recommend_books(user_id, book_title):
    # NCF Predictions
    user_idx = df[df["user_id"] == user_id].index[0]
    book_ids = df["book_id"].values
    user_ids = np.full(len(book_ids), user_idx)

    predictions = ncf_model.predict([user_ids, book_ids])
    top_books = df.iloc[np.argsort(predictions.flatten())[::-1][:5]]["name"].values

    # Content-Based Recommendations
    book_embedding = embedding_model.encode(df[df["name"] == book_title]["description"].values[0])
    similarities = cosine_similarity([book_embedding], df["embeddings"].tolist())[0]
    top_similar_books = df.iloc[np.argsort(similarities)[::-1][:5]]["name"].values

    # Hybrid Recommendation
    hybrid_recommendations = list(set(top_books) & set(top_similar_books))

    return hybrid_recommendations if hybrid_recommendations else list(top_books)[:5]
