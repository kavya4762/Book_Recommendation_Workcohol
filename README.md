# 📚 Book Recommender + GenAI

This is a book recommendation app built using **Streamlit**, **scikit-learn**, and **OpenAI GPT-3.5-turbo**. It suggests books based on your input and generates a personalized message using Generative AI.

---

## 💡 Features

- 🔍 Recommends similar books using TF-IDF and cosine similarity
- 🤖 Uses **OpenAI GPT-3.5-turbo** to generate a friendly, personalized message for book lovers
- 🧠 Combines book titles and authors to enhance recommendation quality
- ⚡ Clean, interactive web interface with Streamlit

---

## 🧑‍💻 Built By

**Satish** and **Kavya G**  
**AI Engineering Interns at Workcohol**, 
**Karunya Institute of Technology and Sciences**

This project was developed as part of our internship at Workcohol, showcasing our learning and application of a Book recommendation System suing Gen AI, also using techniques with hybrid model.

---

## 🛠️ Setup Instructions

### 1. Clone the repository

<pre>
git clone " "
cd "repo-name" </pre>

### 2. Install the required libraries
<pre>
pip install -r requirements.txt </pre>

### 3. Run the app
<pre>
streamlit run app.py </pre>

🔑 OpenAI API Key
To generate personalized messages, this app uses OpenAI GPT-3.5-turbo.

📌 Get your API key from [OpenAI API Dashboard](https://platform.openai.com/account/api-keys) and paste it into the app when prompted. 

📂 Project Files

| File Name             | Description                                      |
|-----------------------|--------------------------------------------------|
| `app.py`              | Main app with book recommender + GenAI           |
| `Books.csv`           | Book metadata (title + author)                  |
| `Ratings.csv`         | Ratings used for filtering recommendations       |
| `requirements.txt`    | Python dependencies                             |
| `hybrid_recommender.py` | Experimental hybrid model (not used in app)    |

🧪 Note: The hybrid_recommender.py script is included as a prototype experiment for a hybrid recommendation system. It is not integrated into the current Streamlit app but can be explored for future development.

🧪 How It Works
1. Enter a book title you like.
2. The app finds similar books using TF-IDF and cosine similarity.
3. GPT-3.5-turbo generates a short and friendly recommendation message.
4. Everything is displayed on a clean Streamlit interface.

🚀 Example
If you type:

The Alchemist

You might see:
 
 Recommended Books
   
   Brida by Paulo Coelho
   
   Siddhartha by Hermann Hesse
   
   Jonathan Livingston Seagull by Richard Bach
 
 Message:
 
 “If you loved The Alchemist, you're going to be inspired by these timeless gems that speak to the soul…”
