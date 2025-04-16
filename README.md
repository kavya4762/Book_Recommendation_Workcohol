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
**AI Engineering Interns at Workcohol, **
**Karunya Institute of Technology and Sciences**

This project was developed as part of our internship at Workcohol, showcasing our learning and application of a Book recommendation System suing Gen AI, also using techniques with hybrid model.

---

## 🛠️ Setup Instructions

### 1. Clone the repository

<pre>
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name </pre>

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

File Name	Description
app.py	Main Streamlit app with book recommendation + GenAI
Books.csv	Book metadata including title and author
Ratings.csv	Ratings data used to filter books
requirements.txt	Required Python dependencies
README.md	This file
hybrid_recommender.py	⚠️ Experimental hybrid recommender (not used in app)
