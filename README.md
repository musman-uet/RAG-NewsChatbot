# RAG-NewsChatbot
An intelligent chatbot that answers real-time queries using Pakistani news articles, powered by Retrieval-Augmented Generation (RAG), LLaMA 3.2B, ChromaDB, and Streamlit.

# 📰 Automated News Chatbot Using RAG-Based LLM

This project is a smart news chatbot that answers your questions using **real-time Pakistani news articles**. It uses **Retrieval-Augmented Generation (RAG)** and the **LLaMA 3.2B Instruct** model to generate accurate, grounded answers.

---

## 📦 Project Structure

```
RAG_NewsChatbot_Project/
├── News_Scrapper/           # Scrapes news articles from URLs using newspaper3k
├── chromadb2/               # Sets up Chroma vector DB for embeddings
├── RAG_with_steam/          # Full RAG pipeline: query handling + chatbot
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 How It Works

1. **Scrape News**  
   Use `News_Scrapper/projectphase1_streamlitapp2.py` to scrape URLs and save article content.

2. **Chunk + Embed**  
   Use `RAG_with_steam/projectphase1_streamlitapp2.py` to:
   - Split news into chunks
   - Embed using `mxbai-embed-large-v1`
   - Store in ChromaDB

3. **Ask Questions**  
   Enter a query using Streamlit interface and get AI-generated answers based on actual news.

---

## 🧠 Tech Stack

- 🗞️ `newspaper3k` for scraping
- 🧩 `ChromaDB` for vector search
- 🧠 `LLaMA 3.2B` via Hugging Face Transformers
- 💬 `Streamlit` for chatbot interface
- 🔎 `LangChain` for chunking and embedding

---

## 🔧 Setup Instructions

```bash
git clone https://github.com/yourusername/RAG-NewsChatbot.git
cd RAG-NewsChatbot
pip install -r requirements.txt
streamlit run RAG_with_steam/projectphase1_streamlitapp2.py
```

✅ Make sure to set your HuggingFace API key:
```bash
export HUGGINGFACEHUB_API_TOKEN=your_token
```

---

## 📌 Notes

- Designed for Google Colab or a system with 16GB RAM + GPU
- Uses `mxbai-embed` for embeddings and `LLaMA` for generation
- You can switch to GPT-3.5 or Mistral for lighter inference

---

## 📄 License

MIT License
