# ⚡ Turbo Conversational RAG with PDFs

A blazing-fast Streamlit app that lets you upload PDF documents and chat with their content using **Groq's ultra-fast LLMs**, **LangChain**, and **FAISS** for retrieval-augmented generation (RAG).


## 🚀 Features

- 🔥 Powered by [Groq](https://groq.com)'s blazing fast LLMs (`Gemma2-9b-it`)
- 🧠 Uses RAG (Retrieval Augmented Generation) for better contextual answers
- 📄 Upload multiple PDF documents
- 🔍 Custom text splitting and FAISS-based semantic search
- 🦮 HuggingFace Embeddings (`all-MiniLM-L6-v2`)
- 💬 Session-aware conversational interface
- 🐍 100% Python, built with [Streamlit](https://streamlit.io/)

---

## 📦 Installation

```bash
git clone https://github.com/UroojAshfa/turbo-pdf-chat.git
cd turbo-pdf-chat
pip install -r requirements.txt
```

---

## 🔐 Environment Variables

Create a `.env` file in the project root with the following keys:

```env
GROQ_API_KEY=your_groq_api_key
HF_TOKEN=your_huggingface_token
LANGCHAIN_PROJECT=RAG
LANGCHAIN_TRACING_V2=true
```

Alternatively, you can set these in Streamlit Cloud's **secrets**:

```toml
# .streamlit/secrets.toml
GROQ_API_KEY = "your_groq_api_key"
HF_TOKEN = "your_huggingface_token"
```

---

## 💻 Running the App

```bash
streamlit run app.py
```

You can now access the app in your browser at `http://localhost:8501`.

---

## 🧠 How It Works

1. **PDF Upload**: Users upload one or more PDF documents.
2. **Text Extraction**: PDFs are parsed and split into semantic chunks.
3. **Vector Store**: Chunks are embedded and stored in a FAISS vector store.
4. **Querying**: User queries are rephrased with chat history and passed to Groq's LLM along with retrieved chunks.
5. **Streaming Responses**: Answers are streamed back to the UI.

---

## 📂 Project Structure

```bash
🔹 app.py                  # Main Streamlit app
🔹 .env                   # Environment variables (not committed)
🔹 requirements.txt       # Python dependencies
🔹 .streamlit/
    └── secrets.toml       # Optional secrets for Streamlit Cloud
```

---

## 🛠️ Dependencies

- `streamlit`
- `langchain`
- `langchain-groq`
- `langchain-community`
- `langchain-huggingface`
- `sentence-transformers`
- `faiss-cpu`
- `python-dotenv`

Install all dependencies with:

```bash
pip install -r requirements.txt
```

---

## ☁️ Deploy to Streamlit Cloud

1. Push your repo to GitHub.
2. Go to [Streamlit Cloud](https://streamlit.io/cloud) and connect your repo.
3. Add your environment variables via `Secrets` tab.
4. Click **Deploy**. Done!

---

## 📌 Tips

- Make sure your `.env` file is **not committed** to version control.
- Use **`st.secrets`** in deployment environments like Streamlit Cloud.
- If embedding initialization fails, check that HuggingFace embeddings are accessible with your token.

---



## 📜 License

MIT License. See `LICENSE` file for details.

