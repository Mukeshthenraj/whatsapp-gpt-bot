
# 📲 WhatsApp GPT Bot -- RAG Product Search + AI Assistant 🤖

A **WhatsApp bot** that combines:\
- ✅ **Baileys** for WhatsApp connection\
- ✅ **Python RAG (Retrieval-Augmented Generation)** for **catalog
product search**\
- ✅ **Ollama + Mistral LLM** for chit-chat & fallback AI responses

This project is my **milestone work**: it connects real-time messaging,
catalog intelligence, and AI responses into a single solution. 🚀

![Node](https://img.shields.io/badge/Node.js-18+-green)\
![Python](https://img.shields.io/badge/Python-3.10+-blue)\
![WhatsApp](https://img.shields.io/badge/WhatsApp-Bot-brightgreen)\
![License](https://img.shields.io/badge/License-MIT-lightgrey)

------------------------------------------------------------------------

## ✨ Features

-   🔍 **Catalog-first answers** → Product queries (by name or
    Bestell-Nr.) answered from structured data (`rag-pdf/`)\
-   🛠️ **German + English support** for catalog queries\
-   🤖 **LLM fallback (Mistral via Ollama)** for chit-chat, jokes, or
    Q&A\
-   📐 **Math guard** → safely computes expressions like `2*(128+64)`\
-   🌦️ **Weather API support** (optional, via OpenWeather)\
-   📰 **News API support** (optional, via NewsAPI)\
-   🌍 **Google search & Wikipedia** integration\
-   🔑 **Secure design** → `.env` file & WhatsApp login info are **never
    committed**

------------------------------------------------------------------------

## 📂 Folder Structure

    whatsapp-gpt-bot/
    ├─ README.md                # This file
    ├─ package.json             # Node dependencies
    ├─ package-lock.json
    ├─ ollama_mistral.js        # Main WhatsApp bot logic
    ├─ start_whatsapp_bot.txt   # Optional run script
    ├─ .env.example             # Safe template for secrets
    ├─ .gitignore               # Keeps secrets & junk out of repo
    │
    ├─ rag-pdf/                 # Catalog + Python RAG search
    │  ├─ search_catalog.py     # Search logic
    │  ├─ prepare_index.py      # Prepares vector index
    │  ├─ kellen_produkte.json  # Extra product list
    │  ├─ catalog_docs.json     # Parsed catalog data
    │  ├─ catalog_vectors.npy   # Embeddings for fast search
    │  └─ requirements.txt      # Python dependencies
    │
    └─ auth_info_baileys/       # WhatsApp session (ignored by Git)

------------------------------------------------------------------------

## ⚡ Quick Start

### 1️⃣ Install dependencies

``` bash
# Node setup
npm install

# Python setup (inside rag-pdf/)
cd rag-pdf
python -m venv .venv
.venv/Scripts/activate    # (Windows)
pip install -r requirements.txt
```

### 2️⃣ Configure environment

Create a `.env` file (not committed) based on `.env.example`:

    PYTHON_BIN=python
    OLLAMA_URL=http://localhost:11434
    OLLAMA_MODEL=mistral
    OLLAMA_TIMEOUT_MS=60000
    DISABLE_OLLAMA=false

    OPENWEATHER_KEY=your_key
    NEWSAPI_KEY=your_key
    GOOGLE_API_KEY=your_key
    GOOGLE_CSE_ID=your_id

### 3️⃣ Pull a model (once)

``` bash
ollama pull mistral
```

### 4️⃣ Run the bot

``` bash
node ollama_mistral.js
```

Scan the QR code in WhatsApp → you're connected! ✅

------------------------------------------------------------------------

## 💡 Usage Examples

-   **Product queries**
    -   `Haben Sie Maurerkelle HaWe?`\
    -   `Preis von Flächenspachtel HaWe`\
    -   `Bestell-Nr. 505.02`
-   **Chit-chat**
    -   `hi` → 🤖 friendly response\
    -   `tell me a joke` → 🤖 AI joke\
    -   `2*(128+64)` → 🧮 384
-   **News / Weather / Search**
    -   `news about AI in Germany`\
    -   `weather in Berlin tomorrow`\
    -   `search WhatsApp bot GitHub`\
    -   `wiki Kemmler`

------------------------------------------------------------------------

## 📸 Screenshots (Demo)

### 🛒 Product Search
![Product Search](./docs/product_search.png)

### 💬 Chat & Chit-Chat
![Chat Demo](./docs/chitchat.png)

------------------------------------------------------------------------

## 🔒 Security

-   `.env` → Not committed (contains secrets)\
-   `auth_info_baileys/` → Not committed (contains WhatsApp login
    session)\
-   `.gitignore` ensures sensitive data never leaks

------------------------------------------------------------------------

## 🚀 Future Improvements

-   Add **Dockerfile** for easy deployment\
-   Expand **catalog sources**\
-   Add **multi-language support** beyond German/English

------------------------------------------------------------------------

## 📝 License

This project is licensed under the **MIT License**.

------------------------------------------------------------------------

## 🙋 Author

👤 **Mukesh Thenraj**\
- 📧 <mukeshthenraj@gmail.com>\
- 🔗 [LinkedIn](https://www.linkedin.com/in/mukeshthenraj)\
- 💻 [GitHub](https://github.com/Mukeshthenraj)

------------------------------------------------------------------------
