Here’s a **clean, professional, interview-ready `README.md`** tailored exactly to your project and repo name.
You can **copy-paste this directly** into GitHub.

---

````md
# 🧠 RAG-Based Psychology Mentor AI

A **Retrieval-Augmented Generation (RAG)** based psychology mentor AI that delivers **empathetic, evidence-grounded guidance** from curated psychology literature, featuring **relevance filtering, ethical guardrails, and crisis interception mechanisms**.

> ⚠️ **Disclaimer:** This project is for educational and research purposes only.  
> It is **not a therapist, psychologist, or medical system** and does **not provide diagnosis or treatment**.

---

## 📌 Overview

This project explores how Large Language Models (LLMs) can be safely applied in **psychology-adjacent domains** without crossing ethical or clinical boundaries.

Instead of acting as a therapist, the system functions as a **college-style mentor**:
- translating psychology concepts into **supportive, human-centered language**
- grounding responses in **retrieved academic sources**
- explicitly avoiding diagnosis, medical claims, or dependency

---

## ✨ Key Features

- **Retrieval-Augmented Generation (RAG)** using ChromaDB  
- **Empathy-aware prompt design** (mentor tone, not clinical)
- **Context relevance filtering** to reduce hallucinations
- **Crisis keyword interception** with safe redirection
- **Explicit ethical boundaries** (non-therapeutic, non-diagnostic)
- **Short-term conversational memory** to prevent over-dependence
- **Dual interfaces**
  - CLI chatbot
  - Streamlit web application
- **Offline local LLM support** via Ollama (Mistral)

---

## 🏗️ System Architecture

```text
Psychology PDFs / CSVs
        ↓
Text Chunking
        ↓
Embeddings (HuggingFace / Ollama)
        ↓
Chroma Vector Database
        ↓
Relevance Filtering
        ↓
RAG Prompt + Safety Guardrails
        ↓
LLM (Mistral via Ollama)
        ↓
User (CLI / Streamlit UI)
````

---

## 🛠️ Tech Stack

* **Language:** Python
* **LLM:** Mistral (via Ollama)
* **Frameworks:** LangChain, Streamlit
* **Vector Database:** ChromaDB
* **Embeddings:**

  * HuggingFace Sentence Transformers
  * Ollama embeddings (fallback)
* **OCR (optional):** Tesseract, pdf2image

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/RAG-Based-Psychology-Mentor-AI.git
cd RAG-Based-Psychology-Mentor-AI
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run Ollama (Required)

Ensure Ollama is running locally and Mistral is pulled:

```bash
ollama pull mistral
```

---

## 📚 Data Ingestion (RAG Setup)

Place your psychology PDFs / CSVs inside the `data/` directory, then run:

```bash
python populate_dataset.py --reset
```

This will:

* Load documents
* Chunk text
* Generate embeddings
* Store them in ChromaDB

---

## 💬 Running the Chatbot

### ▶️ CLI Mode

```bash
python query_data.py
```

### 🌐 Streamlit Web App

```bash
streamlit run streamlit_app.py
```

---

## 🧪 Testing & Evaluation

The project includes **behavioral tests** to evaluate:

* Empathy tone
* Boundary adherence
* Non-diagnostic responses

Run example test:

```bash
python test_empathy_tone.py
```

These tests focus on **response quality**, not exact string matching.

---

## 🛡️ Safety & Ethical Design

This system intentionally includes safeguards:

* ❌ No diagnosis or medical claims
* ❌ No therapy or treatment advice
* ❌ No emotional dependency encouragement
* ✅ Crisis keyword interception with human-resource redirection
* ✅ Context filtering to avoid irrelevant or harmful retrieval

See **`ETHICAL_CONSIDERATIONS.md`** for detailed discussion.

---

## 📂 Project Structure

```text
.
├── query_data.py              # CLI chatbot with RAG + safety guards
├── streamlit_app.py           # Web UI
├── populate_dataset.py        # Data ingestion & vector store setup
├── get_embedding_function.py  # Embedding provider abstraction
├── tests/
│   └── test_empathy_tone.py   # Empathy & boundary evaluation
├── data/                      # Input documents (not tracked)
├── README.md
└── requirements.txt
```

---

## Evaluation Results

The system was evaluated through scenario-based qualitative testing focused on empathy, boundary adherence, and safety behavior rather than exact response matching. Test prompts included common student concerns (e.g., loneliness, academic failure), psychology concept queries, diagnostic-seeking questions, and crisis-related statements. Across these scenarios, the chatbot consistently demonstrated empathetic, non-judgmental tone, accurately translated psychology concepts into accessible language, and refused diagnostic or medical requests while appropriately redirecting users to professional resources. In crisis-oriented prompts, the system reliably interrupted normal conversation flow and escalated responses toward human support mechanisms. These results indicate that the system effectively balances emotional support with ethical constraints, supporting its intended role as a non-clinical psychology mentor rather than a therapeutic agent.


---

## 🔮 Future Improvements

* Semantic (ML-based) crisis detection
* Emotion intensity scaling
* Long-term memory summarization
* Multilingual support
* Human-in-the-loop review mode

---

## 📜 License

This project is licensed under the **MIT License**.
You are free to use, modify, and distribute it with attribution.

---

## 👤 Author

Developed by **Kritansh Uppal**
For educational, research, and portfolio purposes.

---

⭐ If you find this project useful or interesting, consider starring the repository.

```


