# Advanced RAG Chatbot

This project implements a Retrieval-Augmented Generation (RAG) Chatbot that allows users to upload a PDF document or enter a website URL, and ask questions based on the provided content. 

---

## 🚀 Features
- Easy setup and integration with Ollama for model management.
- Run the LLM models completely locally to ensure information security.
- Applied Multi-Query Retriever and ParentDocumentRetriever to improve query response accuracy.
- Interactive user interface using **Streamlit**.

---

## 🖖 Usage Guide

### 1. Clone This Project
First, clone the repository:

```bash
# Clone the repository
git clone https://github.com/HwiTran/Advanced-RAG-Chatbot.git

# Navigate into the project directory
cd Advanced-RAG-Chatbot
```

### 2. Set Up the Environment
Follow these steps to create and activate the Python virtual environment:

```bash
# Create a virtual environment
python -m venv chatbot_env

# Activate the virtual environment
# On Windows
chatbot_env\Scripts\activate
# On macOS/Linux
source chatbot_env/bin/activate
```

### 3. Install Dependencies
Install the required Python packages using:

```bash
pip install -r requirements.txt
```

### 4. Pull Models from Ollama
Retrieve the necessary LLM and embedding models using Ollama. Run these commands:

```bash
# Pull LLM model
ollama pull gemma2

# Pull embedding model
ollama pull bge-m3
```

- To verify the downloaded models, run:

```bash
ollama list
```

### 5. Run the Chatbot
Launch the chatbot with its **Streamlit** user interface:

```bash
streamlit run app.py
```

---

## 🎥 Demo
Here is a preview of the application in action:  


https://github.com/user-attachments/assets/7d0e88a7-235b-47f7-a44b-38dd48ead42e



---


## 📧 Contact
For questions or feedback, please contact: **trandanghuy13456@gmail.com**

-

