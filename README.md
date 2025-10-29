📘 Document Q&A Chatbot (RAG System) 🤖

Status: 🧩 Under Development
This project is currently being built — new features, bug fixes, and improvements are being added regularly.

🧠 Overview

This project is a Retrieval-Augmented Generation (RAG) based chatbot that lets you upload PDFs and ask questions about them.
The system uses:

Cohere API for text embeddings & responses

Chroma VectorDB for semantic search

LangChain for orchestration

Streamlit for the user interface

Once completed, it will allow users to:

Upload PDF documents 📄

Ask natural language questions 💬

Get AI-generated answers from their documents ⚡

View chat history 🕒

🚧 Development Progress
Feature	Status	Description
PDF Upload	✅ Done	Supports uploading multiple PDFs
Text Chunking	✅ Done	Splits large documents into smaller parts
Embedding (Cohere)	⚙️ Testing	Embeds document chunks into vector space
VectorDB (Chroma)	⚙️ Testing	Stores and retrieves relevant chunks
Chat Interface (Streamlit)	🛠️ In progress	Clean UI with chat history
Context-based Responses	🧠 Planned	Answers generated based on retrieved data
Persistent Memory	🔜 Planned	Save chat history and past sessions
🧩 Tech Stack
Component	Technology Used
Frontend	Streamlit
Backend	Python (LangChain, Chroma)
Embeddings	Cohere
Environment	Virtualenv
Version Control	Git + GitHub
🧱 How to Set Up Locally

You can still set up and test the early version 👇

1️⃣ Clone the Repository
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

2️⃣ Create a Virtual Environment
🪟 Windows:
python -m venv venv
venv\Scripts\activate

🍎 Mac/Linux:
python3 -m venv venv
source venv/bin/activate

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Add Your Cohere API Key

Create a .env file in the root directory and add:

COHERE_API_KEY=your_api_key_here

5️⃣ Run the App

Once ready, launch with:

streamlit run app.py

⚙️ Requirements (as of now)
langchain
cohere
chromadb
pypdf
python-dotenv
streamlit


You can install everything later with:

pip install -r requirements.txt

🧩 Folder Structure (Planned)
Document-QA-RAG/
│
├── app.py               # Streamlit interface
├── ingest.py            # PDF processing & vector creation
├── requirements.txt     # Project dependencies
├── .env                 # Cohere API Key
├── data/                # Folder for uploaded PDFs
├── vectorstore/         # Stored embeddings (Chroma DB)
└── README.md            # Project documentation

💡 Future Goals :

🔍 Improve accuracy using hybrid search (semantic + keyword)

🧠 Add multi-PDF context support

🕒 Save and reload chat sessions

🌐 Deploy online (Streamlit Cloud / Render / HuggingFace Spaces)

🤝 Contributing

🧠 Reinforcement Learning using feedback

Contributions are welcome!
If you find bugs, have suggestions, or want to add features — open an issue or submit a pull request.

🧑‍💻 Developer Note

This is a work in progress project, built for learning and exploring Generative AI and RAG systems.
Once completed, it will serve as a full end-to-end Document Q&A chatbot example using Cohere and Chroma.




👤 Developer: [Your Name]
📧 Email: your.email@example.com
