import streamlit as st
from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import cohere
import requests
from langchain_community.chat_models import ChatCohere
from langchain.chains import ConversationalRetrievalChain
import tempfile

load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")

st.set_page_config(
    page_title="📚 Document Q&A Bot",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
    <style>
        .stAlert {
            background-color: #f0f2f6;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 10px;
        }
        .css-1v0mbdj.ebxwdo61 {
            width: 100%;
            max-width: 800px;
            margin: auto;
        }
    </style>
""", unsafe_allow_html=True)

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

st.title("📚 Document Q&A Chatbot")
st.markdown("""
    Upload a PDF document and ask questions about its content.
    The bot will use advanced AI to provide accurate answers based on the document.
""")

def process_pdf(uploaded_file):
    """Process the uploaded PDF file and create a vector store."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        texts = text_splitter.split_documents(documents)

        cohere_client = cohere.Client(COHERE_API_KEY)

        class CohereEmbeddingsWrapper:
            def __init__(self, client, model: str):
                self.client = client
                self.model = model
                self.batch_size = 16
                self.max_text_chars = 4000

            def _to_text_list(self, items):
                """Normalize a list of Documents or strings into plain text strings."""
                out = []
                for item in items:
                    if hasattr(item, "page_content"):
                        txt = item.page_content
                        if hasattr(self, "max_text_chars") and self.max_text_chars:
                            txt = txt[: self.max_text_chars]
                        out.append(txt)
                    elif isinstance(item, dict) and "page_content" in item:
                        txt = item["page_content"]
                        if hasattr(self, "max_text_chars") and self.max_text_chars:
                            txt = txt[: self.max_text_chars]
                        out.append(txt)
                    else:
                        txt = str(item)
                        if hasattr(self, "max_text_chars") and self.max_text_chars:
                            txt = txt[: self.max_text_chars]
                        out.append(txt)
                return out

            def _embed_call(self, texts_batch):
                """Call the Cohere SDK embed safely using the 'texts' kwarg.

                If the SDK raises an error indicating the server requires an 'input_type',
                fall back to an HTTP call that includes input_type. We deliberately avoid
                calling the SDK with 'input_type' to prevent TypeError for unsupported kwargs.
                """
                last_exc = None
                try:
                    resp = self.client.embed(model=self.model, texts=texts_batch)
                    return resp
                except Exception as e:
                    last_exc = e
                    msg = str(e).lower()
                    if "valid input_type" in msg or "input_type" in msg:
                        return self._embed_via_http(texts_batch, require_input_type=True)

                try:
                    return self._embed_via_http(texts_batch, require_input_type=False)
                except Exception as http_e:
                    try:
                        return self._embed_via_http(texts_batch, require_input_type=True)
                    except Exception as http_e2:
                        raise RuntimeError(f"Cohere embed failed. SDK error: {last_exc}; HTTP errors: {http_e}, {http_e2}")

            def _embed_via_http(self, texts_batch, require_input_type: bool = False):
                """Fallback to direct HTTP request to Cohere embed endpoint.

                If require_input_type is True, include the 'input_type' field.
                Returns a simple object with an `embeddings` attribute.
                """
                url = "https://api.cohere.ai/v1/embed"
                headers = {
                    "Authorization": f"Bearer {COHERE_API_KEY}",
                    "Content-Type": "application/json",
                }
                if require_input_type:
                    payload = {
                        "model": self.model,
                        "inputs": texts_batch,
                        "input_type": "text",
                    }
                else:
                    payload = {
                        "model": self.model,
                        "texts": texts_batch,
                    }

                r = requests.post(url, headers=headers, json=payload, timeout=30)
                try:
                    r.raise_for_status()
                except requests.HTTPError as http_err:
                    body = r.text
                    raise RuntimeError(f"Cohere HTTP embed failed with status {r.status_code}: {body}") from http_err
                data = r.json()

                class SimpleResp:
                    def __init__(self, embeddings):
                        self.embeddings = embeddings

                if "embeddings" in data:
                    return SimpleResp(data["embeddings"])
                if "output" in data:
                    return SimpleResp(data["output"])
                for v in data.values():
                    if isinstance(v, list) and len(v) and isinstance(v[0], list):
                        return SimpleResp(v)
                raise RuntimeError(f"Unexpected response shape from Cohere HTTP embed: {data}")

            def _batch_embed(self, texts):
                """Embed texts in batches to avoid very large requests and accumulate embeddings."""
                embeddings = []
                for i in range(0, len(texts), self.batch_size):
                    batch = texts[i : i + self.batch_size]
                    resp = self._embed_call(batch)
                    if hasattr(resp, "embeddings"):
                        embeddings.extend(resp.embeddings)
                    elif hasattr(resp, "outputs"):
                        embeddings.extend(resp.outputs)
                    else:
                        embeddings.extend(list(resp))
                return embeddings

            def embed_documents(self, documents):
                """Embed a list of Documents or strings (for documents)."""
                texts = self._to_text_list(documents)
                return self._batch_embed(texts)

            def embed_query(self, text):
                """Embed a single query string."""
                q = text.page_content if hasattr(text, "page_content") else str(text)
                resp = self._embed_call([q])
                if hasattr(resp, "embeddings"):
                    return resp.embeddings[0]
                if hasattr(resp, "outputs"):
                    return resp.outputs[0]
                return list(resp)[0]

        embeddings = CohereEmbeddingsWrapper(cohere_client, "embed-english-v3.0")
        try:
            num_chunks = len(texts)
            sample_chunk = texts[0].page_content if hasattr(texts[0], "page_content") else str(texts[0])
            st.write(f"Debug: number of chunks={num_chunks}")
            st.write("Debug: sample chunk (first 300 chars):")
            st.code(sample_chunk[:300])

            preview_texts = embeddings._to_text_list(texts[: embeddings.batch_size])
            st.write(f"Debug: first batch size={len(preview_texts)}; showing first 2 items and lengths:")
            for i, t in enumerate(preview_texts[:2]):
                st.write(f"  item {i} length={len(t)} chars")
                st.code(t[:300])
        except Exception:
            pass

        vector_store = Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )

        os.unlink(tmp_path)
        
        return vector_store
    
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None

uploaded_file = st.file_uploader("📎 Upload your PDF document", type=['pdf'])

if uploaded_file:
    with st.spinner("Processing document..."):
        st.session_state.vector_store = process_pdf(uploaded_file)
    if st.session_state.vector_store:
        st.success("✅ Document processed successfully! You can now ask questions about it.")

if st.session_state.vector_store:
    llm = ChatCohere(
        model="command-r",
        temperature=0.7,
        cohere_api_key=COHERE_API_KEY
    )
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=st.session_state.vector_store.as_retriever(
            search_kwargs={"k": 3}
        ),
        return_source_documents=True,
        verbose=True
    )

    for message in st.session_state.chat_history:
        role = "🤖 Assistant:" if message["role"] == "assistant" else "👤 You:"
        st.write(f"{role} {message['content']}")

    question = st.text_input("❓ Ask a question about your document:", placeholder="Type your question here...")
    
    if question:
        with st.spinner("Thinking..."):
            chat_history = [(msg["content"], ans["content"]) 
                          for msg, ans in zip(
                              st.session_state.chat_history[::2], 
                              st.session_state.chat_history[1::2]
                          )] if st.session_state.chat_history else []
            
            response = qa_chain({
                "question": question,
                "chat_history": chat_history
            })
            
            st.session_state.chat_history.extend([
                {"role": "user", "content": question},
                {"role": "assistant", "content": response["answer"]}
            ])
            
            st.write("🤖 Assistant:", response["answer"])

else:
    st.info("👆 Please upload a PDF document to start asking questions.")
