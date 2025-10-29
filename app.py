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

# Load environment variables
load_dotenv()

# Initialize Cohere API key
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# Set page configuration
st.set_page_config(
    page_title="📚 Document Q&A Bot",
    page_icon="🤖",
    layout="wide"
)

# Add custom CSS
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

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

# Title and description
st.title("📚 Document Q&A Chatbot")
st.markdown("""
    Upload a PDF document and ask questions about its content.
    The bot will use advanced AI to provide accurate answers based on the document.
""")

def process_pdf(uploaded_file):
    """Process the uploaded PDF file and create a vector store."""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        # Load and process the PDF
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        texts = text_splitter.split_documents(documents)

        # Create embeddings and vector store using a small Cohere wrapper
        # (avoid compatibility issues between langchain wrapper and cohere SDK)
        cohere_client = cohere.Client(COHERE_API_KEY)

        class CohereEmbeddingsWrapper:
            def __init__(self, client, model: str):
                self.client = client
                self.model = model
                # reduce batch size to avoid large HTTP payloads that can trigger 400s
                self.batch_size = 16
                self.max_text_chars = 4000

            def _to_text_list(self, items):
                """Normalize a list of Documents or strings into plain text strings."""
                out = []
                for item in items:
                    # LangChain Document objects have `page_content`
                    if hasattr(item, "page_content"):
                        txt = item.page_content
                        # truncate overly long texts to a safe size
                        if hasattr(self, "max_text_chars") and self.max_text_chars:
                            txt = txt[: self.max_text_chars]
                        out.append(txt)
                    # Some APIs might pass dict-like objects
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
                    # If server indicates input_type is required, try HTTP fallback including it
                    if "valid input_type" in msg or "input_type" in msg:
                        return self._embed_via_http(texts_batch, require_input_type=True)

                # Otherwise, try HTTP fallback without input_type first, then with it
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
                payload = {
                    "model": self.model,
                    # include 'texts' for newer API versions
                    "texts": texts_batch,
                }
                # Some embed models require the older 'input' field together with input_type.
                # When require_input_type is True, include both for maximum compatibility.
                if require_input_type:
                    payload["input_type"] = "text"
                    payload["input"] = texts_batch

                r = requests.post(url, headers=headers, json=payload, timeout=30)
                try:
                    r.raise_for_status()
                except requests.HTTPError as http_err:
                    # include response body for clearer debugging
                    body = r.text
                    raise RuntimeError(f"Cohere HTTP embed failed with status {r.status_code}: {body}") from http_err
                data = r.json()

                class SimpleResp:
                    def __init__(self, embeddings):
                        self.embeddings = embeddings

                # The HTTP response typically contains 'embeddings'
                if "embeddings" in data:
                    return SimpleResp(data["embeddings"])
                # Some responses might nest differently
                if "output" in data:
                    return SimpleResp(data["output"])
                # Fallback: try to extract any list-of-lists value
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
                    # resp should have attribute `embeddings`
                    if hasattr(resp, "embeddings"):
                        embeddings.extend(resp.embeddings)
                    elif hasattr(resp, "outputs"):
                        embeddings.extend(resp.outputs)
                    else:
                        # fallback: try to treat resp as an iterable of vectors
                        embeddings.extend(list(resp))
                return embeddings

            def embed_documents(self, documents):
                """Embed a list of Documents or strings (for documents)."""
                texts = self._to_text_list(documents)
                return self._batch_embed(texts)

            def embed_query(self, text):
                """Embed a single query string."""
                # Ensure query is a plain string
                q = text.page_content if hasattr(text, "page_content") else str(text)
                resp = self._embed_call([q])
                if hasattr(resp, "embeddings"):
                    return resp.embeddings[0]
                if hasattr(resp, "outputs"):
                    return resp.outputs[0]
                # fallback
                return list(resp)[0]

        embeddings = CohereEmbeddingsWrapper(cohere_client, "embed-english-v3.0")
        # Debug: show number of chunks and a sample to help diagnose embedding payload issues
        try:
            num_chunks = len(texts)
            sample_chunk = texts[0].page_content if hasattr(texts[0], "page_content") else str(texts[0])
            st.write(f"Debug: number of chunks={num_chunks}")
            st.write("Debug: sample chunk (first 300 chars):")
            st.code(sample_chunk[:300])

            # Also show the actual payload for the first embedding batch (after truncation)
            preview_texts = embeddings._to_text_list(texts[: embeddings.batch_size])
            st.write(f"Debug: first batch size={len(preview_texts)}; showing first 2 items and lengths:")
            for i, t in enumerate(preview_texts[:2]):
                st.write(f"  item {i} length={len(t)} chars")
                st.code(t[:300])
        except Exception:
            # if Streamlit not available or texts empty, skip
            pass

        vector_store = Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )

        # Clean up temporary file
        os.unlink(tmp_path)
        
        return vector_store
    
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None

# File upload section
uploaded_file = st.file_uploader("📎 Upload your PDF document", type=['pdf'])

if uploaded_file:
    with st.spinner("Processing document..."):
        st.session_state.vector_store = process_pdf(uploaded_file)
    if st.session_state.vector_store:
        st.success("✅ Document processed successfully! You can now ask questions about it.")

# Chat interface
if st.session_state.vector_store:
    # Initialize Cohere chat model and conversation chain
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

    # Display chat history
    for message in st.session_state.chat_history:
        role = "🤖 Assistant:" if message["role"] == "assistant" else "👤 You:"
        st.write(f"{role} {message['content']}")

    # Question input
    question = st.text_input("❓ Ask a question about your document:", placeholder="Type your question here...")
    
    if question:
        with st.spinner("Thinking..."):
            # Get the conversation history in the format expected by the chain
            chat_history = [(msg["content"], ans["content"]) 
                          for msg, ans in zip(
                              st.session_state.chat_history[::2], 
                              st.session_state.chat_history[1::2]
                          )] if st.session_state.chat_history else []
            
            # Get response from the chain
            response = qa_chain({
                "question": question,
                "chat_history": chat_history
            })
            
            # Update chat history
            st.session_state.chat_history.extend([
                {"role": "user", "content": question},
                {"role": "assistant", "content": response["answer"]}
            ])
            
            # Display the new response
            st.write("🤖 Assistant:", response["answer"])

else:
    st.info("👆 Please upload a PDF document to start asking questions.")
