import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq


# --------------------------------------------------
# BASIC SETUP
# --------------------------------------------------

load_dotenv()

st.set_page_config(
    page_title="RAG Application",
    layout="wide"
)

st.title("📚 RAG Documents App")

st.write(
    """
    Upload PDF documents, build a knowledge base,
    ask questions, and inspect the exact evidence
    retrieved before the AI answers.
    """
)

# --------------------------------------------------
# API KEY
# --------------------------------------------------

with st.sidebar:
    st.header("⚙️ Settings")
    api_key_input = st.text_input(
        "Groq API Key",
        type="password"
    )

api_key = api_key_input or os.getenv("GROQ_API_KEY")

if not api_key:
    st.warning(
        "Enter the Groq API Key in the sidebar "
        "or add it to your .env file."
    )
    st.stop()

os.environ["GROQ_API_KEY"] = api_key

# --------------------------------------------------
# EMBEDDING MODEL
# --------------------------------------------------

if "embeddings" not in st.session_state:
    st.session_state.embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={
            "normalize_embeddings": True
        }
    )

embeddings = st.session_state.embeddings

# --------------------------------------------------
# LLM
# --------------------------------------------------

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

# --------------------------------------------------
# DOCUMENT UPLOAD
# --------------------------------------------------

uploaded_files = st.file_uploader(
    "Upload one or more PDF files",
    type="pdf",
    accept_multiple_files=True
)

col1, col2 = st.columns(2)

with col1:
    chunk_size = st.slider(
        "Chunk Size",
        min_value=400,
        max_value=1400,
        value=900,
        step=100
    )

with col2:
    top_k = st.slider(
        "How Many Chunks Should RAG Retrieve?",
        min_value=2,
        max_value=6,
        value=4,
        step=1
    )

# --------------------------------------------------
# BUILD KNOWLEDGE BASE
# --------------------------------------------------

if st.button("Build Knowledge Base", type="primary"):

    if not uploaded_files:
        st.error("Please upload at least one PDF first.")
        st.stop()

    all_pages = []

    with st.spinner("Reading PDFs and building embeddings..."):

        for uploaded_pdf in uploaded_files:

            temp_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            )

            temp_file.write(uploaded_pdf.getvalue())
            temp_file.close()

            loader = PyPDFLoader(temp_file.name)

            pages = loader.load()

            for page in pages:
                page.metadata["source_file"] = uploaded_pdf.name

            all_pages.extend(pages)

            os.unlink(temp_file.name)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=int(chunk_size * 0.15)
        )

        chunks = text_splitter.split_documents(all_pages)

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )

        st.session_state.vectorstore = vectorstore
        st.session_state.file_count = len(uploaded_files)
        st.session_state.page_count = len(all_pages)
        st.session_state.chunk_count = len(chunks)
        st.session_state.chunk_size_used = chunk_size

    st.success(
        "✅ Knowledge Base Ready! Your PDFs can now be searched."
    )

# --------------------------------------------------
# KNOWLEDGE BASE STATUS
# --------------------------------------------------

if "vectorstore" in st.session_state:

    m1, m2, m3 = st.columns(3)

    m1.metric(
        "Documents",
        st.session_state.file_count
    )

    m2.metric(
        "Pages",
        st.session_state.page_count
    )

    m3.metric(
        "Chunks",
        st.session_state.chunk_count
    )

    st.divider()

    ask_tab, search_tab = st.tabs(
        [
            "Ask with RAG",
            "Semantic Search Only"
        ]
    )

    # ==================================================
    # TAB 1 : ASK WITH RAG
    # ==================================================

    with ask_tab:

        st.subheader("Ask a Question")

        question = st.text_input(
            "Your Question",
            placeholder="Example: What is the project completion date?",
            key="rag_question"
        )

        if st.button(
            "Find Evidence and Answer",
            key="answer_button"
        ):

            if not question.strip():

                st.warning(
                    "Please type a question first."
                )

            else:

                retriever = (
                    st.session_state.vectorstore
                    .as_retriever(
                        search_type="mmr",
                        search_kwargs={
                            "k": top_k,
                            "fetch_k": 20,
                            "lambda_mult": 0.7
                        }
                    )
                )

                retrieved_docs = retriever.invoke(
                    question
                )

                context_parts = []

                for number, doc in enumerate(
                    retrieved_docs,
                    start=1
                ):

                    file_name = doc.metadata.get(
                        "source_file",
                        "Unknown File"
                    )

                    page_number = (
                        doc.metadata.get("page", 0) + 1
                    )

                    context_parts.append(
                        f"[Source {number}: "
                        f"{file_name}, "
                        f"Page {page_number}]\n"
                        f"{doc.page_content}"
                    )

                context = "\n\n".join(
                    context_parts
                )

                prompt = f"""
You are a document question-answering assistant.

Rules:
1. Use ONLY the evidence provided.
2. Use simple language.
3. Mention source references when possible.
4. If the answer is not available in the evidence, say:

"I could not find that answer in the uploaded documents."

DOCUMENT EVIDENCE:

{context}

QUESTION:

{question}
"""

                with st.spinner(
                    "Generating answer..."
                ):

                    response = llm.invoke(
                        prompt
                    )

                    answer = response.content

                st.subheader("🤖 AI Answer")
                st.write(answer)

                st.subheader(
                    "📄 Evidence Retrieved by RAG"
                )

                for number, doc in enumerate(
                    retrieved_docs,
                    start=1
                ):

                    file_name = doc.metadata.get(
                        "source_file",
                        "Unknown File"
                    )

                    page_number = (
                        doc.metadata.get("page", 0)
                        + 1
                    )

                    with st.expander(
                        f"Source {number}: "
                        f"{file_name} "
                        f"(Page {page_number})"
                    ):
                        st.write(
                            doc.page_content
                        )

    # ==================================================
    # TAB 2 : SEMANTIC SEARCH
    # ==================================================

    with search_tab:

        st.subheader(
            "Search the Knowledge Base"
        )

        search_query = st.text_input(
            "Search by Meaning",
            placeholder="Example: Contract duration",
            key="semantic_search"
        )

        if st.button(
            "Search Chunks",
            key="search_button"
        ):

            if not search_query.strip():

                st.warning(
                    "Please type something to search."
                )

            else:

                result = (
                    st.session_state.vectorstore
                    .similarity_search(
                        search_query,
                        k=top_k
                    )
                )

                st.subheader("Results")

                for number, doc in enumerate(
                    result,
                    start=1
                ):

                    file_name = doc.metadata.get(
                        "source_file",
                        "Unknown File"
                    )

                    page_number = (
                        doc.metadata.get("page", 0)
                        + 1
                    )

                    st.markdown(
                        f"### Result {number}"
                    )

                    st.markdown(
                        f"**File:** {file_name}"
                    )

                    st.markdown(
                        f"**Page:** {page_number}"
                    )

                    st.write(
                        doc.page_content
                    )

                    st.divider()

else:

    st.info(
        """
        Workflow:

        1. Upload PDF files
        2. Select chunk size
        3. Build Knowledge Base
        4. Ask questions using RAG
        5. Inspect retrieved evidence
        """
    )