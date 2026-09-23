import os

import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from google import genai

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS


# =========================================================
# CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Engineering Assistant",
    page_icon="🤖",
    layout="wide"
)


# =========================================================
# LOAD API KEY
# =========================================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:

    st.error(
        "GEMINI_API_KEY was not found. "
        "Please add it to your .env file."
    )

    st.stop()


# =========================================================
# GEMINI CLIENT
# =========================================================

client = genai.Client(
    api_key=api_key
)


# =========================================================
# EMBEDDING MODEL
# =========================================================

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=api_key
)


# =========================================================
# TEXT CHUNKING
# =========================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)


# =========================================================
# SESSION STATE
# =========================================================

if "vector_store" not in st.session_state:

    st.session_state.vector_store = None


# =========================================================
# HEADER
# =========================================================

st.title(
    "🤖 AI Engineering Knowledge & Code Review Assistant"
)

st.write(
    "Upload project documentation, ask questions using RAG, "
    "and review Python code with AI."
)


# =========================================================
# PROJECT FILE UPLOAD
# =========================================================

st.header("📂 Project Knowledge Base")

uploaded_files = st.file_uploader(
    "Upload project documentation:",
    type=["pdf", "txt"],
    accept_multiple_files=True
)


# =========================================================
# PROCESS DOCUMENTS
# =========================================================

if uploaded_files:

    all_chunks = []

    metadata = []

    for file in uploaded_files:

        text = ""

        # -------------------------------------------------
        # PDF FILE
        # -------------------------------------------------

        if file.name.lower().endswith(".pdf"):

            pdf = PdfReader(
                file
            )

            for page in pdf.pages:

                text += page.extract_text() or ""

        # -------------------------------------------------
        # TEXT FILE
        # -------------------------------------------------

        elif file.name.lower().endswith(".txt"):

            text = file.getvalue().decode(
                "utf-8",
                errors="ignore"
            )

        # -------------------------------------------------
        # SPLIT DOCUMENT INTO CHUNKS
        # -------------------------------------------------

        chunks = text_splitter.split_text(
            text
        )

        # -------------------------------------------------
        # STORE CHUNKS
        # -------------------------------------------------

        all_chunks.extend(
            chunks
        )

        # -------------------------------------------------
        # STORE SOURCE INFORMATION
        # -------------------------------------------------

        for chunk in chunks:

            metadata.append(
                {
                    "source": file.name,
                    "type": "document"
                }
            )


    # =====================================================
    # CREATE FAISS DATABASE
    # =====================================================

    if all_chunks:

        with st.spinner(
            "Creating embeddings and FAISS database..."
        ):

            st.session_state.vector_store = (
                FAISS.from_texts(
                    all_chunks,
                    embedding=embeddings,
                    metadatas=metadata
                )
            )

        st.success(
            f"Knowledge base created with "
            f"{len(all_chunks)} chunks."
        )

        st.write("Uploaded files:")

        for file in uploaded_files:

            st.write(
                f"📄 {file.name}"
            )


# =========================================================
# PROJECT KNOWLEDGE ASSISTANT
# =========================================================

st.divider()

st.header("💬 Project Knowledge Assistant")

vector_store = st.session_state.vector_store


if vector_store:

    question = st.text_input(
        "Ask a question about your uploaded documents:"
    )

    if question:

        # -------------------------------------------------
        # SEARCH RELEVANT INFORMATION
        # -------------------------------------------------

        with st.spinner(
            "Searching the knowledge base..."
        ):

            relevant_chunks = (
                vector_store.similarity_search(
                    question,
                    k=3
                )
            )


        # -------------------------------------------------
        # CREATE CONTEXT
        # -------------------------------------------------

        context_parts = []

        for document in relevant_chunks:

            source = document.metadata.get(
                "source",
                "Unknown"
            )

            context_parts.append(
                f"Source: {source}\n"
                f"{document.page_content}"
            )


        context = "\n\n---\n\n".join(
            context_parts
        )


        # -------------------------------------------------
        # RAG PROMPT
        # -------------------------------------------------

        prompt = f"""
You are an AI Engineering Knowledge Assistant.

Answer the question using ONLY the information
provided in the retrieved project documents.

Do not invent information.

If the answer is not available in the documents,
say:

"I could not find this information in the uploaded files."

Retrieved information:

{context}

Question:

{question}

Give a clear and concise answer.
"""


        # -------------------------------------------------
        # GEMINI RESPONSE
        # -------------------------------------------------

        with st.spinner(
            "Generating answer..."
        ):

            try:

                response = client.models.generate_content(
                    model="gemini-3.5-flash-lite",
                    contents=prompt
                )

                answer = response.text

            except Exception as e:

                st.error(
                    f"Gemini API error: {e}"
                )

                answer = None


        # -------------------------------------------------
        # DISPLAY ANSWER
        # -------------------------------------------------

        if answer:

            st.subheader("🤖 Answer")

            st.write(
                answer
            )


            # -------------------------------------------------
            # DISPLAY SOURCES
            # -------------------------------------------------

            st.subheader(
                "📚 Retrieved Sources"
            )

            for i, document in enumerate(
                relevant_chunks
            ):

                source = document.metadata.get(
                    "source",
                    "Unknown"
                )

                with st.expander(
                    f"Source {i + 1}: {source}"
                ):

                    st.write(
                        document.page_content
                    )


else:

    st.info(
        "Upload a PDF or TXT file to start "
        "asking questions."
    )


# =========================================================
# AI CODE REVIEW
# =========================================================

st.divider()

st.header("🔍 AI Code Review")

st.write(
    "Upload a Python file and get an AI-assisted "
    "code review."
)


# ---------------------------------------------------------
# PYTHON FILE UPLOAD
# ---------------------------------------------------------

review_file = st.file_uploader(
    "Upload a Python (.py) file:",
    type=["py"],
    key="code_review_file"
)


if review_file:

    # -----------------------------------------------------
    # READ PYTHON CODE
    # -----------------------------------------------------

    code = review_file.getvalue().decode(
        "utf-8",
        errors="ignore"
    )


    st.success(
        f"Python file loaded: {review_file.name}"
    )


    # -----------------------------------------------------
    # SHOW CODE
    # -----------------------------------------------------

    with st.expander(
        "👀 View Python Code"
    ):

        st.code(
            code,
            language="python"
        )


    # -----------------------------------------------------
    # REVIEW MODE
    # -----------------------------------------------------

    review_mode = st.radio(
        "Choose review mode:",
        [
            "General Code Review",
            "Review Against Uploaded Documentation"
        ]
    )


    # -----------------------------------------------------
    # REVIEW BUTTON
    # -----------------------------------------------------

    if st.button(
        "🔍 Review Code",
        type="primary"
    ):


        # =================================================
        # GENERAL CODE REVIEW
        # =================================================

        if review_mode == "General Code Review":

            review_prompt = f"""
You are an AI Engineering Code Review Assistant.

Review the following Python code.

Analyze only what is actually present in the code.
Do not invent problems.

Check:

1. Potential bugs
2. Incorrect logic
3. Error handling
4. Code readability
5. Security concerns if clearly present
6. Performance concerns if clearly present
7. Suggested improvements
8. Test cases

Python file:

{review_file.name}

Code:

{code}

Use this format:

## Summary

Give a short summary of the code.

## Potential Issues

For each issue:

- Issue:
- Why it matters:
- Suggested improvement:

## Good Practices

Mention good practices already present.

## Suggested Test Cases

Give useful test cases.
"""


        # =================================================
        # DOCUMENTATION-BASED REVIEW
        # =================================================

        else:

            if vector_store is None:

                st.warning(
                    "Please upload project documentation "
                    "first."
                )

                st.stop()


            # -------------------------------------------------
            # SEARCH DOCUMENTATION
            # -------------------------------------------------

            requirement_query = f"""
Find project requirements, expected behavior,
functional specifications, constraints, or rules
related to this Python file:

{review_file.name}
"""


            requirement_chunks = (
                vector_store.similarity_search(
                    requirement_query,
                    k=5
                )
            )


            # -------------------------------------------------
            # CREATE DOCUMENTATION CONTEXT
            # -------------------------------------------------

            requirement_parts = []

            for document in requirement_chunks:

                source = document.metadata.get(
                    "source",
                    "Unknown"
                )

                requirement_parts.append(
                    f"Source: {source}\n"
                    f"{document.page_content}"
                )


            requirements_context = (
                "\n\n---\n\n".join(
                    requirement_parts
                )
            )


            # -------------------------------------------------
            # DOCUMENTATION REVIEW PROMPT
            # -------------------------------------------------

            review_prompt = f"""
You are an AI Engineering Code Review Assistant.

Review the Python code against the uploaded
project documentation.

Use ONLY the provided documentation when
checking requirement compliance.

Do not invent requirements.

If the documentation does not provide enough
information, say:

"Cannot be determined from the uploaded documentation."

Project documentation:

{requirements_context}

Python file:

{review_file.name}

Python code:

{code}

Analyze:

1. Requirement compliance
2. Missing functionality
3. Potential bugs
4. Error handling
5. Security concerns if clearly present
6. Readability
7. Performance concerns if clearly present
8. Suggested improvements
9. Test cases

Use this format:

## Requirement Compliance

Explain which documented requirements
the code appears to satisfy.

## Missing or Potential Issues

For each issue:

- Issue:
- Related requirement:
- Why it matters:
- Suggested improvement:

## Good Practices

Mention useful practices already present.

## Suggested Test Cases

Give practical test cases.
"""


        # =================================================
        # SEND CODE TO GEMINI
        # =================================================

        with st.spinner(
            "Analyzing the Python code..."
        ):

            try:

                response = client.models.generate_content(
                    model="gemini-3.5-flash-lite",
                    contents=review_prompt
                )

                review_result = response.text

            except Exception as e:

                st.error(
                    f"Gemini API error: {e}"
                )

                review_result = None


        # =================================================
        # DISPLAY CODE REVIEW
        # =================================================

        if review_result:

            st.subheader(
                "🧑‍💻 Code Review"
            )

            st.markdown(
                review_result
            )


else:

    st.info(
        "Upload a Python (.py) file above to "
        "start the AI code review."
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "AI Engineering Knowledge & Code Review Assistant | "
    "Python • Streamlit • LangChain • Gemini • FAISS • RAG"
)
st.caption(
    "Developed by Dharmarapu Vanisri"
)