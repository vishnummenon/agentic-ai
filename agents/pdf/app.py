import streamlit as st
from typing import Optional
from phi.assistant import Assistant
from phi.storage.assistant.postgres import PgAssistantStorage
from phi.knowledge.pdf import PDFKnowledgeBase, PDFReader
from phi.vectordb.pgvector.pgvector2 import PgVector2
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Database configuration
db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"

# Initialize storage
storage = PgAssistantStorage(
    table_name="pdf_assistant",
    db_url=db_url
)


def load_knowledge_base(pdf_path: str):
    """
    Load a knowledge base from the given PDF file.
    """
    try:
        knowledge_base = PDFKnowledgeBase(
            path=pdf_path,
            vector_db=PgVector2(
                collection="uploaded_pdf",
                db_url=db_url
            ),
            reader=PDFReader(chunk=True)
        )
        knowledge_base.load()
        st.write("Knowledge base successfully loaded.")
        return knowledge_base
    except Exception as e:
        st.error(f"Error loading knowledge base: {e}")
        raise


def create_or_resume_assistant(user: str, knowledge_base, new: bool = False):
    """
    Create or resume an assistant session for the user.
    """
    run_id: Optional[str] = None
    if not new:
        existing_run_ids = storage.get_all_run_ids(user_id=user)
        if len(existing_run_ids) > 0:
            run_id = existing_run_ids[0]

    assistant = Assistant(
        run_id=run_id,
        user_id=user,
        knowledge_base=knowledge_base,
        storage=storage,
        show_tool_calls=True,
        search_knowledge=True,
        read_chat_history=True,
        markdown=True
    )

    if run_id is None:
        run_id = assistant.run_id
        st.write(f"Started a new session: {run_id}")
    else:
        st.write(f"Resumed session: {run_id}")

    return assistant


# Streamlit UI
st.title("PDF Assistant")
st.write("Upload a PDF and ask questions based on its content.")

# User input for name
user_name = st.text_input("Enter your username", value="user")

# File uploader for PDF
uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

# Options to start a new session or resume
new_session = st.checkbox("Start a new session", value=False)

if uploaded_file is not None:
    # Save uploaded file locally
    pdf_path = f"temp_{user_name}.pdf"
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load knowledge base
    st.write("Processing the uploaded PDF...")
    knowledge_base = load_knowledge_base(pdf_path)
    st.write("PDF loaded successfully!")

    # Create or resume assistant session
    assistant = create_or_resume_assistant(
        user=user_name, knowledge_base=knowledge_base, new=new_session)

    # User input for questions
    question = st.text_input("Ask a question about the PDF content")

    if st.button("Ask"):
        if question.strip():
            response = assistant.run(
                message=question, user_id=user_name, markdown=True
            )
            st.write("**Assistant's Response:**")
            st.write(response)
        else:
            st.warning("Please enter a question.")

    # Display session history
    if st.button("Show History"):
        history = assistant.get_chat_history()
        if history:
            st.write("**Chat History:**")
            for entry in history:
                st.write(f"**User:** {entry['user_message']}")
                st.write(f"**Assistant:** {entry['assistant_message']}")
        else:
            st.write("No history available.")
