import streamlit as st
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
from google.generativeai import upload_file, get_file
from pathlib import Path
from dotenv import load_dotenv

import os
import time
import tempfile
import google.generativeai as genai

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

st.set_page_config(
    page_title="Video Analyzer",
    page_icon="📹",
    layout="centered",
)

st.title("Video Analyzer")
st.header("Powered by [Google Gemini](https://www.google.com/gemini)")


@st.cache_resource
def initialize_agent():
    return Agent(
        name="Video Analyzer",
        provider=Gemini(
            id="gemini-2.0-flash-exp"
        ),
        tools=[
            DuckDuckGo()
        ],
        markdown=True
    )


multi_modal_agent = initialize_agent()

video_file = st.file_uploader(
    "Upload a video file",
    type=["mp4", "mov", "avi", "mkv"],
    help="Upload a video file to analyze"
)

if video_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(video_file.read())
        video_path = temp_file.name

    st.video(video_path, format="video/mp4", start_time=0)

    user_query = st.text_area(
        "What would you like to know about this video?",
        placeholder="Ask a question or provide a description",
        help="Ask a question or provide a description to generate a summary"
    )

    if (st.button("Analyze Video", key="analyze_video")) and user_query:
        try:
            with st.spinner("Analyzing video..."):
                processed_video = upload_file(video_path)
                while processed_video.state.name == "PROCESSING":
                    time.sleep(5)
                    processed_video = get_file(processed_video.name)

                analysis_prompt = (
                    f"""
                    Analyze the video for content and context.
                    Respond to the following query using video insights and supplementary web search:
                    {user_query}
                    Provide a detailed, user-friendly and actionable response.
                    """
                )

                response = multi_modal_agent.run(
                    analysis_prompt, videos=[processed_video])

            st.subheader("Analysis Result")
            st.markdown(response.content)

        except Exception as e:
            st.error(f"An error occurred: {e}")
        finally:
            Path(video_path).unlink(missing_ok=True)
else:
    st.info("Upload a video file to get started")

st.markdown(
    """
    <style>
    .stTextArea textarea {
        height: 100px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
