# Cadence MVP Demo (Streamlit App)
# Smart Goal Planner using Gemini Flash 2.0 + Streamlit

# Required installs:
# pip install streamlit google-generativeai pdfplumber --upgrade

import streamlit as st
import pdfplumber
import google.generativeai as genai

# Set your Gemini API Key
GOOGLE_API_KEY = "AIzaSyBKVSCCH0wE8gXFGmgtEnj5ksX91i0b-jw"
genai.configure(api_key=GOOGLE_API_KEY)

st.set_page_config(page_title="📆 Cadence AI", layout="centered")
st.title("📆 Cadence – Smart Goal Planner")

# Persistent session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "academic_text" not in st.session_state:
    st.session_state.academic_text = ""

# File upload
st.header("1. Upload schedule / relevant docs")
documents = st.file_uploader(
    "Upload class/work schedule, syllabus, etc:",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

# Goal input
st.header("2. What are your goals?")
goals = st.text_area("Describe your goals (e.g., gain muscle, learn guitar, score 85+ in DBMS):")
submit = st.button("🧠 Generate My Weekly Plan")

# Extract text from uploaded files
@st.cache_data
def extract_text_from_files(files):
    full_text = ""
    for uploaded_file in files:
        if uploaded_file.name.endswith(".pdf"):
            with pdfplumber.open(uploaded_file) as pdf:
                for page in pdf.pages:
                    table = page.extract_table()
                    if table:
                        for row in table:
                            full_text += "\t".join([str(cell) for cell in row if cell]) + "\n"
                    else:
                        full_text += page.extract_text() or ""
                    full_text += "\n"
        else:
            full_text += uploaded_file.read().decode("utf-8")
    return full_text

# Prompt builder
def build_prompt(goals, academic_text):
    return f"""
You are an intelligent personal assistant designed to help users make meaningful progress toward their short- and long-term goals by creating highly personalized weekly plans.

You are given:
- A paragraph where the user lists their goals (e.g. “gain muscle”, “learn guitar”, “score 85+ in DBMS”). These may vary widely: fitness, creative, academic, career, personal growth, etc.
- Uploaded documents that may include class schedules or other weekly commitments.

Your responsibilities:
1. Parse the schedule and identify all occupied hours across Monday to Saturday. Mark these as unavailable.
2. Build a plan that balances short-term progress (daily effort) with long-term ambition (end goals).
3. Add specific justifications for each session: why this activity now, how it fits their current schedule, and how it serves their bigger objective.
4. Maximize convenience and realism. Adapt to their availability, preferences, and energy levels throughout the day.
5. Prioritize time-sensitive goals (like exams, deadlines) and indicate why they’re scheduled earlier.
6. You may access the internet (real-time retrieval) to gather better recommendations and insights (e.g., looking up hypertrophy training plans, study techniques, or music learning strategies relevant to the user's goal).
7. Ask smart clarifying follow-up questions only when necessary — yes/no or one-word format.
8. Never list existing class timings or academic blocks. Focus only on their goal-based schedule.
9. Avoid overload. Keep sessions light and consistent (20–40 mins typical).
10. If the user provides a follow-up response, regenerate the entire 7-day schedule using the new input.

Output:
- Start with a short explanation of your strategy: how this schedule was designed to fit their goals and avoid conflicts.
- Then give a realistic 7-day plan with specific time slots (e.g., 4:30–5:00 PM).
- After the 7-day schedule, for each individual goal the user listed, include a section like:

Goal: [Goal Title]
Justification:
- Why this approach was selected.
- How the plan supports progress.
- Why the timing and frequency are appropriate.
- Any strategies, methods, or tools used in designing this plan.

Academic Schedule:
{academic_text}

User Goals:
{goals}
"""

# Handle initial plan generation
if submit and goals:
    with st.spinner("Planning your life 💡"):
        if documents:
            academic_text = extract_text_from_files(documents)
            st.session_state.academic_text = academic_text
        else:
            academic_text = st.session_state.get("academic_text", "")

        if academic_text.strip():
            with st.expander("📄 Extracted Academic Info"):
                st.text(academic_text[:3000])
        else:
            st.warning("No academic data extracted from uploaded files.")
            academic_text = "No academic documents were provided or could be read."

        prompt = build_prompt(goals, academic_text)
        model = genai.GenerativeModel("models/gemini-1.5-flash-latest")
        response = model.generate_content(prompt)

        if response.text:
            st.session_state.chat_history.append({"role": "user", "content": goals})
            st.session_state.chat_history.append({"role": "assistant", "content": response.text.strip()})
        else:
            st.error("⚠️ Model returned no response.")

# Display conversation
if st.session_state.chat_history:
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"**You:** {msg['content']}")
        else:
            response = msg["content"]
            if "Goal:" in response:
                main_body, *goal_sections = response.split("Goal:")
                st.markdown(f"**Cadence:**\n\n{main_body.strip()}")
                for section in goal_sections:
                    lines = section.strip().split("\n", 1)
                    title = lines[0].strip()
                    explanation = lines[1].strip() if len(lines) > 1 else ""
                    with st.expander(f"📌 Goal: {title}"):
                        st.markdown(explanation)
            else:
                st.markdown(f"**Cadence:** {response.strip()}")

    # Follow-up input
    followup_input = st.text_input("Ask something or refine the schedule:", key="followup_input")
    if st.button("Send") and followup_input.strip():
        st.session_state.chat_history.append({"role": "user", "content": followup_input})
        full_context = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.chat_history])
        model = genai.GenerativeModel("models/gemini-1.5-flash-latest")
        response = model.generate_content(full_context)

        if response.text:
            st.session_state.chat_history.append({"role": "assistant", "content": response.text.strip()})
        else:
            st.error("⚠️ Could not generate a follow-up response.")
