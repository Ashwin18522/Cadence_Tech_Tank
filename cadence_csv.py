import streamlit as st
import pandas as pd
import pdfplumber
import google.generativeai as genai
import re
import io
from datetime import datetime, timedelta
from streamlit_calendar import calendar

# Set your Gemini API Key
GOOGLE_API_KEY = "AIzaSyBKVSCCH0wE8gXFGmgtEnj5ksX91i0b-jw"
genai.configure(api_key=GOOGLE_API_KEY)

st.set_page_config(page_title="📅 CADENCE AI", layout="wide")
st.title("📆 CADENCE AI – Smart Semester Planner")

# Persistent session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "academic_text" not in st.session_state:
    st.session_state.academic_text = ""
if "schedule_df" not in st.session_state:
    st.session_state.schedule_df = None
if "events" not in st.session_state:
    st.session_state.events = []
if "sem_start" not in st.session_state:
    st.session_state.sem_start = None
if "sem_end" not in st.session_state:
    st.session_state.sem_end = None
if "cie_dates" not in st.session_state:
    st.session_state.cie_dates = []
if "additional_csv_files" not in st.session_state:
    st.session_state.additional_csv_files = []
if "user_start_date" not in st.session_state:
    st.session_state.user_start_date = datetime.now().date()

# --- Sidebar File Uploads ---
st.sidebar.header("📂 Upload Files")
timetable_csv = st.sidebar.file_uploader("Upload Weekly Timetable CSV", type="csv")
academic_csv = st.sidebar.file_uploader("Upload Academic Calendar CSV", type="csv")
additional_csv_files = st.sidebar.file_uploader(
    "Upload Additional CSV Files (assignments, events, etc.)",
    type="csv",
    accept_multiple_files=True
)
documents = st.sidebar.file_uploader(
    "Upload other documents (syllabus, etc.):",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

# Add start date selector
st.sidebar.header("📅 Select Start Date")
user_start_date = st.sidebar.date_input(
    "When do you want your schedule to start?",
    value=st.session_state.user_start_date
)
st.session_state.user_start_date = user_start_date

# Goal input
st.sidebar.header("🎯 Set Goals")
goals = st.sidebar.text_area("Describe your goals (e.g., gain muscle, learn guitar, score 85+ in DBMS):")
submit = st.sidebar.button("🧠 Generate My Plan")

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
def build_prompt(goals, academic_text, start_date):
    return f"""
You are an intelligent personal assistant designed to help users make meaningful progress toward their short- and long-term goals by creating highly personalized weekly plans.

You are given:
- A paragraph where the user lists their goals (e.g. "gain muscle", "learn guitar", "score 85+ in DBMS"). These may vary widely: fitness, creative, academic, career, personal growth, etc.
- Uploaded documents that may include class schedules or other weekly commitments.
- A start date of {start_date.strftime('%Y-%m-%d')} from which the schedule should begin. Ensure all scheduled activities occur on or after this date.

Your responsibilities:
1. Parse the schedule and identify all occupied hours across Monday to Saturday. Mark these as unavailable.
2. Build a plan that balances short-term progress (daily effort) with long-term ambition (end goals).
3. Add specific justifications for each session: why this activity now, how it fits their current schedule, and how it serves their bigger objective.
4. Maximize convenience and realism. Adapt to their availability, preferences, and energy levels throughout the day.
5. Prioritize time-sensitive goals (like exams, deadlines) and indicate why they're scheduled earlier.
6. You may access the internet (real-time retrieval) to gather better recommendations and insights (e.g., looking up hypertrophy training plans, study techniques, or music learning strategies relevant to the user's goal).
7. Ask smart clarifying follow-up questions only when necessary — yes/no or one-word format.
8. Never list existing class timings or academic blocks. Focus only on their goal-based schedule.
9. Avoid overload. Keep sessions light and consistent (20–40 mins typical).
10. If the user provides a follow-up response, regenerate the entire 7-day schedule using the new input.

Output:
- Start with a short explanation of your strategy: how this schedule was designed to fit their goals and avoid conflicts. This will be displayed to the user.
- Then create a 7-day plan with specific time slots, starting from {start_date.strftime('%Y-%m-%d')}.
- VERY IMPORTANT: Format each scheduled activity on its own line using exactly this format:
  Day: HH:MM - HH:MM: Activity
  For example:
  Monday: 09:00 - 10:00: CY245AT
  Monday: 10:00 - 11:00: CS241AT
  Monday: 11:00 - 11:30: Short Break
- After the 7-day schedule, for each individual goal the user listed, include a section like:

Goal: [Goal Title]
Justification:
- Why this approach was selected.
- How the plan supports progress.
- Why the timing and frequency are appropriate.
- Any strategies, methods, or tools used in designing this plan.

The text schedule will be parsed by the system and displayed in a table format to the user. Only your strategy explanation and goal justifications will be shown as text.

Academic Schedule:
{academic_text}

User Goals:
{goals}
"""

# Function to extract schedule from AI response
def extract_schedule(text):
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    schedule_data = []
    
    # Pattern to match time slots in formats like "Monday: 09:00 - 10:00: CY245AT"
    pattern = r'(' + '|'.join(days) + r')[\s:]+((?:\d{1,2}):(?:\d{2}))\s*[-–—to]\s*((?:\d{1,2}):(?:\d{2}))[\s:]+(.+?)(?=$|\n)'
    
    matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
    
    for match in matches:
        day = match.group(1).strip().title()
        start_time = match.group(2).strip()
        end_time = match.group(3).strip()
        subject = match.group(4).strip()
        
        # Ensure start_time and end_time are in HH:MM format
        if len(start_time.split(':')[0]) == 1:
            start_time = f"0{start_time}"
        if len(end_time.split(':')[0]) == 1:
            end_time = f"0{end_time}"
            
        # Remove any trailing punctuation from subject
        subject = re.sub(r'[.:,;]$', '', subject)
        
        schedule_data.append({
            "Day": day,
            "Start Time": start_time,
            "End Time": end_time,
            "Subject": subject
        })
    
    return schedule_data

# Additional function to handle different formats and extract schedule blocks
def extract_schedule_from_text(text):
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    schedule_data = []
    
    # Find sections that might contain schedule blocks
    day_sections = []
    
    # First, try to find structured day sections
    for day in days:
        day_pattern = re.compile(fr'{day}[:\s]*\n', re.IGNORECASE)
        matches = list(day_pattern.finditer(text))
        
        for i, match in enumerate(matches):
            start_pos = match.end()
            end_pos = len(text)
            
            # Find the end of this day's section (next day or end of text)
            if i < len(matches) - 1:
                end_pos = matches[i + 1].start()
            else:
                # Look for the next day
                for next_day in days:
                    if next_day == day:
                        continue
                    next_day_match = re.search(fr'\n{next_day}[:\s]*\n', text[start_pos:], re.IGNORECASE)
                    if next_day_match:
                        end_pos = start_pos + next_day_match.start()
                        break
            
            day_sections.append((day, text[start_pos:end_pos]))
    
    # If we didn't find day sections, look for individual time slots
    if not day_sections:
        # Main pattern for time slots
        time_pattern = re.compile(
            r'(' + '|'.join(days) + r')[\s:]+'  # Day
            r'((?:\d{1,2}):(?:\d{2}))\s*[-–—to]\s*((?:\d{1,2}):(?:\d{2}))[\s:]+' # Time range
            r'(.+?)(?=$|\n)'  # Activity
        , re.IGNORECASE | re.MULTILINE)
        
        matches = time_pattern.finditer(text)
        
        for match in matches:
            day = match.group(1).strip().title()
            start_time = match.group(2).strip()
            end_time = match.group(3).strip()
            subject = match.group(4).strip()
            
            # Ensure start_time and end_time are in HH:MM format
            if len(start_time.split(':')[0]) == 1:
                start_time = f"0{start_time}"
            if len(end_time.split(':')[0]) == 1:
                end_time = f"0{end_time}"
                
            # Remove any trailing punctuation from subject
            subject = re.sub(r'[.:,;]$', '', subject)
            
            schedule_data.append({
                "Day": day,
                "Start Time": start_time,
                "End Time": end_time,
                "Subject": subject
            })
    
    # Process day sections if we found them
    for day, section in day_sections:
        # Look for time slots within each day section
        time_pattern = re.compile(
            r'((?:\d{1,2}):(?:\d{2}))\s*[-–—to]\s*((?:\d{1,2}):(?:\d{2}))[\s:]+(.+?)(?=$|\n)'
        , re.MULTILINE)
        
        matches = time_pattern.finditer(section)
        
        for match in matches:
            start_time = match.group(1).strip()
            end_time = match.group(2).strip()
            subject = match.group(3).strip()
            
            # Ensure start_time and end_time are in HH:MM format
            if len(start_time.split(':')[0]) == 1:
                start_time = f"0{start_time}"
            if len(end_time.split(':')[0]) == 1:
                end_time = f"0{end_time}"
                
            # Remove any trailing punctuation from subject
            subject = re.sub(r'[.:,;]$', '', subject)
            
            schedule_data.append({
                "Day": day.title(),
                "Start Time": start_time,
                "End Time": end_time,
                "Subject": subject
            })
    
    return schedule_data

# Convert schedule data to CSV
def convert_to_csv(schedule_data):
    if not schedule_data:
        return None, None
    
    df = pd.DataFrame(schedule_data)
    
    # Sort by day of week and start time
    day_order = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
    df["Day_Order"] = df["Day"].map(day_order)
    df = df.sort_values(["Day_Order", "Start Time"]).drop("Day_Order", axis=1)
    
    # Create CSV string
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    return csv_buffer.getvalue(), df

# Extract justification and goals from AI response
def extract_justification_and_goals(text):
    # Find where the schedule starts
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_pattern = '|'.join(days)
    schedule_start = re.search(fr'\b({day_pattern})[\s:]+((?:\d{{1,2}}):(?:\d{{2}}))', text, re.IGNORECASE)
    
    if schedule_start:
        justification = text[:schedule_start.start()].strip()
    else:
        # If no schedule pattern found, look for goal section
        goal_start = re.search(r'Goal:', text)
        if goal_start:
            justification = text[:goal_start.start()].strip()
        else:
            # If no clear sections found, use the whole text
            justification = text.strip()
    
    # Extract goal sections
    goal_sections = []
    if "Goal:" in text:
        parts = text.split("Goal:")
        for i in range(1, len(parts)):  # Skip the first part (before first "Goal:")
            goal_sections.append("Goal:" + parts[i].strip())
    
    return justification, goal_sections

# Convert LLM schedule to calendar events
def convert_llm_schedule_to_events(schedule_df, user_start_date):
    events = []
    if schedule_df is None or schedule_df.empty:
        return events
    
    # Use the user-selected start date
    start_date = user_start_date
    
    # Calculate the dates for each day of the week starting from the user's start date
    day_offsets = {}
    current_date = start_date
    
    # Find the first occurrence of each weekday starting from the user's start date
    for _ in range(7):  # Look ahead one week
        day_name = current_date.strftime("%A")
        if day_name not in day_offsets:
            day_offsets[day_name] = current_date
        current_date += timedelta(days=1)

    # Generate events for each schedule entry
    for _, row in schedule_df.iterrows():
        day = row["Day"]
        if day in day_offsets:
            event_date = day_offsets[day]
            start_time = datetime.strptime(row["Start Time"], "%H:%M").time()
            end_time = datetime.strptime(row["End Time"], "%H:%M").time()
            
            start_datetime = datetime.combine(event_date, start_time)
            end_datetime = datetime.combine(event_date, end_time)
            
            events.append({
                "title": row["Subject"],
                "start": start_datetime.isoformat(),
                "end": end_datetime.isoformat(),
                "color": "#9c27b0"  # Purple for AI-generated events
            })
    
    return events

# Process additional CSV files
def process_additional_csv_files():
    if not additional_csv_files:
        return
    
    for csv_file in additional_csv_files:
        try:
            df = pd.read_csv(csv_file)
            
            # Check if the CSV has expected columns for events
            if all(col in df.columns for col in ["Day", "Start Time", "End Time", "Subject"]):
                # This is a schedule-like CSV
                for _, row in df.iterrows():
                    try:
                        day = row["Day"]
                        
                        # Use the user-selected start date
                        start_date = st.session_state.user_start_date
                        day_offsets = {}
                        current_date = start_date
                        
                        for _ in range(7):
                            day_name = current_date.strftime("%A")
                            if day_name not in day_offsets:
                                day_offsets[day_name] = current_date
                            current_date += timedelta(days=1)
                        
                        if day in day_offsets:
                            event_date = day_offsets[day]
                            start_time = datetime.strptime(row["Start Time"], "%H:%M").time()
                            end_time = datetime.strptime(row["End Time"], "%H:%M").time()
                            
                            start_datetime = datetime.combine(event_date, start_time)
                            end_datetime = datetime.combine(event_date, end_time)
                            
                            st.session_state.events.append({
                                "title": row["Subject"],
                                "start": start_datetime.isoformat(),
                                "end": end_datetime.isoformat(),
                                "color": "#f97316"  # Orange for additional events
                            })
                    except Exception as e:
                        st.warning(f"⚠️ Skipping event from {csv_file.name}: {e}")
            
            elif all(col in df.columns for col in ["Date", "Title"]):
                # This is a date-based event CSV
                for _, row in df.iterrows():
                    try:
                        event_date = pd.to_datetime(row["Date"])
                        title = row["Title"]
                        
                        # Check for Start/End Time columns
                        if "Start Time" in df.columns and "End Time" in df.columns:
                            start_time = datetime.strptime(row["Start Time"], "%H:%M").time()
                            end_time = datetime.strptime(row["End Time"], "%H:%M").time()
                            
                            start_datetime = datetime.combine(event_date.date(), start_time)
                            end_datetime = datetime.combine(event_date.date(), end_time)
                        else:
                            # All-day event
                            start_datetime = event_date
                            end_datetime = event_date + timedelta(days=1)
                        
                        st.session_state.events.append({
                            "title": title,
                            "start": start_datetime.isoformat(),
                            "end": end_datetime.isoformat(),
                            "color": "#f97316"  # Orange for additional events
                        })
                    except Exception as e:
                        st.warning(f"⚠️ Skipping event from {csv_file.name}: {e}")
            
            # Show the processed CSV
            with st.expander(f"📋 Additional CSV: {csv_file.name}"):
                st.dataframe(df)
                
        except Exception as e:
            st.error(f"❌ Error processing CSV file {csv_file.name}: {e}")

# Update calendar events
def update_calendar_events():
    # Clear existing events
    st.session_state.events = []
    
    # Add academic calendar events (if available)
    if academic_csv is not None:
        process_academic_calendar()
    
    # Add timetable events (if available)
    if timetable_csv is not None and st.session_state.sem_start and st.session_state.sem_end:
        process_timetable()
    
    # Add additional CSV files (if available)
    if additional_csv_files:
        process_additional_csv_files()
    
    # Add LLM-generated schedule events (if available)
    if st.session_state.schedule_df is not None and not st.session_state.schedule_df.empty:
        llm_events = convert_llm_schedule_to_events(st.session_state.schedule_df, st.session_state.user_start_date)
        st.session_state.events.extend(llm_events)

# Process academic calendar CSV
def process_academic_calendar():
    if academic_csv is None:
        return
        
    try:
        df_academic = pd.read_csv(academic_csv)
        df_academic.columns = df_academic.columns.str.strip()
        
        with st.expander("📗 Academic Calendar"):
            st.dataframe(df_academic)
        
        try:
            sem_start = pd.to_datetime(
                df_academic.loc[df_academic["Activity"] == "Start of Semester", "From"].values[0],
                format="%d-%b-%Y"
            )
            sem_end = pd.to_datetime(
                df_academic.loc[df_academic["Activity"] == "Last Working Day of Semester", "From"].values[0],
                format="%d-%b-%Y"
            )
            st.session_state.sem_start = sem_start
            st.session_state.sem_end = sem_end
        except Exception as e:
            st.warning(f"⚠️ Could not find semester dates: {e}")
            st.session_state.sem_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            st.session_state.sem_end = st.session_state.sem_start + timedelta(days=90)  # Default 90 days semester

        # Extract CIE dates
        cie_dates = []
        cie_activities = ['CIE I (Test & Quiz I)', 'CIE II (Test & Quiz II)', 'Improvement CIE (Quiz & Test)']
        for _, row in df_academic.iterrows():
            if row['Activity'] in cie_activities:
                try:
                    start = pd.to_datetime(row["From"], format="%d-%b-%Y")
                    end = pd.to_datetime(row["To"], format="%d-%b-%Y") + timedelta(days=1) if pd.notna(row["To"]) else start + timedelta(days=1)
                    cie_dates.append((start, end))
                except Exception as e:
                    st.warning(f"⚠️ Skipping CIE event due to error: {e}")
        
        st.session_state.cie_dates = cie_dates

        # Create events from academic calendar
        for _, row in df_academic.iterrows():
            try:
                start = pd.to_datetime(row["From"], format="%d-%b-%Y")
                end = pd.to_datetime(row["To"], format="%d-%b-%Y") + timedelta(days=1) if pd.notna(row["To"]) else start + timedelta(days=1)

                st.session_state.events.append({
                    "title": row["Activity"],
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                    "color": "#16a34a"  # Green for academic events
                })
            except Exception as e:
                st.warning(f"⚠️ Skipping academic event due to error: {e}")
    except Exception as e:
        st.error(f"❌ Error processing academic calendar: {e}")

# Process weekly timetable CSV
def process_timetable():
    if timetable_csv is None or st.session_state.sem_start is None or st.session_state.sem_end is None:
        return
        
    try:
        df_timetable = pd.read_csv(timetable_csv)
        df_timetable.columns = df_timetable.columns.str.strip()
        
        with st.expander("📘 Weekly Timetable"):
            st.dataframe(df_timetable)
        
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
        
        current_date = st.session_state.sem_start
        while current_date <= st.session_state.sem_end:
            weekday_name = current_date.strftime("%A")
            day_rows = df_timetable[df_timetable["Day"] == weekday_name]
            for _, row in day_rows.iterrows():
                try:
                    start_time = datetime.strptime(row["Start Time"], "%H:%M").time()
                    end_time = datetime.strptime(row["End Time"], "%H:%M").time()

                    start_datetime = datetime.combine(current_date, start_time)
                    end_datetime = datetime.combine(current_date, end_time)

                    # Check if the class overlaps with any CIE
                    cancel_class = False
                    for cie_start, cie_end in st.session_state.cie_dates:
                        if (start_datetime >= cie_start and start_datetime < cie_end) or (end_datetime > cie_start and end_datetime <= cie_end):
                            cancel_class = True
                            break

                    if not cancel_class:
                        st.session_state.events.append({
                            "title": row["Subject"],
                            "start": start_datetime.isoformat(),
                            "end": end_datetime.isoformat(),
                            "color": "#2563eb"  # Blue for regular classes
                        })
                except Exception as e:
                    st.warning(f"⚠️ Skipping timetable event due to error: {e}")
            current_date += timedelta(days=1)
    except Exception as e:
        st.error(f"❌ Error processing timetable: {e}")

# Modified display function for AI response
def display_ai_response(response_text):
    # Extract justification and goals
    justification, goal_sections = extract_justification_and_goals(response_text)
    
    # Display justification
    st.markdown(f"*AI Planner:*\n\n{justification}")
    
    # Display goal justifications as expandable sections
    for section in goal_sections:
        if section.startswith("Goal:"):
            goal_lines = section.split('\n', 1)
            goal_title = goal_lines[0].replace('Goal:', '').strip()
            goal_content = goal_lines[1].strip() if len(goal_lines) > 1 else ""
            with st.expander(f"📌 Goal: {goal_title}"):
                st.markdown(goal_content)

# --- Process files if provided ---
update_calendar_events()

# Handle initial plan generation
if submit and goals:
    with st.spinner("Planning your life 💡"):
        # Process documents
        additional_text = ""
        if documents:
            additional_text = extract_text_from_files(documents)
            st.session_state.academic_text = additional_text
        
        # Combine with existing academic context
        if timetable_csv is not None:
            try:
                df_timetable = pd.read_csv(timetable_csv)
                additional_text += f"\nWeekly Timetable:\n{df_timetable.to_string()}\n"
            except Exception as e:
                st.warning(f"⚠️ Error reading timetable CSV: {e}")
        
        if academic_csv is not None:
            try:
                df_academic = pd.read_csv(academic_csv)
                additional_text += f"\nAcademic Calendar:\n{df_academic.to_string()}\n"
            except Exception as e:
                st.warning(f"⚠️ Error reading academic CSV: {e}")
                
        # Add information from additional CSV files
        if additional_csv_files:
            for csv_file in additional_csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    additional_text += f"\nAdditional CSV ({csv_file.name}):\n{df.to_string()}\n"
                except Exception as e:
                    st.warning(f"⚠️ Error reading additional CSV {csv_file.name}: {e}")
        
        academic_text = additional_text
        
        if academic_text.strip():
            with st.expander("📄 Extracted Academic Info"):
                st.text(academic_text[:3000])
        else:
            st.warning("No academic data extracted from uploaded files.")
            academic_text = "No academic documents were provided or could be read."

        prompt = build_prompt(goals, academic_text, st.session_state.user_start_date)
        model = genai.GenerativeModel("models/gemini-1.5-flash-latest")
        response = model.generate_content(prompt)

        if response.text:
            response_text = response.text.strip()
            st.session_state.chat_history.append({"role": "user", "content": goals})
            st.session_state.chat_history.append({"role": "assistant", "content": response_text})
            
            # Try both extraction methods and combine results
            schedule_data1 = extract_schedule(response_text)
            schedule_data2 = extract_schedule_from_text(response_text)
            
            # Combine unique entries from both methods
            all_schedule_data = schedule_data1 + [item for item in schedule_data2 
                                               if item not in schedule_data1]
            
            if all_schedule_data:
                csv_content, schedule_df = convert_to_csv(all_schedule_data)
                st.session_state.schedule_df = schedule_df
                
                # Update calendar events to include LLM-generated schedule
                update_calendar_events()
            else:
                st.warning("Could not extract a schedule from the AI response.")
        else:
            st.error("⚠️ Model returned no response.")

# Display conversation
if st.session_state.chat_history:
    st.subheader("📝 Goal Planning Discussion")
    for i, msg in enumerate(st.session_state.chat_history):
        if msg["role"] == "user":
            st.markdown(f"*You:* {msg['content']}")
        else:
            # For AI responses, use the specialized display function
            display_ai_response(msg["content"])

    # Display schedule table
    if st.session_state.schedule_df is not None and not st.session_state.schedule_df.empty:
        st.subheader("📊 Generated Schedule")
        st.dataframe(st.session_state.schedule_df)
        
        # Create a downloadable CSV from the generated schedule
        csv_content, _ = convert_to_csv(st.session_state.schedule_df.to_dict('records'))
        
        # Allow the user to download the AI-generated schedule
        st.download_button(
            label="📥 Download AI Schedule as CSV",
            data=csv_content,
            file_name="ai_schedule.csv",
            mime="text/csv"
        )
    
    # Follow-up input
    st.subheader("⚡ Refine Your Plan")
    followup_input = st.text_input("Ask something or refine the schedule:", key="followup_input")
    if st.button("Send") and followup_input.strip():
        # Continuation of the code for the follow-up functionality
        if st.button("Send") and followup_input.strip():
            with st.spinner("Refining your plan 💭"):
                # Add previous AI-generated schedule and user goals to the context
                additional_context = ""
                if st.session_state.schedule_df is not None and not st.session_state.schedule_df.empty:
                    additional_context += "Current schedule:\n"
                    for _, row in st.session_state.schedule_df.iterrows():
                        additional_context += f"{row['Day']}: {row['Start Time']} - {row['End Time']}: {row['Subject']}\n"
                
                # Extract the original goals if available in chat history
                original_goals = goals
                for msg in st.session_state.chat_history:
                    if msg["role"] == "user" and len(msg["content"]) > 10:  # Likely the original goals message
                        original_goals = msg["content"]
                        break
                
            # Build a new prompt with the follow-up instructions
            followup_prompt = f"""
You are an intelligent personal assistant designing a personalized weekly plan. 
The user previously provided these goals:
{original_goals}

You previously suggested this schedule starting from {st.session_state.user_start_date.strftime('%Y-%m-%d')}:
{additional_context}

They have now provided this feedback or follow-up: 
"{followup_input}"

Please address their feedback and generate an improved schedule. Remember:
1. All activities should be scheduled on or after {st.session_state.user_start_date.strftime('%Y-%m-%d')}.
2. Format each scheduled activity exactly as:
   Day: HH:MM - HH:MM: Activity

Start with a brief explanation of the changes you made to address their feedback, followed by the full 7-day schedule, and finally include justifications for how your schedule addresses each of their goals.
"""
            
            model = genai.GenerativeModel("models/gemini-1.5-flash-latest")
            response = model.generate_content(followup_prompt)
            
            if response.text:
                response_text = response.text.strip()
                st.session_state.chat_history.append({"role": "user", "content": followup_input})
                st.session_state.chat_history.append({"role": "assistant", "content": response_text})
                
                # Extract the updated schedule
                schedule_data1 = extract_schedule(response_text)
                schedule_data2 = extract_schedule_from_text(response_text)
                
                # Combine unique entries
                all_schedule_data = schedule_data1 + [item for item in schedule_data2 
                                                   if item not in schedule_data1]
                
                if all_schedule_data:
                    csv_content, schedule_df = convert_to_csv(all_schedule_data)
                    st.session_state.schedule_df = schedule_df
                    
                    # Update calendar events to include the updated schedule
                    update_calendar_events()
                else:
                    st.warning("Could not extract an updated schedule from the AI response.")
            else:
                st.error("⚠️ Model returned no response.")
                
            # Force a rerun to display the updated conversation and schedule
            st.rerun()

# Display calendar view
st.subheader("📅 Calendar View")

if st.session_state.events:
    # Initialize calendar view
    calendar_options = {
        "headerToolbar": {
            "left": "today prev,next",
            "center": "title",
            "right": "dayGridMonth,timeGridWeek,timeGridDay"
        },
        "initialView": "timeGridWeek",
        "selectable": True,
        "editable": False,
        "navLinks": True,
        "dayMaxEvents": True,
        "timeZone": "local",
        "slotMinTime": "07:00:00",
        "slotMaxTime": "21:00:00",
        "contentHeight": "auto",
    }
    
    calendar(events=st.session_state.events, options=calendar_options)
else:
    st.info("📝 Your calendar will appear here after generating a schedule.")

# Footer
st.markdown("---")
st.markdown("🧠 **CADENCE AI** - Smart Semester Planning with AI")