import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# =============================
# PAGE CONFIGURATION
# =============================
st.set_page_config(
    page_title="SafeStudy AI",
    page_icon="📘",
    layout="centered"
)

# =============================
# APP TITLE
# =============================
st.title("📘 SafeStudy AI")
st.subheader("Learn the right way — responsibly and safely")

# =============================
# LOAD MODEL (CPU SAFE)
# =============================
@st.cache_resource
def load_model():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# =============================
# SAFETY SYSTEM PROMPT
# =============================
SYSTEM_PROMPT = """
You are SafeStudy AI, a responsible study assistant.

Rules:
- Do NOT complete exams, quizzes, or graded assignments.
- Explain concepts step by step instead of giving final answers.
- Refuse requests that promote cheating or harm.
- Keep responses age-appropriate and safe.
"""

# =============================
# SIDEBAR CONTROLS
# =============================
st.sidebar.header("📚 Study Mode")

task = st.sidebar.radio(
    "Choose a task:",
    [
        "Explain a concept",
        "Summarize notes",
        "Create practice questions"
    ]
)

# =============================
# USER INPUT
# =============================
user_input = st.text_area(
    "Enter your topic or question:",
    height=180,
    placeholder="Example: Explain Newton's First Law in simple terms"
)

generate_button = st.button("Generate Response")

# =============================
# RESPONSE GENERATION FUNCTION
# =============================
def generate_response(task, user_input):
    if not user_input.strip():
        return "⚠️ Please enter a topic or question."

    prompt = f"""
{SYSTEM_PROMPT}

Task: {task}

Student Input:
{user_input}

Response:
"""

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# =============================
# DISPLAY OUTPUT
# =============================
if generate_button:
    with st.spinner("Thinking..."):
        response = generate_response(task, user_input)
        st.markdown("### ✅ SafeStudy AI Response")
        st.write(response)

# =============================
# FOOTER
# =============================
st.markdown("---")
st.caption("SafeStudy AI promotes learning, not cheating.")
