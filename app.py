import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ==============================
# PAGE CONFIGURATION
# ==============================
st.set_page_config(
    page_title="SafeStudy AI",
    page_icon="📘",
    layout="centered"
)

st.title("📘 SafeStudy AI")
st.subheader("Learn the right way — responsibly and safely")

# ==============================
# LOAD MODEL (FAST + CPU SAFE)
# ==============================
@st.cache_resource
def load_model():
    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32
    )
    model.eval()
    return tokenizer, model

with st.spinner("Loading model (first startup may take ~20 seconds)..."):
    tokenizer, model = load_model()

# ==============================
# SAFETY SYSTEM PROMPT
# ==============================
SYSTEM_PROMPT = (
    "You are SafeStudy AI, a responsible study assistant.\n"
    "Rules:\n"
    "- Do NOT complete exams, quizzes, or graded assignments.\n"
    "- Explain concepts step by step instead of giving final answers.\n"
    "- Refuse requests that promote cheating or harm.\n"
    "- Keep responses age-appropriate and educational.\n"
)

# ==============================
# SIDEBAR CONTROLS
# ==============================
st.sidebar.header("📚 Study Mode")
task = st.sidebar.radio(
    "Choose a task:",
    [
        "Explain a concept",
        "Summarize notes",
        "Create practice questions"
    ]
)

# ==============================
# USER INPUT
# ==============================
user_input = st.text_area(
    "Enter your topic or question:",
    height=160,
    placeholder="Example: Explain photosynthesis in simple terms"
)

generate_button = st.button("Generate Response")

# ==============================
# RESPONSE GENERATION
# ==============================
def generate_response(task, text):
    if not text.strip():
        return "Please enter a topic or question."

    prompt = (
        SYSTEM_PROMPT
        + "\nTask: " + task
        + "\nStudent Input: " + text
        + "\nResponse:"
    )

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ==============================
# OUTPUT
# ==============================
if generate_button:
    with st.spinner("Thinking..."):
        response = generate_response(task, user_input)
        st.markdown("### ✅ SafeStudy AI Response")
        st.write(response)

# ==============================
# FOOTER
# ==============================
st.markdown("---")
st.caption("SafeStudy AI promotes learning, not cheating.")
