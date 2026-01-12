import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

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
# LOAD MODEL (CACHED)
# =============================
@st.cache_resource
def load_model():
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
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
# USER INPUT BOX
# =============================
user_input = st.text_area(
    "Enter your topic or question:",
    height=180,
    placeholder="Example: Explain Newton's First Law in simple terms"
)

generate_button = st.button("Generate Response")

# =============================
# RESPONSE FUNCTION
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

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.5,
            top_p=0.9,
            do_sample=True
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# =============================
# GENERATE OUTPUT
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
