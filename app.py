import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

st.set_page_config(page_title="SafeStudy AI", layout="centered")

st.title("SafeStudy AI")
st.write("A responsible AI study assistant.")

@st.cache_resource
def load_model():
    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

with st.spinner("Loading model. First startup may take about 20 seconds."):
    tokenizer, model = load_model()

SYSTEM_PROMPT = (
    "You are SafeStudy AI, a responsible study assistant.\n"
    "Rules:\n"
    "- Do not complete exams, quizzes, or graded assignments.\n"
    "- Explain concepts step by step instead of giving final answers.\n"
    "- Refuse requests that promote cheating or harm.\n"
    "- Keep responses educational and age appropriate.\n"
)

st.sidebar.header("Study Mode")
task = st.sidebar.radio(
    "Choose a task",
    ["Explain a concept", "Summarize notes", "Create practice questions"]
)

user_input = st.text_area("Enter your topic or question", height=150)
generate_button = st.button("Generate response")

def generate_response(task, text):
    if not text.strip():
        return "Please enter a topic or question."
    prompt = (
        SYSTEM_PROMPT +
        "\nTask: " + task +
        "\nStudent input: " + text +
        "\nResponse:"
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=200)
    return tokenizer.decode(output[0], skip_special_tokens=True)

if generate_button:
    with st.spinner("Generating response..."):
        result = generate_response(task, user_input)
        st.subheader("SafeStudy AI Response")
        st.write(result)

st.markdown("---")
st.caption("SafeStudy AI promotes learning, not cheating.")
