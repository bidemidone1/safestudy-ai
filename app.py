import gradio as gr
from transformers import pipeline

# 1. Load the model pipeline
# Using a small, efficient model for the demo
pipe = pipeline("text-generation", model="HuggingFaceH4/zephyr-7b-beta")

def medical_assistant(user_input):
    # Safety Check: Simple Keyword Filter
    emergency_keywords = ["chest pain", "suicide", "stroke", "bleeding"]
    if any(k in user_input.lower() for k in emergency_keywords):
        return "🚨 EMERGENCY DETECTED: Please contact your local emergency services (911) immediately. I am an AI and cannot assist with medical emergencies."

    # Prompt Engineering for Reliability
    system_prompt = f"System: You are a medical information bot. Provide general info only. User: {user_input}"
    
    response = pipe(system_prompt, max_new_tokens=200, do_sample=True)[0]['generated_text']
    
    # Cleaning response to remove system prompt
    final_output = response.split("User:")[-1].strip()
    
    return f"{final_output}\n\n---\n*Disclaimer: This is for informational purposes only. Consult a doctor for medical advice.*"

# 2. Build the Gradio Interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏥 SafeStep Health Assistant")
    gr.Markdown("Ask general medical questions. Diagnoses and emergencies are automatically blocked.")
    
    with gr.Row():
        input_text = gr.Textbox(label="Enter your question (e.g., 'What are the symptoms of Vitamin D deficiency?')")
    
    submit_btn = gr.Button("Get Information")
    output_text = gr.Markdown(label="Response")

    submit_btn.click(fn=medical_assistant, inputs=input_text, outputs=output_text)

demo.launch()
