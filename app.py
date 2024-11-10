# Import necessary libraries
import gradio as gr
from transformers import pipeline
from datasets import load_dataset
import torch
import soundfile as sf

# Initialize models and data outside of functions to avoid reloading on every call
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
synthesizer = pipeline("text-to-speech", model="microsoft/speecht5_tts")

# Load speaker embedding from CMU Arctic dataset
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

# Minimum character threshold for valid input
MIN_INPUT_LENGTH = 20  # Minimum characters required for summarization and TTS

# Function to summarize, speak, and highlight text
def summarize_and_speak(text, speed=1.0, font_size=16, font_style="Arial", line_spacing=1.5, background_color="#121212"):
    # Check if input is long enough
    if len(text.strip()) < MIN_INPUT_LENGTH:
        return "Not enough input. Please provide more text.", None

    # Summarize the input text
    summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
    summarized_text = summary[0]['summary_text']

    # Convert summarized text to speech
    speech = synthesizer(summarized_text, forward_params={"speaker_embeddings": speaker_embedding})
    sf.write("output_speech.wav", speech["audio"], samplerate=speech["sampling_rate"])

    # Adjust text appearance settings (for readability)
    html_text = f"""
    <div style="font-size: {font_size}px; font-family: {font_style}; line-height: {line_spacing}; background-color: {background_color}; padding: 20px; color: white; border-radius: 8px;">
        <p>{summarized_text}</p>
    </div>
    """

    return html_text, "output_speech.wav"

# Gradio Interface
iface = gr.Interface(
    fn=summarize_and_speak,
    inputs=[
        gr.Textbox(
            placeholder="Paste or type the text you want to simplify and listen to...",
            label="Input Text for Summarization & Audio",
            lines=5
        ),
        gr.Slider(minimum=0.5, maximum=2.0, step=0.1, value=1.0, label="Speech Speed"),  # Updated default to value
        gr.Slider(minimum=10, maximum=40, step=1, value=16, label="Font Size"),           # Updated default to value
        gr.Dropdown(choices=["Arial", "Comic Sans MS", "Helvetica", "OpenDyslexic"], label="Font Style"),
        gr.Slider(minimum=1.0, maximum=2.5, step=0.1, value=1.5, label="Line Spacing"),   # Updated default to value
        gr.ColorPicker(value="#121212", label="Background Color")                         # Updated default to value
    ],
    outputs=[
        gr.HTML(label="✨ Simplified Summary", elem_id="summary-text"),
        gr.Audio(label="🎧 Listen to the Audio Output")
    ],
    title="📚 ReadAlly: Your Literacy Companion",
    description=(
        "Welcome to **ReadAlly**, your personal literacy assistant for simplified reading and "
        "text-to-speech. Perfect for individuals with reading disabilities, dyslexia, or anyone "
        "seeking easy access to complex texts. "
        "\n\n**How It Works**:\n1. Enter or paste the text you wish to simplify.\n2. Click "
        "'Generate Summary and Audio'.\n3. Enjoy the summarized text and listen to it as audio!"
    ),
    theme="compact",  # Keep the theme as compact for responsiveness
    css="""
        body {
            font-family: 'Arial', sans-serif;
            color: #333;
            text-align: center;
            transition: all 0.3s ease;
        }
        /* Dark Mode */
        .dark-mode {
            background-color: #121212;
            color: white;
        }
        .dark-mode h1 {
            color: #1E90FF;
        }
        .dark-mode .output_audio label {
            font-weight: 600;
            color: #1E90FF;
        }
        .dark-mode .gradio-container {
            background: #1e1e1e;
        }
        /* Light Mode */
        .light-mode {
            background-color: #f4f4f4;
            color: #333;
        }
        .light-mode h1 {
            color: #1E90FF;
        }
        .light-mode .output_audio label {
            font-weight: 600;
            color: #1E90FF;
        }
        .light-mode .gradio-container {
            background: #fff;
        }
        .input_textarea label, .output_textbox label {
            font-weight: 600;
        }
        .output_textbox textarea {
            background-color: #f0f0f0;
            color: #333;
            border-radius: 5px;
        }
        .gradio-container {
            border-radius: 10px;
            padding: 20px;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100%;
        }
        .gradio-interface {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }
    """
)

# Launch the Gradio app
iface.launch(share=True)
