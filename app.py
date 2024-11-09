# Import necessary libraries
import gradio as gr
from transformers import pipeline
from datasets import load_dataset
import torch
import soundfile as sf

# Initialize models and data outside of functions to avoid reloading on every call
# Summarization pipeline setup
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# TTS pipeline setup
synthesizer = pipeline("text-to-speech", model="microsoft/speecht5_tts")

# Load speaker embedding from CMU Arctic dataset
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

# Define function to summarize and convert text to audio
def summarize_and_speak(text):
    # Summarize the input text
    summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
    summarized_text = summary[0]['summary_text']
    
    # Convert summarized text to speech
    speech = synthesizer(summarized_text, forward_params={"speaker_embeddings": speaker_embedding})
    sf.write("output_speech.wav", speech["audio"], samplerate=speech["sampling_rate"])

    return summarized_text, "output_speech.wav"

# Define Gradio interface
iface = gr.Interface(
    fn=summarize_and_speak,
    inputs=gr.Textbox(
        placeholder="Paste or type the text you want to simplify and listen to...",
        label="Input Text for Summarization & Audio",
        lines=5
    ),
    outputs=[
        gr.Textbox(label="✨ Simplified Summary", elem_id="summary-text", interactive=False),
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
    theme="dark",  # Enable dark mode
    css="""
        body { 
            font-family: 'Arial', sans-serif; 
            background-color: #121212; 
            color: #fff; 
            text-align: center; 
        }
        h1 { color: #1E90FF; }
        .output_audio label { font-weight: 600; color: #1E90FF; }
        .gradio-container { 
            border-radius: 10px; 
            padding: 20px; 
            background: #1e1e1e; 
            display: flex; 
            justify-content: center; 
            align-items: center; 
            height: 100%; 
        }
        .input_textarea label { 
            font-weight: 600; 
            color: #ffffff; 
        }
        .output_textbox label { 
            font-weight: 600; 
            color: #ffffff; 
        }
        .output_textbox textarea { 
            background-color: #333; 
            color: #fff; 
            border-radius: 5px;
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
