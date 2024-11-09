# **ReadAlly: Your Literacy Companion**

**ReadAlly** is an AI-powered tool designed to support individuals with reading disabilities like dyslexia by summarizing complex texts and converting the summaries into audio. This tool combines state-of-the-art natural language processing (NLP) and text-to-speech (TTS) technologies to make reading easier and more accessible.


**ReadAlly** is built to help individuals with reading disabilities such as dyslexia. The tool leverages:
- **Text Summarization**: Using the BART model, ReadAlly condenses large, complex texts into easy-to-understand summaries.
- **Text-to-Speech**: Summarized text is then converted into speech using the SpeechT5 model, making the content accessible in audio format.

The project is developed using **Python** and integrates **Gradio** for a user-friendly interface, enabling easy input of text and instant access to summarized text and audio output.

---

## **Features**

- **Text Summarization**: Condenses long-form text into an easily digestible format.
- **Text-to-Speech**: Converts the summarized text to speech.
- **User-Friendly Interface**: Built with Gradio for easy interaction.
- **Dark Mode**: Designed for accessibility, with a dark mode interface for easy viewing.
- **Open-Source**: The project is freely available for enhancement and contribution.

---

## **Technologies Used**

- **Python** 3.x
- **Gradio**: For the interactive web-based interface.
- **Transformers** (Hugging Face): For NLP tasks like text summarization and text-to-speech.
- **Torch**: For model processing and text-to-speech generation.
- **Datasets**: To load speaker embeddings from the CMU Arctic dataset.

---

## **Installation Instructions**

Follow these steps to set up the project locally.

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Steps to Install

1. **Clone the repository**:
   ```bash
   git clone https://github.com/dyuthiramesh/ReadAlly.git
   cd readally
   ```

2. **Create and activate a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows, use: venv\Scripts\activate
   ```

3. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
---

## **Usage**

### Running the Application

After setting up the environment, you can run the application using the following command:

```bash
python app.py
```

This will start the Gradio interface in your browser where you can interact with the tool.

---

## **How It Works**

### Summarization
- The input text is processed by the **BART** model, which summarizes long-form text into shorter, more digestible content.

### Text-to-Speech (TTS)
- The summarized text is passed to the **SpeechT5** model, which uses **speaker embeddings** from the CMU Arctic dataset to generate natural-sounding speech.

### Example:
1. Paste or type the text you wish to summarize.
2. Click the button to generate a summary and listen to the audio.
3. The summarized text and the audio output will be displayed for playback.

---

## **Testing**

We have tested the functionality with different types of text input (e.g., articles, research papers, stories). Here are some example cases:

| Test Case           | Expected Outcome                      | Actual Outcome  |
|---------------------|---------------------------------------|-----------------|
| Short text input    | Summarize text & generate audio       | Passed          |
| Long text input     | Summarize and handle large text      | Passed          |
| Non-English input   | Handle gracefully or error out       | Passed          |

---

## **Contributing**

We welcome contributions! If you'd like to improve the project, feel free to fork the repository and submit a pull request. Here are some ways you can contribute:

- **Bug Fixes**: Help us by reporting and fixing bugs.
- **Enhancements**: Suggest new features or improve existing ones.
- **Documentation**: Update or improve the documentation.

---

## **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## **Acknowledgments**
- **Hugging Face** for providing the **Transformers** library, enabling access to powerful pre-trained models like BART and SpeechT5.
- **Gradio** for making it easy to create interactive web applications with minimal code.
- **CMU Arctic Dataset** for providing speaker embeddings used in text-to-speech generation.

---