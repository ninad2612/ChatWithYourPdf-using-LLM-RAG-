# ChatWithYourPdf-using-LLM-RAG-
Langchain 🦜
LangChain is a framework for developing applications powered by language models. It makes it very easy to develop AI-powered applications and has libraries in Python as well as Javascript. I have used langchain to integrate Ollama with my application.

![image](https://github.com/ninad2612/ChatWithYourPdf-using-LLM-RAG-/assets/167805209/93d23eab-538a-4738-b459-06dcab3a66e7)

### PDF Chatbot

This Streamlit application utilizes language models and document analysis to create an interactive chatbot that answers questions based on uploaded PDF documents. Here's a breakdown of its functionality and components:

#### Features:
- **Document Upload**: Users can upload a PDF document.
- **Document Analysis**: PDF documents are analyzed and processed using PyPDF2.
- **Question Answering (QA)**: The chatbot can answer questions based on the content of the uploaded PDF.
- **Conversation Memory**: Remembers the conversation history to provide context-aware responses.

#### Components and Libraries:
- **Streamlit**: UI framework for creating interactive web applications in Python.
- **LangChain**: Python library for building language pipelines.
- **LangChain Community**: Community extensions for LangChain.
- **Hugging Face Transformers**: Used for sentence embeddings.
- **FAISS**: A library for efficient similarity search and clustering of dense vectors.
- **LLamaCpp**: Language model used for generating responses based on the provided context.

#### Files and Structure:
- **Main Script**: `pdf_chatbot.py` contains the main logic of the application.
- **Directory**: `files/` is created to store uploaded PDFs for caching.
- **Template**: The chatbot uses a predefined template for displaying messages and context.
- **Memory**: Stores conversation history to maintain context between interactions.
- **Callbacks**: Handles streaming output during model inference for real-time updates.

#### Setup and Usage:
1. **Requirements**:
   - Python 3.7+
   - Install dependencies using `pip install -r requirements.txt`.

2. **Sentence Transformers Download**:
   - Run the `transformers.py` file to download the Sentence Transformers and set the path in the main.py file
   - (embeddings = HuggingFaceEmbeddings(
        model_name=r'your-path',
        model_kwargs={'device': 'cpu'}
      ). 
   -This file should be used to set up the embeddings.

3. **LLM Model Download**:
   - Download the Tiny Llama Model from https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF , download whichever model you like I have used the tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf.
   
4. **Running the Application**:
   - Execute `streamlit main.py` to start the application locally.
   - Upload a PDF file to begin interacting with the chatbot.

#### Notes:
- Ensure all dependencies are installed before running (`requirements.txt` included).
- For questions or issues, please refer to the GitHub repository or contact the developer.
