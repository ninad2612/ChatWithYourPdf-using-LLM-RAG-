import streamlit as st
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import LlamaCpp
import os

if not os.path.exists('files'):
    os.mkdir('files')

template = """You are a knowledgeable chatbot, here to help users.

Context: {context}
History: {history}

Query: {question}
"""

if 'template' not in st.session_state:
    st.session_state.template = template

if 'prompt' not in st.session_state:
    st.session_state.prompt = PromptTemplate(
        input_variables=["history", "context", "question"],
        template=st.session_state.template,
    )

if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="history",
        return_messages=True,
        input_key="question"
    )

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def process_pdf(uploaded_file):
    if not os.path.isfile("files/"+uploaded_file.name+".pdf"):
        with st.status("Analyzing your document..."):
            bytes_data = uploaded_file.read()
            with open("files/"+uploaded_file.name+".pdf", "wb") as f:
                f.write(bytes_data)

            loader = PyPDFLoader("files/"+uploaded_file.name+".pdf")
            data = loader.load()

            return data
    else:
        st.write("File already analyzed. Using cached data...")
        loader = PyPDFLoader("files/"+uploaded_file.name+".pdf")
        data = loader.load()
        return data

st.title("PDF Chatbot")

uploaded_file = st.file_uploader("Upload your PDF", type='pdf')

if uploaded_file is not None:
    data = process_pdf(uploaded_file)

    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=r'E:\LLM\sentencemodel',
        model_kwargs={'device': 'cpu'}
    )

    # Create vector store using FAISS
    vdb = FAISS.from_documents(
        documents=data,
        embedding=embeddings
    )

    # Initialize LLM
    llm = LlamaCpp(
        model_path=r"E:\LLM\TinyLLama\tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf",
        max_tokens=512,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        verbose=True,
        temperature=0.4,
        stop=['User :'],
        n_ctx=4096
    )

    # Initialize QA chain
    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type='stuff',
            retriever=vdb.as_retriever(),
            verbose=True,
            chain_type_kwargs={
                "verbose": True,
                "prompt": st.session_state.prompt,
                "memory": st.session_state.memory,
            }
        )

    # Chat input
    if user_input := st.text_input("You:"):
        user_message = {"role": "user", "message": user_input}
        st.session_state.chat_history.append(user_message)

        # Get response from QA chain
        with st.spinner("Assistant is typing..."):
            response = st.session_state.qa_chain(user_input)

        # Display response
        with st.container():
            st.markdown(f"**Assistant:** {response['result']}")
            st.session_state.chat_history.append({"role": "assistant", "message": response['result']})

else:
    st.write("Please upload a PDF file.")
