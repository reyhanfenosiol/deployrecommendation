import streamlit as st
from streamlit_chat import message
import tempfile
import os
import requests
from langchain.document_loaders import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain

DB_FAISS_PATH = 'vectorstore/db_faiss'
os.makedirs(DB_FAISS_PATH, exist_ok=True)

# Fungsi untuk mengunduh model
def download_model(url, save_path):
    if not os.path.exists(save_path):
        print(f"Downloading model from {url}...")
        response = requests.get(url, stream=True)
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print(f"Model downloaded to {save_path}")
    else:
        print(f"Model already exists at {save_path}")

# Unduh model jika belum ada
MODEL_URL = "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/resolve/main/llama-2-7b-chat.ggmlv3.q8_0.bin"
MODEL_PATH = "llama-2-7b-chat.ggmlv3.q8_0.bin"
download_model(MODEL_URL, MODEL_PATH)

# Fungsi untuk memuat model
def load_llm():
    llm = CTransformers(
        model=MODEL_PATH,
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5
    )
    return llm

st.title("🏆✨Recommendation System of Top Brand Award Brand Excellence 🏅🥇")
st.markdown("<h3 style='text-align: center; color: white;'>Built by <a href='https://github.com/AIAnytime'>Mochammad Reyhan Mauluddi - 6032222003</a></h3>", unsafe_allow_html=True)

uploaded_file = st.sidebar.file_uploader("Upload your Data", type="csv")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    try:
        loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8", csv_args={'delimiter': ','})
        data = loader.load()
    except Exception as e:
        st.error(f"Error loading CSV: {str(e)}")
        st.stop()

    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs={'device': 'cpu'}
    )

    db = FAISS.from_documents(data, embeddings)
    db.save_local(DB_FAISS_PATH)

    llm = load_llm()
    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever())

    def conversational_chat(query):
        result = chain({"question": query, "chat_history": st.session_state['history']})
        st.session_state['history'].append((query, result["answer"]))
        return result["answer"]

    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = [f"Hello! Ask me anything about {uploaded_file.name} 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey! 👋"]

    response_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Query:", placeholder="Talk to your CSV data here (:", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            output = conversational_chat(user_input)
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=f"{i}_user", avatar_style="big-smile")
                message(st.session_state["generated"][i], key=f"{i}", avatar_style="thumbs")
