from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import google.generativeai as genai
import sympy as sp
import streamlit as st
import os
import time

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():

    # - If the answer is not in provided context just say, "answer is not available in the context".
    # - Make the most important part of your answer COLORFUL like: ":red[Text1]" 
    prompt_template = """
    ## Follow instructions below:
    - Answer the question as detailed as possible from the provided context.
    - Make sure to provide all the details.
    - Don't provide the wrong answer.

    \n\n

    ## Context:\n {context}?\n

    ## Question: \n{question}\n

    ## Answer: Answer here by using Markdown
    """

    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, stream=True)

    # if "chat" not in st.session_state:
    #     st.session_state.chat = model.start_chat(history=[])

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


def user_input(user_question, stream=True):
    USER_NAME = "user"
    ASSISTANT_NAME = "assistant"

    # 以前のチャットログを表示
    for chat in st.session_state.chat_log:
        with st.chat_message(chat["name"]):
            st.write(chat["msg"])

    # 最新のメッセージを表示
    with st.chat_message(USER_NAME):
        st.write(user_question)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    with st.chat_message(ASSISTANT_NAME):
        assistant_msg = ""
        assistant_response_area = st.empty()
        if stream:
            length = len(response["output_text"])
            quo = length//10
            for i in range(quo):
                show_text = response["output_text"][i*10:i*10+10]
                assistant_msg += show_text
                assistant_response_area.write(assistant_msg)
                time.sleep(0.05)
            assistant_msg += response["output_text"][quo*10:]
            assistant_response_area.write(assistant_msg)

        else:
            assistant_msg = response["output_text"]
            assistant_response_area.write(assistant_msg)

    # セッションにチャットログを追加
    st.session_state.chat_log.append({"name": USER_NAME, "msg": user_question})
    st.session_state.chat_log.append({"name": ASSISTANT_NAME, "msg": assistant_msg})


def main():
    # st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini-1.5-flash 💁")

    if "chat_log" not in st.session_state:
        st.session_state.chat_log = []

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")


    # user_question = st.text_input("Ask a Question from the PDF Files")
    # user_msg = st.chat_input("ここにメッセージを入力")
    user_question = st.chat_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)


if __name__ == "__main__":
    
    # with st.sidebar:
    #     API_KEY = st.text_input("Please fill the GOOGLE_API_KEY", type="password")
    #     genai.configure(api_key=API_KEY)

    main()
