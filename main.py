import streamlit as st
import tempfile
import os

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from langchain_community.document_loaders import YoutubeLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# Setup API keys and models
os.environ["GOOGLE_API_KEY"] = "paste your google api key" 

model = ChatGoogleGenerativeAI(model='gemini-2.5-pro')
e_model = GoogleGenerativeAIEmbeddings(model='models/embedding-001')

def till_vector_store(url, pdf_path):
    if url:
        loader = YoutubeLoader.from_youtube_url(youtube_url=url, language=["en"])
        docs = loader.load()
    else:
        loader = PyPDFLoader(file_path=pdf_path)
        docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    split_docs = splitter.split_documents(docs)

    vector_store = Chroma(embedding_function=e_model)
    vector_store.add_documents(split_docs)
    return vector_store, split_docs

def get_complete_context(top_k_docs):
    return " ".join([doc.page_content for doc in top_k_docs])

context_runnable = RunnableLambda(get_complete_context)
parser = StrOutputParser()

# Prompts
prompt = PromptTemplate(
    template= """You are my trainer and your the expert tutor.
                Help me to answer the below query by only using the below provided pdf_context(notes) and youtube_context,
                make sure the answers are simple and easy to understand, so that the person with zero knowldge on the topic will also understand 
                always provide the answers in paragraph wise under three sections Notes solution (based on pdf_context), 
                                                                                  youtube solution(based on youtube_context) 
                                                                                and overview (both pdf_context and youtube_context).
                and then if required mention the necessary using bullet points.
                and also provide the reference page number from pdf_context that you are using in order to answer the query.
                after answering, try to provide the human understandable analogy if possible by explicitly mentioning it under 'analogy section' at the end.
                
                if the pdf_context is not available then under notes solution section clearly mention 'no such notes context',
                if the youtube_context is not available then under youtube_context solution section clearly mention 'no such youtube context',
                If the query is out of context(that is no youtube_context and pdf_context) then you can simply display 
                'out of context ask question relating to uploaded pdf and url'.

                query->{query}
                pdf_context->{pdf_context}
                youtube_context->{youtube_context}
            """,        
    input_variables=['query', 'pdf_context', 'youtube_context']
)

quiz_prompt = PromptTemplate(
    template="""Based on the following pdf content,
    generate 10 MCQs with 4 options each in different line
    and provide the correct answer for each question mentioning the question number
    at the end (that is after displaying all the 10 questions)
    Content: {pdf_content}""", 
    input_variables=['pdf_content']
)

summary_prompt = PromptTemplate(
    template="""Summarize this YouTube transcript in detailed format:
    Transcript: {youtube_content}""", 
    input_variables=['youtube_content']
)

abstract_prompt = PromptTemplate(
    template="""Extract a topic-wise abstract as bullet points for each topic,
    separate them under the topic name, by using the provided below content,\ncontent->{pdf_content}""",
    input_variables=['pdf_content']
)

# --- UI Setup ---
st.title("YouSchool")
st.header("Your Study Buddy 🤖")

# Tabs
main_tabs = st.tabs(["Q&A", "Others"])

# Tab 1: Upload & Ask
with main_tabs[0]:
    st.subheader("Upload & Ask Your Question")

    pdf = st.file_uploader("Upload PDF", type=["pdf"])
    url = st.text_input("Paste YouTube URL")
    query = st.text_input("Ask a question based on the PDF/YouTube:")

    if st.button("Answer"):
        if not query or not pdf:
            st.warning("Please upload a PDF and type a question.")
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(pdf.read())
                pdf_path = tmp_file.name

            st.session_state["pdf_path"] = pdf_path
            st.session_state["url"] = url

            pdf_vs, pdf_docs = till_vector_store('', pdf_path)
            yt_vs, yt_docs = till_vector_store(url, '')

            st.session_state["pdf_vs"] = pdf_vs
            st.session_state["yt_vs"] = yt_vs
            st.session_state["pdf_docs"] = pdf_docs
            st.session_state["yt_docs"] = yt_docs

            retr_pdf = pdf_vs.as_retriever(search_kwargs={"k": 5})
            retr_yt = yt_vs.as_retriever(search_kwargs={"k": 5})
            context_query = RunnableParallel({
                'query': RunnablePassthrough(),
                'pdf_context': retr_pdf | context_runnable | parser,
                'youtube_context': retr_yt | context_runnable | parser,
            }) 
            chain = context_query | prompt | model | parser
            result = chain.invoke(query)
            st.success(result)

with main_tabs[1]:
    sub_tabs = st.tabs(["Quiz", "Abstract", "YouTube Summary"])

    if st.button("Generate All"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf.read())
            pdf_path = tmp_file.name

        st.session_state["pdf_path"] = pdf_path
        st.session_state["url"] = url

        pdf_vs, pdf_docs = till_vector_store('', pdf_path)
        yt_vs, yt_docs = till_vector_store(url, '')

        st.session_state["pdf_vs"] = pdf_vs
        st.session_state["yt_vs"] = yt_vs
        st.session_state["pdf_docs"] = pdf_docs
        st.session_state["yt_docs"] = yt_docs

        st.success("Generating.....")

        with sub_tabs[0]:
            st.subheader("Generate Quiz (MCQs) from PDF")
            content = get_complete_context(st.session_state["pdf_docs"])
            prompt = quiz_prompt.format(pdf_content=content)
            chain_1 = model | parser
            result = chain_1.invoke(prompt)
            st.write(result)

        with sub_tabs[1]:
            st.subheader("Topic-wise Abstract from PDF")
            content = get_complete_context(st.session_state["pdf_docs"])
            prompt = abstract_prompt.format(pdf_content=content)
            chain_3 = model | parser
            result = chain_3.invoke(prompt)
            st.write(result)

        with sub_tabs[2]:
            st.subheader("Summarize YouTube Video")
            yt_content = get_complete_context(st.session_state["yt_docs"])
            prompt = summary_prompt.format(youtube_content=yt_content)
            chain_2 = model | parser
            result = chain_2.invoke(prompt)
            st.write(result)
