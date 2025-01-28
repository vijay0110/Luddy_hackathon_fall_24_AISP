import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain import hub
from langchain_ollama.llms import OllamaLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import PromptTemplate
from textblob import TextBlob


from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import login
from langchain_chroma import Chroma
import os
import base64

def load_pdf(file_path):
    """Load text from a PDF file."""
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def load_logo(logo_path):
    with open(logo_path, "rb") as file:
        encoded_logo = base64.b64encode(file.read()).decode("utf-8")
    return encoded_logo

def analyze_sentiment(query):
    """Analyze sentiment of the input query."""
    sentiment = TextBlob(query).sentiment.polarity
    if sentiment > 0.1:
        return "positive"
    elif sentiment < -0.1:
        return "negative"
    else:
        return "neutral"


def generate_prompt(query, sentiment):
    """Generate a sentiment-aware prompt."""
    if sentiment == "positive":
        return f"The user is optimistic and curious. Here's the query: {query}"
    elif sentiment == "negative":
        return f"The user seems concerned or upset. Provide a detailed and reassuring response to the query: {query}"
    else:
        return f"The user has a neutral tone. Answer the query directly: {query}"


def ask_rag_chain(query):
    """Process a query with sentiment analysis and pass it to the RAG chain."""
    # Analyze sentiment
    sentiment = analyze_sentiment(query)
    # print(f"Detected sentiment: {sentiment}")
    # Generate sentiment-aware prompt
    prompt = generate_prompt(query, sentiment)
    # print(f"Generated prompt: {prompt}")
    # Pass prompt to the RAG chain
    return prompt

def main():
    st.set_page_config(layout="wide")
    st.title("Automated Insights and Summarization Platform")
    
    # Path to your local logo file
    logo_path = "logo.jpeg"  # Replace with the actual path to your logo

    # Embed the logo with the title
    encoded_logo = load_logo(logo_path)
    st.markdown(
    f"""
    <style>
        .top-right-logo {{
            position: fixed;
            top: 70px;
            right: 370px;
            z-index: 10;
        }}
    </style>
    <div class="top-right-logo">
        <img src="data:image/jpeg;base64,{encoded_logo}" alt="Logo" style="width:100px; height:auto;">
    </div>
    """,
    unsafe_allow_html=True,
    )

    tab1, tab2, tab3 = st.tabs(["Upload the File", "Document Summary", "Q/A"])


    with tab1 :
        st.write("Upload a PDF, Word, or text file to process its content.")
        uploaded_file = st.file_uploader("Upload your file", type=["pdf", "docx", "txt"])
        if uploaded_file is not None:
            with open("uploaded_file.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())

        
            # Process file based on type
            docs = load_pdf("uploaded_file.pdf")

            custom_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
                You are an expert assistant with access to relevant documents. 
                Use the following context to answer the question as accurately and positively as possible.

                Always provide a response, even if the exact information is not available in the documents. 
                If necessary, use logical reasoning or general knowledge to give a constructive and helpful answer.

                Context:
                {context}

                Question:
                {question}

                Your Positive Response:
            """
            )

            prompt = custom_prompt
            
            # # Replace 'your_huggingface_token' with your actual token
            login('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')
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=250)
            splits = text_splitter.split_documents(docs)
            vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

            retriever = vectorstore.as_retriever()
            llm = OllamaLLM(model="llama3.2")
            
            rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
            )

        with tab2:
            if uploaded_file is not None:
                sum_ques = '''
                You are an AI assistant tasked with summarizing the content of a document. Your goal is to provide a concise summary and highlight the key points clearly and accurately. Summarize the document in a clear and concise manner, ensuring that important information is easily accessible. Avoid including unnecessary details or sensitive information while maintaining the essence of the content.
                '''
                sum_reponse = rag_chain.invoke(sum_ques)
                st.subheader("Summary of the Document")
                st.write(sum_reponse)

        with tab3:
            if uploaded_file is not None:
                st.subheader("Q/A")
                question = st.text_input('Please enter your question:')
                if question:
                    st.write(rag_chain.invoke(ask_rag_chain(question)))


if __name__ == "__main__":
    main()
