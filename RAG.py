import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

# Ensure page config is set before anything else
st.set_page_config(layout="wide")

def main():
    # Initialize session state keys if not present
    st.session_state.setdefault("retrieve_chain", None)
    
    st.subheader("RAG Chatbot", divider="rainbow")
    
    with st.sidebar:
        st.sidebar.title("Data Loader")
        st.image("./rag.png", width=500)
        pdf_docs = st.file_uploader(
            label="Upload Your PDFs",
            accept_multiple_files=True,
        )
        if st.button("Submit"):
            with st.spinner("Loading..."):
                pdf_content = ""
                for pdf in pdf_docs:
                    pdf_reader = PdfReader(pdf)
                    for page in pdf_reader.pages:
                        text = page.extract_text()
                        if text is not None:
                            pdf_content += text
                st.write(pdf_content)
                
                if not pdf_content.strip():
                    st.error("No content extracted from the uploaded PDFs.")
                    return
                
                text_splitter = CharacterTextSplitter(
                    separator="\n",
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len,
                )
                chunks = text_splitter.split_text(pdf_content)
                st.write(chunks)

                # Create vector store using FastEmbedEmbeddings
                ollama_embeddings = FastEmbedEmbeddings()
                vector_store = FAISS.from_texts(
                    texts=chunks, embedding=ollama_embeddings
                )
                # Create the language model using Ollama
                llm = Ollama(model="mistral")
                prompt = ChatPromptTemplate.from_template(
                    """
                    Answer the following question based only on the provided context:
                    <context>
                      {context}
                    </context>
                    Question: {input}
                    """
                )
                document_chain = create_stuff_documents_chain(llm, prompt)
                retriever = vector_store.as_retriever()
                retrieval_chain = create_retrieval_chain(retriever, document_chain)
                st.session_state.retrieve_chain = retrieval_chain

    st.subheader("Chatbot zone")
    user_question = st.text_input("Ask your question :")
    if user_question:
        if st.session_state.retrieve_chain:
            response = st.session_state.retrieve_chain.invoke({"input": user_question})
            st.markdown(response["answer"], unsafe_allow_html=True)
        else:
            st.error("Please load data to initialize the chatbot.")

if __name__ == "__main__":
    main()