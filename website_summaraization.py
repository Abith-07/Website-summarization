import streamlit as st
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import faiss
import time
from mistralai import Mistral
from secretsapi import mistral_API

api_key = mistral_API  
client = Mistral(api_key=api_key)
model="mistral-medium-latest"
models = client.models.list()
print(models)

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

faiss_index = faiss.IndexFlatL2(embedding_model.get_sentence_embedding_dimension())

def extract_text_from_url(url):
    """Extract visible text content from a web page given its URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        # Extract visible text
        texts = soup.stripped_strings
        return list(texts)
    except Exception as e:
        st.error(f"Error fetching URL content: {e}")
        return []

def chunk_text(text_list, chunk_size=500):
    chunks = []
    for text in text_list:
        if text.strip():
            chunks.extend([text[i:i+chunk_size] for i in range(0, len(text), chunk_size)])
    return chunks

def add_data_to_faiss_db(faiss_index, text_chunks):
    try:
        embeddings = embedding_model.encode(text_chunks, convert_to_tensor=False)
        faiss_index.add(embeddings)
    except Exception as e:
        st.error(f"Error adding data to FAISS: {e}")

def summarize_document_with_mistral(context):
    prompt_template = """
    You are a highly skilled document summarization assistant. Using the provided context, generate a concise and informative summary of the document. The summary should highlight the main points and key details while maintaining clarity and coherence.

    Context:
    {context}

    Summary:
    """
    prompt = prompt_template.format(context=context)

    message = [{"role": "user", "content": prompt}]
    chat_response = client.chat.complete(
        model=model,
        messages=message
    )

    return chat_response.choices[0].message.content.strip()

def summarize_document(faiss_index, text_chunks):
    context = "\n".join(text_chunks)
    summary = summarize_document_with_mistral(context)
    return summary


def main():
    st.set_page_config("Website Summarizer")
    st.header("Website Summarizer🌐")

    # Initialize session states
    if 'uploaded' not in st.session_state:
        st.session_state.uploaded = False
    if 'text_chunks' not in st.session_state:
        st.session_state.text_chunks = []

    # Initial text and input box
    st.write("### Text: Upload Website URL")
    url = st.text_input("Enter the URL here")

    if url:
        if not st.session_state.uploaded:
            if st.button("Upload to FAISS DB"):
                with st.spinner("Fetching content from URL..."):
                    text_list = extract_text_from_url(url)

                if not text_list:
                    st.error("No text extracted from the provided URL.")
                else:
                    st.session_state.text_chunks = chunk_text(text_list)
                    with st.spinner("Uploading to FAISS DB..."):
                        add_data_to_faiss_db(faiss_index, st.session_state.text_chunks)
                        st.session_state.uploaded = True
                        st.success("URL successfully added to FAISS DB")

        if st.session_state.uploaded:
            if st.button("Summarize URL"):
                with st.spinner("Summarizing content..."):
                    summary = summarize_document(faiss_index, st.session_state.text_chunks)
                    st.write("## Summary:")
                    st.write(summary)
    else:
        st.info("Please upload the URL to begin.")

main()
