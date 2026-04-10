# Import standard libraries for file handling and text processing
import os, pathlib, textwrap, glob

# Load documents from various sources (URLs, text files, PDFs)
from langchain_community.document_loaders import UnstructuredURLLoader, TextLoader, PyPDFLoader

# Split long texts into smaller, manageable chunks for embedding
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Vector store to store and retrieve embeddings efficiently using FAISS
from langchain.vectorstores import FAISS

#text embeddings using OpenAI or Hugging Face models
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings, SentenceTransformerEmbeddings

# local LLMs (e.g., via Ollama) for response generation
from langchain.llms import Ollama

#a retrieval chain that combines a retriever, a prompt, and an LLM
from langchain.chains import ConversationalRetrievalChain

# prompts for the RAG system
from langchain.prompts import PromptTemplate

