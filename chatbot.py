
import os, pathlib, textwrap, glob


from langchain_community.document_loaders import UnstructuredURLLoader, TextLoader, PyPDFLoader


from langchain.text_splitter import RecursiveCharacterTextSplitter


from langchain.vectorstores import FAISS


from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings, SentenceTransformerEmbeddings


from langchain.llms import Ollama
from langchain_ollama import OllamaLLM


from langchain.chains import ConversationalRetrievalChain


from langchain.prompts import PromptTemplate

try:

    loader = UnstructuredURLLoader(URLS)
    raw_docs = loader.load()
    print(f"Fetched {len(raw_docs)} documents from the web.")
except Exception as e:
    print("⚠️  Web fetch failed, using offline copies:", e)
    raw_docs = []
    loader = UnstructuredHTMLLoader("/Shipping _ BigCommerce Developer Center.html")
    data = loader.load() 
    print(f"Loaded {len(raw_docs)} offline documents.")

chunks = []

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 300,
    chunk_overlap  = 30,
    length_function = len,
)

chunks = text_splitter.split_documents(raw_docs)


print(f"✅ {len(chunks)} chunks ready for embedding")

# Load an embedding model and test it by embedding a random sentence "Hello world!""
embedding_model = SentenceTransformerEmbeddings(model_name="thenlper/gte-small")
query = "Hello World!"
query_vector = embedding_model.embed_query(query)

print(f"Vector length: {len(query_vector)}")
print(f"First 5 values: {query_vector[:5]}")

vector_store = FAISS.from_documents(chunks, embedding_model)
vector_store.save_local("faiss_index_gte_small")
print(f"Total number of embeddings stored: {vector_store.index.ntotal}")

llm = Ollama(
    model="gemma3:1b",
    temperature=0.1
)
response = llm.invoke("Hello! This is a test to verify you are running correctly.")
print(response)

SYSTEM_TEMPLATE = """
You are a **Customer Support Chatbot**. Use only the information in CONTEXT to answer.
If the answer is not in CONTEXT, respond with “I'm not sure from the docs.”

Rules:
1) Use ONLY the provided <context> to answer.
2) If the answer is not in the context, say: "I don't know based on the retrieved documents."
3) Be concise and accurate. Prefer quoting key phrases from the context.
4) When possible, cite sources as [source: source] using the metadata.

CONTEXT:
{context}

USER:
{question}
"""

retriever = vector_store.as_retriever(search_type="similarity",
    search_kwargs={"k": 5} )
prompt = ChatPromptTemplate.from_template(SYSTEM_TEMPLATE)

def rag_step(question: str):
    # Run one RAG pass (retrieve -> format context -> prompt -> LLM)
    # Step 1: Retrieve the top-k relevant documents for the question
    # Step 2: format them into a context string using `format_docs` function
    # Step 3: format the prompt template with the context and the question
    # Step 4: Call the LLM and return both the answer and the source docs
    source_docs = retriever.invoke(question)
    context_string = format_docs(source_docs)
    final_prompt = prompt.format(context=context_string, question=question)
    answer = llm.invoke(final_prompt)

    return {'answer':answer, 'source_docs':source_docs}


def format_docs(docs):
    # Turn retrieved documents into a single context string
    return "\n\n".join(doc.page_content for doc in docs)

test_questions = [
    "If I'm not happy with my purchase, what is your refund policy and how do I start a return?",
    "How long will delivery take for a standard order, and where can I track my package once it ships?",
    "What's the quickest way to contact your support team, and what are your operating hours?",
]

for q in test_questions:
    result = rag_step(q)
    print(f"\nQ: {q}\nA: {result['answer']}...")





