import os
import gradio as gr
import pandas as pd
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document

# -------------------
# Load environment
# -------------------
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# -------------------
# Load Excel Data
# -------------------
def load_excel(file_path="data.xlsx"):
    df = pd.read_excel(file_path)
    docs = []
    for _, row in df.iterrows():
        text = f"Q: {row['Question']}\nA: {row['Answer']}"
        docs.append(Document(page_content=text))
    return docs

documents = load_excel()

# Split into chunks
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# -------------------
# Local embeddings (BAAI bge model)
# -------------------
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en",   # can change to bge-base-en or bge-large-en
    model_kwargs={'device': 'cpu'},   # use 'cuda' if you have GPU
    encode_kwargs={'normalize_embeddings': True}  # recommended for BGE
)
vectorstore = FAISS.from_documents(docs, embeddings)

# -------------------
# LLM (Groq)
# -------------------
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=groq_api_key
)

# -------------------
# Conversational RAG
# -------------------
memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory,
    verbose=True
)

# -------------------
# Gradio Chat Function
# -------------------
def chat(user_input, history):
    result = qa_chain({"question": user_input})
    answer = result["answer"]
    history.append((user_input, answer))
    return history, history

# -------------------
# Gradio UI
# -------------------
with gr.Blocks() as demo:
    gr.Markdown("## 📊 LangChain + Groq + Excel RAG Chatbot (BAAI Embeddings)")

    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="Ask a question about the Excel data...")
    clear = gr.Button("Clear Chat")

    state = gr.State([])

    msg.submit(chat, [msg, state], [chatbot, state])
    clear.click(lambda: ([], []), None, [chatbot, state])

if __name__ == "__main__":
    demo.launch()
