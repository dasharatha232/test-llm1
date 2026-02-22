import os
import gradio as gr
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

from langchain.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
    TextLoader
)

# -------------------
# Load environment
# -------------------
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=groq_api_key
)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# -------------------
# Process File
# -------------------
def process_file(file):
    if file is None:
        return None, gr.update(interactive=False), gr.update(interactive=False)

    file_path = file.name

    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".docx"):
        loader = UnstructuredWordDocumentLoader(file_path)
    elif file_path.endswith(".xlsx") or file_path.endswith(".xls"):
        loader = UnstructuredExcelLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)
    else:
        raise ValueError("Unsupported file type")

    documents = loader.load()

    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    docs = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    vectorstore = FAISS.from_documents(docs, embeddings)

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        verbose=False
    )

    # ✅ Enable chat input
    # ❌ Disable process button after completion
    return qa_chain, gr.update(interactive=True), gr.update(interactive=False)

# -------------------
# Enable Process Button When File Uploaded
# -------------------
def enable_process(file):
    if file is None:
        return gr.update(interactive=False)
    return gr.update(interactive=True)

# -------------------
# Chat Function
# -------------------
def chat(user_input, history, qa_chain):
    if qa_chain is None:
        history.append((user_input, "⚠️ Please upload and process a document first."))
        return history, history

    result = qa_chain({"question": user_input})
    answer = result["answer"]

    history.append((user_input, answer))
    return history, history,gr.update(value="")

# -------------------
# UI
# -------------------
with gr.Blocks() as demo:
    gr.Markdown("## 📄 Upload Document RAG Chatbot")

    file_input = gr.File(label="Upload Document")
    process_btn = gr.Button("Process Document", interactive=False)

    chatbot = gr.Chatbot()
    msg = gr.Textbox(
        placeholder="Ask a question...",
        interactive=False   # disabled initially
    )

    clear = gr.Button("Clear Chat")

    state = gr.State([])
    qa_state = gr.State(None)

    # Enable process button only when file is uploaded
    file_input.change(
        enable_process,
        inputs=file_input,
        outputs=process_btn
    )

    # Process file
    process_btn.click(
        process_file,
        inputs=file_input,
        outputs=[qa_state, msg, process_btn]
    )

    # Chat submit
    msg.submit(
        chat,
        inputs=[msg, state, qa_state],
        outputs=[chatbot, state, msg]
    )

    # Clear chat
    clear.click(lambda: [], None, chatbot)

if __name__ == "__main__":
    demo.launch()
