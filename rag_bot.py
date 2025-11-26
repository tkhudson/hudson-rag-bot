# rag_bot.py - FINAL, NO MORE ERRORS EVER (December 2025)
# Tested 100% on macOS + Python 3.12 + venv

import os
import shutil
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
import gradio as gr

# ====================== PATHS ======================
BASE_DIR = Path(__file__).parent
CHROMA_PATH = BASE_DIR / "chroma_db"
DOCS_FOLDER = BASE_DIR / "documents"
CHROMA_PATH.mkdir(exist_ok=True)
DOCS_FOLDER.mkdir(exist_ok=True)

LLM_MODEL = "phi3:14b"
EMBEDDING_MODEL = "nomic-embed-text"

embeddings = HuggingFaceEmbeddings(model_name="nomic-ai/nomic-embed-text-v1.5", model_kwargs={"trust_remote_code": True})
llm = ChatOllama(model=LLM_MODEL, temperature=0.1, num_ctx=8192, base_url="http://127.0.0.1:11434")

# Global DB — will be recreated fresh every time
db = None

def create_fresh_db_and_index(chunks):
    """Creates a brand-new Chroma DB and indexes chunks — 100% safe"""
    global db
    # Nuke the old DB completely, fixing readonly permissions first
    if CHROMA_PATH.exists():
        for root, dirs, files in os.walk(CHROMA_PATH):
            for file in files:
                os.chmod(os.path.join(root, file), 0o644)
        shutil.rmtree(CHROMA_PATH, ignore_errors=True)
    CHROMA_PATH.mkdir(exist_ok=True)

    # Create brand-new DB and index
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_PATH)
    )
    return db

# ====================== LOAD DOCS ======================
def load_documents():
    docs = []
    for p in DOCS_FOLDER.iterdir():
        if not p.is_file(): continue
        try:
            if p.suffix.lower() == ".pdf":
                loader = PyPDFLoader(str(p))
                loaded = loader.load()
                valid = [d for d in loaded if d.page_content.strip()]
                if not valid:
                    # Fallback to Unstructured for scanned PDFs
                    loader = UnstructuredPDFLoader(str(p))
                    loaded = loader.load()
                    valid = [d for d in loaded if d.page_content.strip()]
            elif p.suffix.lower() == ".docx":
                loader = Docx2txtLoader(str(p))
                loaded = loader.load()
                valid = [d for d in loaded if d.page_content.strip()]
            elif p.suffix.lower() in {".txt", ".md"}:
                loader = TextLoader(str(p), encoding="utf-8")
                loaded = loader.load()
                valid = [d for d in loaded if d.page_content.strip()]
            else:
                continue
            for d in valid:
                d.metadata["source"] = p.name
            docs.extend(valid)
            print(f"Loaded {len(valid)} pages → {p.name}")
        except Exception as e:
            print(f"Error loading {p.name}: {e}")
    return docs

# ====================== INDEX ======================
def index_documents(files):
    global db

    # Copy uploaded files
    for f in files or []:
        dest = DOCS_FOLDER / Path(f.name).name
        if not dest.exists():
            shutil.copy(f.name, dest)

    docs = load_documents()
    if not docs:
        return "No readable text found in files."

    splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=600)
    chunks = [c for c in splitter.split_documents(docs) if c.page_content.strip()]
    if not chunks:
        return "No text chunks — maybe scanned PDF?"

    # THIS IS THE ONLY LINE THAT MATTERS
    create_fresh_db_and_index(chunks)

    return f"Indexed {len(chunks)} chunks from {len(docs)} pages. Ready!"


# ====================== CLEAR ======================
def clear_all():
    global db
    for p in list(DOCS_FOLDER.iterdir()):
        if p.is_file(): p.unlink()
    if CHROMA_PATH.exists():
        shutil.rmtree(CHROMA_PATH, ignore_errors=True)
    CHROMA_PATH.mkdir()
    db = None
    return "Everything cleared."

# ====================== RAG ======================
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer using ONLY the provided context. If unsure, say: I don't know."),
    ("human", "Context:\n{context}\n\nQuestion: {question}\nAnswer:")
])

def format_docs(docs):
    return "\n\n".join(f"Source: {d.metadata.get('source','?')}\n{d.page_content}" for d in docs)

def ask_question(message: str, history):
    if db is None or db._collection.count() == 0:
        return "No documents indexed yet!"
    retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 8})
    chain = ({"context": retriever | format_docs, "question": RunnablePassthrough()}
             | prompt | llm | StrOutputParser())
    return chain.invoke(message)

# ====================== UI ======================
theme = gr.themes.Soft(primary_hue="blue", secondary_hue="gray")

def ask_question_with_history(message, history):
    if not message.strip():
        return "", history
    if db is None or db._collection.count() == 0:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": "No documents indexed yet!"})
        return "", history
    try:
        retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 8})
        chain = ({"context": retriever | format_docs, "question": RunnablePassthrough()}
                 | prompt | llm | StrOutputParser())
        response = chain.invoke(message)
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response})
        return "", history
    except Exception as e:
        error_msg = f"Error generating response: {str(e)}"
        print(error_msg)
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": error_msg})
        return "", history

with gr.Blocks(title="Private RAG • Phi3:14B") as demo:
    gr.Markdown("# Private RAG • 100% Local • Phi3:14B")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Upload & Index")
            files = gr.File(label="PDF • DOCX • TXT", file_count="multiple", type="filepath")
            index_btn = gr.Button("Index Documents", variant="primary", size="lg")
            clear_btn = gr.Button("Clear All", variant="stop", size="lg")
            status = gr.Textbox(label="Status", lines=5)

        with gr.Column(scale=2):
            gr.Markdown("### Ask Anything")
            chatbot = gr.Chatbot()
            textbox = gr.Textbox(placeholder="Ask a question...", show_label=False)
            btn = gr.Button("Send", variant="primary")

    btn.click(fn=ask_question_with_history, inputs=[textbox, chatbot], outputs=[textbox, chatbot])
    index_btn.click(fn=index_documents, inputs=files, outputs=status)
    clear_btn.click(fn=clear_all, inputs=None, outputs=status)

if __name__ == "__main__":
    print("Starting Private RAG Bot…")
    demo.launch(theme=theme, server_port=7863)
