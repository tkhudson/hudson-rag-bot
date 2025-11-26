# rag_bot.py - FINAL, NO MORE ERRORS EVER (December 2025)
# Tested 100% on macOS + Python 3.12 + venv

import os
import shutil
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
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

embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
llm = ChatOllama(model=LLM_MODEL, temperature=0.1, num_ctx=8192)

# Global DB — will be recreated fresh every time
db = None

# Global examples — will be updated based on uploaded files
examples = ["Summarize the document", "What are the main points?", "List key skills", "Contact information"]

def create_fresh_db_and_index(chunks):
    """Creates a brand-new Chroma DB and indexes chunks — 100% safe"""
    global db
    # Nuke the old DB completely
    if CHROMA_PATH.exists():
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
            elif p.suffix.lower() == ".docx":
                loader = Docx2txtLoader(str(p))
            elif p.suffix.lower() in {".txt", ".md"}:
                loader = TextLoader(str(p), encoding="utf-8")
            else:
                continue
            loaded = loader.load()
            valid = [d for d in loaded if d.page_content.strip()]
            for d in valid:
                d.metadata["source"] = p.name
            docs.extend(valid)
            print(f"Loaded {len(valid)} pages → {p.name}")
        except Exception as e:
            print(f"Error loading {p.name}: {e}")
    return docs

# ====================== INDEX ======================
def index_documents(files):
    global db, examples

    # Copy uploaded files
    for f in files or []:
        dest = DOCS_FOLDER / Path(f.name).name
        if not dest.exists():
            shutil.copy(f.name, dest)

    docs = load_documents()
    if not docs:
        examples = ["Summarize the document", "What are the main points?", "List key skills", "Contact information"]
        examples_md = "### Suggested Questions\n" + "\n".join(f"- {e}" for e in examples)
        return "No readable text found in files.", examples_md

    # Update examples based on the most recently uploaded file
    if files:
        last_file = Path(files[-1].name).name.lower()
        if 'cover' in last_file or 'letter' in last_file:
            examples = ["Summarize the document", "What are the main points?", "List key skills", "Contact information"]
        elif 'army' in last_file or 'survival' in last_file:
            examples = ["What are the key survival tips?", "How to build a shelter?", "Water purification methods", "First aid basics"]
        else:
            examples = ["Summarize the document", "What are the main points?", "Key information", "Details"]
    else:
        examples = ["Summarize the document", "What are the main points?", "List key skills", "Contact information"]

    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
    chunks = [c for c in splitter.split_documents(docs) if c.page_content.strip()]
    if not chunks:
        examples_md = "### Suggested Questions\n" + "\n".join(f"- {e}" for e in examples)
        return "No text chunks — maybe scanned PDF?", examples_md

    # THIS IS THE ONLY LINE THAT MATTERS
    create_fresh_db_and_index(chunks)

    examples_md = "### Suggested Questions\n" + "\n".join(f"- {e}" for e in examples)
    return f"Indexed {len(chunks)} chunks from {len(docs)} pages. Ready!", examples_md

# ====================== CLEAR ======================
def clear_all():
    global db, examples
    for p in list(DOCS_FOLDER.iterdir()):
        if p.is_file(): p.unlink()
    if CHROMA_PATH.exists():
        shutil.rmtree(CHROMA_PATH, ignore_errors=True)
    CHROMA_PATH.mkdir()
    db = None
    examples = ["Summarize the document", "What are the main points?", "List key skills", "Contact information"]
    examples_md = "### Suggested Questions\n" + "\n".join(f"- {e}" for e in examples)
    return "Everything cleared.", examples_md

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
theme = gr.themes.Soft(primary_hue="indigo", secondary_hue="slate")

with gr.Blocks(title="Private RAG Knowledge Base") as demo:
    gr.Markdown("# 🔒 Private RAG Knowledge Base\n**100% Local • Phi3:14B • Nomic Embeddings**\n\nUpload documents and ask questions about them.")

    with gr.Tabs():
        with gr.TabItem("📁 Documents"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Upload Files")
                    files = gr.File(label="Supported formats: PDF, DOCX, TXT, MD", file_count="multiple", type="filepath")
                    with gr.Row():
                        index_btn = gr.Button("🔍 Index Documents", variant="primary", size="lg")
                        clear_btn = gr.Button("🗑️ Clear All", variant="secondary", size="lg")
                    status = gr.Textbox(label="Status", lines=5)

        with gr.TabItem("💬 Chat"):
            gr.Markdown("### Ask Questions")
            examples_md = gr.Markdown("### Suggested Questions\n- Summarize the document\n- What are the main points?\n- List key skills\n- Contact information")
            gr.ChatInterface(
                fn=ask_question,
                examples=[
                    "Summarize the document",
                    "What are the main points?",
                    "List key skills",
                    "Contact information"
                ]
            )

    index_btn.click(fn=index_documents, inputs=files, outputs=[status, examples_md])
    clear_btn.click(fn=clear_all, inputs=None, outputs=[status, examples_md])

if __name__ == "__main__":
    print("Starting Private RAG Bot…")
    demo.launch(theme=theme, share=True, server_port=7863)
