from langchain_community.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import (
    StreamingStdOutCallbackHandler,
)  # for streaming resposne
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA



loader = PyPDFDirectoryLoader("../GPT_data/PDFd")
documents = loader.load_and_split()
len(documents)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,
    chunk_overlap=64
)
texts = text_splitter.split_documents(documents)
len(texts)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma.from_documents(texts, embeddings, persist_directory="db")

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
model_path = "../models/llama-2-7b.Q8_0.gguf"
n_gpu_layers = -1
n_batch = 512
llm = LlamaCpp(
    model_path=model_path,
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    callback_manager=callback_manager,
    verbose=False,
    # temperature=1
)
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    verbose=False,
)

res = qa(f"""
   Do I get a free famcard if I have a ultra membership? 
""")
