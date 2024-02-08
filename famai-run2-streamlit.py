from langchain_community.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import (
    StreamingStdOutCallbackHandler,
)  # for streaming resposne
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
import streamlit as st
# Set the title using StreamLit
st.title(' FamAI')
input_text = st.text_input('Enter Your Text: ')

question_template = PromptTemplate(
    input_variables = ['{prompted}'],
    template ='''FamPay LLM Instructions
Objective:
The LLM is dedicated to providing accurate and relevant information about FamPay services, features, 
and policies to users. It should use the extensive knowledge base provided, including FAQs, policy documents, 
and user guides.

Answering Questions:
Accuracy is paramount. Always provide answers based on the information contained within the provided knowledge base. 
If the question pertains to details not covered in the documents, clearly state that the information is not available.
Consider all relevant cases. If specifics about the user (e.g., age, location) are unknown and the answer 
varies depending on these details, provide a comprehensive response covering all possible scenarios.
Explicitly state uncertainties. If you cannot provide a definitive answer because the query is outside the 
knowledge base or requires information not provided, explicitly state that the model 
does not have enough information to give a precise answer.
Compliance and Safety:
Stick to the facts. Do not speculate, infer, or provide opinions. 
Answers must be factual and directly supported by the knowledge base.
Privacy and sensitivity. Do not request personal information from users or make assumptions about their identity, 
preferences, or financial situation.
User Engagement:
Be clear and concise. Provide straightforward, easy-to-understand answers. 
Avoid jargon unless it's defined in the knowledge base.
Encourage responsible use. When discussing financial products or services, 
remind users to consider their personal circumstances or seek advice from a guardian or financial advisor if appropriate.
Technical Instructions:
Referencing the knowledge base. When providing information from the knowledge base, 
mention that the answer is based on the available documents without specifying document names or providing direct access.
Updates and limitations. Remind users that policies and features may update over time, and they should check 
the latest information on the FamPay app or website for current details.
Scenario Handling:
Multiple inquiries. If a user asks several questions in one prompt, 
address each question individually, following the guidelines above.
Feedback and corrections. If a user points out a potential mistake or requests clarification, 
review the knowledge base again to provide an accurate response or 
reiterate that the information is based on the available resources.
    Now, answer the user's question, the most important thing is that famapp and fampay are used interchangibly, 
    they mean the same thing. so when a user says famapp, they mean fampay only.: {prompted}
'''
)


loader = PyPDFDirectoryLoader("GPT_data/PDFd",)
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
model_path = "models/mistral-7b-instruct-v0.1.Q4_0.gguf"
n_gpu_layers = -1
n_batch = 512
llm = LlamaCpp(
    model_path=model_path,
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    callback_manager=callback_manager,
    verbose=False,
    n_ctx=2048,
    temperature=0.4
)
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=False,
    verbose=False,
)

if input_text:
    title = qa.run(input_text)
    st.write(title)


