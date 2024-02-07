from langchain_community.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import (
    StreamingStdOutCallbackHandler,
)  # for streaming resposne

from langchain_community.document_loaders import UnstructuredMarkdownLoader

markdown_path = "../lottie-docs/docs/Introduction.md"

loader = UnstructuredMarkdownLoader(markdown_path)

data = loader.load()

model_path = "models/mistral-7b-instruct-v0.1.Q4_0.gguf"
template = """Question: {question}
Answer: """
prompt = PromptTemplate(template=template, input_variables=["question"])
# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
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
llm_chain = LLMChain(prompt=prompt, llm=llm)
question = "tell me things that ai can do"
llm_chain.run(question)
