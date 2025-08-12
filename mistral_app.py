from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import Ollama  # Local LLM from Ollama

# Load the local LLM (change "mistral" to "llama2" if you pulled that)
llm = Ollama(model="mistral")

# Create the prompt template
prompt = PromptTemplate(
    input_variables=["question"],
    template="You are a helpful assistant. Answer the following question clearly:\n{question}"
)

# Create the chain
chain = LLMChain(llm=llm, prompt=prompt)

# Run interactive Q&A
while True:
    question = input("Ask a question (or type 'quit' to exit): ")
    if question.lower() == "quit":
        break
    response = chain.invoke({"question": question})  # new method instead of run()
    print("Answer:", response["text"])
