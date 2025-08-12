from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os

# Make sure your API key is set
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

# Create LLM instance
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Create a prompt template
prompt = PromptTemplate(
    input_variables=["question"],
    template="You are a helpful assistant. Answer the following question:\n{question}"
)

# Create a chain
chain = LLMChain(llm=llm, prompt=prompt)

if __name__ == "__main__":
    while True:
        question = input("Ask a question (or type 'quit' to exit): ")
        if question.lower() in ["quit", "exit"]:
            break
        response = chain.run(question)
        print("Answer:", response)
