from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

def summarize_pdf(file_path):
    # Initialize the PDF loader
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    # Initialize the language model
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

    # Create a custom prompt with a system message
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an assistant that summarizes documents. "
                   "Focus only on the topic: 'Illustration of the sampling process of a student-corpus pair'."),
        ("human", "{text}")
    ])

    # Load the summarization chain with custom prompt
    chain = load_summarize_chain(llm, chain_type="map_reduce", combine_prompt=prompt)
    
    # Run the chain on the documents
    summary = chain.invoke(documents)
    
    return summary

if __name__ == "__main__":
    summary = summarize_pdf("docs/test.pdf")

    print("Summary of the PDF:")
    print(summary['output_text'])
