from langchain_aws import ChatBedrock, BedrockEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from pinecone import Pinecone
import os
from dotenv import load_dotenv

load_dotenv()

def get_chatbot():
    # Initialize Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    # Load embeddings
    embeddings = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v2:0",
        region_name="us-east-2"
    )

    # Connect to Pinecone vector store
    vectorstore = PineconeVectorStore(
        index_name=os.getenv("PINECONE_INDEX"),
        embedding=embeddings
    )

    # Load Claude via AWS Bedrock
    llm = ChatBedrock(
        model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0",
        region_name="us-east-2",
        model_kwargs={"max_tokens": 1000}
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    return {"llm": llm, "retriever": retriever, "history": []}

def ask(chatbot, question):
    retriever = chatbot["retriever"]
    llm = chatbot["llm"]
    history = chatbot["history"]

    # Get relevant docs
    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])

    # Build messages
    messages = [
        ("system", f"You are a helpful AI assistant. Use the following context to answer questions.\n\nContext:\n{context}"),
    ]
    
    # Add history
    for h in history:
        messages.append(("human", h["question"]))
        messages.append(("assistant", h["answer"]))
    
    messages.append(("human", question))

    # Get response
    prompt = ChatPromptTemplate.from_messages(messages)
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({})

    # Save to history
    chatbot["history"].append({"question": question, "answer": answer})

    return answer