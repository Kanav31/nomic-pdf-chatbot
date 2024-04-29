import PyPDF2
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
import chainlit as cl
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

# Loading environment variables from .env file
load_dotenv() 

# Function to initialize conversation chain with GROQ language model
groq_api_key = os.environ['GROQ_API_KEY']

# Initializing GROQ chat with provided API key, model name, and settings
llm_groq = ChatGroq(
    groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768",
    temperature=0.2
)

# Define chat bot name and image
chat_bot_name = "EmpSmart Bot"
chat_bot_image_path = "./robot.jpeg"

@cl.on_chat_start
async def on_chat_start():
    # Specify the path to the PDF file you want to process
    pdf_file_path = "./pdfs/HR_Policy_Manual.pdf"

    # Read the PDF file
    pdf = PyPDF2.PdfReader(pdf_file_path)
    pdf_text = ""
    for page in pdf.pages:
        pdf_text += page.extract_text()

    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=50)
    texts = text_splitter.split_text(pdf_text)

    # Create a metadata for each chunk
    metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]

    # Create a Chroma vector store
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    docsearch = await cl.make_async(Chroma.from_texts)(
        texts, embeddings, metadatas=metadatas
    )
    
    # Initialize message history for conversation
    message_history = ChatMessageHistory()
    
    # Memory for conversational context
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    # Create a chain that uses the Chroma vector store
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm_groq,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        memory=memory,
        return_source_documents=True,
    )

    # Let the user know that the system is ready
    msg = cl.Message(content=f"Processing pdf done. You can now ask questions!", elements=[
        cl.Text(content=f"Welcome to {chat_bot_name}! How can I assist you today?"),
        cl.Image(name="bot_image", display="inline", path=chat_bot_image_path)
    ])
    await msg.send()
    
    # Store the chain in user session
    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):
     # Retrieve the chain from user session
    chain = cl.user_session.get("chain") 
    # Callbacks happen asynchronously/parallel 
    cb = cl.AsyncLangchainCallbackHandler()
    
    # Call the chain with user's message content
    res = await chain.ainvoke(message.content, callbacks=[cb])
    answer = res["answer"]
    source_documents = res["source_documents"] 

    text_elements = [] # Initialize list to store text elements
    
    # Process source documents if available
    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            # Create the text element referenced in the message
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=source_name)
            )
        source_names = [text_el.name for text_el in text_elements]
        
         # Add source references to the answer
        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"
    # Return results
    await cl.Message(content=answer, elements=text_elements).send()

# @cl.on_message
# async def main(message: cl.Message):
#     if cl.context.session.client_type == "copilot":
#         fn = cl.CopilotFunction(name="test", args={"msg": message.content})
#         res = await fn.acall()
#         await cl.Message(content=res).send()
#     else:
#         # Retrieve the chain from user session
#         chain = cl.user_session.get("chain") 
#         # Callbacks happen asynchronously/parallel 
#         cb = cl.AsyncLangchainCallbackHandler()
        
#         # Call the chain with user's message content
#         res = await chain.ainvoke(message.content, callbacks=[cb])
#         answer = res["answer"]
#         source_documents = res["source_documents"] 

#         text_elements = [] # Initialize list to store text elements
        
#         # Process source documents if available
#         if source_documents:
#             for source_idx, source_doc in enumerate(source_documents):
#                 source_name = f"source_{source_idx}"
#                 # Create the text element referenced in the message
#                 text_elements.append(
#                     cl.Text(content=source_doc.page_content, name=source_name)
#                 )
#             source_names = [text_el.name for text_el in text_elements]
            
#             # Add source references to the answer
#             if source_names:
#                 answer += f"\nSources: {', '.join(source_names)}"
#             else:
#                 answer += "\nNo sources found"
#         # Return results
#         await cl.Message(content=answer, elements=text_elements).send()