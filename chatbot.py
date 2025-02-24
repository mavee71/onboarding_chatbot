#import streamlit
import streamlit as st
import os
from dotenv import load_dotenv

# import pinecone
from pinecone import Pinecone, ServerlessSpec

# import langchain
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# Streamlit app layout
st.title("Oracle ERP FCS Customer Onboarding Assistant")
st.write("Interact with the assistant to obtain information about the onboarding process:")

# initialize pinecone database
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# initialize pinecone database
index = os.environ['PINECONE_INDEX_NAME']
namespace = os.environ['PINECONE_NAMESPACE_NAME']

# initialize embeddings model + vector store
embeddings = OpenAIEmbeddings(model="text-embedding-3-large",api_key=os.environ.get("OPENAI_API_KEY"))
vector_store = PineconeVectorStore.from_existing_index(index_name=index, embedding=embeddings, namespace=namespace)

# initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

    st.session_state.messages.append(SystemMessage("You are a world class assistant for question-answering tasks. "))

# display chat messages from history on app rerun
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# create the bar where we can type messages
prompt = st.chat_input("Please, ask your question or type 'quit' to end the session.")

# did the user submit a prompt?
if prompt:
    # check if the user wants to end the session
    if prompt.lower() == 'quit':
        st.write("Session ended by user. Goodbye!")
        st.stop()

    # add the message from the user (prompt) to the screen with streamlit
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append(HumanMessage(prompt))

    # initialize the llm
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7
    )

    # creating and invoking the retriever
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.5},
    )

    docs = retriever.invoke(prompt)
    docs_text = "".join(d.page_content for d in docs)

    # creating the system prompt
    system_prompt = """You are world class assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question abstractly. Do not simply repeat the context. 
    If you don't know the answer, politely inform the question is outside of your knowledge scope. 
    Context: {context}:"""

    # Populate the system prompt with the retrieved context
    system_prompt_fmt = system_prompt.format(context=docs_text)

    print("-- SYS PROMPT --")
    print(system_prompt_fmt)

    # adding the system prompt to the message history
    st.session_state.messages.append(SystemMessage(system_prompt_fmt))

    # invoking the llm
    result = llm.invoke(st.session_state.messages).content

    # adding the response from the llm to the screen (and chat)
    with st.chat_message("assistant"):
        st.markdown(result)

    st.session_state.messages.append(AIMessage(result))