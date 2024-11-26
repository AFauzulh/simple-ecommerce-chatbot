import os
import json
import requests

import numpy as np

from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.tools import tool

import streamlit as st

from agents import get_agent_tools, refine_query, agent_executor_invoker


if "llm" not in st.session_state:
    st.session_state.llm = ChatOllama(model="llama3.2", base_url="http://host.docker.internal:11434")
    st.session_state.prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer the following question, as if you are a ecommerce robo-advisor. Answer the question based on the result of the provided tools. If the tools doesn't result anything, please say so and say that you will using your own knowledge instead.",
            ),
            (
                "human", 
                "{input}"
            ),
            MessagesPlaceholder("agent_scratchpad")
        ]
    )
    
    st.session_state.tools = get_agent_tools()
    st.session_state.agent = create_tool_calling_agent(st.session_state.llm, st.session_state.tools, st.session_state.prompt)
    st.session_state.agent_executor = AgentExecutor(agent=st.session_state.agent, tools=st.session_state.tools, verbose=True, return_intermediate_steps=True)
    
st.title("Simple Ecommerce Chatbot")
query = st.text_input("Ask anything about our ecommerce")

if query:
    query = refine_query(query)
    response = st.session_state.agent_executor.invoke({"input": query})
    st.write(response["output"])

    with st.expander("Retrieved Information"):
        st.write(response["intermediate_steps"])