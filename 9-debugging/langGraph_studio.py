from typing import Annotated
from typing_extensions import TypedDict
from langchain_groq import ChatGroq
from langgraph.graph import START, END
from langgraph.graph.state import StateGraph
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from langchain_community.tools import tool

import os
from dotenv import load_dotenv

load_dotenv()

os.environ["GROQ_API_KEY"] = os.environ.get("GROQ_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.environ.get("LANGCHAIN_API_KEY")

llm = ChatGroq(model="qwen/qwen3-32b")


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def make_default_graph():

    # create a chatbot node...
    def llm_tool(state: State):
        return {"messages": [llm.invoke(state["messages"])]}

    builder = StateGraph(State)

    builder.add_node(llm_tool, "llm_tool")

    builder.add_edge(START, "llm_tool")
    builder.add_edge("llm_tool", END)

    graph = builder.compile()
    return graph


def graph_with_tools():
    """Make a tool calling agent"""

    @tool
    def add(a: int, b: int):
        """add two numbers"""
        return a + b

    llm_with_tools = llm.bind_tools([add])

    # creating a chatbot node
    def llm_tool(state: State):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    builder = StateGraph(State)

    builder.add_node(llm_tool, "llm_tool")
    builder.add_node("tools", ToolNode([add]))

    builder.add_edge(START, "llm_tool")
    builder.add_conditional_edges(
        "llm_tool",
        # if the latest message from LLM is a toolcall -> tools_condition route to tools
        # if the latest message from LLM is not a toolcall -> tools_condition route to END
        tools_condition,
    )
    builder.add_edge("tools", "llm_tool")

    graph = builder.compile()
    return graph


agent = graph_with_tools()
