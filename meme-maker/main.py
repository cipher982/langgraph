import os
import logging
from typing import TypedDict, Annotated, Sequence, Literal
from functools import lru_cache
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, END, add_messages

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use environment variables for sensitive information
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# tools = [TavilySearchResults(max_results=1, api_key=TAVILY_API_KEY)]
tools = []

@lru_cache(maxsize=2)
def _get_model(model_name: str,temperature: float, max_tokens: int) -> ChatOpenAI:
    model = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        openai_api_key=OPENAI_API_KEY
    )
    return model
    # return model.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

class GraphConfig(TypedDict):
    model_name: str
    temperature: float
    max_tokens: int

def get_config(config: GraphConfig) -> GraphConfig:
    defaults = {
        "model_name": "gpt-4o-mini",
        "temperature": 0.9,
        "max_tokens": 256,
    }
    return {**defaults, **config}

def agent1_node(state: AgentState, config: GraphConfig) -> AgentState:
    messages = state["messages"]
    full_config = get_config(config)
    model = _get_model(full_config["model_name"], full_config["temperature"], full_config["max_tokens"])
    
    if len(messages) == 1:  # Initial prompt
        human_message = messages[0].content
        prompt = f"We're creating a meme about {human_message}. Let's search for some information and brainstorm funny ideas. What do you suggest?"
    else:
        last_message = messages[-1].content
        prompt = f"Based on our previous discussion, let's continue developing our meme idea. Use the search tool if needed. {last_message}"
    
    response = model.invoke([HumanMessage(content=prompt)])
    logger.info(f"Agent 1: {response.content}")
    return {"messages": state["messages"] + [response]}

def agent2_node(state: AgentState, config: GraphConfig) -> AgentState:
    messages = state["messages"]
    full_config = get_config(config)
    model = _get_model(full_config["model_name"], full_config["temperature"], full_config["max_tokens"])
    last_message = messages[-1].content
    
    if "FINAL_MEME:" not in last_message:
        prompt = f"Great ideas! Now, let's finalize our meme. Create a funny meme based on our discussion about {messages[0].content}. Use the search tool if needed for additional context or inspiration. Start your response with 'FINAL_MEME:'"
    else:
        prompt = "The meme is complete. Let's end the process."
    
    response = model.invoke([HumanMessage(content=prompt)])
    logger.info(f"Agent 2: {response.content}")
    return {"messages": state["messages"] + [response]}


def should_continue(state: AgentState) -> Literal["agent1", "agent2", "action", "end"]:
    messages = state["messages"]
    last_message = messages[-1]
    if isinstance(last_message, AIMessage):
        if "FINAL_MEME:" in last_message.content:
            logger.info("FINAL_MEME detected. Ending meme creation process.")
            return "end"
        if hasattr(last_message, 'additional_kwargs') and last_message.additional_kwargs.get('tool_calls'):
            logger.info("Tool call detected. Executing action.")
            return "action"
    elif isinstance(last_message, BaseMessage) and last_message.type == "tool":
        logger.info("Tool message received. Continuing meme creation process.")
        return "agent1" if len(messages) % 2 == 0 else "agent2"
    logger.info("Continuing meme creation process")
    return "agent1" if len(messages) % 2 == 0 else "agent2"

# Define the function to execute tools
tool_node = ToolNode(tools)

workflow = StateGraph(AgentState, config_schema=GraphConfig)
workflow.add_node("agent1", agent1_node)
workflow.add_node("agent2", agent2_node)
workflow.add_node("action", tool_node)
workflow.set_entry_point("agent1")

for agent in ["agent1", "agent2", "action"]:
    workflow.add_conditional_edges(
        agent,
        should_continue,
        {
            "agent1": "agent1",
            "agent2": "agent2",
            "action": "action",
            "end": END,
        }
    )

graph = workflow.compile()

if __name__ == "__main__":
    topic = "Always Sunny in Philadelphia"
    result = graph.invoke({"messages": [HumanMessage(content=topic)]})
    logger.info(f"Meme for '{topic}': {result['messages'][-1].content}")