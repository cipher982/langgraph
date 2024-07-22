import os
import logging
from typing import TypedDict, Annotated, Sequence, Literal
from functools import lru_cache
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, add_messages

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use environment variables for sensitive information
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

@lru_cache(maxsize=2)
def _get_model(model_name: str = "gpt-3.5-turbo", temperature: float = 0.7, max_tokens: int = 150) -> ChatOpenAI:
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        openai_api_key=OPENAI_API_KEY
    )

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

class GraphConfig(TypedDict):
    model_name: Literal["gpt-4o-mini"]
    temperature: float
    max_tokens: int

def agent1_node(state: AgentState) -> AgentState:
    messages = state["messages"]
    model = _get_model()
    
    if len(messages) == 1:  # Initial prompt
        human_message = messages[0].content
        prompt = f"We're creating a meme about {human_message}. Let's brainstorm some funny ideas. What do you suggest?"
    else:
        last_message = messages[-1].content
        prompt = f"Based on our previous discussion, let's continue developing our meme idea. {last_message}"
    
    response = model.invoke([HumanMessage(content=prompt)])
    logger.info(f"Agent 1: {response.content}")
    return {"messages": state["messages"] + [AIMessage(content=response.content)]}

def agent2_node(state: AgentState) -> AgentState:
    messages = state["messages"]
    model = _get_model()
    last_message = messages[-1].content
    
    if "FINAL_MEME:" not in last_message:
        prompt = f"Great ideas! Now, let's finalize our meme. Create a funny meme based on our discussion about {messages[0].content}. Start your response with 'FINAL_MEME:'"
    else:
        prompt = "The meme is complete. Let's end the process."
    
    response = model.invoke([HumanMessage(content=prompt)])
    logger.info(f"Agent 2: {response.content}")
    return {"messages": state["messages"] + [AIMessage(content=response.content)]}


def should_continue(state: AgentState) -> Literal["agent1", "agent2", "end"]:
    messages = state["messages"]
    last_message = messages[-1].content
    if "FINAL_MEME:" in last_message:
        logger.info("FINAL_MEME detected. Ending meme creation process.")
        return "end"
    logger.info("Continuing meme creation process")
    return "agent1" if len(messages) % 2 == 0 else "agent2"

workflow = StateGraph(AgentState, config_schema=GraphConfig)
workflow.add_node("agent1", agent1_node)
workflow.add_node("agent2", agent2_node)
workflow.set_entry_point("agent1")
for agent in ["agent1", "agent2"]:
    workflow.add_conditional_edges(
        agent,
        should_continue,
        {
            "agent1": "agent1",
            "agent2": "agent2",
            "end": END,
        }
    )
graph = workflow.compile()

if __name__ == "__main__":
    topic = "grumpy cat"
    result = graph.invoke({"messages": [HumanMessage(content=topic)]})
    logger.info(f"Meme for '{topic}': {result['messages'][-1].content}")