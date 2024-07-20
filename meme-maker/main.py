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
def _get_model(model_name: str, temperature: float = 0.7, max_tokens: int = 150) -> ChatOpenAI:
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

def agent1_node(state: AgentState, config: GraphConfig) -> dict:
    messages = state["messages"]
    model = _get_model(config["model_name"], config["temperature"], config["max_tokens"])
    
    if len(messages) == 1:  # Initial prompt
        human_message = messages[0].content
        prompt = f"We're creating a meme about {human_message}. Let's brainstorm some funny ideas. What do you suggest?"
    else:
        last_message = messages[-1].content
        if "FINAL_MEME" in last_message:
            return {"messages": [AIMessage(content="Great job! We've completed our meme.")]}
        prompt = f"Based on our previous discussion, let's continue developing our meme idea. {last_message}"
    
    response = model.invoke([HumanMessage(content=prompt)])
    logger.info(f"Agent 1: {response.content}")
    return {"messages": [AIMessage(content=response.content)]}

def agent2_node(state: AgentState, config: GraphConfig) -> dict:
    messages = state["messages"]
    model = _get_model(config["model_name"], config["temperature"], config["max_tokens"])
    last_message = messages[-1].content
    
    if "template" in last_message.lower():
        prompt = "Great template idea! Now let's finalize our meme. Write out the final joke and describe the image. Start your response with 'FINAL_MEME:'"
    else:
        prompt = f"I like your ideas! Let's build on them. Can you suggest a meme template that would work well with our {messages[0].content} theme?"
    
    response = model.invoke([HumanMessage(content=prompt)])
    logger.info(f"Agent 2: {response.content}")
    return {"messages": [AIMessage(content=response.content)]}

def should_continue(state: AgentState) -> str:
    messages = state["messages"]
    last_message = messages[-1].content
    if last_message.strip().startswith("FINAL_MEME:"):
        return "end"
    return "agent1" if messages[-1].type == "ai" and not messages[-1].content.startswith("Agent 1:") else "agent2"

workflow = StateGraph(AgentState, config_schema=GraphConfig)
workflow.add_node("agent1", agent1_node)
workflow.add_node("agent2", agent2_node)
workflow.set_entry_point("agent1")
workflow.add_conditional_edges("agent1", should_continue, {"agent2": "agent2", "end": END})
workflow.add_conditional_edges("agent2", should_continue, {"agent1": "agent1", "end": END})

graph = workflow.compile()

def create_meme(topic: str) -> str:
    config = {
        "model_name": "gpt-4o-mini",
        "temperature": 0.7,
        "max_tokens": 150
    }
    try:
        result = graph.invoke({"messages": [HumanMessage(content=topic)]}, config=config)
        final_messages = [msg for msg in result["messages"] if msg.content.strip().startswith("FINAL_MEME:")]
        if not final_messages:
            raise ValueError("No final meme was generated.")
        return final_messages[-1].content.split("FINAL_MEME:", 1)[-1].strip()
    except Exception as e:
        logger.error(f"Error creating meme: {str(e)}")
        return f"Error: {str(e)}"

if __name__ == "__main__":
    topic = "grumpy cat"
    meme = create_meme(topic)
    logger.info(f"Meme for '{topic}': {meme}")