import os
import logging
from typing import TypedDict, Annotated, Sequence, Literal, List
from functools import lru_cache
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, END, add_messages
import asyncio

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use environment variables for sensitive information
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

tools = []  # Placeholder for future tool integration

@lru_cache(maxsize=2)
def _get_model(model_name: str, temperature: float, max_tokens: int) -> ChatOpenAI:
    try:
        model = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            openai_api_key=OPENAI_API_KEY
        )
        return model
    except Exception as e:
        logger.error(f"Error creating model: {e}")
        raise

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    turn_count: int

class GraphConfig(TypedDict):
    model_name: str
    temperature: float
    max_tokens: int
    max_turns: int

def get_config(config: GraphConfig) -> GraphConfig:
    defaults = {
        "model_name": "gpt-4o-mini", # dont change this
        "temperature": 0.7,
        "max_tokens": 256,
        "max_turns": 5
    }
    return {**defaults, **config}

def get_conversation_history(messages: List[BaseMessage], n: int = 3) -> str:
    return "\n".join([f"{'User' if i == 0 else 'Bob' if i % 2 == 1 else 'Alice'}: {m.content}" for i, m in enumerate(messages[-n:])])


async def bob_node(state: AgentState, config: GraphConfig) -> AgentState:
    messages = state["messages"]
    full_config = get_config(config)
    model = _get_model(full_config["model_name"], full_config["temperature"], full_config["max_tokens"])
    
    history = get_conversation_history(messages)
    prompt = f"You are Bob, a mad scientist that thinks unconventionally. Your task is to work with Alice to brainstorm innovative and possibly unusual ideas for the following problem, considering the conversation history:\n\n{history}\n\nDon't hold back on creativity!"
    
    try:
        response = await model.ainvoke([HumanMessage(content=prompt)])
        logger.info(f"(Bob) {response.content}")
        return {"messages": state["messages"] + [response], "turn_count": state["turn_count"] + 1}
    except Exception as e:
        logger.error(f"Error in Bob's response: {e}")
        return state

async def alice_node(state: AgentState, config: GraphConfig) -> AgentState:
    messages = state["messages"]
    full_config = get_config(config)
    model = _get_model(full_config["model_name"], full_config["temperature"], full_config["max_tokens"])
    
    history = get_conversation_history(messages)
    prompt = f"You are Alice, a practical and grounded thinker and bobs assistant. Your task is to evaluate and refine the ideas proposed in the following conversation, providing a realistic assessment and practical improvements:\n\n{history}"
    
    try:
        response = await model.ainvoke([HumanMessage(content=prompt)])
        logger.info(f"(Alice) {response.content}")
        return {"messages": state["messages"] + [response], "turn_count": state["turn_count"] + 1}
    except Exception as e:
        logger.error(f"Error in Alice's response: {e}")
        return state


def should_continue(state: AgentState) -> Literal["bob", "alice", "action", "end"]:
    if state["turn_count"] >= get_config({})["max_turns"]:
        return "end"
    if len(state["messages"]) % 2 == 1:
        return "bob"
    else:
        return "alice"


tool_node = ToolNode(tools)

workflow = StateGraph(AgentState, config_schema=GraphConfig)
workflow.add_node("bob", bob_node)
workflow.add_node("alice", alice_node)
workflow.add_node("action", tool_node)
workflow.set_entry_point("bob")

for agent in ["bob", "alice", "action"]:
    workflow.add_conditional_edges(
        agent,
        should_continue,
        {
            "bob": "bob",
            "alice": "alice",
            "action": "action",
            "end": END,
        }
    )

graph = workflow.compile()


## Running the graph locally
async def run_conversation(task: str):
    try:
        result = await graph.ainvoke({"messages": [HumanMessage(content=task)], "turn_count": 0})
        return result["messages"]
    except Exception as e:
        logger.error(f"Error in conversation: {e}")
        return []


async def main():
    while True:
        task = input("Enter your task or question (or 'quit' to exit): ")
        if task.lower() == 'quit':
            break
        
        _ = await run_conversation(task)
        
        follow_up = input("Do you have any follow-up questions? (yes/no): ")
        if follow_up.lower() != 'yes':
            break


if __name__ == "__main__":
    asyncio.run(main())