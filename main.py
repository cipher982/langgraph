from typing import List, Tuple, TypedDict, Annotated
from operator import add
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import MessageGraph, END

# Define our state
class MemeState(TypedDict):
    messages: Annotated[List[Tuple[str, str]], add]

# Create our agents
agent1 = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
agent2 = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

def agent1_node(state):
    messages = state
    if len(messages) == 1:  # Initial prompt
        human_message = messages[0].content
        prompt = f"We're creating a meme about {human_message}. Let's brainstorm some funny ideas. What do you suggest?"
    else:
        last_message = messages[-1].content
        if "FINAL_MEME" in last_message:
            response_content = "Great job! We've completed our meme."
            print(f"Agent 1: {response_content}")
            return messages + [AIMessage(content=response_content)]
        prompt = f"Based on our previous discussion, let's continue developing our meme idea. {last_message}"
    
    response = agent1.invoke([HumanMessage(content=prompt)])
    print(f"Agent 1: {response.content}")
    return messages + [AIMessage(content=response.content)]

def agent2_node(state):
    messages = state
    last_message = messages[-1].content
    
    if "template" in last_message.lower():
        prompt = "Great template idea! Now let's finalize our meme. Write out the final joke and describe the image. Start your response with 'FINAL_MEME:'"
    else:
        prompt = f"I like your ideas! Let's build on them. Can you suggest a meme template that would work well with our {messages[0].content} theme?"
    
    response = agent2.invoke([HumanMessage(content=prompt)])
    print(f"Agent 2: {response.content}")
    return messages + [AIMessage(content=response.content)]

def should_continue(state):
    last_message = state[-1].content
    return not last_message.strip().startswith("FINAL_MEME:")

# Update the create_meme function
def create_meme(topic: str):
    initial_state = [HumanMessage(content=topic)]
    result = compiled_graph.invoke(initial_state)
    final_message = [msg for msg in result if msg.content.strip().startswith("FINAL_MEME:")][-1].content
    return final_message.split("FINAL_MEME:", 1)[-1].strip()

# Define the condition function
def check_final_meme(state):
    last_message = state[-1].content
    if last_message.strip().startswith("FINAL_MEME:"):
        return END
    return "agent1_node" if state[-1].type == "ai" and state[-1].content.startswith("Agent 2:") else "agent2_node"

# Update the graph edges
graph = MessageGraph()
graph.add_node("agent1_node", agent1_node)
graph.add_node("agent2_node", agent2_node)
graph.add_edge("agent1_node", "agent2_node")
graph.add_conditional_edges("agent2_node", check_final_meme)
graph.set_entry_point("agent1_node")

# Compile the graph
compiled_graph = graph.compile()