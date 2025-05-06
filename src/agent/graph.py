from langgraph.graph import StateGraph, MessagesState
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langchain_community.tools import DuckDuckGoSearchResults


class State(MessagesState):
    pass


async def my_node(state: State) -> dict:
    model = init_chat_model("anthropic:claude-3-5-haiku-latest")
    agent = create_react_agent(
        model,
        [DuckDuckGoSearchResults()],
    )

    response = await agent.ainvoke(state)

    return {"messages": [response["messages"][-1]]}


graph = StateGraph(State).add_node(my_node).set_entry_point("my_node").compile()
