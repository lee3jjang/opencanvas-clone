from typing import Literal

from langgraph.graph import StateGraph

from web_search.nodes import (
    classify_message,
    query_generator,
    search,
    search_or_end_conditional,
)
from web_search.state import WebSearchState


graph = (
    StateGraph(WebSearchState)
    .add_node(classify_message)
    .add_node(query_generator)
    .add_node(search)
    .set_entry_point("classify_message")
    .add_conditional_edges("classify_message", search_or_end_conditional)
    .add_edge("query_generator", "search")
    .set_finish_point("search")
    .compile()
)
