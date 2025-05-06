from langgraph.graph import MessagesState
from typing_extensions import TypedDict


class SearchResult(TypedDict):
    snippet: str
    title: str
    link: str


class WebSearchState(MessagesState):
    query: str
    web_search_results: list[SearchResult]
    should_search: bool
