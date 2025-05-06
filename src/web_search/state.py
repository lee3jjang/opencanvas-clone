from langgraph.graph import MessagesState


class SearchResult:
    snippet: str
    title: str
    link: str


class WebSearchState(MessagesState):
    query: str
    web_search_results: list[SearchResult]
    should_search: bool
