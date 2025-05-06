import datetime
from typing import Literal

from langchain_anthropic.chat_models import ChatAnthropic
from langchain_community.tools import DuckDuckGoSearchResults
from pydantic import BaseModel, Field

from zuzu_agent.agents.web_search.prompts import (
    CLASSIFIER_PROMPT,
    QUERY_GENERATOR_PROMPT,
)
from zuzu_agent.agents.web_search.state import SearchResult, WebSearchState
from zuzu_agent.shared.utils import format_messages


class ClassificationSchema(BaseModel):
    """The classification of the user's latest message."""

    should_search: bool = Field(
        description="Whether or not to search the web based on the user's latest message."
    )


async def classify_message(state: WebSearchState) -> dict:
    model = ChatAnthropic(
        model="claude-3-5-haiku-latest",
        temperature=0,
    ).with_structured_output(ClassificationSchema)
    latest_message_content = state["messages"][-1].content
    formatted_prompt = CLASSIFIER_PROMPT.format(message=latest_message_content)
    response: ClassificationSchema = await model.ainvoke([("user", formatted_prompt)])
    return {"should_search": response.should_search}


async def query_generator(state: WebSearchState) -> dict:
    model = ChatAnthropic(
        model="claude-3-5-haiku-latest",
        temperature=0,
    )

    formatted_messages = format_messages(state["messages"])
    additional_context = f"The current date is {datetime.date.today()}"
    formatted_prompt = QUERY_GENERATOR_PROMPT.format(
        conversation=formatted_messages,
        additional_context=additional_context,
    )

    response = await model.ainvoke([("user", formatted_prompt)])

    return {"query": response.content}


async def search(state: WebSearchState) -> dict:
    query = state["query"]
    retriever = DuckDuckGoSearchResults(output_format="list")

    results: list[SearchResult] = await retriever.ainvoke(query)

    return {"web_search_results": results}


async def search_or_end_conditional(
    state: WebSearchState,
) -> Literal["__end__", "query_generator"]:
    if state["should_search"]:
        return "query_generator"
    return "__end__"
