from typing import Literal
import datetime
from web_search.state import SearchResult, WebSearchState
from pydantic import BaseModel, Field
from langchain_anthropic.chat_models import ChatAnthropic
from langchain_core.messages import BaseMessage
from langchain_community.tools import DuckDuckGoSearchResults

from web_search.utils import format_messages

CLASSIFIER_PROMPT = """You're a helpful AI assistant tasked with classifying the user's latest message.
The user has enabled web search for their conversation, however not all messages should be searched.

Analyze their latest message in isolation and determine if it warrants a web search to include additional context.

<message>
{message}
</message>"""


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


QUERY_GENERATOR_PROMPT = """You're a helpful AI assistant tasked with writing a query to search the web.
You're provided with a list of messages between a user and an AI assistant.
The most recent message from the user is the one you should update to be a more search engine friendly query.

Try to keep the new query as similar to the message as possible, while still being search engine friendly.

Only answer result query.

Here is the conversation between the user and the assistant, in order of oldest to newest:

<conversation>
{conversation}
</conversation>

<additional_context>
{additional_context}
</additional_context>

generated query is:
>>> """


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
