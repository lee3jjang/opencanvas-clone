from typing import cast

from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.graph import MessagesState, StateGraph

from zuzu_agent.shared.utils import format_messages

TITLE_SYSTEM_PROMPT = """You are tasked with generating a concise, descriptive title for a conversation between a user and an AI assistant. The title should capture the main topic or purpose of the conversation.

Guidelines for title generation:
- Keep titles extremely short (ideally 2-5 words)
- Focus on the main topic or goal of the conversation
- Use natural, readable language
- Avoid unnecessary articles (a, an, the) when possible
- Do not include quotes or special characters
- Capitalize important words
- Make sure the title is generated in Korean

Use the 'generate_title_tool' tool to output your title."""

TITLE_USER_PROMPT = """Based on the following conversation, generate a very short and descriptive title for:

{conversation}"""


@tool
def generate_title_tool(title: str) -> None:
    """Generate a concise title for the conversation.

    Args:
        title (str): The generated title for the conversation.
    """


class ThreadTitleState(MessagesState):
    title: str


async def generate_title(state: ThreadTitleState) -> dict:
    model = init_chat_model("anthropic:claude-3-5-haiku-latest").bind_tools(
        [generate_title_tool]
    )

    formatted_user_prompt = TITLE_USER_PROMPT.format(
        conversation=format_messages(state["messages"]),
    )

    result = await model.ainvoke(
        [
            ("system", TITLE_SYSTEM_PROMPT),
            ("user", formatted_user_prompt),
        ]
    )

    title = cast(str, result.tool_calls[0]["args"]["title"])

    return {"title": title}


graph = (
    StateGraph(ThreadTitleState)
    .add_node(generate_title)
    .set_entry_point("generate_title")
    .compile()
)
