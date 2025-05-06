from langchain_core.messages import BaseMessage


def format_messages(messages: list[BaseMessage]) -> str:
    def _get_content_from_msg(msg: BaseMessage) -> str:
        message_content = ""
        if isinstance(msg.content, str):
            message_content = msg.content
        elif isinstance(msg.content, list[str | dict]):
            for c in msg.content:
                if isinstance(c, str):
                    message_content += c
                elif isinstance(c, dict):
                    if "text" in c.keys():
                        message_content += c["text"]
        return message_content

    formatted_messages_list = []
    for i, msg in enumerate(messages):
        msg_content = _get_content_from_msg(msg)
        formatted_messages_list.append(
            f"<{msg.type} index={i}>\n{msg_content}\n</{msg.type}>"
        )

    return "\n".join(formatted_messages_list)
