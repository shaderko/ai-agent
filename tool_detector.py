from typing import Callable, List, Optional, Dict
from ollama import chat, Message
import json
from tools import ToolManager


class ToolDetector:
    def __init__(self, model_name: str = "llama3.2", tools: Dict[str, Callable] = {}):
        """
        Initializes the ToolDetector with the specified model.

        Args:
            model_name (str): The name of the Ollama model used for tool detection.
        """
        self.model_name = model_name
        self.tool_manager = ToolManager(tools)
        self.system_prompt = (
            "You are a tool detection assistant. Determine if the user's input requires invoking a tool. "
            "If so, specify the tool name and necessary arguments in JSON format. Otherwise, respond with 'NO_TOOL'."
            "If you need more information to get results from a tool, respond with 'INFO_REQ' followed by the requirements."
        )

    def detect_tool(self, user_input: str) -> Optional[List[Message]]:
        """
        Analyzes the user input to determine if a tool should be used.

        Args:
            user_input (str): The input message from the user.

        Returns:
            Optional[Dict]: A dictionary containing the tool name and arguments if a tool is needed; otherwise, None.
        """
        messages = [
            Message(role="system", content=self.system_prompt),
            Message(role="user", content=user_input),
        ]

        try:
            r = chat(
                model=self.model_name,
                messages=messages,
                stream=False,
                tools=self.tool_manager.tools,
            )

            print("[ToolDetector] ", r.message.content)

            if r.message.content and r.message.content.find("INFO_REQ") != -1:
                return [r.message]

            messages = []

            if r.message.tool_calls:
                for tool_call in r.message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = tool_call.function.arguments

                    # Execute the tool function if available
                    function_to_call = self.tool_manager.available_functions.get(
                        tool_name
                    )
                    if function_to_call:
                        output = function_to_call(**tool_args)
                        # Append the tool's output to the conversation history
                        tool_message = Message(role="tool", content=str(output))
                        messages.append(tool_message)

            print(
                "[ToolDetector] ",
                json.dumps(messages),
            )

            return messages

        except Exception as e:
            print(f"[ToolDetector Error] {e}")
            tool_message = Message(role="tool", content=str(e))
            return [tool_message]
