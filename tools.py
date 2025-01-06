from typing import Callable, Dict
import requests
import json


def add_numbers(first: int, second: int) -> int:
    return first + second


def get_weather(latitude: float, longitude: float) -> dict:
    """
    Retrieves the current weather for the specified latitude and longitude using Open-Meteo.

    Args:
        latitude (float): Latitude of the location.
        longitude (float): Longitude of the location.

    Returns:
        dict: A dictionary containing weather details or an error message.
    """
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "current_weather": True,
            "timezone": "auto",
        }

        response = requests.get(url, params=params)
        data = response.json()

        if "current_weather" in data:
            weather = data["current_weather"]
            return {
                "temperature": f"{weather['temperature']}°C",
                "windspeed": f"{weather['windspeed']} km/h",
                "weather_code": weather["weathercode"],
            }
        else:
            return {"error": "Weather data not available for the specified location."}

    except Exception as e:
        return {"error": str(e)}


tools_json = [
    {
        "name": "add_numbers",
        "description": "Add two numbers together",
        "parameters": {
            "type": "object",
            "properties": {
                "first": {
                    "type": "integer",
                    "description": "First number.",
                },
                "second": {
                    "type": "integer",
                    "description": "Second number",
                },
            },
            "required": ["first", "second"],
        },
    },
    {
        "name": "get_weather",
        "description": "Retrieve the current weather for a specified location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "Location of where to get the current weather for.",
                }
            },
            "required": ["location"],
        },
    },
]


class ToolManager:
    def __init__(self, available_functions: Dict[str, Callable]):
        self.available_functions = available_functions
        self.tools = tools_json
        print("[ToolManager] Initialized with ", json.dumps(tools_json))

    def get_available_tools(self):
        return "\n".join(
            f"Function {tool['name']} to {tool['description']}:\n{tool}"
            for tool in tools_json
        )

    def execute_tool(self, tool_name: str, **kwargs):
        function = self.available_functions.get(tool_name)
        if not function:
            return f"Error: Tool '{tool_name}' not found."

        try:
            return function(**kwargs)
        except Exception as e:
            return f"Error executing tool '{tool_name}': {str(e)}"
