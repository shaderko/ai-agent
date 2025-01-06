import asyncio
from typing import Callable, Dict
from tool_detector import ToolDetector
from tools import add_numbers, get_weather
from whisper_live.client import TranscriptionClient
from ollama import chat, Message


class AsyncTranscriptionProcessor:
    def __init__(
        self,
        host="localhost",
        port=9090,
        use_vad=True,
        log_transcription=True,
        tools: Dict[str, Callable] = {},
        model="gemma2",
    ):
        self.tool_detector = ToolDetector(tools=tools)

        self.chat_messages = [
            Message(
                role="system",
                content=(
                    "You are an AI assistant primarily for chatting. "
                    "Start the chat with 'Hey, how are you doing?'. "
                    "Always respond to the user and keep answers short (max 3-5 sentences). "
                    "If needed, ask the user to provide more information."
                    f"You have these tools, which are called by a different model: {self.tool_detector.tool_manager.get_available_tools()}"
                ),
            )
        ]
        self.log_transcription = log_transcription
        self.task_queue = asyncio.Queue()
        self.shutdown_event = asyncio.Event()
        self.loop = asyncio.get_event_loop()
        self.transcription_client = TranscriptionClient(
            host=host,
            port=port,
            use_vad=use_vad,
            msg_callback=self.sync_on_message,  # Pass the synchronous wrapper
            log_transcription=False,
        )
        self.worker_task = None

        self.main_model_name = model

    def sync_on_message(self, message):
        """
        Synchronous callback that schedules the asynchronous enqueue_message coroutine.
        This allows TranscriptionClient to remain synchronous while integrating with asyncio.
        """
        # Schedule the enqueue_message coroutine in the event loop
        asyncio.run_coroutine_threadsafe(self.enqueue_message(message), self.loop)

    async def enqueue_message(self, message):
        """
        Enqueue a new message to the task queue.
        """
        await self.task_queue.put(message)
        if self.log_transcription:
            print(f"Enqueued message: {message}")

    async def worker(self):
        """
        Background worker that processes messages from the queue sequentially.
        """
        while not self.shutdown_event.is_set():
            try:
                message = await self.task_queue.get()
                await self.process_message(message)
                self.task_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in worker: {e}")

    async def process_message(self, message):
        """
        Process a single message by sending it to the Ollama API.
        """
        self.chat_messages.append(Message(role="user", content=message))

        if self.tool_detector:
            tools_output = self.tool_detector.detect_tool(message)
            if tools_output:
                string_out = ""
                for x in tools_output:
                    self.chat_messages.append(x)
                    string_out += f"{x['message']['content']}\n"
                final_prompt = (
                    "Using the tool's output below, provide a concise and relevant response to the user."
                    "\n\nTool Output:"
                    f"\n{string_out}"
                )
                self.chat_messages.append(Message(role="system", content=final_prompt))

        # Step 5: Send the updated conversation to the main chat model
        try:
            response = chat(
                model=self.main_model_name, messages=self.chat_messages, stream=True
            )

            message_content = ""
            for chunk in response:
                content = chunk["message"]["content"]
                print(content, end="", flush=True)
                message_content += content
            print()  # For newline after the response

            # Append the assistant's response to the conversation history
            self.chat_messages.append(
                Message(role="assistant", content=message_content)
            )

        except Exception as e:
            print(
                f"[Process Error] An error occurred while processing the message: {e}"
            )

        # try:
        #     response = chat(
        #         model="gemma2",
        #         messages=self.chat_messages,
        #         stream=True,
        #         tools=self.tools,
        #     )

        #     message_content = ""
        #     for chunk in response:
        #         print(chunk["message"]["content"], end="", flush=True)
        #         message_content += chunk["message"]["content"]
        #     r_message = Message(role="assistant", content=message_content)

        #     # Append the assistant's response to the conversation history
        #     self.chat_messages.append(r_message)

        #     # Handle tool calls if any
        #     if r_message.tool_calls:
        #         for tool_call in r_message.tool_calls:
        #             tool_name = tool_call.function.name
        #             tool_args = tool_call.function.arguments

        #             # Execute the tool function if available
        #             function_to_call = available_functions.get(tool_name)
        #             if function_to_call:
        #                 output = function_to_call(**tool_args)
        #                 # Append the tool's output to the conversation history
        #                 tool_message = Message(role="tool", content=str(output))
        #                 self.chat_messages.append(tool_message)
        #                 print(f"[Tool Output] {tool_message.content}")

        #                 # **Critical Step:** Prompt the assistant to generate a final response based on the tool's output
        #                 # This prevents the assistant from re-triggering tool calls in a loop
        #                 final_prompt = "For improving the answer to my last question use the following context, this is a system prompt do not repeat the last message to the user."
        #                 await self.process_message(
        #                     final_prompt
        #                 )  # Recursively process the final prompt
        #             else:
        #                 print(f"[Tool Error] Function '{tool_name}' not found.")
        #                 tool_message = Message(
        #                     role="tool",
        #                     content=f"[Tool Error] Function '{tool_name}' not found.",
        #                 )
        #                 self.chat_messages.append(tool_message)

        #                 final_prompt = "For improving the answer to my last question use the following context, this is a system prompt do not repeat the last message to the user."
        #                 await self.process_message(final_prompt)
        # except Exception as e:
        #     print(
        #         f"[Process Error] An error occurred while processing the message: {e}"
        #     )

    async def run_transcription_client(self):
        """
        Runs the TranscriptionClient in a separate thread to prevent blocking the event loop.
        """
        await self.loop.run_in_executor(None, self.transcription_client)

    async def run_worker(self):
        """
        Starts the background worker coroutine.
        """
        self.worker_task = asyncio.create_task(self.worker())

    async def run(self):
        """
        Starts both the transcription client and the worker.
        """
        await asyncio.gather(self.run_worker(), self.run_transcription_client())

    async def shutdown(self):
        """
        Shuts down the processor gracefully by signaling the worker to stop,
        waiting for the queue to be empty, and closing the HTTP session.
        """
        # Signal the worker to stop
        self.shutdown_event.set()

        # Wait for the queue to be empty
        await self.task_queue.join()

        # Cancel the worker task
        if self.worker_task:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                print("Worker task cancelled.")


# Example usage
async def main():
    processor = AsyncTranscriptionProcessor(
        tools={"add_numbers": add_numbers, "get_weather": get_weather}
    )

    try:
        # Start the transcription client and worker
        await processor.run()
    except KeyboardInterrupt:
        print("Received exit signal.")
    finally:
        # Shutdown the processor gracefully
        await processor.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
