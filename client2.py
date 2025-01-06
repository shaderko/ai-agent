import asyncio
import os
from typing import List, Dict
from whisper_live.client import TranscriptionClient
from ollama import chat, Message
from tools import ToolManager, google_search, run_cli_command
from tool_detector import ToolDetector


class AsyncTranscriptionProcessor:
    def __init__(
        self,
        host="localhost",
        port=9090,
        use_vad=True,
        log_transcription=False,
        main_model_name: str = "chat-model",
        tool_detector: ToolDetector = None,
        tool_manager: ToolManager = None,
        tools=[],
    ):
        self.context = []
        self.tool_detector = tool_detector
        self.tool_manager = tool_manager

        self.chat_messages = [
            Message(
                role="system",
                content=(
                    "You are an AI assistant primarily for chatting. "
                    "Always ask 'Hey, how are you doing?'. "
                    "Always respond to the user and keep answers short (max 3-5 sentences). "
                    "If needed, ask the user to provide more information."
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
            msg_callback=self.sync_on_message,
            log_transcription=False,
        )
        self.tools = tools
        self.worker_task = None
        self.main_model_name = main_model_name

    def sync_on_message(self, message):
        """
        Synchronous callback that schedules the asynchronous enqueue_message coroutine.
        This allows TranscriptionClient to remain synchronous while integrating with asyncio.
        """
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
        Process a single message by determining if a tool is needed and responding accordingly.
        """
        # Append the user message to the conversation history
        self.chat_messages.append(Message(role="user", content=message))

        # Step 1: Detect if a tool is needed
        tool_info = None
        if self.tool_detector and self.tool_manager:
            tools_output = self.tool_detector.detect_tool(message)
            if tools_output:
                for x in tools_output:
                    self.chat_messages.append(x)
                final_prompt = (
                    "Using the tool's output below, provide a concise and relevant response to the user."
                    "\n\nTool Output:"
                    f"\n{tool_output}"
                )
            self.chat_messages.append(Message(role="system", content=final_prompt))

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
        waiting for the queue to be empty, and closing the transcription client.
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

        # Shutdown the transcription client
        self.transcription_client.shutdown()
        print("Transcription client shut down.")


async def main():
    available_functions = {
        "google_search": google_search,
        "run_cli_command": run_cli_command,
    }
    tool_manager = ToolManager(available_functions=available_functions)

    tool_detector = ToolDetector(model_name="llama3.1")  # Replace with your model name

    # Initialize AsyncTranscriptionProcessor
    processor = AsyncTranscriptionProcessor(
        tools=[],  # Tools are managed separately
        main_model_name="gemma2",  # Replace with your main chat model name
        tool_detector=tool_detector,
        tool_manager=tool_manager,
        log_transcription=True,  # Enable logging for debugging
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