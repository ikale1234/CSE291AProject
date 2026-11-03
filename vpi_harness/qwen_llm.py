"""AgentDojo pipeline element for a vision-language model served by vLLM's OpenAI-compatible server
(Qwen3-VL-8B-Instruct by default), using the model's native tool calling.

Qwen3 emits Hermes-style `<tool_call>{"name": ..., "arguments": {...}}</tool_call>` blocks. Two modes:
  - `tool_choice="auto"` (native): vLLM parses the blocks into OpenAI `tool_calls`. Needs the server to be
    started with `--enable-auto-tool-choice --tool-call-parser hermes`.
  - `tool_choice="none"` (text): for servers without those flags. vLLM still renders the tools into the prompt
    (unless started with `--exclude-tools-when-tool-choice-none`), and we parse the blocks from the text.
In both modes, a block that cannot be parsed is recorded as a `parse_error`, so malformed calls are counted as
invalid rather than as "no tool".
"""

import json
import re
import time
from collections.abc import Sequence

import openai
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.llms.openai_llm import _function_to_openai, _messages_to_openai
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionCall, FunctionsRuntime
from agentdojo.types import ChatAssistantMessage, ChatMessage, text_content_block_from_string

_TOOL_CALL_BLOCK = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", re.DOTALL)
_TRANSIENT_ERRORS = (openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError)


def to_vllm_messages(messages: Sequence[ChatMessage], model: str) -> list[dict]:
    """AgentDojo messages -> OpenAI chat messages accepted by vLLM and Qwen's chat template.

    Images returned by tools go in a user message after the tool results (see `_messages_to_openai`).
    AgentDojo sends the system prompt with OpenAI's `developer` role, which Qwen's template does not know,
    and tool results as lists of text parts, which we join into a string.
    """
    converted = []
    for message in _messages_to_openai(messages, model):
        message = dict(message)
        if message["role"] == "developer":
            message["role"] = "system"
        if message["role"] in ("system", "tool") and isinstance(message.get("content"), list):
            message["content"] = "\n".join(part["text"] for part in message["content"] if part["type"] == "text")
        converted.append(message)
    return converted


def _parse_text_tool_calls(content: str) -> tuple[list[FunctionCall], str | None]:
    """Parses `<tool_call>{"name": ..., "arguments": {...}}</tool_call>` blocks that vLLM left in the text."""
    calls = []
    for i, block in enumerate(_TOOL_CALL_BLOCK.findall(content)):
        try:
            data = json.loads(block)
            arguments = data.get("arguments", {})
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            calls.append(FunctionCall(function=data["name"], args=arguments, id=f"text_call_{i}"))
        except (json.JSONDecodeError, KeyError, TypeError, AttributeError, ValueError):
            return [], f"malformed <tool_call> block in text: {block[:200]!r}"
    return calls, None


def parse_assistant_message(message) -> tuple[ChatAssistantMessage, str | None]:
    """Returns the assistant message and a parse error (None if the output was well-formed)."""
    content = message.content
    blocks = [text_content_block_from_string(content)] if content else None
    if message.tool_calls:
        calls = []
        for tool_call in message.tool_calls:
            try:
                arguments = json.loads(tool_call.function.arguments or "{}")
                calls.append(FunctionCall(function=tool_call.function.name, args=arguments, id=tool_call.id))
            except (json.JSONDecodeError, ValueError):
                error = f"malformed arguments for {tool_call.function.name}: {tool_call.function.arguments[:200]!r}"
                return ChatAssistantMessage(role="assistant", content=blocks, tool_calls=None), error
        return ChatAssistantMessage(role="assistant", content=blocks, tool_calls=calls), None
    if content and "<tool_call>" in content:
        calls, error = _parse_text_tool_calls(content)
        return ChatAssistantMessage(role="assistant", content=blocks, tool_calls=calls or None), error
    return ChatAssistantMessage(role="assistant", content=blocks, tool_calls=None), None


class VLLMChatLLM(BasePipelineElement):
    """Queries the model once and appends its reply. Per-request info (latency, token usage, finish reason,
    raw text, parse error) is returned in `extra_args["llm"]`; `extra_args["seed"]` sets the sampling seed."""

    def __init__(
        self,
        client: openai.OpenAI,
        model: str,
        temperature: float = 0.7,
        top_p: float = 0.8,
        max_tokens: int = 512,
        tool_choice: str = "auto",
    ) -> None:
        self.client = client
        self.tool_choice = tool_choice
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

    @retry(
        retry=retry_if_exception_type(_TRANSIENT_ERRORS),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=2, max=30),
        reraise=True,
    )
    def _create(self, messages: list[dict], tools: list, seed: int | None):
        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore -- plain dicts in the OpenAI format
            tools=tools or openai.NOT_GIVEN,
            tool_choice=self.tool_choice if tools else openai.NOT_GIVEN,  # type: ignore
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            seed=seed if seed is not None else openai.NOT_GIVEN,
        )

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        request_messages = to_vllm_messages(messages, self.model)
        tools = [_function_to_openai(f) for f in runtime.functions.values()]
        start = time.perf_counter()
        completion = self._create(request_messages, tools, extra_args.get("seed"))
        latency = time.perf_counter() - start

        choice = completion.choices[0]
        output, parse_error = parse_assistant_message(choice.message)
        usage = completion.usage
        info = {
            "latency_s": latency,
            "prompt_tokens": usage.prompt_tokens if usage else None,
            "completion_tokens": usage.completion_tokens if usage else None,
            "finish_reason": choice.finish_reason,
            "raw_content": choice.message.content,
            "parse_error": parse_error,
        }
        return query, runtime, env, [*messages, output], {**extra_args, "llm": info}
