"""
Middlewares:
Most important part of the langchain, where almost it adds harness and tools
Useful for handling
- logging, metrics
- tool selections
- retries and fallbacks
- rate limits

before_agent -> before_model -> (wrap_model_calls) -> model -> (wrap_tool_calls) -> before_model


Tool error	Catch tool execution exceptions and convert them to error messages for the model.
Tool retry	Automatically retry failed tool calls with exponential backoff.
Model retry	Automatically retry failed model calls with exponential backoff.
Model fallback	Automatically fallback to alternative models when primary fails.
Summarization	Automatically summarize conversation history when approaching token limits.
Human-in-the-loop	Pause execution for human approval of tool calls.
Model call limit	Limit the number of model calls to prevent excessive costs.
Tool call limit	Control tool execution by limiting call counts.
PII detection	Detect and handle Personally Identifiable Information (PII).
To-do list	Equip agents with task planning and tracking capabilities.
LLM tool selector	Use an LLM to select relevant tools before calling main model.
Provider tool search	Defer tools behind providers’ server-side tool search, surfacing them on demand.
Shell tool	Expose a persistent shell session to agents for command execution.
Filesystem	Provide agents with a filesystem for storing context and long-term memories.
Subagent	Add the ability to spawn subagents.
File search	Provide Glob and Grep search tools over filesystem files.
Context editing	Manage conversation context by trimming or clearing tool uses.
LLM tool emulator	Emulate tool execution using an LLM for testing purposes.


As implementing all can lead to huge time, as their syntax also defers we will directly use in building project
"""





