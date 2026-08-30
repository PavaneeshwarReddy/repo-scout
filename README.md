# repo-scout
An autonomous multi-agent due-diligence system built with LangChain, LangGraph, and Deep Agents that performs AST-based static code analysis, security auditing, and automated report synthesis for GitHub repositories.


### Framework Roles at a Glance

* **LangChain:** **Tool & Model Framework** — Wraps external integrations (GitHub API, Search APIs, AST code analyzer) into structured tools that models can consume.
* **Deep Agents:** **Cognitive Harness** — Manages multi-step task planning (via a built-in `todo` list tool), context-quarantined sub-agent delegation, and Virtual File System (VFS) memory management.
* **LangGraph:** **Stateful Graph Runtime** — Controls the top-level execution state, manages memory checkpoints across turns, and enforces Human-in-the-Loop (HITL) approval gates.
* **Abstract Syntax Tree (AST):** **Deterministic Pre-Filter** — Parses raw Python code programmatically to extract structural metadata, imports, and dangerous function calls in milliseconds before sending anything to an LLM.

---

### Step-by-Step Workflow: User Perspective vs. Backend Mechanics

#### Stage 1: Request Submission & State Initialization

* **User Perspective:** You enter a target repository into the interface (e.g., `"Audit `pallets/flask` for production readiness"`) and click Start.
* **Backend Mechanics:**
* **LangGraph** receives the request and instantiates a new state object (`DiligenceState`).
* It assigns a unique execution thread ID and saves the initial state into its **Memory Checkpointer**. This guarantees that the workflow can be paused, resumed, or inspected at any point.



#### Stage 2: Autonomous Task Planning & Workspace Setup

* **User Perspective:** The UI displays a progress state: *"Initializing audit plan and setting up isolated workspace..."*
* **Backend Mechanics:**
* The main manager agent—initialized via **Deep Agents**—receives the objective.
* It uses its built-in planning tool (`todo_write`) to generate a step-by-step audit strategy.
* It initializes a Virtual File System workspace containing virtual markdown files (`/code_audit.md`, `/security_audit.md`, `/final_diligence_report.md`).



#### Stage 3: Sub-Agent Execution & AST Parsing

* **User Perspective:** You see parallel logs showing live progress: *"Auditing codebase structure..."* and *"Querying external security databases..."*
* **Backend Mechanics:**
* **Context-Quarantined Delegation:** The Deep Agent manager spawns two isolated sub-agents:
1. **Code Analyst Sub-Agent:** Calls a custom **LangChain tool** that fetches target code via the GitHub API. It passes this code directly into Python's native **AST module**. The AST parser programmatically scans the code tree, identifying dangerous calls (`eval()`, `exec()`, unhandled exceptions) and mapping class/function skeletons. This deterministic step extracts critical code metrics in milliseconds without wasting LLM tokens. The sub-agent writes its findings directly to `/code_audit.md`.
2. **Security Analyst Sub-Agent:** Concurrently calls web search tools via **LangChain** to research reported CVEs, maintainer activity, and community issues. It writes its findings to `/security_audit.md`.


* **Context Quarantine:** Neither sub-agent clutters the main manager's context window with raw search results or code blobs; they pass back only high-level status updates while saving detailed notes to the Virtual File System.



#### Stage 4: Executive Report Synthesis

* **User Perspective:** The UI updates: *"Consolidating audit findings into executive report..."*
* **Backend Mechanics:**
* The main Deep Agent manager reads `/code_audit.md` and `/security_audit.md` from the Virtual File System workspace.
* Using **LangChain** prompt structures and model abstractions, the manager consolidates both domain reports into a polished executive summary, saving it as `/final_diligence_report.md`.



#### Stage 5: Human-in-the-Loop (HITL) Review Gate

* **User Perspective:** Execution pauses automatically. The interface presents the full drafted report for your review, alongside two options: **[Approve & Finalize]** or **[Request Revision]** with a feedback text box.
* **Backend Mechanics:**
* **LangGraph** hits a predefined `interrupt_before` node in its state graph.
* It saves the graph state to memory and pauses execution completely, releasing server resources while waiting for your response.



#### Stage 6: Finalization or Self-Correction Loop

* **User Perspective:**
* If you click **Approve**, the system notifies you that the audit is complete.
* If you enter feedback (e.g., *"Re-check dependency licenses in detail"*), the system shows *"Refining report based on user feedback..."* and updates the report accordingly.


* **Backend Mechanics:**
* **LangGraph** resumes execution and evaluates conditional graph logic based on your input:
* **Approval Path:** Transitions state to the final node and completes execution.
* **Revision Path:** Passes your human feedback into the state graph and routes back to the Deep Agent manager. The manager updates its task list (`todo_write`), dispatches sub-agents to gather additional data, updates `/final_diligence_report.md`, and loops back to the HITL gate for re-approval.