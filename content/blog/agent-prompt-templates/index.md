---
title: "Agent Fine-Tuning Starts with the Trajectory: A Review of Prompt Templates"
summary: "How ReAct, ModelScope-Agent, AutoGPT, and ToolLLM structure reasoning, tool use, reflection, and restart behaviour in AI agents."
date: 2026-11-15
authors:
  - me
categories:
  - Literature Reviews
tags:
  - AI Agents
  - Agent Fine-Tuning
  - Tool Use
  - LLMs
cover:
  icon:
    name: "🤖"
image:
  caption: "A literature review of prompt templates used in tool-using AI agents."
---

## Why do prompt templates matter for agent fine-tuning?

An AI agent is more than a language model answering a question once. It may reason about a task, select a tool, execute an action, read the result, and decide what to do next. A **prompt template** defines the structure of this interaction. It tells the model which fields to produce and how an action, its input, an observation, and a final response should be separated.

This is closely connected to agent fine-tuning, but the two ideas should not be confused. A prompt template specifies the desired **trajectory format** at inference time. Fine-tuning uses examples of such trajectories - for example, tool-call records or successful recovery from failed calls - to train a model to follow the format more reliably. The template therefore provides a useful starting point for deciding what training data an agent should contain.

This post reviews four representative designs: ReAct, ModelScope-Agent, AutoGPT, and ToolLLM. The comparison focuses on their prompt structure, rather than attempting to rank the underlying models.

## Comparison of mainstream agent prompt templates

| Template | Description | Advantages | Limitations |
|---|---|---|---|
| **ReAct** [1] | The prompt defines a **Question**, followed by repeated rounds of **Thought**, **Action**, **Action Input**, and **Observation**, before a **Final Answer** is returned. | Simple, readable trajectory design that interleaves reasoning with external actions. | The basic template does not prescribe a dedicated reflection or recovery stage. Tool arguments and multi-step tool state may require extra conventions. |
| **ModelScope-Agent** [2] | ModelScope-Agent uses a relatively direct tool-use format. Its prompt conventions use thought delimiters such as `<\|startofthink\|>` and `<\|endofthink\|>`, together with tool fields such as `command_name` and `args`. | Direct and lightweight format for expressing reasoning and a tool invocation. | The exact prompt syntax is implementation- and version-dependent. A reflection or correction loop is not inherent in the minimal format. |
| **AutoGPT** [3] | AutoGPT uses more detailed instructions for planning, tool invocation, constraints, available resources, and best practices. Tool outputs are commonly represented in a structured form, including JSON-based interfaces. | Explicit constraints can encourage planning, structured tool use, and checking actions against task requirements. | Long system prompts and repeated context can increase token use, latency, and inference cost. |
| **ToolLLM** [4] | ToolLLM builds tool-use capabilities at scale and uses structured action trajectories. Its framework supports a `give_up_and_restart` operation, allowing a new attempt to incorporate records from unsuccessful earlier attempts. | Supports a broader tool-use workflow, including recovery from failed trajectories and training for a large tool/API space. | More complex trajectories require longer prompts. Replaying or restarting a trajectory may substantially increase the number of tokens and tool calls. |

## From templates to fine-tuning data

The table above can be translated into a practical data-design question: *what should one training example teach the model to do?* A minimal ReAct-style sample can contain a user request, one reasoning step, an action, an action input, an observation, and a final answer. A more capable agent dataset should also include:

- valid and invalid tool calls;
- tool schemas and argument-validation examples;
- observations from successful and unsuccessful executions;
- examples where the agent revises a plan after an error;
- explicit stopping behaviour, including when the model should ask the user for clarification rather than call another tool.

For supervised fine-tuning, it is especially important that the expected output is unambiguous. If one example represents a tool call as JSON while another uses free-form text, the model may learn an inconsistent interface. A stable schema also makes it easier to validate generated tool calls programmatically.

## Practical implications

ReAct remains a useful baseline because it separates reasoning, action, and observation with minimal overhead. It is often appropriate when the number of tools is small and the workflow is straightforward. More constrained designs, such as AutoGPT-style prompts, are useful when an agent must satisfy rules about available tools, resources, budgets, or output formatting.

ToolLLM highlights a second requirement for agent fine-tuning: data should not contain only perfect trajectories. Real tool use includes malformed arguments, unavailable tools, incomplete observations, and unproductive plans. Training data that records how an agent identifies a failure, changes its plan, or stops safely is likely more useful than data containing only successful single-pass tool calls.

There is also a deployment trade-off. Richer prompts and restart mechanisms can make an agent more controllable, but every repeated plan, observation, and tool result consumes context length and inference budget. The best format is therefore task-dependent: a small deterministic workflow benefits from a compact schema, while an open-ended research or engineering agent may justify more explicit planning and recovery steps.

## Limitations of this comparison

This table is a high-level comparison of representative prompt and trajectory designs. The exact behaviour of an agent depends on the model, decoding settings, tool descriptions, orchestration code, and the quality of its fine-tuning data. In particular, ModelScope-Agent and AutoGPT are software frameworks whose prompt implementations can change over time. Their rows should therefore be read as implementation-level summaries, not as fixed theoretical definitions.

## References

1. Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y. (2023). ReAct: Synergizing Reasoning and Acting in Language Models. *International Conference on Learning Representations (ICLR).* https://arxiv.org/abs/2210.03629

2. ModelScope. (n.d.). *ModelScope-Agent: An Agent Framework for Building Customised Intelligent Agents.* GitHub repository. https://github.com/modelscope/modelscope-agent

3. Significant Gravitas. (2023). *AutoGPT: An Experimental Open-Source Attempt to Make GPT-4 Fully Autonomous.* GitHub repository. https://github.com/Significant-Gravitas/AutoGPT

4. Qin, Y., Liang, S., Ye, Y., Zhu, K., Yan, L., Lu, Y., Ni, Y., Hu, X., et al. (2024). ToolLLM: Facilitating Large Language Models to Master 16000+ Real-World APIs. *International Conference on Learning Representations (ICLR).* https://arxiv.org/abs/2307.16789
