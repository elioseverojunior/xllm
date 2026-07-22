---
name: instructions.md
created_by: elioseverojunior@gmail.com
---

# Intructions

I want to build a new Project called `xllm` pure in rust lang. However it's must follow the existing project `llama.cpp`.
Our project must have everyting that `llama.cpp` has and other enhancements
the will make our project much better that the original one. I want ensure
the we will implement LLM and vLLM supporting.
i want to focus to run on modles using CPU, it must be fast.
On feature I want to enable GPu models support (AMD and Nvidia GPUs), so we need to think on design that will handle this.

## Rust SDK Requirements

- Must use Rust (stable) that allow new features.
- We must focus on Security-first and Performance

## AI Requirements

You must to write the needed agents, skill, prompts and other configs including ai-agentics.
I want AI agnotic configurations that can be rused by Claude, Codex, Copilot, Bedrock and so ones.
I want the AI be alble to do fully autonomus work.
Check the [docs/guidelines/contribution.md](docs/guidelines/contribution.md)

## Coding Specifications

Ensure we use TDD before write any codeline.
Ensure the development principles:

1. The codebase must have 100% of coverage.
2. TDD (If we need to change an existing code without TDD, first write the
   test using TDD and ensure the test is working and then start to
   refactor the context we intended to modify, always doing TDD
   implementation loop red -> code -> loop test till green.)
3. KISS (Keep It Simple, Stupid)
4. DRY (Don’t Repeat Yourself)
5. YAGNI (You Aren’t Gonna Need It)
6. TDA (Tell Don’t Ask)
7. SOLID (Use the SOLID Principles that make sense to the project).

Write the plan into [docs/plan/XLLM_PLAN.md](docs/plan/XLLM_PLAN.md)
