# Differential Harness — Research Memo

**Status:** draft v0.1 · **Date:** 2026-05-01

## 1. Thesis

A *differential harness* is an agent harness whose every meaningful choice
— prompts, retrieval parameters, tool routing, control flow, guards,
context-management policies — is exposed as a typed parameter and
optimized end-to-end against a task metric. "Differential" here generalizes
beyond numeric gradients: any signal that can be **attributed to a specific
harness decision** counts (true gradients, bandit rewards, textual
gradients, evolutionary fitness, process-supervised judges).

The thesis is not new optimizers. The thesis is that the *unit of
optimization* should move up a level: from the model weights, and even from
the prompt, to the **harness program itself**.

## 2. Why now

Three independent lines of work converged in late 2025 / early 2026:

1. **Harness engineering** became a named discipline. Hashimoto's "Engineer
   the Harness" (Feb 2026), Fowler's *guides vs sensors* framing, and the
   OpenAI/Anthropic posts established that for production agents the
   harness — not the model — is the dominant lever for reliability.

2. **Textual-gradient frameworks matured.** TextGrad and Microsoft's Trace
   reframed an LLM pipeline's execution trace as an optimization graph;
   DSPy's GEPA showed reflective program evolution beats hand-tuning at
   scale.

3. **Outer-loop harness optimizers shipped.** Meta-Harness (search over
   harness *code* with full-trace context) and AutoHarness (synthesize a
   constraint-enforcing code harness from environment feedback) demonstrate
   that the harness itself is a viable optimization target, not just a
   container.

Combine them and the natural construct is: a harness whose source of truth
is a **declarative program with optimizable parameters**, executed against
**rich traces**, optimized by **mixed signals**, and shipped with
**continuous shadow evaluation**.

## 3. Conceptual model

### 3.1 The harness as a typed DAG

Model the harness as a directed acyclic graph of typed steps:

```
input ─► Router ─► Retriever ─► Planner ─► [Tool*] ─► Critic ─► Output
                       │                       ▲
                       └──── Memory ◄──────────┘
```

Each node is a **module** with:
- **typed inputs/outputs** (so signals can be routed correctly),
- **explicit parameters** (prompt template, top-k, threshold, model id, ...),
- **a forward implementation** (LLM call, retrieval op, deterministic code),
- **an emit() that writes a trace record** (inputs, params, output, cost, latency).

The *graph itself* is also a parameter — branches and loop bodies can be
gated by routers whose policies are tunable.

### 3.2 The trace as the optimization graph

Following Trace and TextGrad, the **execution trace is the optimization
graph**. A backward pass over the trace propagates whatever signal is
available — numeric gradient, scalar reward, natural-language critique —
back to the parameters of each module that contributed to the output.

This unifies what would otherwise be three disjoint loops (prompt
optimization, RL fine-tuning, retrieval tuning) under one programming
model.

### 3.3 Signals are heterogeneous

Different boundaries admit different signals. The differential harness
treats this as first-class: each parameter declares which signal types it
accepts.

| Boundary | Differentiable? | Preferred signal |
|---|---|---|
| Continuous prompt embeddings, soft top-k | yes | numeric gradient |
| Reranker weights, small policies | partial | RL / supervised on traces |
| Discrete tool / model routing | no | contextual bandit |
| Prompt text, instructions, few-shots | no | textual gradient |
| Whole-program structure, control flow | no | evolutionary / Meta-Harness search |
| Guards, judges, thresholds | no | process-supervised RL |

The optimizer's job is to dispatch the right signal to the right knob, not
to pretend everything is a gradient.

## 4. Architecture sketch

Five layers, each replaceable:

1. **Module library** — typed building blocks (Router, Retriever, Planner,
   Tool, Critic, Memory, Guard). Each declares its parameter schema and
   accepted signals.
2. **Compiler** — turns a declarative program into an executable harness;
   resolves parameter bindings, instruments emit() calls, builds the trace
   schema.
3. **Runtime** — executes the harness, captures traces, enforces budgets
   (cost, latency, tokens), exposes guards as tunable sensors rather than
   frozen rules.
4. **Optimizer** — runs the inner loop (per-trace signal propagation) and
   the outer loop (program-level proposals à la Meta-Harness). Holds a
   library of signal handlers (gradient, bandit, textual, evolutionary).
5. **Eval & promotion** — held-out task suites, shadow-traffic mirroring,
   snapshot/rollback. A candidate harness is **promoted** only after
   passing budget and safety SLOs.

## 5. Hard problems

- **Credit assignment over long trajectories.** Outcome reward is sparse;
  attach signal close to decisions via process supervision and trace-local
  judges. Trace's "minimal subgraph" idea is a useful default.
- **Non-stationarity.** Underlying models, prices, and tools shift weekly.
  Routers should be online learners (contextual bandits) rather than
  compiled-once policies.
- **Eval cost.** A naïve outer loop calls the harness thousands of times.
  Mitigations: smaller "reflection" models for proposal, cached tool
  responses, Pareto selection, escalate-on-promise.
- **Safety as a parameter, not a wall.** Guards have thresholds; thresholds
  are tunable; tunable means optimizable. Constrain the optimizer with
  hard safety SLOs, not just soft penalties.
- **Reproducibility.** Versioned traces + versioned harness snapshots are
  non-negotiable. Without them, optimization regresses silently.
- **Verbosity / drift collapse.** Documented failure mode of textual
  gradients: prompts grow until they damage performance. Use length /
  perplexity priors or Pareto eval against a smaller-prompt baseline.

## 6. Evaluation strategy

- **Offline:** held-out task suite per capability (routing accuracy,
  retrieval recall@k, end-to-end task success, cost per task).
- **Online shadow:** mirror a fraction of production traffic to the
  candidate harness; compare against the incumbent on success, cost, and
  safety violations.
- **Promotion gate:** strictly better on the primary metric, no regression
  beyond ε on any guard, within the cost budget.
- **Rollback:** every promoted snapshot is one command from being undone.

## 7. Out of scope (for v0)

- Training model weights. The harness optimizes around frozen models.
- Multi-tenant per-user personalization. Single global harness first.
- Cross-organization federation of traces.

## 8. Open questions

1. Is the trace schema worth standardizing, or should each module own its
   own? (Trade-off: portability vs. expressiveness.)
2. How much of the outer loop should run *inside* the harness as a
   self-modifying agent vs. *outside* as a separate optimizer process?
   Meta-Harness chose outside; that probably generalizes.
3. Should the parameter space be typed strongly enough that the optimizer
   can refuse invalid moves, or weakly enough that it can discover
   surprising ones?
4. What is the smallest useful "differential harness" — a router + a single
   tool + a judge — that demonstrates the loop without being a toy?
5. How do we make the safety SLO part of the parameter space without
   making it gameable?

## 9. References

- Hashimoto, *My AI Adoption Journey* — "Engineer the Harness" (Feb 2026).
- Fowler, *Harness engineering for coding agent users* (2026).
- OpenAI, *Harness engineering: leveraging Codex in an agent-first world*.
- Anthropic, *Effective context engineering for AI agents*.
- *Meta-Harness: End-to-End Optimization of Model Harnesses* (2026).
- *AutoHarness: improving LLM agents by automatically synthesizing a code
  harness* (arXiv:2603.03329).
- Cheng et al., *Trace: AutoDiff for agents* (NeurIPS 2024).
- Yuksekgonul et al., *TextGrad* (arXiv:2406.07496).
- Khattab et al., *DSPy* (arXiv:2310.03714); *GEPA* (2025).
