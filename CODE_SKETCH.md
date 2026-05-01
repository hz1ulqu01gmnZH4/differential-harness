# Differential Harness — Abstract Code Sketch

**Status:** illustrative pseudocode, not runnable · **Companion to:** `RESEARCH_MEMO.md`

This memo shows what the *shape* of a differential harness API could look
like. Types are Python-flavored but intentionally informal. The point is to
make the moving parts concrete enough to argue about, not to ship.

---

## 1. Core types

```python
# A parameter is the unit of optimization. It carries its current value, a
# schema for valid values, and a list of signal types it accepts.
@dataclass
class Param[T]:
    name: str
    value: T
    schema: Schema[T]            # discrete | continuous | text | code
    accepts: set[SignalKind]     # {GRADIENT, BANDIT, TEXT, EVO, RL}
    history: list[Update]        # for rollback and audit


# A signal is anything that can be attributed back to a parameter.
class Signal: ...

class GradientSignal(Signal):    grad: Tensor
class BanditSignal(Signal):      arm: str; reward: float; ctx: dict
class TextSignal(Signal):        critique: str; severity: float
class RewardSignal(Signal):      scalar: float; turn: int


# A trace record is what each module emits per invocation.
@dataclass
class TraceRecord:
    module: str
    inputs: dict
    params_snapshot: dict          # parameter values at call time
    output: Any
    cost: Cost                     # tokens, dollars, latency
    children: list["TraceRecord"]  # for nested calls (planner → tool → ...)
    timestamp: float
```

## 2. Module interface

Every harness step implements the same protocol. The protocol is small on
purpose: a `forward` that does the work, and a `params` that exposes the
optimizable surface.

```python
class Module(Protocol):
    name: str

    def params(self) -> list[Param]: ...

    def forward(self, *inputs, ctx: RunCtx) -> Any:
        # ctx provides emit(), budget checks, the current trace node, and
        # access to shared services (cache, memory, telemetry).
        ...

    def accept(self, signals: list[Signal]) -> list[Update]:
        # Translate inbound signals into proposed parameter updates.
        # The optimizer decides whether to apply them.
        ...
```

A retrieval module, sketched:

```python
class Retriever(Module):
    name = "retriever"

    def __init__(self):
        self.top_k       = Param("top_k",   8,    IntRange(1, 64),
                                 accepts={BANDIT, EVO})
        self.rerank_temp = Param("rerank_temp", 0.7, FloatRange(0, 2),
                                 accepts={GRADIENT, BANDIT})
        self.query_tmpl  = Param("query_tmpl",
                                 "Rewrite for retrieval: {q}",
                                 TextSchema(),
                                 accepts={TEXT, EVO})

    def forward(self, query, ctx):
        rewritten = llm(self.query_tmpl.value.format(q=query))
        cands     = vector_store.search(rewritten, k=self.top_k.value)
        ranked    = soft_topk(cands, temp=self.rerank_temp.value)  # diff'able
        ctx.emit(inputs={"query": query}, output=ranked)
        return ranked
```

Note three things:
- `top_k` only accepts `BANDIT` / `EVO` — it's discrete; no gradients.
- `rerank_temp` accepts `GRADIENT` because the soft-top-k op is differentiable.
- `query_tmpl` is a string parameter optimized by textual gradients or
  evolutionary proposals.

## 3. The harness as a typed DAG

A harness is a small declarative program. The compiler walks it, wires up
trace emission, and produces a callable.

```python
harness = Harness(
    name="research-agent-v3",
    nodes={
        "router":    Router(models=["sonnet-4.6", "opus-4.7", "haiku-4.5"]),
        "retriever": Retriever(),
        "planner":   Planner(),
        "tools":     ToolBelt([Search(), Calc(), Code()]),
        "critic":    Judge(),
        "memory":    Memory(policy="recency+salience"),
    },
    edges=[
        ("input",     "router"),
        ("router",    "retriever"),
        ("retriever", "planner"),
        ("planner",   "tools"),     # may loop
        ("tools",     "critic"),
        ("critic",    "output"),
        # cross edges
        ("memory",    "planner"),
        ("planner",   "memory"),
    ],
    budgets=Budgets(usd=0.40, latency_ms=8000, tokens=120_000),
    guards=[NoSecrets(), MaxToolCalls(12), ApprovalForWrite()],
)
```

`Budgets` and `Guards` are not magic — their thresholds are themselves
`Param`s, exposed to the optimizer with hard SLO constraints.

## 4. Forward pass = instrumented execution

The runtime executes the harness and accumulates a trace tree. Nothing
unusual; the only contract is that every module call goes through `ctx`
so emission and budget enforcement happen automatically.

```python
def run(harness: Harness, task: Task) -> Result:
    ctx = RunCtx(harness=harness, task=task, budget=harness.budgets)
    with ctx.trace_root():
        out = harness.execute(task.input, ctx=ctx)
    return Result(output=out, trace=ctx.trace, cost=ctx.cost)
```

## 5. Backward pass = signal propagation

This is the load-bearing piece. After a run (or a batch of runs), we have
a trace and an outcome. The backward pass walks the trace and dispatches
signals to module parameters.

```python
def backward(traces: list[TraceRecord],
             outcomes: list[Outcome],
             objective: Metric) -> dict[ParamId, list[Signal]]:

    signals: dict[ParamId, list[Signal]] = defaultdict(list)

    for trace, outcome in zip(traces, outcomes):
        # Step 1: outcome-level scalar signal (always available).
        scalar = objective(outcome)

        # Step 2: localize signal using credit-assignment policy.
        # Options: uniform, last-decision, learned attribution, judge-based.
        attribution = credit_assigner.assign(trace, scalar)

        # Step 3: per-node, decide which signal kind(s) to emit.
        for node in walk(trace):
            module = harness.module(node.module)
            for p in module.params():
                if GRADIENT in p.accepts and node.has_grad():
                    signals[p.id].append(GradientSignal(node.grad_for(p)))
                if BANDIT in p.accepts and node.is_choice():
                    signals[p.id].append(
                        BanditSignal(arm=node.choice, reward=attribution[node]))
                if TEXT in p.accepts:
                    critique = text_judge(trace, node, outcome)
                    signals[p.id].append(TextSignal(critique=critique,
                                                    severity=abs(scalar)))
    return signals
```

A `text_judge` is itself a module; its prompt is a `Param`; therefore the
critic that produces textual gradients is *itself optimizable*. The fixed
point is intentional — it's what TextGrad and Trace exploit.

## 6. The optimizer

The optimizer applies signals to parameters, respecting per-signal
update rules and global constraints (safety SLOs, no-regression on
held-out evals).

```python
class Optimizer:
    handlers: dict[SignalKind, SignalHandler]   # one per signal kind

    def step(self, signals, eval_set, slos) -> Harness:
        proposals: list[Update] = []
        for param_id, sigs in signals.items():
            kind = pick_kind(sigs)               # priority order
            proposals += self.handlers[kind].propose(param_id, sigs)

        candidate = apply(self.harness, proposals)

        # Gate: must beat incumbent on metric *and* satisfy SLOs.
        if eval_set.score(candidate) <= eval_set.score(self.harness):
            return self.harness                  # reject
        if any(slo.violated(candidate, eval_set) for slo in slos):
            return self.harness                  # reject
        snapshot(self.harness)                   # for rollback
        return candidate
```

## 7. Outer loop: Meta-Harness style

The above optimizes parameters within a fixed program graph. To search
over the *graph itself* (add a guard, swap in a new tool, change the
control-flow pattern), the outer loop borrows from Meta-Harness:

```python
def meta_loop(seed: Harness, tasks: TaskSuite, budget: MetaBudget):
    archive = TraceArchive()              # all prior candidates' traces
    current = seed
    while not budget.exhausted():
        proposer = ProposerAgent(model="opus-4.7", tools=[archive.fs()])
        proposal = proposer.propose(current, archive)   # returns new harness code

        result = evaluate(proposal, tasks)
        archive.add(proposal, result)
        if result.dominates(current):
            current = promote(proposal)
    return current
```

The proposer reads prior traces *as a filesystem* (Meta-Harness's key
trick: ~10M tokens of context per step) and writes the next candidate as
code, not as a parameter delta. This is how structural changes enter the
search.

## 8. Eval and promotion

```python
class EvalSuite:
    offline:  list[Task]                  # held-out tasks
    shadow:   ShadowMirror                # fraction of prod traffic
    metrics:  list[Metric]                # primary + guardrail

    def score(self, harness) -> Score: ...
    def regressions(self, candidate, incumbent) -> list[Regression]: ...

class PromotionGate:
    def admit(self, candidate, incumbent, suite) -> bool:
        if not suite.score(candidate).beats(suite.score(incumbent)):
            return False
        if suite.regressions(candidate, incumbent):
            return False
        if any(s.violations(candidate) for s in self.slos):
            return False
        return True
```

Promotion is one direction; rollback is always available because every
promoted harness is snapshotted with its trace archive entry.

## 9. End-to-end pseudocode

```python
def differential_harness_loop():
    harness   = load_seed()
    tasks     = TaskSuite.from_yaml("eval/tasks.yaml")
    optimizer = Optimizer(handlers=default_handlers())

    for epoch in range(N):
        # 1. Run a batch of tasks, collect traces and outcomes.
        traces, outcomes = zip(*[run(harness, t) for t in tasks.sample(32)])

        # 2. Backward pass: derive heterogeneous signals.
        signals = backward(traces, outcomes, objective=tasks.metric)

        # 3. Inner step: apply signals; promotion-gated.
        harness = optimizer.step(signals, eval_set=tasks, slos=SLOS)

        # 4. Periodically: outer step over harness *structure*.
        if epoch % META_EVERY == 0:
            harness = meta_loop(harness, tasks, budget=MetaBudget(usd=20))

        publish(harness, epoch)
```

## 10. What we are *not* writing

- A new model trainer. Models stay frozen.
- A new tracing protocol. Reuse OpenTelemetry / OpenAI Agents SDK traces.
- A new RL library. Wrap an existing bandit lib for routing; lean on
  TextGrad/Trace for the textual-gradient handler; lean on DSPy/GEPA for
  evolutionary proposals.

The novelty is the **integration surface** — one parameter system, one
trace schema, one promotion gate — wrapping all of these so a single
harness can absorb signal from each.

## 11. Smallest useful instance (suggested v0 target)

A three-node harness:

```
input → Router(2 models) → Tool(1: web search) → Judge → output
```

with these parameters:
- `Router.policy` (bandit over 2 models, context = task embedding),
- `Tool.query_tmpl` (textual gradient),
- `Judge.criteria` (textual gradient),
- `budget.usd_per_task` (constant, hard SLO).

Run on ~200 questions with ground-truth answers. Show that the loop
improves task success without breaking the cost SLO, and that rollback
works. That's the minimum viable differential harness.
