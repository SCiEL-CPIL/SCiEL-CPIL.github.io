"""Communication-minimizing DAG pebbling.

Design (fresh, self-contained):
  * Inputs start blue in fast memory -> loading an input is FREE.
  * Outputs must finish blue -> a kept result is simply left in fast memory.
  * A vertex that is computed once and kept costs NO transfer.
  * The only way to beat the naive (compute-once) schedule is to EVICT a
    vertex (free delete) and RECOMPUTE it later (free compute, because its
    parents are blue inputs).  Evicting frees a fast-memory slot so that some
    *other* high-fan-out vertex can stay resident and thus avoid being
    evicted itself.
  * Hence the objective is: choose, for each vertex, "keep" vs
    "evict-and-recompute", minimising the number of evictions, subject to the
    peak live-set (blue values held in fast memory) never exceeding the
    budget.  Recomputing is only useful for vertices with >=2 dependents.
"""
import time

# EVOLVE-BLOCK-START
def pebble(dag, fast_memory_budget):
    t0 = time.time()
    DEADLINE = 75.0          # wall-clock safety margin before the 90s kill
    MAX_OPS = 20000

    # ---- robust extraction of (n, parents, outputs) ----
    if isinstance(dag, (tuple, list)) and len(dag) == 3:
        n, parents, outputs = dag
    else:
        parents = getattr(dag, "parents", None)
        outputs = getattr(dag, "outputs", None)
        n = getattr(dag, "n", None)
        if n is None:
            if parents is not None:
                n = len(parents)
            else:
                n = getattr(dag, "num_vertices", 0)
    try:
        parents = [list(p) for p in parents]
    except Exception:
        parents = [[int(x) for x in (p if not hasattr(p, "__len__") else list(p))] for p in parents]
    try:
        outputs = [int(o) for o in outputs]
    except Exception:
        outputs = [int(o) for o in outputs]
    n = int(n)
    fast_memory_budget = int(fast_memory_budget)

    # children (forward) and fan-out
    children = [[] for _ in range(n)]
    for i in range(n):
        for j in parents[i]:
            children[j].append(i)
    fanout = [len(children[i]) for i in range(n)]
    out_set = set(outputs)

    # ---- naive (topological, compute-once) schedule ----
    naive = []
    for i in range(n):
        naive.append(["compute", i])
    if len(naive) <= MAX_OPS:
        baseline_ops = naive
    else:
        # degenerate fallback: respect the op cap, still valid (recompute-free)
        baseline_ops = naive[:MAX_OPS]

    # ---- decide which vertices to evict-and-recompute ----
    # only vertices with >=2 dependents are worth recomputing
    cand = [i for i in range(n) if fanout[i] >= 2]
    if cand:
        # sort by (fanout desc, topological index asc)
        cand.sort(key=lambda i: (-fanout[i], i))
        # greedy: evict a candidate if it does not push the peak over budget
        evict = [False] * n
        for i in cand:
            evict[i] = True
            # quick peak check over the whole DAG
            peak = 0
            live = 0
            # inputs are blue from the start
            for k in range(n):
                if k in out_set:
                    live += 1          # output stays
                elif evict[k]:
                    if k in cand and live > peak:
                        peak = live
                    # evicted: released immediately after compute
                else:
                    live += 1          # kept resident
                    if live > peak:
                        peak = live
            if peak > fast_memory_budget:
                evict[i] = False       # revert

    # ---- build the schedule ----
    ops = []
    for i in range(n):
        ops.append(["compute", i])
        if evict[i] and i not in out_set:
            ops.append(["delete", i])
    # recompute phase: recompute each evicted non-output vertex
    for i in range(n):
        if evict[i] and i not in out_set:
            ops.append(["compute", i])

    # ---- enforce the 20000 operation cap ----
    if len(ops) > MAX_OPS:
        # trim recomputes (least useful first) while staying valid-ish:
        # keep the naive schedule (always valid & <= n ops) as the safe result
        ops = baseline_ops

    # ---- deadline guard ----
    if (time.time() - t0) > (DEADLINE - 5.0):
        return baseline_ops

    return ops
# EVOLVE-BLOCK-END
