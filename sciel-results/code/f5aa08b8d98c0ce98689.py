"""Communication-minimizing DAG pebbling (self-contained, no external imports)."""
import time

# EVOLVE-BLOCK-START


def _time_budget():
    # Evaluator kills after 90s wall-clock; leave a safety margin.
    return 75.0


def _naive_ops(dag):
    """Transfers of the naive topological schedule (no eviction, no recompute).

    Each input loaded once, each output stored once.
    """
    n = dag["n_vertices"]
    parents = dag["parents"]
    outputs = dag["output_vertices"]
    ops = len(outputs)
    for v in range(n):
        if len(parents[v]) == 0:
            ops += 1
    return ops


def pebble(dag, fast_memory_budget):
    """Return a list of [operation, vertex] pebbling the DAG.

    Operations: 'load', 'store', 'compute', 'delete'.
    Inputs start blue (in slow memory); all outputs must finish blue (in fast
    memory); every red value occupies one fast-memory slot, including newly
    computed results.
    """
    start = time.time()
    deadline = start + _time_budget()

    n = dag["n_vertices"]
    parents = dag["parents"]
    outputs = list(dag["output_vertices"])
    output_set = set(outputs)
    budget = fast_memory_budget

    # Fast path: budget large enough to hold every vertex simultaneously.
    if budget >= n:
        return _pebble_no_evict(dag)

    # Otherwise: greedy LRU pebbling with correct bookkeeping.
    schedule = _greedy_lru(dag, budget, deadline)

    # Guarantee outputs end in fast memory (blue). If any output was evicted,
    # re-materialize it at the end (recompute path). This keeps validity.
    schedule = _ensure_outputs_blue(dag, budget, schedule, deadline)

    return schedule


def _pebble_no_evict(dag):
    """Schedule when budget >= n: no eviction, no recompute."""
    n = dag["n_vertices"]
    parents = dag["parents"]
    outputs = dag["output_vertices"]
    ops = []
    for v in range(n):
        if len(parents[v]) == 0:
            ops.append(["load", v])
    for v in range(n):
        ops.append(["compute", v])
    for v in outputs:
        ops.append(["store", v])
    return ops


def _greedy_lru(dag, budget, deadline):
    """Greedy LRU pebbling in topological order.

    Invariant: a vertex is in fast memory iff it is 'red'. We process vertices
    in topological order. To compute v, all parents must be red; if a parent
    is blue (evicted or not yet loaded), we must bring it red again:
      - if it is an input: 'load' it (evicting if necessary);
      - if it is a computed non-input: recompute it (which recursively brings
        its parents red), then 'compute' it.
    After computing v, evict (LRU) to stay within budget, but never evict a
    vertex that is still needed as a parent of a not-yet-computed vertex AND
    is cheaper to keep than to recompute -- a simple heuristic is to never
    evict outputs and to prefer evicting vertices with no remaining children.
    """
    n = dag["n_vertices"]
    parents = dag["parents"]
    outputs = dag["output_vertices"]
    output_set = set(outputs)

    # Children and remaining-child counts (for eviction preference).
    children = [[] for _ in range(n)]
    for v in range(n):
        for p in parents[v]:
            children[p].append(v)
    remaining = [len(children[v]) for v in range(n)]

    ops = []
    in_fast = set()
    computed = set()  # vertices whose value has been computed at least once
    lru = {}
    t = 0

    def touch(v):
        nonlocal t
        t += 1
        lru[v] = t

    def evict_one():
        # Choose victim: prefer non-output, then max remaining children is bad
        # (we want to evict vertices with fewest remaining uses), then LRU.
        best = None
        best_key = None
        for v in in_fast:
            if v in output_set:
                continue  # never evict outputs (must end blue)
            key = (remaining[v], lru.get(v, 0))
            if best_key is None or key < best_key:
                best_key = key
                best = v
        if best is None:
            # All in fast are outputs; must evict something (fallback: LRU).
            best = min(in_fast, key=lambda v: lru.get(v, 0))
        return best

    def bring_red(v):
        """Ensure vertex v is in fast memory (red)."""
        if v in in_fast:
            touch(v)
            return
        if len(parents[v]) == 0:
            # input: load it.
            if len(in_fast) >= budget:
                e = evict_one()
                ops.append(["delete", e])
                in_fast.discard(e)
            ops.append(["load", v])
            in_fast.add(v)
            touch(v)
        else:
            # non-input: recompute it (bring parents red, then compute).
            for p in parents[v]:
                bring_red(p)
            if len(in_fast) >= budget:
                e = evict_one()
                ops.append(["delete", e])
                in_fast.discard(e)
            ops.append(["compute", v])
            in_fast.add(v)
            computed.add(v)
            touch(v)

    # Load inputs lazily (on demand) to save fast slots early.
    for v in range(n):
        if time.time() > deadline:
            break
        for p in parents[v]:
            bring_red(p)
        # Compute v.
        if len(in_fast) >= budget:
            e = evict_one()
            ops.append(["delete", e])
            in_fast.discard(e)
        ops.append(["compute", v])
        in_fast.add(v)
        computed.add(v)
        touch(v)
        # Update remaining counts of parents.
        for p in parents[v]:
            remaining[p] -= 1
        # Store output.
        if v in output_set:
            ops.append(["store", v])

    return ops


def _ensure_outputs_blue(dag, budget, ops, deadline):
    """Verify (and fix) that all outputs end in fast memory.

    In _greedy_lru we never evict outputs, so they remain red at the end.
    This is a safety check; if an output is missing, re-materialize it.
    """
    n = dag["n_vertices"]
    parents = dag["parents"]
    outputs = set(dag["output_vertices"])

    # Simulate to see which outputs are in fast at the end.
    in_fast = set()
    for op, v in ops:
        if op == "load" or op == "compute":
            in_fast.add(v)
        elif op == "delete":
            in_fast.discard(v)
        # store does not remove from fast.

    missing = outputs - in_fast
    if not missing:
        return ops

    # Re-materialize missing outputs (recompute path).
    extra = []
    # Build a simple recompute: for each missing output, recompute it.
    # (Rare path; keep simple.)
    for v in sorted(missing):
        # Recompute v and its ancestors as needed.
        stack = [v]
        seen = set()
        order = []
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            order.append(x)
            for p in parents[x]:
                stack.append(p)
        for x in reversed(order):
            if len(parents[x]) == 0:
                extra.append(["load", x])
            else:
                extra.append(["compute", x])
        extra.append(["store", v])
    return ops + extra


# EVOLVE-BLOCK-END
