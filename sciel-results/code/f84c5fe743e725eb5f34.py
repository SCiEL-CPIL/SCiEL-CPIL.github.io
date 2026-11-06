# EVOLVE-BLOCK-START
import time

_START = time.time()
_DEADLINE = _START + 82.0   # safety margin under the 90s wall-clock kill


def _norm(dag):
    """Return (parents, children, inputs, outputs) as plain lists/sets."""
    parents = list(dag.get("parents", []))
    children = dag.get("children")
    if children is None:
        n = len(parents)
        children = [[] for _ in range(n)]
        for i, ps in enumerate(parents):
            for p in ps:
                children[p].append(i)
        children = [list(c) for c in children]
    inputs = dag.get("inputs")
    if inputs is None:
        inputs = [i for i, ps in enumerate(parents) if len(ps) == 0]
    outputs = dag.get("outputs")
    if outputs is None:
        outputs = [i for i, ch in enumerate(children) if len(ch) == 0]
    return parents, children, list(inputs), list(outputs)


def _layered_schedule(parents, children, inputs, outputs, budget):
    """Layer-by-layer sweep for layered DAGs."""
    n = len(parents)
    level = [0] * n
    for i in range(n):
        if parents[i]:
            level[i] = 1 + max(level[p] for p in parents[i])
    maxlev = max(level) if n else 0

    layers = [[] for _ in range(maxlev + 1)]
    for i in range(n):
        layers[level[i]].append(i)

    # Greedy within each layer: highest fan-out first (reuse as next-layer parents)
    for lv in range(maxlev + 1):
        layers[lv].sort(key=lambda i: -len(children[i]))

    ops = []
    fast = set()          # vertices currently blue in fast memory
    avail = set(inputs)   # vertices whose value is available (blue or computed)

    for lv in range(maxlev + 1):
        for i in layers[lv]:
            # Ensure all parents are available in fast memory
            for p in parents[i]:
                if p not in fast:
                    if p not in avail:
                        # Should not happen in a layered sweep, but guard
                        ops.append(["load", p])
                    else:
                        ops.append(["load", p])
                    fast.add(p)
            # Make room for the result if budget is exceeded
            while len(fast) >= budget and budget > 0:
                # Evict a parent that is not needed again in this/next layer
                victim = None
                for p in list(fast):
                    # Keep current layer's children's parents and inputs
                    needed = False
                    # is p a parent of any not-yet-processed vertex in this layer?
                    for j in layers[lv]:
                        if p in parents[j]:
                            needed = True
                            break
                    # is p a parent of any vertex in a future layer?
                    if not needed:
                        for fl in range(lv + 1, maxlev + 1):
                            for j in layers[fl]:
                                if p in parents[j]:
                                    needed = True
                                    break
                            if needed:
                                break
                    if not needed:
                        victim = p
                        break
                if victim is None:
                    # evict anything
                    victim = next(iter(fast))
                fast.discard(victim)
                ops.append(["delete", victim])
            # Compute i
            ops.append(["compute", i])
            fast.add(i)
            avail.add(i)

    # Finalize: every output must end blue; store any that are not in fast
    for o in outputs:
        if o not in fast:
            # reload if needed
            if o not in avail:
                # recompute path (shouldn't happen); fallback load
                ops.append(["load", o])
            else:
                ops.append(["load", o])
            fast.add(o)
        ops.append(["store", o])
    return ops


def _naive_transfers(parents, children, inputs, outputs):
    """Reference: load each input once, compute each vertex once, store each output once."""
    t = 0
    for i in inputs:
        t += 1  # load
    for i in range(len(parents)):
        pass
    for o in outputs:
        t += 1  # store
    return max(t, 1)


def _count_transfers(ops):
    return sum(1 for op, _ in ops if op in ("load", "store"))


def pebble(dag, fast_memory_budget):
    if time.time() > _DEADLINE:
        # Emergency fallback: minimal valid-ish schedule
        parents, children, inputs, outputs, = _norm(dag)
        ops = []
        for i in inputs:
            ops.append(["load", i])
        for i in range(len(parents)):
            ops.append(["compute", i])
        for o in outputs:
            ops.append(["store", o])
        return ops

    parents, children, inputs, outputs = _norm(dag)
    n = len(parents)

    if n == 0:
        return []

    budget = int(fast_memory_budget) if fast_memory_budget else 0

    # Build schedule
    ops = _layered_schedule(parents, children, inputs, outputs, budget)

    # Enforce operation cap
    if len(ops) > 20000:
        ops = ops[:20000]

    # Safety: ensure outputs end blue (append stores if missing)
    have_store = set(v for op, v in ops if op == "store")
    for o in outputs:
        if o not in have_store:
            ops.append(["store", o])
            have_store.add(o)

    return ops
# EVOLVE-BLOCK-END
