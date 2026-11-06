"""Communication-minimizing DAG pebbling via re-use-aware scheduling + fast-memory caching."""
# EVOLVE-BLOCK-START
import time

def _topo_order(dag):
    """Return vertex order that respects parent-before-child (topological)."""
    parents = dag.get("parents", {})
    vertices = list(parents.keys())
    if not vertices:
        return []
    # Kahn's algorithm
    in_deg = {v: 0 for v in vertices}
    children = {v: [] for v in vertices}
    for v, ps in parents.items():
        for p in ps:
            in_deg[v] = in_deg.get(v, 0) + 1
            children.setdefault(p, []).append(v)
    # Ensure all referenced parents are in the vertex set
    for v, ps in parents.items():
        for p in ps:
            if p not in in_deg:
                in_deg[p] = 0
                children.setdefault(p, [])
    from collections import deque
    q = deque([v for v in vertices if in_deg[v] == 0])
    order = []
    while q:
        v = q.popleft()
        order.append(v)
        for c in children.get(v, []):
            in_deg[c] -= 1
            if in_deg[c] == 0:
                q.append(c)
    # Fallback: append any missing (cycle or isolated)
    if len(order) != len(in_deg):
        for v in in_deg:
            if v not in order:
                order.append(v)
    return order

def _count_transfers(ops):
    """Count load/store operations (the communication measure)."""
    c = 0
    for op, v in ops:
        if op in ("load", "store"):
            c += 1
    return c

def _naive_transfers(dag):
    """Baseline: load each input once, compute each vertex, store each output."""
    parents = dag.get("parents", {})
    outputs = dag.get("outputs", [])
    is_input = {}
    for v, ps in parents.items():
        is_input[v] = (len(ps) == 0)
    loads = sum(1 for v in parents if is_input.get(v, False))
    stores = len(outputs)
    return loads + stores

def _re_use_order(dag, order):
    """Order vertices by descending re-use count (number of children), stable topo."""
    parents = dag.get("parents", {})
    children = {v: [] for v in parents}
    for v, ps in parents.items():
        for p in ps:
            children.setdefault(p, []).append(v)
    re_use = {v: len(children.get(v, [])) for v in parents}
    # Process in reverse topo so children scheduled before parents when possible,
    # but we want parents computed before children. Use a priority queue by re-use.
    in_deg = {v: 0 for v in parents}
    for v, ps in parents.items():
        for p in ps:
            in_deg[v] = in_deg.get(v, 0) + 1
    from heapq import heappush, heappop
    import itertools
    counter = itertools.count()
    heap = []
    for v in parents:
        if in_deg[v] == 0:
            heappush(heap, (-re_use[v], next(counter), v))
    order2 = []
    while heap:
        _, _, v = heappop(heap)
        order2.append(v)
        for c in children.get(v, []):
            in_deg[c] -= 1
            if in_deg[c] == 0:
                heappush(heap, (-re_use[c], next(counter), c))
    if len(order2) != len(parents):
        for v in parents:
            if v not in order2:
                order2.append(v)
    return order2

def _greedy_pebble(dag, fast_memory_budget, order):
    """Simulate pebbling: load inputs, compute, cache intermediates in fast memory.
    Returns (ops, transfers)."""
    parents = dag.get("parents", {})
    outputs = set(dag.get("outputs", []))
    is_input = {v: (len(ps) == 0) for v, ps in parents.items()}

    # fast memory: vertex -> value slot (we track which are cached)
    fast = {}   # v -> True (cached in fast memory)
    ops = []
    transfers = 0
    # inputs start blue (available in slow memory)
    for v in order:
        if is_input[v]:
            # input already blue in slow memory; if we need it, load it
            # We'll load it when first needed. For now, mark as available in slow.
            continue
        # ensure parents available
        for p in parents[v]:
            if p not in fast:
                # p not in fast memory: load from slow (or recompute)
                # if p is input, it's in slow; else it was stored to slow
                ops.append(["load", p])
                transfers += 1
                fast[p] = True
        # compute v
        ops.append(["compute", v])
        # decide whether to keep v in fast memory
        if len(fast) < fast_memory_budget:
            fast[v] = True
        else:
            # evict a vertex that is NOT an output and has lowest future re-use
            # (simple: evict an input or a non-output intermediate)
            evictable = [x for x in fast if x not in outputs]
            if evictable:
                # evict one with lowest re-use (approx: inputs first)
                evictable.sort(key=lambda x: (x in is_input and is_input[x], 0))
                # prefer evicting inputs
                ev = evictable[0]
                del fast[ev]
                ops.append(["delete", ev])
            # v stays in fast
            fast[v] = True
    # store outputs
    for v in outputs:
        if v not in fast:
            # need it available; if input, load
            if is_input.get(v, False):
                ops.append(["load", v])
                transfers += 1
                fast[v] = True
        ops.append(["store", v])
        transfers += 1
    return ops, transfers

def pebble(dag, fast_memory_budget):
    t0 = time.time()
    parents = dag.get("parents", {})
    outputs = dag.get("outputs", [])
    order = _topo_order(dag)
    re_use_order = _re_use_order(dag, order)

    best_ops = None
    best_transfers = None
    for o in (re_use_order, order):
        ops, transfers = _greedy_pebble(dag, fast_memory_budget, o)
        if best_transfers is None or transfers < best_transfers:
            best_transfers = transfers
            best_ops = ops
        if time.time() - t0 > 85:
            break
    if best_ops is None:
        ops, _ = _greedy_pebble(dag, fast_memory_budget, order)
        best_ops = ops
    return best_ops
# EVOLVE-BLOCK-END
