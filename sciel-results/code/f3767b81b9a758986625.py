"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

def _topo_order(parents, outputs, vertices):
    vset = set(vertices)
    children = {v: [] for v in vertices}
    indeg = {v: 0 for v in vertices}
    for v, ps in parents.items():
        if v not in vset:
            continue
        for p in ps:
            if p in vset:
                children[p].append(v)
                indeg[v] += 1
    q = deque([v for v in vertices if indeg[v] == 0])
    topo = []
    while q:
        u = q.popleft()
        topo.append(u)
        for w in children.get(u, []):
            indeg[w] -= 1
            if indeg[w] == 0:
                q.append(w)
    if len(topo) != len(vertices):
        topo = vertices
    return topo, children

def _pebble_postorder(dag, budget):
    # Corrected postorder pebbling with proper loaded-state tracking.
    # Inputs start blue (loaded). Pebble each output's subtree recursively,
    # deleting intermediates no longer needed, reloading/recomputing when a
    # parent was deleted but is still required by a later sibling.
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
    if not vertices:
        return []
    vset = set(vertices)
    outset = set(outputs)

    topo, children = _topo_order(parents, outputs, vertices)
    post = list(reversed(topo))

    ops = []
    loaded = set()
    # Inputs (vertices with no parents) start blue -> already loaded.
    for v in vertices:
        if not parents.get(v, []):
            loaded.add(v)
    stored = set()

    def pebble(v):
        # Ensure v is computed and blue (in fast memory). Reload/recompute if not.
        if v in loaded:
            return
        ps = parents.get(v, [])
        for p in ps:
            pebble(p)
        ops.append(["compute", v])
        loaded.add(v)
        # Unpebble parents that are no longer needed by any pending vertex.
        for p in ps:
            if p in outset or p in stored:
                continue
            needed = False
            for w in children.get(p, []):
                if w != v and w not in loaded and w not in stored:
                    needed = True
                    break
            if not needed:
                ops.append(["delete", p])
                loaded.discard(p)

    for v in post:
        if v in outset:
            pebble(v)
            ops.append(["store", v])
            stored.add(v)
            loaded.discard(v)
        # Non-output vertices are pebbled lazily via their output descendants.

    return ops[:20000]

def _pebble_fallback(dag, budget):
    # Greedy baseline: load parents, compute, store outputs, delete when possible.
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
    if not vertices:
        return []
    vset = set(vertices)
    ops = []
    outset = set(outputs)
    topo, children = _topo_order(parents, outputs, vertices)
    for v in topo:
        for p in parents.get(v, []):
            ops.append(["load", p])
        ops.append(["compute", v])
        if v in outset:
            ops.append(["store", v])
        for p in parents.get(v, []):
            still = any(p in parents.get(w, []) for w in topo[topo.index(v)+1:] if w in vset)
            if not still and p not in outset:
                ops.append(["delete", p])
    return ops[:20000]

def pebble(dag, fast_memory_budget):
    start_t = time.time()
    deadline = start_t + 50  # safety margin before 60s / 90s kill

    candidates = []

    # 1) Corrected postorder pebbling (polynomial, minimizes transfers).
    try:
        res = _pebble_postorder(dag, fast_memory_budget)
        if res is not None:
            candidates.append(res)
    except Exception:
        pass

    # 2) Greedy fallback.
    try:
        res = _pebble_fallback(dag, fast_memory_budget)
        if res is not None:
            candidates.append(res)
    except Exception:
        pass

    # Pick the candidate with the fewest transfers (load + store), within op limit.
    def transfers(seq):
        return sum(1 for op, _ in seq if op in ("load", "store"))

    best = None
    best_score = float("inf")
    for c in candidates:
        if c is None:
            continue
        if len(c) > 20000:
            continue
        s = transfers(c)
        if s < best_score:
            best_score = s
            best = c
    if best is not None:
        return best
    # last resort
    try:
        return _pebble_fallback(dag, fast_memory_budget)[:20000]
    except Exception:
        return []
# EVOLVE-BLOCK-END
