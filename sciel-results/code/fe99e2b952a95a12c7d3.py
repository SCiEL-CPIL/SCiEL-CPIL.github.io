"""Rewritten pebbling solver: layered frontier schedule with greedy LRU/least-remaining-use eviction."""
# EVOLVE-BLOCK-START
import time

def pebble(dag, fast_memory_budget):
    t0 = time.time()
    DEADLINE = 70.0  # safety margin under 90s kill

    def _elapsed():
        return time.time() - t0

    # ---- Parse the DAG ----
    # dag format (from baseline): dict with 'parents' mapping vertex -> list of parents,
    # 'inputs' list, 'outputs' list. We also accept a few common variants.
    parents = {}
    inputs = []
    outputs = []
    if isinstance(dag, dict):
        parents = dict(dag.get('parents', dag.get('parent', {})))
        inputs = list(dag.get('inputs', dag.get('input', [])))
        outputs = list(dag.get('outputs', dag.get('output', [])))
    elif isinstance(dag, (list, tuple)):
        # maybe list of (vertex, parents)
        for item in dag:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                v, ps = item[0], item[1]
                parents[v] = list(ps)
        inputs = [v for v, ps in parents.items() if not ps]
        # outputs: vertices with no children
        children_set = set()
        for v, ps in parents.items():
            for p in ps:
                children_set.add(p)
        outputs = [v for v in parents if v not in children_set]

    # Normalize vertex keys
    vertices = list(parents.keys())
    if not vertices:
        return []

    # Build children map
    children = {v: [] for v in vertices}
    for v, ps in parents.items():
        for p in ps:
            if p not in children:
                children[p] = []
            children[p].append(v)

    inputs_set = set(inputs)
    outputs_set = set(outputs)
    if not inputs:
        inputs = [v for v in vertices if not parents.get(v)]
        inputs_set = set(inputs)
    if not outputs:
        outputs = [v for v in vertices if not children.get(v)]
        outputs_set = set(outputs)

    # ---- Layer by longest path depth ----
    depth = {}
    order = []
    # Kahn topological order
    indeg = {v: len(parents.get(v, [])) for v in vertices}
    from collections import deque
    q = deque([v for v in vertices if indeg[v] == 0])
    topo = []
    while q:
        v = q.popleft()
        topo.append(v)
        for c in children.get(v, []):
            indeg[c] -= 1
            if indeg[c] == 0:
                q.append(c)
    if len(topo) != len(vertices):
        # cycle fallback: use given order
        topo = vertices
    for v in topo:
        ps = parents.get(v, [])
        depth[v] = 0 if not ps else max(depth.get(p, 0) for p in ps) + 1
    max_depth = max(depth.values()) if depth else 0
    layers = [[] for _ in range(max_depth + 1)]
    for v in topo:
        layers[depth[v]].append(v)

    # ---- Compute remaining uses (how many times a value is needed downstream) ----
    # For each vertex, count distinct descendants that need it (approx by fan-out propagation).
    # Use a simple DP: need[v] = number of children + sum of propagated needs, but to avoid
    # double counting we count each descendant once via a reachability-free heuristic:
    # remaining[v] = number of vertices in the subtree rooted at v (distinct).
    subtree_size = {}
    for v in reversed(topo):
        s = 1
        for c in children.get(v, []):
            s += subtree_size.get(c, 1)
        subtree_size[v] = s

    # ---- Schedule ----
    ops = []
    fast = {}  # vertex -> True if currently in fast memory (blue)
    budget = max(1, int(fast_memory_budget))

    # Precompute, for each vertex, the set of children (next layer needs).
    # We process layer by layer. Within a layer, compute vertices whose parents are all available.
    # A value is "needed" if it has children not yet computed.

    # Track which vertices have been computed (their value produced at least once).
    computed = set()

    def is_needed(v):
        # needed if any child is not yet computed
        for c in children.get(v, []):
            if c not in computed:
                return True
        return False

    def evict_one():
        # Choose victim: among fast values that are NOT needed, pick the one with smallest subtree_size
        # (cheapest to recompute). If all are needed, pick smallest subtree_size overall.
        candidates = [v for v in fast if not is_needed(v)]
        if not candidates:
            candidates = list(fast.keys())
        if not candidates:
            return None
        # prefer non-needed; among them smallest subtree_size; tie-break by fewer remaining children
        def keyf(v):
            rem_children = sum(1 for c in children.get(v, []) if c not in computed)
            return (subtree_size.get(v, 1), rem_children)
        v = min(candidates, key=keyf)
        del fast[v]
        ops.append(['store', v])
        return v

    # Inputs start blue (in fast memory) per contract.
    for v in inputs:
        if len(fast) >= budget:
            evict_one()
        fast[v] = True
        ops.append(['load', v])

    # Process layers
    for d in range(max_depth + 1):
        layer = layers[d]
        # Within layer, compute in topo order (already topo-sorted globally)
        for v in layer:
            if _elapsed() > DEADLINE:
                break
            if v in computed:
                continue
            ps = parents.get(v, [])
            # Ensure parents are in fast memory
            for p in ps:
                if p not in fast:
                    # Need to load or recompute p
                    # If p is an input, load; else recompute (its parents must be available)
                    if p in inputs_set:
                        if len(fast) >= budget:
                            evict_one()
                        fast[p] = True
                        ops.append(['load', p])
                    else:
                        # recompute p: recursively ensure p's parents, compute p
                        # Use a stack-based recompute
                        def recompute(x):
                            if x in fast:
                                return
                            pxs = parents.get(x, [])
                            for pp in pxs:
                                recompute(pp)
                            if len(fast) >= budget:
                                evict_one()
                            fast[x] = True
                            ops.append(['compute', x])
                            computed.add(x)
                        recompute(p)
            # Now compute v
            if len(fast) >= budget:
                evict_one()
            fast[v] = True
            ops.append(['compute', v])
            computed.add(v)
        # End of layer: store values no longer needed (to free budget for next layer)
        for v in list(fast.keys()):
            if _elapsed() > DEADLINE:
                break
            if not is_needed(v):
                del fast[v]
                ops.append(['store', v])

    # Ensure all outputs are blue at the end
    for v in outputs:
        if v not in fast:
            if v in inputs_set:
                if len(fast) >= budget:
                    evict_one()
                fast[v] = True
                ops.append(['load', v])
            else:
                def recompute2(x):
                    if x in fast:
                        return
                    pxs = parents.get(x, [])
                    for pp in pxs:
                        recompute2(pp)
                    if len(fast) >= budget:
                        evict_one()
                    fast[x] = True
                    ops.append(['compute', x])
                    computed.add(x)
                recompute2(v)
        # keep it (don't store outputs)

    # Trim if too many ops
    if len(ops) > 20000:
        ops = ops[:20000]

    return ops
# EVOLVE-BLOCK-END
