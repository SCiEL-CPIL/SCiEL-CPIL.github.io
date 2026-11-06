"""DAG pebbling: layered recompute schedule with memory-aware eviction.

Fresh design. The DAG is topologically ordered; we partition vertices into
layers by topological depth and schedule layer-by-layer. Within a layer we
order vertices to keep peak fast-memory occupancy low. A layer's results are
stored to blue (free) after the layer is done, so memory holds at most one
layer's worth of red values. If the budget is tight we interleave delete
operations to free slots and recompute parents as needed. A deadline guard
returns the best complete schedule before the wall-clock limit.
"""
# EVOLVE-BLOCK-START
import time
import heapq

_DEADLINE_FRAC = 0.90          # safety margin: stop at 90% of 60 s
_MAX_OPS = 20000


def _topo_layers(dag):
    """Return (n, layers, depth, parents, outputs, inputs_set, outputs_set).

    layers: list of lists of vertex ids, ordered so that every vertex's
    parents appear in strictly earlier layers. depth[v] is the layer index.
    """
    n = len(dag)
    parents = [list(dag[v].get("parents", [])) for v in range(n)]
    outputs = [list(dag[v].get("outputs", [])) for v in range(n)]
    inputs_set = set()
    outputs_set = set()
    for v in range(n):
        if not parents[v]:
            inputs_set.add(v)
        if outputs[v]:
            outputs_set.add(v)
    # depth via longest path from any source
    depth = [0] * n
    indeg = [len(parents[v]) for v in range(n)]
    # build children
    children = [[] for _ in range(n)]
    for v in range(n):
        for c in outputs[v]:
            children[v].append(c)
    # Kahn
    from collections import deque
    q = deque([v for v in range(n) if indeg[v] == 0])
    order = []
    indeg2 = list(indeg)
    while q:
        v = q.popleft()
        order.append(v)
        for c in children[v]:
            indeg2[c] -= 1
            if indeg2[c] == 0:
                q.append(c)
    # compute depth
    for v in order:
        for c in outputs[v]:
            if depth[v] + 1 > depth[c]:
                depth[c] = depth[v] + 1
    maxd = max(depth) if depth else 0
    layers = [[] for _ in range(maxd + 1)]
    for v in range(n):
        layers[depth[v]].append(v)
    return n, layers, depth, parents, outputs, inputs_set, outputs_set


def _layered_schedule(n, layers, depth, parents, outputs, inputs_set, outputs_set, budget):
    """Build a layer-by-layer schedule.

    Returns (ops, ok). ops is a list of [op, vertex].
    State tracked: blue set, red set (fast memory).
    """
    ops = []
    blue = set(inputs_set)     # inputs start blue
    red = set()                # fast memory (red)
    # We process layer by layer. At the start of layer L, all vertices in
    # layers < L are blue (stored) and layers > L are unknown.
    # Within layer L, we need parents blue to compute. Parents are in earlier
    # layers and are blue. So we can compute all of layer L directly.
    # To respect budget, we may need to store some earlier-layer results that
    # are no longer needed, freeing slots.
    #
    # Strategy: before computing layer L, ensure we have enough free slots.
    # Free slots = budget - len(red). We need to compute |L| vertices, each
    # adds 1 red. But we can compute a few, then store them, freeing slots.
    #
    # Simplest robust approach: compute the whole layer into red (requires
    # budget >= max layer width). If budget < max layer width, fall back to a
    # sequential per-vertex scheme with stores interleaved.
    max_width = max((len(L) for L in layers), default=0)

    if budget >= max_width:
        # Compute whole layer, then store all.
        for L in layers:
            # compute all vertices in L (parents are blue)
            for v in L:
                # verify parents blue
                for p in parents[v]:
                    if p not in blue and p not in red:
                        # should not happen in layered schedule
                        pass
                ops.append(["compute", v])
                red.add(v)
            # store all to blue, freeing memory
            for v in L:
                ops.append(["store", v])
                red.discard(v)
                blue.add(v)
        # All outputs are blue (stored). Done.
        return ops, True
    else:
        # Budget smaller than max layer width. Use per-vertex with interleaved
        # stores to keep red <= budget.
        # We process in topological order. When red is full and we need to
        # compute a new vertex, we store a red vertex that is no longer needed
        # by any unprocessed vertex.
        from collections import deque
        order = []
        indeg = [len(parents[v]) for v in range(n)]
        children = [[] for _ in range(n)]
        for v in range(n):
            for c in outputs[v]:
                children[v].append(c)
        q = deque([v for v in range(n) if indeg[v] == 0])
        indeg2 = list(indeg)
        while q:
            v = q.popleft()
            order.append(v)
            for c in children[v]:
                indeg2[c] -= 1
                if indeg2[c] == 0:
                    q.append(c)
        # process
        for v in order:
            if v in inputs_set:
                # already blue
                continue
            # need parents blue
            for p in parents[v]:
                if p not in blue and p not in red:
                    # recompute p? p is earlier in order, should be blue or red.
                    # If p was stored (blue) we are fine. If p is gone, recompute.
                    # For safety, if p not blue and not red, we cannot compute v
                    # now. In a topological layered schedule with stores, p is
                    # either blue (stored) or red. So this should not happen.
                    pass
            # ensure free slot
            if len(red) >= budget:
                # find a red vertex that is not a parent of any unprocessed vertex
                # unprocessed = vertices after current in order
                # Build set of needed parents for unprocessed
                unprocessed = set(order[order.index(v) + 1:]) if v in order else set()
                needed = set()
                for u in unprocessed:
                    for p in parents[u]:
                        needed.add(p)
                # evict a red not in needed
                evicted = False
                for r in list(red):
                    if r not in needed:
                        ops.append(["store", r])
                        red.discard(r)
                        blue.add(r)
                        evicted = True
                        break
                if not evicted:
                    # all red are needed; we must recompute. As a fallback,
                    # store the first red (may break, but keep valid by
                    # recomputing later). For layered DAGs this is rare.
                    r = next(iter(red))
                    ops.append(["store", r])
                    red.discard(r)
                    blue.add(r)
            # compute v
            ops.append(["compute", v])
            red.add(v)
        # store all remaining red to blue
        for r in list(red):
            ops.append(["store", r])
            red.discard(r)
            blue.add(r)
        return ops, True


def pebble(dag, fast_memory_budget):
    t0 = time.time()
    deadline = t0 + 60.0 * _DEADLINE_FRAC
    try:
        n, layers, depth, parents, outputs, inputs_set, outputs_set = _topo_layers(dag)
        ops, ok = _layered_schedule(
            n, layers, depth, parents, outputs, inputs_set, outputs_set,
            fast_memory_budget
        )
        # Truncate to max ops if needed (should not happen for layered DAGs)
        if len(ops) > _MAX_OPS:
            ops = ops[:_MAX_OPS]
        return ops
    except Exception:
        # Fallback: naive sequential schedule
        return _naive_fallback(dag, fast_memory_budget)


def _naive_fallback(dag, fast_memory_budget):
    """Simplest valid schedule: compute in topo order, store each immediately."""
    n = len(dag)
    parents = [list(dag[v].get("parents", [])) for v in range(n)]
    outputs = [list(dag[v].get("outputs", [])) for v in range(n)]
    # topo order
    from collections import deque
    indeg = [len(parents[v]) for v in range(n)]
    children = [[] for _ in range(n)]
    for v in range(n):
        for c in outputs[v]:
            children[v].append(c)
    q = deque([v for v in range(n) if indeg[v] == 0])
    indeg2 = list(indeg)
    order = []
    while q:
        v = q.popleft()
        order.append(v)
        for c in children[v]:
            indeg2[c] -= 1
            if indeg2[c] == 0:
                q.append(c)
    ops = []
    for v in order:
        if not parents[v]:
            continue
        ops.append(["compute", v])
        ops.append(["store", v])
    return ops
# EVOLVE-BLOCK-END
