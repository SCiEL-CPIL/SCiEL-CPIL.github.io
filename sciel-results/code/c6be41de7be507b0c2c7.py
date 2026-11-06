"""Rewritten pebbling solver: layered-DAG communication minimizer with deadline guard."""
import time
import heapq
import random

_DEADLINE = 85.0


def _elapsed_ok(t0):
    return (time.time() - t0) < _DEADLINE


def _topo_levels(dag):
    parents = dag.get('parents', None)
    if parents is None:
        edges = dag.get('edges', [])
        n = dag.get('n', dag.get('num_vertices', 0))
        if n == 0 and edges:
            n = max(max(p, c) for p, c in edges) + 1
        parents = [[] for _ in range(n)]
        for p, c in edges:
            parents[c].append(p)
    n = len(parents)
    outputs = dag.get('outputs', [])

    children = [[] for _ in range(n)]
    indeg = [0] * n
    for v in range(n):
        for p in parents[v]:
            children[p].append(v)
            indeg[v] += 1

    depth = [0] * n
    order = []
    q = [v for v in range(n) if indeg[v] == 0]
    indeg2 = indeg[:]
    while q:
        v = q.pop()
        order.append(v)
        for c in children[v]:
            depth[c] = max(depth[c], depth[v] + 1)
            indeg2[c] -= 1
            if indeg2[c] == 0:
                q.append(c)
    if len(order) < n:
        order += [v for v in range(n) if v not in set(order)]

    maxd = max(depth) if depth else 0
    levels = [[] for _ in range(maxd + 1)]
    for v in order:
        levels[depth[v]].append(v)
    return levels, parents, outputs, n, indeg, children


def _naive_transfers(dag, fast_memory_budget):
    levels, parents, outputs, n, indeg, children = _topo_levels(dag)
    transfers = 0
    for v in range(n):
        if not parents[v]:
            transfers += 1
    transfers += len(outputs)
    return max(transfers, 1)


def _simulate(ops, dag, fast_memory_budget):
    levels, parents, outputs, n, indeg, children = _topo_levels(dag)
    out_set = set(outputs)
    state = set()  # vertices currently in fast memory
    for v in range(n):
        if not parents[v]:
            state.add(v)
    transfers = 0
    for op, v in ops:
        if op == 'load':
            if v in state:
                return None, False, False
            state.add(v)
            if len(state) > fast_memory_budget:
                return None, False, False
            transfers += 1
        elif op == 'store':
            if v not in state:
                return None, False, False
            state.remove(v)
            transfers += 1
        elif op == 'compute':
            for p in parents[v]:
                if p not in state:
                    return None, False, False
            if v not in state:
                state.add(v)
                if len(state) > fast_memory_budget:
                    return None, False, False
        elif op == 'delete':
            if v not in state:
                return None, False, False
            state.remove(v)
        else:
            return None, False, False
    for o in outputs:
        if o not in state:
            return None, False, False
    return transfers, True, True


def pebble(dag, fast_memory_budget):
    t0 = time.time()
    levels, parents, outputs, n, indeg, children = _topo_levels(dag)
    out_set = set(outputs)
    naive = _naive_transfers(dag, fast_memory_budget)

    # Greedy: process in topological order. Keep values in fast memory as long as possible.
    # When budget is full and we need to compute, evict the "least useful" value.
    # Use: a value is useful if it's an output or has remaining children that need it.

    remaining = [len(children[v]) for v in range(n)]
    order = []
    for lvl in levels:
        order.extend(sorted(lvl))

    state = set()
    for v in range(n):
        if not parents[v]:
            state.add(v)
    ops = []
    transfers = 0

    def evict():
        nonlocal transfers
        # Find least useful value to store (evict from fast memory)
        # Priority: 0 = not output, no remaining children; 1 = not output, has remaining; 2 = output
        best = None
        best_key = None
        for v in state:
            if v in out_set:
                key = (2, v)
            elif remaining[v] == 0:
                key = (0, v)
            else:
                key = (1, v)
            if best_key is None or key < best_key:
                best_key = key
                best = v
        if best is not None:
            state.remove(best)
            ops.append(['store', best])
            transfers += 1
            return True
        return False

    for v in order:
        if not _elapsed_ok(t0):
            break
        # Ensure parents are in fast memory
        for p in parents[v]:
            if p not in state:
                ops.append(['load', p])
                state.add(p)
                transfers += 1
                if len(state) > fast_memory_budget:
                    evict()
        # Make room if needed
        if v not in state and len(state) >= fast_memory_budget:
            evict()
        # Compute
        ops.append(['compute', v])
        if v not in state:
            state.add(v)

        # If not output and no remaining children, evict to free space
        if v not in out_set and remaining[v] == 0 and len(state) > 1:
            state.remove(v)
            ops.append(['store', v])
            transfers += 1

        for p in parents[v]:
            remaining[p] -= 1

    # Ensure outputs are in fast memory
    for o in outputs:
        if o not in state:
            ops.append(['load', o])
            state.add(o)
            transfers += 1

    if len(ops) > 20000:
        ops = ops[:20000]

    tr, valid, ok = _simulate(ops, dag, fast_memory_budget)
    if not valid or not ok:
        # Fallback
        ops = _naive_schedule(dag, fast_memory_budget)
        tr, valid, ok = _simulate(ops, dag, fast_memory_budget)
        if not valid:
            return []

    return ops


def _naive_schedule(dag, fast_memory_budget):
    levels, parents, outputs, n, indeg, children = _topo_levels(dag)
    out_set = set(outputs)
    remaining = [len(children[v]) for v in range(n)]

    state = set()
    for v in range(n):
        if not parents[v]:
            state.add(v)
    ops = []

    order = []
    for lvl in levels:
        order.extend(sorted(lvl))

    for v in order:
        for p in parents[v]:
            if p not in state:
                ops.append(['load', p])
                state.add(p)
        ops.append(['compute', v])
        if v not in state:
            state.add(v)
        if v not in out_set and remaining[v] == 0:
            ops.append(['store', v])
            state.remove(v)
        for p in parents[v]:
            remaining[p] -= 1

    for o in outputs:
        if o not in state:
            ops.append(['load', o])
            state.add(o)

    return ops
