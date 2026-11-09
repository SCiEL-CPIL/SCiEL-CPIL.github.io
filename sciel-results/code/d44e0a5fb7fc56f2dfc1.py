"""Communication-minimizing DAG pebbling — memory-aware topological scheduler."""

def _extract(dag):
    """Defensively extract (n, parents, inputs, outputs) from dag."""
    parents = None
    n = None
    inputs = None
    outputs = None

    def get_dict(d, keys):
        for k in keys:
            if k in d:
                return d[k]
        return None

    if isinstance(dag, dict):
        parents = get_dict(dag, ('parents','parent','preds','in_edges','input_edges'))
        n = get_dict(dag, ('n','num_vertices','num_nodes','size','num'))
        inputs = get_dict(dag, ('inputs','input','sources','source'))
        outputs = get_dict(dag, ('outputs','output','sinks','sink'))
        edges = get_dict(dag, ('edges','arcs','links'))
    else:
        # object with attributes
        for attr in ('parents','parent','preds','in_edges'):
            if hasattr(dag, attr):
                parents = getattr(dag, attr); break
        for attr in ('n','num_vertices','num_nodes','size','num'):
            if hasattr(dag, attr):
                n = getattr(dag, attr); break
        for attr in ('inputs','input','sources','source'):
            if hasattr(dag, attr):
                inputs = getattr(dag, attr); break
        for attr in ('outputs','output','sinks','sink'):
            if hasattr(dag, attr):
                outputs = getattr(dag, attr); break
        edges = getattr(dag, 'edges', None)

    if parents is None and edges is not None:
        if n is None:
            mx = -1
            for e in edges:
                mx = max(mx, e[0], e[1])
            n = mx + 1
        parents = [[] for _ in range(n)]
        for e in edges:
            parents[e[1]].append(e[0])

    if n is None and parents is not None:
        n = len(parents)

    parents = [list(p) for p in parents]

    if inputs is None:
        inputs = [i for i in range(n) if len(parents[i]) == 0]
    if outputs is None:
        cc = [0]*n
        for i in range(n):
            for p in parents[i]:
                cc[p] += 1
        outputs = [i for i in range(n) if cc[i] == 0]

    return n, parents, list(inputs), list(outputs)


def pebble(dag, fast_memory_budget):
    n, parents, inputs, outputs = _extract(dag)
    input_set = set(inputs)
    output_set = set(outputs)

    children = [[] for _ in range(n)]
    for v in range(n):
        for p in parents[v]:
            children[p].append(v)

    # topological order
    from collections import deque
    indeg = [len(parents[v]) for v in range(n)]
    q = deque([v for v in range(n) if indeg[v] == 0])
    topo = []
    indeg2 = list(indeg)
    while q:
        v = q.popleft()
        topo.append(v)
        for c in children[v]:
            indeg2[c] -= 1
            if indeg2[c] == 0:
                q.append(c)
    if len(topo) != n:
        topo = list(range(n))

    remaining = [len(children[v]) + (1 if v in output_set else 0) for v in range(n)]

    ops = []
    in_fast = [False]*n
    for v in inputs:
        in_fast[v] = True
    fast_count = len(inputs)
    budget = max(1, int(fast_memory_budget))

    for v in topo:
        if v in input_set:
            continue
        for p in parents[v]:
            if not in_fast[p]:
                ops.append(('load', p))
                in_fast[p] = True
                fast_count += 1
        ops.append(('compute', v))
        in_fast[v] = True
        fast_count += 1
        # decrement remaining for parents (they've served this child)
        for p in parents[v]:
            remaining[p] -= 1

        while fast_count > budget:
            best = -1
            best_score = None
            for u in range(n):
                if not in_fast[u] or u in output_set:
                    continue
                cand = (remaining[u], u)
                if best_score is None or cand < best_score:
                    best_score = cand
                    best = u
            if best == -1:
                for u in range(n):
                    if in_fast[u] and u in input_set:
                        best = u; break
                if best == -1:
                    break
            ops.append(('store', best))
            in_fast[best] = False
            fast_count -= 1

    for v in outputs:
        if not in_fast[v]:
            for p in parents[v]:
                if not in_fast[p]:
                    ops.append(('load', p))
                    in_fast[p] = True
                    fast_count += 1
            ops.append(('compute', v))
            in_fast[v] = True
            fast_count += 1

    if len(ops) > 20000:
        ops = ops[:20000]
    return ops
