"""Communication-minimizing DAG pebbling via state-space shortest path."""
from collections import deque


def _extract(dag):
    """Return (vertices, parents_dict, inputs_set, outputs_set) or None."""
    parents = None
    outputs = None
    if isinstance(dag, dict):
        if 'parents' in dag:
            parents = {k: list(v) for k, v in dag['parents'].items()}
            outputs = set(dag.get('outputs', []))
        elif 'edges' in dag:
            par = {}
            for a, b in dag['edges']:
                par.setdefault(b, []).append(a)
            parents = par
            outputs = set(dag.get('outputs', []))
        else:
            vals = list(dag.values())
            if vals and all(isinstance(v, (list, tuple, set)) for v in vals):
                parents = {k: list(v) for k, v in dag.items()}
    if parents is None and hasattr(dag, 'parents'):
        parents = {k: list(v) for k, v in dict(dag.parents).items()}
    if parents is None:
        return None
    if outputs is None and hasattr(dag, 'outputs'):
        outputs = set(dag.outputs)
    verts = set(parents.keys())
    for ps in parents.values():
        verts.update(ps)
    if outputs is None:
        allp = set()
        for ps in parents.values():
            allp.update(ps)
        outputs = set(v for v in verts if v not in allp)
    else:
        outputs = set(outputs)
    verts = list(verts)
    for v in verts:
        parents.setdefault(v, [])
    inputs = set(v for v in verts if not parents[v])
    return verts, parents, inputs, outputs


def _naive(dag, budget):
    """Valid greedy schedule: topological order, compute when free else load."""
    info = _extract(dag)
    if info is None:
        return []
    vertices, parents, inputs, outputs = info
    indeg = {v: len(parents.get(v, [])) for v in vertices}
    children = {v: [] for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            children.setdefault(p, []).append(v)
    q = deque([v for v in vertices if indeg[v] == 0])
    topo = []
    seen = set()
    while q:
        v = q.popleft()
        if v in seen:
            continue
        seen.add(v)
        topo.append(v)
        for c in children.get(v, []):
            indeg[c] -= 1
            if indeg[c] == 0:
                q.append(c)
    topo += [v for v in vertices if v not in seen]
    mem = set(inputs)
    ops = []
    outset = set(outputs)

    def free_one():
        cand = [x for x in mem if x not in outset]
        d = cand[0] if cand else (next(iter(mem)) if mem else None)
        if d is not None:
            ops.append(['delete', d])
            mem.discard(d)

    for v in topo:
        if v in mem:
            continue
        if len(mem) >= budget:
            free_one()
        if all(p in mem for p in parents.get(v, [])):
            ops.append(['compute', v])
        else:
            ops.append(['load', v])
        mem.add(v)
    for o in outputs:
        if o not in mem:
            if len(mem) >= budget:
                free_one()
            ops.append(['load', o])
            mem.add(o)
    return ops


def _search(vertices, parents, inputs, outputs, budget):
    """Dijkstra over memory-state space; minimize number of loads."""
    import heapq
    pf = {v: frozenset(parents.get(v, [])) for v in vertices}
    start = frozenset(inputs)
    if outputs <= set(start):
        return []
    outset = set(outputs)
    INF = float('inf')
    dist = {start: 0}
    prev = {}
    heap = [(0, start)]
    expansions = 0
    MAX_EXPAND = 200000
    MAX_DIST = 400000
    goal = None
    while heap:
        d, s = heapq.heappop(heap)
        if d > dist.get(s, INF):
            continue
        if outset <= set(s):
            goal = s
            break
        expansions += 1
        if expansions > MAX_EXPAND or len(dist) > MAX_DIST:
            return None
        sset = set(s)
        if len(sset) < budget:
            for v in vertices:
                if v in sset:
                    continue
                ns = s | {v}
                if pf[v] <= sset:  # compute: free
                    if d < dist.get(ns, INF):
                        dist[ns] = d
                        prev[ns] = (s, 'compute', v)
                        heapq.heappush(heap, (d, ns))
                nd = d + 1  # load: cost 1
                if nd < dist.get(ns, INF):
                    dist[ns] = nd
                    prev[ns] = (s, 'load', v)
                    heapq.heappush(heap, (nd, ns))
        for v in sset:  # delete: free
            ns = s - {v}
            if d < dist.get(ns, INF):
                dist[ns] = d
                prev[ns] = (s, 'delete', v)
                heapq.heappush(heap, (d, ns))
    if goal is None:
        return None
    ops = []
    cur = goal
    while cur != start:
        ps, op, v = prev[cur]
        ops.append([op, v])
        cur = ps
    ops.reverse()
    return ops


def pebble(dag, fast_memory_budget):
    try:
        budget = int(fast_memory_budget)
    except Exception:
        budget = 1
    if budget < 1:
        budget = 1
    info = _extract(dag)
    if info is None:
        return _naive(dag, budget)
    vertices, parents, inputs, outputs = info
    if not outputs or not vertices:
        return []
    # Use exact search only when the state space is tractable.
    if len(vertices) <= 22:
        res = _search(vertices, parents, inputs, outputs, budget)
        if res is not None and len(res) <= 20000:
            return res
    return _naive(dag, budget)
