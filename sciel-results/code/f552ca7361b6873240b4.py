# EVOLVE-BLOCK-START
import time
import heapq
from collections import deque

def _topo_order(dag):
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
    if not vertices:
        return [], {}
    vset = set(vertices)
    children = {v: [] for v in vertices}
    indeg = {v: len([p for p in parents.get(v, []) if p in vset]) for v in vertices}
    for v, ps in parents.items():
        for p in ps:
            if p in children:
                children[p].append(v)
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
    # topological order
    indeg = {v: len(parents.get(v, [])) for v in vertices}
    children = {v: [] for v in vertices}
    for v, ps in parents.items():
        for p in ps:
            if p in children:
                children[p].append(v)
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

def _pebble_postorder(dag, budget):
    # Efficient pebbling via postorder with recompute-on-demand.
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
    if not vertices:
        return []
    vset = set(vertices)
    outset = set(outputs)
    children = {v: [] for v in vertices}
    for v, ps in parents.items():
        for p in ps:
            if p in children:
                children[p].append(v)

    indeg = {v: len(parents.get(v, [])) for v in vertices}
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
    post = list(reversed(topo))

    ops = []
    loaded = set()
    stored = set()

    def pebble(v):
        if v in loaded:
            return
        ps = parents.get(v, [])
        for p in ps:
            pebble(p)
        ops.append(["compute", v])
        loaded.add(v)
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

    return ops[:20000]

def _pebble_greedy_lfu(dag, budget):
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
    if not vertices:
        return []
    vset = set(vertices)
    outset = set(outputs)
    children = {v: [] for v in vertices}
    for v, ps in parents.items():
        for p in ps:
            if p in children:
                children[p].append(v)
                
    indeg = {v: len(parents.get(v, [])) for v in vertices}
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
        
    freq = {v: 0 for v in vertices}
    for v, ps in parents.items():
        for p in ps:
            freq[p] += 1

    ops = []
    loaded = {}
    stored = set()
    
    topo_idx = {v: i for i, v in enumerate(topo)}

    for v in topo:
        for p in parents.get(v, []):
            if p not in loaded:
                if len(loaded) >= budget:
                    best_victim = None
                    best_score = (float('inf'), float('inf'))
                    for lp in loaded:
                        if lp in stored:
                            continue
                        rem_children = 0
                        for c in children.get(lp, []):
                            if topo_idx.get(c, 0) > topo_idx.get(v, 0):
                                rem_children += 1
                        score = (rem_children, freq[lp])
                        if score < best_score:
                            best_score = score
                            best_victim = lp
                    if best_victim is not None:
                        ops.append(["delete", best_victim])
                        loaded.pop(best_victim)
                
                ops.append(["load", p])
                loaded[p] = v
        
        ops.append(["compute", v])
        
        if v in outset:
            ops.append(["store", v])
            stored.add(v)
            loaded.pop(v, None)
        else:
            loaded[v] = v

    return ops[:20000]

def _pebble_astar(dag, budget, deadline):
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
    n = len(vertices)
    if n == 0:
        return []
    if n > 14:
        return None
    idx = {v: i for i, v in enumerate(vertices)}
    out_mask = 0
    for o in outputs:
        out_mask |= (1 << idx[o])
    par_mask = [0] * n
    for v, ps in parents.items():
        i = idx[v]
        m = 0
        for p in ps:
            m |= (1 << idx[p])
        par_mask[i] = m

    def popcount(x):
        return bin(x).count("1")

    def heuristic(loaded):
        return popcount(out_mask & ~loaded)

    start = 0
    goal = out_mask
    counter = 0
    open_heap = [(heuristic(start), 0, counter, start)]
    g_score = {start: 0}
    parent_op = {}
    best = None
    while open_heap:
        if time.time() > deadline:
            break
        f, g, _, state = heapq.heappop(open_heap)
        if state == goal:
            seq = []
            s = goal
            while s != start:
                prev, op, v = parent_op[s]
                seq.append([op, v])
                s = prev
            seq.reverse()
            best = seq
            break
        if g > g_score.get(state, float("inf")):
            continue
        for v in vertices:
            i = idx[v]
            bit = 1 << i
            if not (state & bit):
                ns = state | bit
                if popcount(ns) > budget:
                    continue
                ng = g + 1
                if ng < g_score.get(ns, float("inf")):
                    g_score[ns] = ng
                    parent_op[ns] = (state, "load", v)
                    counter += 1
                    heapq.heappush(open_heap, (ng + heuristic(ns), ng, counter, ns))
            if (state & bit) and (out_mask & bit):
                ns = state & ~bit
                ng = g + 1
                if ng < g_score.get(ns, float("inf")):
                    g_score[ns] = ng
                    parent_op[ns] = (state, "store", v)
                    counter += 1
                    heapq.heappush(open_heap, (ng + heuristic(ns), ng, counter, ns))
            if (state & bit) and not (out_mask & bit):
                ns = state & ~bit
                ng = g + 1
                if ng < g_score.get(ns, float("inf")):
                    g_score[ns] = ng
                    parent_op[ns] = (state, "delete", v)
                    counter += 1
                    heapq.heappush(open_heap, (ng + heuristic(ns), ng, counter, ns))
            if (state & par_mask[i]) == par_mask[i]:
                ns = state | bit
                if popcount(ns) > budget:
                    continue
                ng = g + 1
                if ng < g_score.get(ns, float("inf")):
                    g_score[ns] = ng
                    parent_op[ns] = (state, "compute", v)
                    counter += 1
                    heapq.heappush(open_heap, (ng + heuristic(ns), ng, counter, ns))
    return best

def pebble(dag, fast_memory_budget):
    start_t = time.time()
    deadline = start_t + 50

    candidates = []

    try:
        res = _pebble_postorder(dag, fast_memory_budget)
        if res is not None:
            candidates.append(res)
    except Exception:
        pass

    try:
        res = _pebble_greedy_lfu(dag, fast_memory_budget)
        if res is not None:
            candidates.append(res)
    except Exception:
        pass

    if time.time() < deadline:
        try:
            res = _pebble_astar(dag, fast_memory_budget, deadline)
            if res is not None and len(res) <= 20000:
                candidates.append(res)
        except Exception:
            pass

    try:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        res = pebble_baseline(dag, fast_memory_budget)
        if res is not None:
            candidates.append(res)
    except Exception:
        pass
    try:
        res = _pebble_fallback(dag, fast_memory_budget)
        if res is not None:
            candidates.append(res)
    except Exception:
        pass

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
    return _pebble_fallback(dag, fast_memory_budget)[:20000]
# EVOLVE-BLOCK-END
