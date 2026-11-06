"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
from collections import deque

def _pebble_fallback(dag, budget):
    # Greedy baseline: load parents, compute, store outputs, delete when possible.
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
    if not vertices:
        return []
    vset = set(vertices)
    ops = []
    def cost(v):
        ps = parents.get(v, [])
        if not ps:
            return 1
        return 1 + sum(cost(p) for p in ps)
    cost_cache = {}
    def ccost(v):
        if v not in cost_cache:
            ps = parents.get(v, [])
            cost_cache[v] = 1 + (sum(ccost(p) for p in ps) if ps else 0)
        return cost_cache[v]
    # Process vertices in topological order, load all needed, compute, store outputs, delete non-outputs.
    # Determine topological order.
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
        topo = vertices  # cycle fallback
    outset = set(outputs)
    for v in topo:
        # load parents if not loaded
        for p in parents.get(v, []):
            ops.append(["load", p])
        ops.append(["compute", v])
        if v in outset:
            ops.append(["store", v])
        # delete parents that are no longer needed
        for p in parents.get(v, []):
            still = any(p in parents.get(w, []) for w in topo[topo.index(v)+1:] if w in vset)
            if not still and p not in outset:
                ops.append(["delete", p])
    return ops[:20000]

def _pebble_astar(dag, budget, deadline):
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
    n = len(vertices)
    if n == 0:
        return []
    idx = {v: i for i, v in enumerate(vertices)}
    out_mask = 0
    for o in outputs:
        out_mask |= (1 << idx[o])
    # Precompute parents bitmasks and children sets
    par_mask = [0] * n
    children = [set() for _ in range(n)]
    for v, ps in parents.items():
        i = idx[v]
        m = 0
        for p in ps:
            j = idx[p]
            m |= (1 << j)
            children[j].add(i)
        par_mask[i] = m
    
    # Precompute sub-DAG masks: for each vertex, the set of all ancestors (including itself)
    subdag_mask = [0] * n
    def compute_subdag(i):
        if subdag_mask[i] != 0:
            return subdag_mask[i]
        m = (1 << i) | par_mask[i]
        mm = par_mask[i]
        while mm:
            lb = mm & (-mm)
            j = lb.bit_length() - 1
            m |= compute_subdag(j)
            mm ^= lb
        subdag_mask[i] = m
        return m
    for i in range(n):
        compute_subdag(i)
    
    # Cost (number of vertices in sub-DAG) per vertex
    cost = [0] * n
    for i in range(n):
        cost[i] = bin(subdag_mask[i]).count("1")

    def popcount(x):
        return bin(x).count("1")

    # Stronger admissible heuristic:
    # For each missing output, we need at least (number of ancestors not in loaded) transfers
    # but ancestors can be shared. Lower bound: sum over missing outputs of (missing ancestors in subdag)
    # but that overcounts shared parts. Better: for each missing output, need at least 1 load+compute per missing ancestor
    # Simple admissible: count of missing vertices that are ancestors of some missing output
    # Even simpler: popcount of (union of subdags of missing outputs) & ~loaded
    def heuristic(loaded):
        missing_outs = out_mask & ~loaded
        if missing_outs == 0:
            return 0
        # Union of all subdag masks for missing outputs
        union_mask = 0
        mm = missing_outs
        while mm:
            lb = mm & (-mm)
            j = lb.bit_length() - 1
            union_mask |= subdag_mask[j]
            mm ^= lb
        # Number of vertices in union that are not loaded
        needed = union_mask & ~loaded
        return popcount(needed)

    start = 0
    goal = out_mask
    # A*
    counter = 0
    open_heap = [(heuristic(start), 0, counter, start)]
    g_score = {start: 0}
    parent_op = {}
    best = None
    start_t = time.time()
    max_ops_limit = 20000
    
    # Reconstruct ops from path
    def reconstruct(state, start_mask):
        seq = []
        while state != start_mask:
            prev, op, v = parent_op[state]
            seq.append([op, v])
            state = prev
        seq.reverse()
        return seq

    while open_heap:
        if time.time() > deadline:
            break
        f, g, _, state = heapq.heappop(open_heap)
        if state == goal:
            best = reconstruct(state, start)
            break
        if g > g_score.get(state, float("inf")):
            continue
        # Generate successors: for each vertex v, possible ops
        for v in vertices:
            i = idx[v]
            bit = 1 << i
            # load v if not loaded
            if not (state & bit):
                ns = state | bit
                ng = g + 1
                if ng < g_score.get(ns, float("inf")):
                    g_score[ns] = ng
                    parent_op[ns] = (state, "load", v)
                    counter += 1
                    heapq.heappush(open_heap, (ng + heuristic(ns), ng, counter, ns))
            # store v if loaded and is output
            if (state & bit) and (out_mask & bit):
                ns = state & ~bit
                ng = g + 1
                if ng < g_score.get(ns, float("inf")):
                    g_score[ns] = ng
                    parent_op[ns] = (state, "store", v)
                    counter += 1
                    heapq.heappush(open_heap, (ng + heuristic(ns), ng, counter, ns))
            # delete v if loaded and not output
            if (state & bit) and not (out_mask & bit):
                ns = state & ~bit
                ng = g + 1
                if ng < g_score.get(ns, float("inf")):
                    g_score[ns] = ng
                    parent_op[ns] = (state, "delete", v)
                    counter += 1
                    heapq.heappush(open_heap, (ng + heuristic(ns), ng, counter, ns))
            # compute v if all parents loaded
            if (state & par_mask[i]) == par_mask[i]:
                ns = state | bit
                ng = g + 1
                if ng < g_score.get(ns, float("inf")):
                    g_score[ns] = ng
                    parent_op[ns] = (state, "compute", v)
                    counter += 1
                    heapq.heappush(open_heap, (ng + heuristic(ns), ng, counter, ns))
    return best

def pebble(dag, fast_memory_budget):
    deadline = time.time() + 45  # safety margin before 60s (and 90s kill)
    try:
        result = _pebble_astar(dag, fast_memory_budget, deadline)
        if result is not None and len(result) <= 20000:
            return result
    except Exception:
        pass
    # Fallback to greedy baseline
    try:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)
    except Exception:
        return _pebble_fallback(dag, fast_memory_budget)
# EVOLVE-BLOCK-END
