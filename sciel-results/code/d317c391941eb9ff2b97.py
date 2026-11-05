"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
from collections import deque, defaultdict

def _get_layer_assignment(dag):
    """Assign layers to vertices. Returns dict vertex->layer, or None if not layered."""
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
    if not vertices:
        return {}
    vset = set(vertices)
    
    # Build children map
    children = defaultdict(list)
    for v, ps in parents.items():
        for p in ps:
            if p in vset:
                children[p].append(v)
    
    # Compute longest path from any source to determine layer
    # layer[v] = max(layer[p] for p in parents[v]) + 1, or 0 if no parents
    layer = {}
    indeg = {v: len([p for p in parents.get(v, []) if p in vset]) for v in vertices}
    q = deque([v for v in vertices if indeg[v] == 0])
    for v in q:
        layer[v] = 0
    
    processed = set()
    while q:
        u = q.popleft()
        if u in processed:
            continue
        processed.add(u)
        for w in children[u]:
            if w not in layer:
                layer[w] = layer[u] + 1
            else:
                layer[w] = max(layer[w], layer[u] + 1)
            indeg[w] -= 1
            if indeg[w] == 0:
                q.append(w)
    
    if len(processed) != len(vertices):
        return None  # cycle or disconnected
    
    return layer

def _pebble_layered(dag, budget):
    """Optimal pebbling for layered DAGs: process layer by layer."""
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
    if not vertices:
        return []
    vset = set(vertices)
    outset = set(outputs)
    
    layer = _get_layer_assignment(dag)
    if layer is None:
        return None
    
    # Group vertices by layer
    layers = defaultdict(list)
    for v in vertices:
        layers[layer[v]].append(v)
    
    max_layer = max(layers.keys()) if layers else 0
    ops = []
    loaded = set()  # vertices currently in fast memory
    
    # For each layer, we need to have all its vertices loaded before computing them
    # Strategy: for layer i, load all vertices in layer i that aren't already loaded,
    # compute them, then delete layer i-1 vertices that are no longer needed
    
    # Precompute which vertices are needed by which later layers
    # For each vertex v in layer i, determine if it's needed by any vertex in layer i+1 or later
    children = defaultdict(list)
    for v, ps in parents.items():
        for p in ps:
            if p in vset:
                children[p].append(v)
    
    # Process layers in order
    for li in range(max_layer + 1):
        layer_vertices = layers.get(li, [])
        if not layer_vertices:
            continue
        
        # Load all vertices in this layer that need to be computed
        for v in layer_vertices:
            if v not in loaded:
                # Load v from slow memory
                ops.append(["load", v])
                loaded.add(v)
        
        # Compute all vertices in this layer
        for v in layer_vertices:
            # All parents should be loaded (they're in earlier layers)
            # But we may have deleted some - need to ensure parents are available
            ps = [p for p in parents.get(v, []) if p in vset]
            for p in ps:
                if p not in loaded:
                    # Parent was deleted, need to reload
                    ops.append(["load", p])
                    loaded.add(p)
            ops.append(["compute", v])
        
        # After computing layer li, we can delete layer li-1 vertices that are no longer needed
        if li > 0:
            prev_layer = layers.get(li - 1, [])
            for v in prev_layer:
                if v in outset:
                    # Outputs must be stored, not deleted
                    if v in loaded:
                        ops.append(["store", v])
                        loaded.discard(v)
                    continue
                # Check if v is needed by any vertex in layer li or later
                needed = False
                for w in children.get(v, []):
                    if layer.get(w, -1) >= li:
                        needed = True
                        break
                if not needed and v in loaded:
                    ops.append(["delete", v])
                    loaded.discard(v)
        
        # If this is the last layer, store all outputs
        if li == max_layer:
            for v in layer_vertices:
                if v in outset and v in loaded:
                    ops.append(["store", v])
                    loaded.discard(v)
    
    # Store any remaining outputs
    for v in vertices:
        if v in outset and v in loaded:
            ops.append(["store", v])
            loaded.discard(v)
    
    return ops[:20000]

def _pebble_greedy_optimal(dag, budget):
    """Greedy pebbling with optimal deletion strategy for general DAGs."""
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
    if not vertices:
        return []
    vset = set(vertices)
    outset = set(outputs)
    
    # Build children map
    children = defaultdict(list)
    for v, ps in parents.items():
        for p in ps:
            if p in vset:
                children[p].append(v)
    
    # Topological order
    indeg = {v: len([p for p in parents.get(v, []) if p in vset]) for v in vertices}
    q = deque([v for v in vertices if indeg[v] == 0])
    topo = []
    while q:
        u = q.popleft()
        topo.append(u)
        for w in children[u]:
            indeg[w] -= 1
            if indeg[w] == 0:
                q.append(w)
    if len(topo) != len(vertices):
        topo = vertices
    
    ops = []
    loaded = set()
    computed = set()
    
    # For each vertex in topological order, ensure its parents are loaded,
    # compute it, then delete parents that are no longer needed
    for v in topo:
        ps = [p for p in parents.get(v, []) if p in vset]
        for p in ps:
            if p not in loaded:
                ops.append(["load", p])
                loaded.add(p)
        ops.append(["compute", v])
        loaded.add(v)
        computed.add(v)
        
        # Delete parents that are no longer needed by any uncomputed vertex
        for p in ps:
            if p in outset:
                continue
            # Check if p is needed by any vertex not yet computed
            needed = False
            for w in children.get(p, []):
                if w not in computed:
                    needed = True
                    break
            if not needed and p in loaded:
                ops.append(["delete", p])
                loaded.discard(p)
        
        # If v is an output, store it
        if v in outset:
            ops.append(["store", v])
            loaded.discard(v)
    
    # Store any remaining outputs
    for v in vertices:
        if v in outset and v in loaded:
            ops.append(["store", v])
            loaded.discard(v)
    
    return ops[:20000]

def _pebble_astar_small(dag, budget, deadline):
    """A* for small instances (n <= 12) to find optimal solution."""
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
    n = len(vertices)
    if n == 0:
        return []
    if n > 12:  # Limit to very small instances
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
            if p in idx:
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
                ng = g + 1
                if ng < g_score.get(ns, float("inf")):
                    g_score[ns] = ng
                    parent_op[ns] = (state, "compute", v)
                    counter += 1
                    heapq.heappush(open_heap, (ng + heuristic(ns), ng, counter, ns))
    return best

def _count_transfers(seq):
    return sum(1 for op, _ in seq if op in ("load", "store"))

def _validate_and_optimize(dag, ops):
    """Validate ops and try to remove redundant loads/stores."""
    if not ops:
        return ops
    parents = dag.get("parents", {})
    outputs = set(dag.get("outputs", []))
    vertices = set(list(parents.keys()) + list(outputs))
    
    # Simulate to check validity and identify redundancies
    loaded = set()
    computed = set()
    stored = set()
    valid_ops = []
    
    for op, v in ops:
        if v not in vertices:
            continue
        if op == "load":
            if v in loaded:
                continue  # redundant load
            loaded.add(v)
            valid_ops.append([op, v])
        elif op == "store":
            if v not in loaded or v not in outputs:
                continue  # invalid store
            loaded.discard(v)
            stored.add(v)
            valid_ops.append([op, v])
        elif op == "delete":
            if v not in loaded or v in outputs:
                continue  # invalid delete
            loaded.discard(v)
            valid_ops.append([op, v])
        elif op == "compute":
            ps = [p for p in parents.get(v, []) if p in vertices]
            if not all(p in loaded for p in ps):
                continue  # invalid compute
            loaded.add(v)
            computed.add(v)
            valid_ops.append([op, v])
    
    # Check all outputs are stored
    if stored != outputs:
        # Try to fix by adding missing stores
        for o in outputs - stored:
            if o in loaded:
                valid_ops.append(["store", o])
                stored.add(o)
            elif o in computed:
                valid_ops.append(["load", o])
                valid_ops.append(["store", o])
                stored.add(o)
    
    return valid_ops[:20000]

def pebble(dag, fast_memory_budget):
    start_t = time.time()
    deadline = start_t + 55  # Safety margin before 60s kill
    
    candidates = []
    
    # 1) Try A* for very small instances
    if time.time() < deadline:
        try:
            res = _pebble_astar_small(dag, fast_memory_budget, deadline)
            if res is not None and len(res) <= 20000:
                candidates.append(res)
        except Exception:
            pass
    
    # 2) Try layered pebbling
    if time.time() < deadline:
        try:
            res = _pebble_layered(dag, fast_memory_budget)
            if res is not None:
                candidates.append(res)
        except Exception:
            pass
    
    # 3) Try greedy optimal
    if time.time() < deadline:
        try:
            res = _pebble_greedy_optimal(dag, fast_memory_budget)
            if res is not None:
                candidates.append(res)
        except Exception:
            pass
    
    # 4) Fallback baseline
    try:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        res = pebble_baseline(dag, fast_memory_budget)
        if res is not None:
            candidates.append(res)
    except Exception:
        pass
    
    # Validate and pick best
    best = None
    best_score = float("inf")
    for c in candidates:
        if c is None or len(c) > 20000:
            continue
        validated = _validate_and_optimize(dag, c)
        if not validated:
            continue
        s = _count_transfers(validated)
        if s < best_score:
            best_score = s
            best = validated
    
    if best is not None:
        return best
    
    # Last resort
    try:
        return _pebble_greedy_optimal(dag, fast_memory_budget)
    except Exception:
        return []
# EVOLVE-BLOCK-END
