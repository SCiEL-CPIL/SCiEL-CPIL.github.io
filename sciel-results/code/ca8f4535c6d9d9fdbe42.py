"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque
import heapq
import sys

sys.setrecursionlimit(100000)


def _parse_dag(dag):
    if isinstance(dag, dict):
        parents = dict(dag.get('parents', {}))
        outputs = list(dag.get('outputs', []))
        vertices = dag.get('vertices', None)
        inputs = dag.get('inputs', None)
    else:
        parents = dict(getattr(dag, 'parents', {}))
        outputs = list(getattr(dag, 'outputs', []))
        vertices = getattr(dag, 'vertices', None)
        inputs = getattr(dag, 'inputs', None)

    if vertices is not None:
        all_vertices = set(vertices)
    else:
        all_vertices = set(parents.keys())
        for v, ps in parents.items():
            for p in ps:
                all_vertices.add(p)
        for o in outputs:
            all_vertices.add(o)

    if inputs is not None:
        input_vertices = set(inputs)
    else:
        input_vertices = set(v for v in all_vertices if not parents.get(v, []))

    return all_vertices, input_vertices, outputs, parents


def _pebble_small_optimal(all_vertices, input_vertices, outputs, parents, budget):
    """Dijkstra for tiny instances (state space is small)."""
    init_black = frozenset(input_vertices)
    init_fast = frozenset()
    start = (init_fast, init_black)
    dist = {start: 0}
    prev = {}
    pq = [(0, start)]

    vertices_list = sorted(all_vertices)
    outputs_set = frozenset(outputs)

    while pq:
        cost, state = heapq.heappop(pq)
        if cost > dist.get(state, float('inf')):
            continue
        fast, black = state

        if all(o in black for o in outputs_set):
            ops = []
            curr = state
            while curr != start:
                p_state, op = prev[curr]
                ops.append(op)
                curr = p_state
            ops.reverse()
            return ops

        for v in vertices_list:
            # Compute
            if v not in fast and v not in black:
                ps = parents.get(v, [])
                if all(p in fast or p in black for p in ps):
                    nf = fast | frozenset({v})
                    if len(nf) <= budget:
                        ns = (nf, black)
                        if cost < dist.get(ns, float('inf')):
                            dist[ns] = cost
                            prev[ns] = (state, ['compute', v])
                            heapq.heappush(pq, (cost, ns))
            # Load
            if v in black and v not in fast:
                nf = fast | frozenset({v})
                if len(nf) <= budget:
                    ns = (nf, black)
                    nc = cost + 1
                    if nc < dist.get(ns, float('inf')):
                        dist[ns] = nc
                        prev[ns] = (state, ['load', v])
                        heapq.heappush(pq, (nc, ns))
            # Delete
            if v in fast:
                nf = fast - frozenset({v})
                ns = (nf, black)
                if cost < dist.get(ns, float('inf')):
                    dist[ns] = cost
                    prev[ns] = (state, ['delete', v])
                    heapq.heappush(pq, (cost, ns))
            # Store
            if v in fast and v not in black:
                nb = black | frozenset({v})
                ns = (fast, nb)
                nc = cost + 1
                if nc < dist.get(ns, float('inf')):
                    dist[ns] = nc
                    prev[ns] = (state, ['store', v])
                    heapq.heappush(pq, (nc, ns))

    return []


def _naive_schedule(all_vertices, input_vertices, outputs, parents, budget):
    """Count transfers for a naive sequential schedule for baseline comparison."""
    children = defaultdict(list)
    for v, ps in parents.items():
        for p in ps:
            children[p].append(v)

    # Topological order
    layer = {}
    queue = deque(list(input_vertices))
    for v in input_vertices:
        layer[v] = 0

    while queue:
        u = queue.popleft()
        for c in children[u]:
            if c not in layer:
                if all(p in layer for p in parents.get(c, [])):
                    layer[c] = max(layer[p] for p in parents[c]) + 1
                    queue.append(c)

    for v in all_vertices:
        if v not in layer:
            layer[v] = 0

    # Naive: load parent, compute, store, delete
    transfers = 0
    for v in sorted(all_vertices, key=lambda x: layer.get(x, 0)):
        if v in input_vertices:
            continue
        ps = parents.get(v, [])
        for p in ps:
            transfers += 1  # load
        transfers += 1  # store
    return transfers


def pebble(dag, fast_memory_budget):
    start_time = time.time()
    DEADLINE = start_time + 50.0

    def elapsed():
        return time.time() - start_time

    def time_check():
        return elapsed() > DEADLINE

    try:
        all_vertices, input_vertices, outputs, parents = _parse_dag(dag)
    except Exception:
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except ImportError:
            raise

    # For small DAGs, use optimal Dijkstra
    if len(all_vertices) <= 12:
        result = _pebble_small_optimal(all_vertices, input_vertices, outputs, parents, fast_memory_budget)
        if result is not None and len(result) > 0:
            return result

    # Build auxiliary structures
    children = defaultdict(list)
    for v, ps in parents.items():
        for p in ps:
            children[p].append(v)

    # Compute layer index (longest path from any input)
    layer = {}
    queue = deque(list(input_vertices))
    for v in input_vertices:
        layer[v] = 0

    while queue:
        u = queue.popleft()
        for c in children[u]:
            if c not in layer:
                if all(p in layer for p in parents.get(c, [])):
                    layer[c] = max(layer[p] for p in parents[c]) + 1
                    queue.append(c)

    for v in all_vertices:
        if v not in layer:
            layer[v] = 0

    max_L = max(layer.values()) if all_vertices else 0

    # Group by layer
    outputs_set = set(outputs)
    layer_verts = defaultdict(list)
    for v in all_vertices:
        layer_verts[layer[v]].append(v)

    # Sort within layer: outputs first, then by decreasing number of children
    for L in range(max_L + 1):
        layer_verts[L].sort(key=lambda v: (0 if v in outputs_set else 1, -len(children.get(v, []))))

    # Precompute next_use layer for each vertex
    next_use = {}
    for v in all_vertices:
        nl = max_L + 1
        for c in children.get(v, []):
            cl = layer.get(c, 0)
            if cl < nl:
                nl = cl
        next_use[v] = nl

    # Greedy schedule
    slow_mem = set(input_vertices)
    fast_mem = set()
    ops = []

    def evict_one(current_layer):
        """Evict the best vertex from fast_mem using a refined heuristic."""
        best = None
        best_score = -float('inf')
        for u in fast_mem:
            # Outputs that haven't been stored must be stored first
            if u in outputs_set and u not in slow_mem:
                continue
            nu = next_use.get(u, max_L + 1)
            if nu > max_L:
                # Never needed again: best candidate to evict
                score = 1000.0
            else:
                dist = nu - current_layer
                # Lower dist = needed sooner = worse to evict
                score = -dist * 100.0
                if u in input_vertices:
                    # Inputs are cheap to reload
                    score += 50.0
            if score > best_score:
                best_score = score
                best = u

        if best is None:
            # All candidates are un-stored outputs; force evict the one with highest next_use
            best = max(fast_mem, key=lambda u: next_use.get(u, max_L + 1))

        if best in outputs_set and best not in slow_mem:
            ops.append(['store', best])
            slow_mem.add(best)
        else:
            ops.append(['delete', best])
        fast_mem.discard(best)

    def ensure_space(n_needed, current_layer):
        """Ensure at least n_needed slots are available."""
        while len(fast_mem) + n_needed > fast_memory_budget and len(fast_mem) > 0:
            evict_one(current_layer)

    # Process each layer
    for L in range(max_L + 1):
        if time_check():
            break
        lv = layer_verts.get(L, [])
        for v in lv:
            if v in input_vertices:
                continue
            if time_check():
                break

            ps = parents.get(v, [])
            need_load = []
            for p in ps:
                if p not in fast_mem and p in slow_mem:
                    need_load.append(p)

            ensure_space(len(need_load), L)

            for p in need_load:
                ops.append(['load', p])
                fast_mem.add(p)
                slow_mem.discard(p)

            ops.append(['compute', v])
            fast_mem.add(v)
            slow_mem.discard(v)

            if len(fast_mem) > fast_memory_budget:
                evict_one(L)

    # Store any remaining outputs
    for o in outputs:
        if time_check():
            break
        if o in fast_mem:
            ops.append(['store', o])
            slow_mem.add(o)
            fast_mem.discard(o)
        elif o not in slow_mem:
            ops.append(['store', o])
            slow_mem.add(o)

    if len(ops) > 20000:
        ops = ops[:20000]

    return ops
# EVOLVE-BLOCK-END
