"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque
import heapq

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    DEADLINE = start_time + 55.0

    def elapsed():
        return time.time() - start_time

    def time_check():
        return elapsed() > DEADLINE

    try:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        has_baseline = True
    except ImportError:
        has_baseline = False

    try:
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
    except Exception:
        if has_baseline:
            return pebble_baseline(dag, fast_memory_budget)
        raise

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

    # If the DAG is small enough, use Dijkstra for optimal solution
    if len(all_vertices) <= 20:
        return pebble_dijkstra(dag, fast_memory_budget, all_vertices, input_vertices, outputs, parents, time_check)

    # Otherwise, use improved greedy
    # Precompute children and future use counts
    children = defaultdict(list)
    for v, ps in parents.items():
        for p in ps:
            children[p].append(v)
    
    # Count total uses (fan-out) for each vertex
    use_count = {v: len(children[v]) for v in all_vertices}
    
    # Topological sort to process in order
    in_deg = {v: len(parents.get(v, [])) for v in all_vertices}
    queue = deque([v for v in all_vertices if in_deg[v] == 0])
    topo_order = []
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children[v]:
            in_deg[c] -= 1
            if in_deg[c] == 0:
                queue.append(c)
        
    # If topo sort didn't include all, there's a cycle or disconnected graph, 
    # but DAG implies no cycles. 
    # Ensure all vertices are in topo_order
    if len(topo_order) < len(all_vertices):
        # Add missing vertices (should not happen in valid DAG)
        for v in all_vertices:
            if v not in topo_order:
                topo_order.append(v)

    slow_mem = set(input_vertices)
    fast_mem = set()
    ops = []
    
    # To allow "reloading" of computed nodes, we need to know if they are in slow_mem.
    # We will simulate the state.
    
    # We process vertices in topological order.
    # However, we can choose the order of computation within topological constraints 
    # to optimize memory usage. 
    # A simple greedy: Process vertices in topological order. 
    # When a vertex is ready (all parents in fast_mem), compute it.
    # But we might want to delay computation to keep parents in memory?
    # Actually, computing a vertex adds it to fast_mem. 
    # If we compute too many, we evict.
    
    # Let's use a priority queue for "ready" vertices.
    # Priority: High fan-out first? Or Low fan-out first?
    # If we compute a high fan-out vertex early, it stays in memory and can be used by many children, 
    # reducing loads for those children.
    # So, compute high fan-out vertices first?
    # But we need their parents.
    
    # Let's stick to Topological Order but with a twist:
    # We maintain a set of "ready" vertices (parents in fast_mem).
    # We pick the one with the highest "value" to compute next.
    # Value: use_count[v] (how many children it has).
    
    ready = set()
    for v in topo_order:
        if v in input_vertices:
            # Inputs are initially in slow_mem. 
            # We need to load them to fast_mem to compute children.
            # But we don't compute inputs. We just load them when needed.
            pass

    # Initialize: Inputs are in slow_mem. 
    # We don't load them until needed.
    
    # To make this work, we need to dynamically determine readiness.
    # A vertex is ready if all its parents are in fast_mem.
    
    # We iterate through topo_order.
    # For each v in topo_order:
    #   If v is input: 
    #     It's "ready" to be loaded.
    #   If v is not input:
    #     It becomes ready when all parents are in fast_mem.
    
    # This is tricky because loading an input takes space.
    
    # Alternative: 
    # 1. Start with all inputs in slow_mem.
    # 2. Fast_mem is empty.
    # 3. We have a set of "available" vertices to compute/load.
    #    - Inputs are available to load.
    #    - Non-inputs are available to compute if parents are in fast_mem.
    
    # Let's use a loop that continues until all outputs are stored.
    # State: fast_mem, slow_mem, computed (set of vertices that have been computed at least once)
    
    # Actually, the standard greedy for pebbling is:
    # 1. Identify vertices whose parents are in fast_mem.
    # 2. Among those, pick one to compute.
    # 3. If fast_mem is full, evict.
    
    # Let's implement a simple greedy that processes in topological order but 
    # loads parents just-in-time.
    
    # This is essentially the same as the current program but with better eviction.
    # Let's refine the eviction logic.
    
    # Precompute next use layers (as in current) to estimate urgency.
    layer = {}
    for v in input_vertices:
        layer[v] = 0
    queue = deque(list(input_vertices))
    while queue:
        v = queue.popleft()
        for c in children[v]:
            if c not in layer:
                all_p_done = all(p in layer for p in parents.get(c, []))
                if all_p_done:
                    layer[c] = max(layer[p] for p in parents.get(c, [])) + 1
                    queue.append(c)
    for v in all_vertices:
        if v not in layer:
            layer[v] = 0

    layers = defaultdict(list)
    for v in all_vertices:
        layers[layer[v]].append(v)
    max_L = max(layers.keys()) if layers else 0

    # Precompute next use layers for each vertex
    next_use_layers = {v: [] for v in all_vertices}
    for v in all_vertices:
        for c in children.get(v, []):
            cl = layer.get(c, 0)
            next_use_layers[v].append(cl)
        next_use_layers[v].sort()

    slow_mem = set(input_vertices)
    fast_mem = set()
    ops = []
    
    # We will process layer by layer, but within a layer, we can choose order.
    # Actually, let's just iterate through all vertices in topological order.
    # But we need to ensure parents are loaded.
    
    # Let's use the layer-based approach but improve eviction.
    
    for L in range(max_L + 1):
        if time_check():
            break
        lv = layers.get(L, [])
        if not lv:
            continue

        # For each vertex in this layer, ensure parents are in fast_mem
        for v in lv:
            if v in input_vertices:
                continue
            if time_check():
                break
            
            # Load any parents not in fast_mem
            for p in parents.get(v, []):
                if p not in fast_mem:
                    if p in slow_mem:
                        ops.append(['load', p])
                        fast_mem.add(p)
                    else:
                        # This should not happen if we maintain state correctly,
                        # unless we deleted it and didn't store it.
                        # If we didn't store it, we must recompute it.
                        # But p is a parent, so it was computed earlier.
                        # If it was deleted and not stored, we lost it.
                        # To avoid this, we should store nodes before deleting if they have future uses.
                        pass
            
            ops.append(['compute', v])
            fast_mem.add(v)
            
            # Evict if needed
            while len(fast_mem) > fast_memory_budget and not time_check():
                best = None
                best_score = float('inf')
                
                for u in fast_mem:
                    # Calculate score for eviction
                    # Lower score = better to evict
                    
                    # 1. If u is an output, we must store it. 
                    #    Storing it allows us to delete it from fast_mem.
                    #    Score: 0 (we will store and delete, cost 1 transfer)
                    if u in outputs:
                        score = 0.0
                    else:
                        # 2. Check if u is in slow_mem
                        if u in slow_mem:
                            # Can reload with 1 transfer
                            # Score based on future usage
                            nu = max_L + 1
                            for nl in next_use_layers[u]:
                                if nl > L:
                                    nu = nl
                                    break
                            if nu > max_L:
                                score = 1.0 # Never needed again, evict
                            else:
                                dist = nu - L
                                score = 1.0 + dist * 0.1
                        else:
                            # Not in slow_mem. Must recompute if needed again.
                            # Recomputation is expensive.
                            # Score should be high (bad to evict)
                            nu = max_L + 1
                            for nl in next_use_layers[u]:
                                if nl > L:
                                    nu = nl
                                    break
                            if nu > max_L:
                                score = 2.0 # Never needed again, but not in slow mem. Just delete.
                            else:
                                dist = nu - L
                                # High penalty for evicting non-stored nodes
                                score = 10.0 + dist * 0.1
                    
                    if score < best_score:
                        best_score = score
                        best = u

                if best is None:
                    break

                if best in outputs:
                    ops.append(['store', best])
                    slow_mem.add(best)
                    fast_mem.discard(best)
                elif best in slow_mem:
                    # Just delete, can reload later
                    ops.append(['delete', best])
                    fast_mem.discard(best)
                else:
                    # Not in slow_mem. 
                    # If it has future uses, we should store it first?
                    # Storing costs 1. Deleting costs 0.
                    # If we store, we can reload later (1). Total 2.
                    # If we delete, we must recompute (costly).
                    # So store it.
                    ops.append(['store', best])
                    slow_mem.add(best)
                    ops.append(['delete', best])
                    fast_mem.discard(best)

    # Store remaining outputs
    for o in outputs:
        if time_check():
            break
        if o in fast_mem:
            ops.append(['store', o])
            slow_mem.add(o)
            fast_mem.discard(o)

    if len(ops) > 20000:
        ops = ops[:20000]

    return ops

def pebble_dijkstra(dag, budget, all_vertices, input_vertices, outputs, parents, time_check):
    # State: (frozenset(fast_red), frozenset(black))
    init_black = frozenset(input_vertices)
    init_fast = frozenset()
    
    dist = {}
    prev = {}
    
    start_state = (init_fast, init_black)
    dist[start_state] = 0
    
    pq = [(0, start_state)]
    
    while pq:
        if time_check():
            break
        cost, state = heapq.heappop(pq)
        fast, black = state
        
        if cost > dist.get(state, float('inf')):
            continue
            
        if all(o in black for o in outputs):
            ops = []
            curr = state
            while curr != start_state:
                p_state, op = prev[curr]
                ops.append(op)
                curr = p_state
            ops.reverse()
            return ops
            
        # Generate neighbors
        # 1. Compute
        for v in all_vertices:
            if v in fast or v in black:
                continue
            ps = parents.get(v, [])
            if all(p in fast or p in black for p in ps):
                new_fast = fast | frozenset([v])
                if len(new_fast) <= budget:
                    new_state = (new_fast, black)
                    new_cost = cost
                    if new_cost < dist.get(new_state, float('inf')):
                        dist[new_state] = new_cost
                        prev[new_state] = (state, ['compute', v])
                        heapq.heappush(pq, (new_cost, new_state))
                        
        # 2. Store
        for v in fast:
            if v not in black:
                new_black = black | frozenset([v])
                new_state = (fast, new_black)
                new_cost = cost + 1
                if new_cost < dist.get(new_state, float('inf')):
                    dist[new_state] = new_cost
                    prev[new_state] = (state, ['store', v])
                    heapq.heappush(pq, (new_cost, new_state))
                    
        # 3. Delete
        for v in fast:
            new_fast = fast - frozenset([v])
            new_state = (new_fast, black)
            new_cost = cost
            if new_cost < dist.get(new_state, float('inf')):
                dist[new_state] = new_cost
                prev[new_state] = (state, ['delete', v])
                heapq.heappush(pq, (new_cost, new_state))
                
        # 4. Load
        for v in black:
            if v not in fast:
                new_fast = fast | frozenset([v])
                if len(new_fast) <= budget:
                    new_state = (new_fast, black)
                    new_cost = cost + 1
                    if new_cost < dist.get(new_state, float('inf')):
                        dist[new_state] = new_cost
                        prev[new_state] = (state, ['load', v])
                        heapq.heappush(pq, (new_cost, new_state))

    return []

# EVOLVE-BLOCK-END
