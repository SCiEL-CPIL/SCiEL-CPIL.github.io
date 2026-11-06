"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    """
    Optimize DAG pebbling by minimizing transfers.
    
    Strategy: Greedy with smart eviction considering:
    1. Next-use distance
    2. Fan-out (reload cost)
    3. Recomputation cost (parents count)
    4. Whether value can be recomputed cheaply
    """
    start_time = time.time()
    deadline = start_time + 55
    
    # Parse DAG structure
    if isinstance(dag, dict):
        vertices = dag.get('vertices', [])
        edges = dag.get('edges', [])
        outputs = dag.get('outputs', [])
        if not vertices:
            vertices = dag.get('nodes', [])
            edges = dag.get('arcs', [])
            outputs = dag.get('sinks', [])
    elif isinstance(dag, (list, tuple)):
        vertices = []
        edges = []
        outputs = []
        for item in dag:
            if isinstance(item, (list, tuple)):
                if len(item) >= 3:
                    v, parents, is_out = item[0], item[1], item[2]
                    vertices.append(v)
                    if is_out:
                        outputs.append(v)
                    for p in parents:
                        edges.append((p, v))
                elif len(item) == 2:
                    v, parents = item[0], item[1]
                    vertices.append(v)
                    for p in parents:
                        edges.append((p, v))
    else:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)
    
    if not vertices:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)
    
    # Build adjacency structures
    parents_map = defaultdict(list)
    children_map = defaultdict(list)
    all_vertices = set()
    output_set = set()
    
    for v in vertices:
        all_vertices.add(v)
    
    for edge in edges:
        if len(edge) == 2:
            p, c = edge[0], edge[1]
            parents_map[c].append(p)
            children_map[p].append(c)
            all_vertices.add(p)
            all_vertices.add(c)
    
    for v in outputs:
        output_set.add(v)
        all_vertices.add(v)
    
    if not all_vertices:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)
    
    # Topological sort
    in_degree = {v: 0 for v in all_vertices}
    for v in all_vertices:
        for p in parents_map.get(v, []):
            in_degree[v] += 1
    
    queue = deque([v for v in all_vertices if in_degree[v] == 0])
    topo_order = []
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children_map.get(v, []):
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    if len(topo_order) != len(all_vertices):
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)
    
    if time.time() > deadline:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)
    
    # Identify input and output vertices
    input_vertices = [v for v in topo_order if not parents_map.get(v)]
    output_vertices = [v for v in topo_order if v in output_set]
    if not output_vertices:
        output_vertices = [v for v in topo_order if not children_map.get(v)]
    
    topo_pos = {v: i for i, v in enumerate(topo_order)}
    
    # Precompute children positions and fan-out
    children_positions = {}
    for v in all_vertices:
        if v in children_map:
            children_positions[v] = sorted([topo_pos[c] for c in children_map[v]])
        else:
            children_positions[v] = []
    
    fan_out = {v: len(children_map.get(v, [])) for v in all_vertices}
    
    # Precompute recomputation cost: number of parents that would need to be in memory
    recompute_cost = {v: len(parents_map.get(v, [])) for v in all_vertices}
    
    # Greedy pebbling with improved eviction
    operations = []
    fast_mem = set()
    slow_mem = set()
    computed = set()
    
    # Track remaining uses for each vertex
    remaining_uses = {}
    for v in all_vertices:
        remaining_uses[v] = len(children_positions.get(v, []))
    
    # Process in topological order
    for idx, v in enumerate(topo_order):
        if time.time() > deadline:
            break
        
        # Load parents that are not in fast memory
        for p in parents_map.get(v, []):
            if p not in fast_mem:
                operations.append(['load', p])
                fast_mem.add(p)
                slow_mem.discard(p)
        
        # Compute v
        operations.append(['compute', v])
        computed.add(v)
        fast_mem.add(v)
        
        # Decrement remaining uses for parents
        for p in parents_map.get(v, []):
            if p in remaining_uses:
                remaining_uses[p] -= 1
        
        # If over budget, evict
        while len(fast_mem) > fast_memory_budget:
            best_to_evict = None
            best_score = float('inf')
            
            for cand in fast_mem:
                # Don't evict outputs if we can avoid it
                if cand in output_set:
                    continue
                    
                rem = remaining_uses.get(cand, 0)
                if rem <= 0:
                    # No more uses, evict this one
                    score = -1.0
                else:
                    # Find next use position
                    next_pos = None
                    cps = children_positions.get(cand, [])
                    for pos in cps:
                        if pos > idx:
                            next_pos = pos
                            break
                    if next_pos is not None:
                        dist = next_pos - idx
                        # Score: higher = more important to keep
                        # Balance: next-use proximity, fan-out benefit, recompute cost
                        reload_benefit = fan_out.get(cand, 1)
                        recompute_penalty = recompute_cost.get(cand, 1)
                        # If recompute is cheap, less important to keep
                        # If next use is far, less important to keep
                        score = (reload_benefit * 10.0) / (1.0 + dist) - recompute_penalty * 0.5
                    else:
                        score = -1.0
                
                if score < best_score:
                    best_score = score
                    best_to_evict = cand
            
            if best_to_evict is None:
                best_to_evict = next(iter(fast_mem))
            
            operations.append(['store', best_to_evict])
            fast_mem.discard(best_to_evict)
            slow_mem.add(best_to_evict)
    
    # Ensure all outputs are in fast memory at the end
    for v in output_vertices:
        if v not in fast_mem:
            operations.append(['load', v])
            fast_mem.add(v)
            slow_mem.discard(v)
    
    # Only store non-output vertices that are no longer needed
    for v in list(fast_mem):
        if v not in output_set and remaining_uses.get(v, 0) <= 0:
            operations.append(['store', v])
            fast_mem.discard(v)
            slow_mem.add(v)
    
    # Limit operations
    if len(operations) > 20000:
        operations = operations[:20000]
    
    return operations
# EVOLVE-BLOCK-END
