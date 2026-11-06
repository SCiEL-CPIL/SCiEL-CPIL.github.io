"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
from collections import defaultdict, deque

def _parse_dag(dag):
    """Parse DAG structure into vertices, edges, outputs."""
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
        return None, None, None
    
    if not vertices:
        return None, None, None
    
    return vertices, edges, outputs

def _topo_sort(all_vertices, parents_map, children_map):
    """Topological sort using Kahn's algorithm."""
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
    
    return topo_order

def _compute_recomp_cost(topo_order, parents_map, children_map):
    """Compute recomputation cost for each vertex (number of compute ops to recompute)."""
    recomp_cost = {}
    for v in topo_order:
        if not parents_map.get(v):
            recomp_cost[v] = 0  # Inputs are free to load
        else:
            recomp_cost[v] = 1 + sum(recomp_cost.get(p, 0) for p in parents_map[v])
    return recomp_cost

def _compute_need_schedule(topo_order, children_map):
    """Compute all positions where each vertex is needed (as parent of a child)."""
    need_positions = defaultdict(list)
    topo_pos = {v: i for i, v in enumerate(topo_order)}
    for v in topo_order:
        for c in children_map.get(v, []):
            need_positions[v].append(topo_pos[c])
    return need_positions, topo_pos

def pebble(dag, fast_memory_budget):
    """
    Optimize DAG pebbling by minimizing transfers.
    
    Strategy: Greedy with recomputation-cost-aware eviction.
    - Process in topological order
    - Load parents just-in-time
    - Evict based on recomputation cost vs. urgency (how soon needed again)
    """
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    # Parse DAG
    vertices, edges, outputs = _parse_dag(dag)
    if vertices is None:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)
    
    # Build adjacency
    parents_map = defaultdict(list)
    children_map = defaultdict(list)
    all_vertices = set()
    output_set = set(outputs)
    
    for v in vertices:
        all_vertices.add(v)
    for edge in edges:
        if len(edge) == 2:
            p, c = edge[0], edge[1]
            parents_map[c].append(p)
            children_map[p].append(c)
            all_vertices.add(p)
            all_vertices.add(c)
    
    if not all_vertices:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)
    
    # Topological sort
    topo_order = _topo_sort(all_vertices, parents_map, children_map)
    if len(topo_order) != len(all_vertices):
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)
    
    if time.time() > deadline:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)
    
    # Precompute
    recomp_cost = _compute_recomp_cost(topo_order, parents_map, children_map)
    need_positions, topo_pos = _compute_need_schedule(topo_order, children_map)
    
    output_vertices = [v for v in topo_order if v in output_set]
    if not output_vertices:
        output_vertices = [v for v in topo_order if not children_map.get(v)]
    
    # Greedy pebbling
    operations = []
    fast_mem = set()
    slow_mem = set()
    
    # Track next need index for each vertex (for efficient lookup)
    next_need_idx = {}  # vertex -> index into need_positions[v] list
    
    for v in topo_order:
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
        fast_mem.add(v)
        
        # If over budget, evict
        while len(fast_mem) > fast_memory_budget:
            # Score each candidate for eviction:
            # We want to evict the vertex with the lowest "value" of keeping it
            # Value = high if needed soon OR expensive to recompute
            # Evict the one with lowest value
            
            best_evict = None
            best_score = float('inf')  # We want to minimize this score (evict lowest value)
            
            for candidate in fast_mem:
                # Get next need position
                np_list = need_positions.get(candidate, [])
                ni = next_need_idx.get(candidate, 0)
                if ni < len(np_list):
                    next_need_pos = np_list[ni]
                else:
                    next_need_pos = float('inf')
                
                # Urgency: how soon is it needed? (smaller = more urgent = higher value)
                urgency = max(0, next_need_pos - topo_pos[v]) if next_need_pos != float('inf') else 0
                
                # Recomputation cost
                rc = recomp_cost.get(candidate, 0)
                
                # Value of keeping: high urgency + high recomp cost = keep
                # Score for eviction: low urgency + low recomp cost = evict
                # Use: score = urgency * (rc + 1)  -> lower score = better to evict
                score = urgency * (rc + 1)
                
                # Outputs should never be evicted (or very high penalty)
                if candidate in output_set:
                    score = float('inf')
                
                if score < best_score:
                    best_score = score
                    best_evict = candidate
            
            if best_evict is None or best_evict in output_set:
                # Can't evict outputs, try next non-output
                non_outputs = [c for c in fast_mem if c not in output_set]
                if not non_outputs:
                    break
                best_evict = non_outputs[0]
            
            operations.append(['store', best_evict])
            fast_mem.discard(best_evict)
            slow_mem.add(best_evict)
            
            # Advance next need index for evicted vertex
            ni = next_need_idx.get(best_evict, 0)
            np_list = need_positions.get(best_evict, [])
            # Find first need position > current topo_pos
            while ni < len(np_list) and np_list[ni] <= topo_pos[v]:
                ni += 1
            next_need_idx[best_evict] = ni
        
        # Advance next need index for v (it's now computed and in fast_mem)
        next_need_idx[v] = 0
    
    # Ensure all outputs are in fast memory at the end
    for v in output_vertices:
        if v not in fast_mem:
            operations.append(['load', v])
            fast_mem.add(v)
            slow_mem.discard(v)
    
    # Final cleanup: store everything that's not an output
    for v in list(fast_mem):
        if v not in output_set:
            operations.append(['store', v])
            fast_mem.discard(v)
            slow_mem.add(v)
    
    if len(operations) > 20000:
        operations = operations[:20000]
    
    return operations
# EVOLVE-BLOCK-END
