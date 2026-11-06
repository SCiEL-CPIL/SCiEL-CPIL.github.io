"""Rewritten pebbling solver with improved eviction and recomputation awareness."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    """
    Optimize DAG pebbling by minimizing transfers.
    
    Strategy: Topological processing with demand-aware eviction.
    1. Process in topological order
    2. Evict based on: remaining uses, fan-out, and recompute cost
    3. Avoid evicting vertices that are about to be used again
    """
    start_time = time.time()
    deadline = start_time + 50  # Safety margin for 60s limit
    
    # Parse DAG structure
    if isinstance(dag, dict):
        vertices = dag.get('vertices', [])
        edges = dag.get('edges', [])
        outputs = dag.get('outputs', [])
        parents_dict = dag.get('parents', None)
        if not vertices:
            vertices = dag.get('nodes', [])
            edges = dag.get('arcs', [])
            outputs = dag.get('sinks', [])
            parents_dict = dag.get('parents', None)
    elif isinstance(dag, (list, tuple)):
        vertices = []
        edges = []
        outputs = []
        parents_dict = None
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
            elif isinstance(item, dict):
                v = item.get('vertex', item.get('node', item.get('id', None)))
                if v is None:
                    continue
                parents = item.get('parents', item.get('deps', []))
                is_out = item.get('is_output', item.get('is_out', False))
                vertices.append(v)
                if is_out:
                    outputs.append(v)
                for p in parents:
                    edges.append((p, v))
    else:
        return []
    
    if not vertices:
        return []
    
    # Build adjacency structures
    parents_map = defaultdict(list)
    children_map = defaultdict(list)
    all_vertices = set()
    output_set = set()
    
    for v in vertices:
        all_vertices.add(v)
    
    if parents_dict:
        for v, plist in parents_dict.items():
            if v not in all_vertices and not isinstance(v, int):
                all_vertices.add(v)
            for p in plist:
                if p not in all_vertices:
                    all_vertices.add(p)
                parents_map[v].append(p)
                children_map[p].append(v)
    else:
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
        return []
    
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
        return []
    
    if time.time() > deadline:
        return []
    
    # Identify input and output vertices
    input_vertices = [v for v in topo_order if not parents_map.get(v)]
    output_vertices = [v for v in topo_order if v in output_set]
    if not output_vertices:
        output_vertices = [v for v in topo_order if not children_map.get(v)]
    
    topo_pos = {v: i for i, v in enumerate(topo_order)}
    
    # Precompute: for each vertex, list of children positions (sorted)
    children_positions = {}
    for v in all_vertices:
        if v in children_map:
            children_positions[v] = sorted([topo_pos[c] for c in children_map[v]])
        else:
            children_positions[v] = []
    
    # Count total uses (fan-out) for each vertex
    fan_out = {v: len(children_map.get(v, [])) for v in all_vertices}
    
    # Compute depth for eviction priority
    depth = {}
    for v in topo_order:
        if not parents_map.get(v):
            depth[v] = 0
        else:
            depth[v] = max(depth.get(p, 0) for p in parents_map[v]) + 1
    
    # Compute recompute cost: number of ancestors (approximation of work to recompute)
    recompute_cost = {}
    for v in topo_order:
        if not parents_map.get(v):
            recompute_cost[v] = 0
        else:
            recompute_cost[v] = 1 + max(recompute_cost.get(p, 0) for p in parents_map[v])
    
    # Greedy pebbling
    operations = []
    fast_mem = set()
    slow_mem = set()
    computed = set()
    
    # Track remaining uses for each vertex
    remaining_uses = {}
    for v in all_vertices:
        remaining_uses[v] = len(children_positions.get(v, []))
    
    # Track next use position for each vertex (index into children_positions)
    next_use_idx = {v: 0 for v in all_vertices}
    
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
        
        # Decrement remaining uses for parents (they were just used)
        for p in parents_map.get(v, []):
            if p in remaining_uses:
                remaining_uses[p] -= 1
                # Advance next_use_idx for p
                while next_use_idx[p] < len(children_positions.get(p, [])) and \
                      children_positions[p][next_use_idx[p]] <= idx:
                    next_use_idx[p] += 1
        
        # If over budget, evict
        while len(fast_mem) > fast_memory_budget:
            # Score each candidate: lower score = more important to keep
            # We want to evict the vertex with the highest "eviction score"
            # Eviction score = - (remaining_uses * weight1 + fan_out * weight2 + recompute_cost * weight3)
            # Vertices with no remaining uses get evicted first
            best_to_evict = None
            best_score = float('-inf')
            
            for cand in fast_mem:
                # Don't evict outputs if we can avoid it
                if cand in output_set:
                    continue
                    
                rem = remaining_uses.get(cand, 0)
                
                if rem <= 0:
                    # No more uses, evict this one with highest priority
                    score = 1000.0 + depth.get(cand, 0) * 0.1
                else:
                    # Find next use distance
                    nui = next_use_idx.get(cand, 0)
                    cps = children_positions.get(cand, [])
                    if nui < len(cps):
                        next_pos = cps[nui]
                        dist = next_pos - idx
                        # Higher remaining uses, higher fan-out, higher recompute cost = more important
                        # Smaller distance to next use = more important
                        score = (rem * 10.0 + fan_out.get(cand, 1) * 5.0 + recompute_cost.get(cand, 0) * 2.0) / (1.0 + dist)
                    else:
                        score = 0.0
                
                if score > best_score:
                    best_score = score
                    best_to_evict = cand
            
            if best_to_evict is None:
                # All are outputs, evict one anyway (least important output)
                best_to_evict = min(fast_mem, key=lambda x: remaining_uses.get(x, 0))
            
            operations.append(['store', best_to_evict])
            fast_mem.discard(best_to_evict)
            slow_mem.add(best_to_evict)
    
    # Ensure all outputs are in fast memory at the end
    for v in output_vertices:
        if v not in fast_mem:
            operations.append(['load', v])
            fast_mem.add(v)
            slow_mem.discard(v)
    
    # Final cleanup: store non-output vertices
    for v in list(fast_mem):
        if v not in output_set:
            operations.append(['store', v])
            fast_mem.discard(v)
            slow_mem.add(v)
    
    # Limit operations
    if len(operations) > 20000:
        operations = operations[:20000]
    
    return operations
# EVOLVE-BLOCK-END
