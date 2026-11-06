"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    """
    Communication-minimizing DAG pebbling.
    dag structure expected:
      - dag['parents']: dict vertex -> list of parent vertices (topologically ordered)
      - dag['outputs']: list of output vertices
      - dag['inputs']: list of input vertices (start blue)
    Returns list of [operation, vertex] where operation in {load, store, compute, delete}.
    """
    start_time = time.time()
    DEADLINE = 80.0  # wall-clock limit with safety margin

    try:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
    except Exception:
        pebble_baseline = None

    try:
        parents = dag.get('parents', {})
        outputs = dag.get('outputs', [])
        inputs = dag.get('inputs', [])
        
        # Build reverse adjacency (children)
        children = defaultdict(list)
        for v, ps in parents.items():
            for p in ps:
                children[p].append(v)
        
        all_vertices = set(parents.keys()) | set(inputs) | set(outputs)
        
        # Topological order
        in_degree = {v: len(parents.get(v, [])) for v in all_vertices}
        for v in inputs:
            in_degree[v] = 0
        
        queue = [v for v in all_vertices if in_degree.get(v, 0) == 0]
        topo_order = []
        while queue:
            queue.sort()
            u = queue.pop(0)
            topo_order.append(u)
            for c in children.get(u, []):
                in_degree[c] -= 1
                if in_degree[c] == 0:
                    queue.append(c)
        
        # Map vertex to index in topo_order
        topo_index = {v: i for i, v in enumerate(topo_order)}
        
        # Precompute: for each vertex, the set of children (direct dependents)
        # and the "last_use" index: the highest topo_index of any child
        last_use = {}
        for v in all_vertices:
            if v in children and children[v]:
                last_use[v] = max(topo_index[c] for c in children[v])
            else:
                last_use[v] = topo_index.get(v, -1)
        
        # Also ensure outputs are kept until the end
        must_be_blue = set(outputs)
        
        # Strategy: Use a more intelligent pebbling strategy
        # 1. First, compute a "recomputation cost" for each vertex
        # 2. Use this to decide what to keep in memory vs recompute
        
        # Precompute depth (longest path from input)
        depth = {}
        for v in topo_order:
            ps = parents.get(v, [])
            if not ps:
                depth[v] = 0
            else:
                depth[v] = max(depth[p] for p in ps) + 1
        
        # Precompute: for each vertex, how many times it's needed in the future
        future_uses = {}
        for v in all_vertices:
            count = 0
            for c in children.get(v, []):
                count += 1
            future_uses[v] = count
        
        # Dynamic programming approach for optimal pebbling
        # State: set of vertices in fast memory
        # But this is exponential, so we use a heuristic
        
        # Better approach: Use a priority-based eviction strategy
        # When memory is full, evict the vertex with the lowest "value"
        # Value = (future_uses * depth) / (1 + last_use_distance)
        
        ops = []
        fast_mem = set()
        
        def get_eviction_priority(v):
            """
            Lower priority means evict first.
            We want to keep vertices that are:
            1. Used many times in the future (high future_uses)
            2. Expensive to recompute (high depth)
            3. Needed soon (low last_use_distance)
            """
            if v in must_be_blue:
                return float('inf')  # Never evict outputs
            
            # Distance to last use
            current_index = len(ops)  # Approximate current position
            last_use_dist = max(0, last_use[v] - current_index)
            
            # Value: higher means keep
            value = future_uses[v] * (depth[v] + 1) / (1 + last_use_dist)
            return value
        
        def ensure_in_fast_mem(v, current_index):
            """
            Ensure vertex v is in fast_mem.
            Uses smart eviction when memory is full.
            """
            if v in fast_mem:
                return True
            
            ps = parents.get(v, [])
            if not ps:
                # It's an input
                if len(fast_mem) >= fast_memory_budget:
                    # Find the vertex with lowest priority to evict
                    best_evict = None
                    best_priority = float('inf')
                    for u in fast_mem:
                        if u == v:
                            continue
                        p = get_eviction_priority(u)
                        if p < best_priority:
                            best_priority = p
                            best_evict = u
                    
                    if best_evict is not None:
                        ops.append(['delete', best_evict])
                        fast_mem.discard(best_evict)
                    else:
                        return False
                
                ops.append(['load', v])
                fast_mem.add(v)
                return True
            else:
                # It's a computed vertex
                # First, ensure all its parents are in fast_mem
                for p in ps:
                    if not ensure_in_fast_mem(p, current_index):
                        return False
                
                # Now compute v
                if len(fast_mem) >= fast_memory_budget:
                    # Find the vertex with lowest priority to evict
                    best_evict = None
                    best_priority = float('inf')
                    for u in fast_mem:
                        if u == v:
                            continue
                        p = get_eviction_priority(u)
                        if p < best_priority:
                            best_priority = p
                            best_evict = u
                    
                    if best_evict is not None:
                        ops.append(['delete', best_evict])
                        fast_mem.discard(best_evict)
                    else:
                        return False
                
                ops.append(['compute', v])
                fast_mem.add(v)
                return True
        
        # Process in topological order
        for i, v in enumerate(topo_order):
            if time.time() > DEADLINE:
                break
            
            if not ensure_in_fast_mem(v, i):
                break
            
            # Proactively delete vertices that are no longer needed
            for u in list(fast_mem):
                if u == v:
                    continue
                if last_use[u] < i and u not in must_be_blue:
                    ops.append(['delete', u])
                    fast_mem.discard(u)
        
        # Ensure all outputs are in fast_mem at the end
        for o in outputs:
            if o not in fast_mem:
                ensure_in_fast_mem(o, len(topo_order))
        
        # Check operation count
        if len(ops) <= 20000:
            return ops
        else:
            # Too many ops, try baseline or truncate
            if pebble_baseline:
                return pebble_baseline(dag, fast_memory_budget)
            return ops[:20000]
            
    except Exception as e:
        # Fallback to baseline if available
        if pebble_baseline:
            return pebble_baseline(dag, fast_memory_budget)
        return []
# EVOLVE-BLOCK-END
