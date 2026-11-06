# EVOLVE-BLOCK-START
import time
from collections import deque

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Parse DAG
    vertices = dag.get('vertices', [])
    edges = dag.get('edges', [])
    inputs = set(dag.get('inputs', []))
    outputs = set(dag.get('outputs', []))
    
    if not vertices:
        return []
    
    # Build adjacency
    adj = {v: [] for v in vertices}
    radj = {v: [] for v in vertices}
    for u, v in edges:
        adj[u].append(v)
        radj[v].append(u)
    
    # Topological sort
    in_degree = {v: len(radj[v]) for v in vertices}
    queue = deque([v for v in vertices if in_degree[v] == 0])
    topo_order = []
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for w in adj[v]:
            in_degree[w] -= 1
            if in_degree[w] == 0:
                queue.append(w)
    
    # If topo_order doesn't contain all vertices, there's a cycle - return empty
    if len(topo_order) != len(vertices):
        return []
    
    # State
    fast_memory = set()  # vertices currently in fast memory (red)
    stored = set()  # vertices in slow memory (blue) that have been stored
    ops = []
    
    # For each vertex, track how many children it has and which are unprocessed
    unprocessed_children = {v: set(adj[v]) for v in vertices}
    
    # Process in topological order
    for v in topo_order:
        if time.time() > deadline:
            break
        
        # If v is an input, it's initially blue. We'll load it when needed.
        # For non-inputs, we need to compute it.
        
        if v not in inputs:
            # Ensure all parents are in fast memory
            for p in radj[v]:
                if p not in fast_memory:
                    # Load p from slow memory
                    ops.append(['load', p])
                    fast_memory.add(p)
            
            # Compute v
            ops.append(['compute', v])
            fast_memory.add(v)
        
        # Now decide what to do with v
        # v is now in fast_memory (either loaded if input, or computed)
        
        # Check if v has any unprocessed children
        has_future_uses = len(unprocessed_children[v]) > 0
        
        if not has_future_uses:
            # No future uses. Store v (unless it's an output, but outputs also need to be stored at end)
            ops.append(['store', v])
            fast_memory.remove(v)
            stored.add(v)
        else:
            # Has future uses. Check if fast memory is over budget
            if len(fast_memory) > fast_memory_budget:
                # Need to evict
                # Evict vertices with fewest unprocessed children
                # Priority: more unprocessed children = higher priority (keep)
                
                def priority(x):
                    return len(unprocessed_children[x])
                
                # Find min priority
                min_prio = min(priority(x) for x in fast_memory)
                # Evict all with min priority until under budget
                to_evict = [x for x in fast_memory if priority(x) == min_prio]
                to_evict.sort()  # deterministic
                for x in to_evict:
                    if len(fast_memory) <= fast_memory_budget:
                        break
                    ops.append(['store', x])
                    fast_memory.remove(x)
                    stored.add(x)
        
        # Update unprocessed_children for all parents of v
        for p in radj[v]:
            if v in unprocessed_children[p]:
                unprocessed_children[p].remove(v)
    
    # At the end, ensure all outputs are stored (blue)
    for v in outputs:
        if v in fast_memory:
            ops.append(['store', v])
            fast_memory.remove(v)
            stored.add(v)
    
    # Check limit
    if len(ops) > 20000:
        ops = ops[:20000]
    
    return ops
# EVOLVE-BLOCK-END
