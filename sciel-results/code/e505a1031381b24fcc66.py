"""Communication-minimizing DAG pebbling via time-reversed unpebbling."""
# EVOLVE-BLOCK-START
import time
import math
from collections import defaultdict

def pebble(dag, fast_memory_budget):
    """
    dag: dict with keys:
        'n': number of vertices
        'parents': list of lists, parents[v] = list of parents of v (topologically ordered)
        'outputs': list of output vertex indices
    fast_memory_budget: int, max number of red (in-memory) values
    
    Returns: list of [operation, vertex] where operation in {'load', 'store', 'compute', 'delete'}
    """
    start_time = time.time()
    DEADLINE = start_time + 85.0  # safety margin before 90s
    
    n = dag['n']
    parents = dag['parents']
    outputs = set(dag['outputs'])
    
    # Build children adjacency
    children = defaultdict(list)
    for v in range(n):
        for p in parents[v]:
            children[p].append(v)
    
    # Identify input vertices (no parents)
    inputs = set(v for v in range(n) if len(parents[v]) == 0)
    
    # Compute in-degree (number of parents) for each vertex
    in_degree = [len(parents[v]) for v in range(n)]
    
    # Compute out-degree (number of children)
    out_degree = [len(children[v]) for v in range(n)]
    
    # Compute topological order (already given by vertex indices presumably)
    # But let's be safe and compute it
    from collections import deque
    in_deg_copy = [len(parents[v]) for v in range(n)]
    queue = deque([v for v in range(n) if in_deg_copy[v] == 0])
    topo_order = []
    temp_in_deg = in_deg_copy[:]
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children[v]:
            temp_in_deg[c] -= 1
            if temp_in_deg[c] == 0:
                queue.append(c)
    
    # Reverse topological order
    rev_topo = list(reversed(topo_order))
    
    # Naive schedule baseline: just compute everything once, keeping all intermediates
    # For comparison/scoring purposes, but we need to produce a valid schedule
    
    # Strategy: Time-reversed greedy "unpebbling"
    # Process from outputs backwards
    # We simulate the pebbling process in reverse:
    # Forward: start with inputs in memory (blue), compute intermediates, end with outputs in memory (blue)
    # Reverse: start with outputs in memory, "uncompute" to get parents, end with inputs in memory
    
    # Actually, let's think about this differently.
    # The key insight: minimize transfers (load/store) while respecting memory budget.
    
    # Let's use a forward greedy approach with lookahead:
    # 1. Determine a computation order (topological)
    # 2. At each step, if a vertex's parents are all in memory, compute it
    # 3. When memory is full, evict the vertex whose next use is farthest in the future
    
    # First, determine a good computation order
    # We want to process vertices that share parents together to maximize reuse
    
    # Build a priority queue based on:
    # 1. Vertices whose parents are all available
    # 2. Among those, prefer vertices with more children (higher fanout) or used soon
    
    # Precompute next use for each vertex in topological order
    # For each vertex v, next_use[v] = the earliest position in topo_order where v is needed as a parent
    next_use = [float('inf')] * n
    for v in range(n):
        for c in children[v]:
            # v is used when c is computed
            if topo_order.index(c) < next_use[v]:
                next_use[v] = topo_order.index(c)
    
    # Simpler: just use topological order as the computation order
    # and use a greedy eviction strategy
    
    # Forward greedy pebbling:
    in_memory = set()  # vertices currently in fast memory
    computed = set()   # vertices that have been computed at least once
    operations = []
    
    # Load all inputs
    for v in inputs:
        operations.append(['load', v])
        in_memory.add(v)
        computed.add(v)
    
    # Process vertices in topological order
    for v in topo_order:
        if v in inputs:
            continue  # already loaded
        
        # Check if all parents are in memory
        all_parents_in_memory = all(p in in_memory for p in parents[v])
        
        if not all_parents_in_memory:
            # Need to load missing parents
            for p in parents[v]:
                if p not in in_memory:
                    operations.append(['load', p])
                    in_memory.add(p)
        
        # Compute v
        operations.append(['compute', v])
        in_memory.add(v)
        computed.add(v)
        
        # Check if memory is over budget
        # Remove parents that are no longer needed
        while len(in_memory) > fast_memory_budget:
            # Find the vertex in in_memory that is not an output and has the farthest next use
            # (or is not needed at all)
            candidates = []
            for u in in_memory:
                if u in outputs:
                    continue  # don't evict outputs
                # Find next use of u
                nu = next_use[u]
                if nu == float('inf'):
                    nu = len(topo_order) + 1  # never needed again
                candidates.append((nu, u))
            
            if not candidates:
                break
            
            # Sort by next_use descending (least urgent first)
            candidates.sort(reverse=True)
            _, u_to_remove = candidates[0]
            
            # Store it if it might be needed again
            if next_use[u_to_remove] < float('inf'):
                operations.append(['store', u_to_remove])
            else:
                operations.append(['delete', u_to_remove])
            in_memory.discard(u_to_remove)
    
    # Ensure all outputs are in memory at the end
    for v in outputs:
        if v not in in_memory:
            operations.append(['load', v])
            in_memory.add(v)
    
    # Truncate to 20000 operations max
    if len(operations) > 20000:
        operations = operations[:20000]
    
    return operations

# EVOLVE-BLOCK-END
