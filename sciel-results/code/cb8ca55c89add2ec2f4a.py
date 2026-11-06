"""Communication-minimizing DAG pebbling with bounded fast memory."""
import time

# EVOLVE-BLOCK-START
def pebble(dag, fast_memory_budget):
    """
    Generate a pebbling schedule for a DAG under a fast-memory budget.
    
    Args:
        dag: A dict with keys:
            - 'vertices': list of vertex ids (topologically ordered)
            - 'parents': dict mapping vertex -> list of parent vertices
            - 'outputs': list of output vertex ids
            - 'inputs': list of input vertex ids (start blue, free)
            - 'num_vertices': int (optional, can be inferred)
        fast_memory_budget: int, max number of red values in fast memory
        
    Returns:
        list of [operation, vertex] where operation is one of
        'load', 'store', 'compute', 'delete'
    """
    start_time = time.time()
    DEADLINE = 55.0  # safety margin before 60s limit
    
    # Parse DAG structure
    vertices = dag.get('vertices', [])
    parents = dag.get('parents', {})
    outputs = dag.get('outputs', [])
    inputs = dag.get('inputs', [])
    n = len(vertices)
    
    if n == 0:
        return []
    
    # Build children map and identify inputs
    children = {v: [] for v in vertices}
    in_degree = {v: 0 for v in vertices}
    for v in vertices:
        ps = parents.get(v, [])
        in_degree[v] = len(ps)
        for p in ps:
            if p in children:
                children[p].append(v)
    
    # Inputs are vertices with no parents (or explicitly given)
    input_set = set(inputs) if inputs else set(v for v in vertices if in_degree[v] == 0)
    
    # Output set
    output_set = set(outputs)
    
    # Compute naive schedule (topological order, load parents, compute, store outputs)
    # For scoring, we need to minimize transfers vs naive
    # Naive: for each vertex in topo order, load all parents (if not already loaded),
    #        compute, then store if output
    
    # Strategy: Process vertices in topological order.
    # For each vertex, ensure its parents are in fast memory (load if not).
    # Compute the vertex (it becomes red, consuming a slot).
    # If the vertex is an output, we must keep it red until the end (or store it).
    # We need to manage the fast memory budget by evicting (storing) values.
    
    # A simple but effective strategy:
    # 1. Process in topological order
    # 2. Keep track of which values are in fast memory (red)
    # 3. When computing a vertex, ensure all parents are in fast memory
    #    - If a parent is not in fast memory, we need to load it (transfer from slow)
    #    - But wait: inputs start blue (in slow memory), so loading an input is a transfer
    #    - For non-input vertices, they were computed and stored (red -> slow)
    # 4. After computing, if fast memory is full, evict a non-essential value
    #    (store it back to slow memory)
    # 5. Outputs must remain red at the end
    
    # Let's implement a priority-based eviction strategy:
    # - Values needed soon (high frequency of future use) should be kept
    # - Values not needed again can be evicted
    
    # Precompute: for each vertex, how many times it's used (fan-out)
    # and the next time it's needed
    future_uses = {v: 0 for v in vertices}
    for v in vertices:
        for child in children[v]:
            future_uses[v] += 1
    
    # Process in topological order
    fast_mem = set()  # vertices currently in fast memory (red)
    stored = set()    # vertices that have been computed and stored to slow
    operations = []
    
    # Inputs start blue (in slow memory), not in fast memory
    # We need to load inputs when needed
    
    def load_vertex(v):
        """Load vertex v from slow memory to fast memory."""
        nonlocal fast_mem
        if v not in fast_mem:
            operations.append(['load', v])
            fast_mem.add(v)
    
    def store_vertex(v):
        """Store vertex v from fast memory to slow memory."""
        nonlocal fast_mem
        if v in fast_mem:
            operations.append(['store', v])
            fast_mem.remove(v)
            stored.add(v)
    
    def compute_vertex(v):
        """Compute vertex v (all parents must be in fast memory)."""
        nonlocal fast_mem
        operations.append(['compute', v])
        fast_mem.add(v)
        stored.discard(v)
    
    def delete_vertex(v):
        """Delete vertex v from fast memory (only valid if not needed again)."""
        nonlocal fast_mem
        if v in fast_mem:
            operations.append(['delete', v])
            fast_mem.remove(v)
    
    # For eviction, we need to decide which value to evict
    # Strategy: evict the value with the lowest future use count
    # If tie, evict the one that was loaded earliest (LIFO is bad, use LIFO for stack-like)
    # Actually, let's use: evict the value with lowest (future_uses, -load_time)
    # i.e., least used first, then most recently loaded last (keep recently loaded)
    
    load_order = {}
    load_counter = [0]
    
    def evict_if_needed(exclude=None):
        """Evict values from fast memory until budget is met, excluding specified set."""
        nonlocal fast_mem
        if exclude is None:
            exclude = set()
        while len(fast_mem) > fast_memory_budget:
            # Find candidate to evict
            candidates = [v for v in fast_mem if v not in exclude]
            if not candidates:
                break
            # Sort by future_uses (ascending), then by load_order (ascending, evict oldest)
            candidates.sort(key=lambda v: (future_uses[v], load_order.get(v, 0)))
            v = candidates[0]
            store_vertex(v)
    
    # Main loop: process vertices in topological order
    for v in vertices:
        if time.time() - start_time > DEADLINE:
            break
        
        # If this is an input, it starts in slow memory (blue)
        # If it's not an input, its parents have been processed
        
        # Ensure all parents are in fast memory
        for p in parents.get(v, []):
            if p not in fast_mem:
                load_vertex(p)
                load_order[p] = load_counter[0]
                load_counter[0] += 1
                evict_if_needed(exclude=set(parents.get(v, [])))
        
        # Now compute v
        compute_vertex(v)
        load_order[v] = load_counter[0]
        load_counter[0] += 1
        
        # If v is not an output and has no future uses, we can delete it
        # But be careful: deleting is only valid if we never need it again
        if v not in output_set and future_uses[v] == 0 and fast_memory_budget > 0:
            # Check if we can delete
            delete_vertex(v)
        else:
            # May need to evict to make room
            # But only evict if we're over budget
            evict_if_needed(exclude=output_set)
    
    # Ensure all outputs are in fast memory at the end
    for o in outputs:
        if o not in fast_mem:
            load_vertex(o)
            # Try to make room
            evict_if_needed(exclude=output_set)
    
    return operations
# EVOLVE-BLOCK-END
