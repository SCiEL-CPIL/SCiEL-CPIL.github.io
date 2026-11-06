def pebble(dag, fast_memory_budget):
    """
    Pebble a DAG to minimize communication (transfers between slow and fast memory).
    
    Returns a list of [operation, vertex] where operation is one of:
      'load'    - load vertex from slow (blue) to fast (red)
      'store'   - store vertex from fast (red) to slow (blue)
      'compute' - compute vertex from its parents (must be red)
      'delete'  - delete vertex from fast memory (red -> gone)
    
    Constraints:
      - At most fast_memory_budget red vertices at any time
      - Inputs start blue
      - All outputs must finish blue
      - At most 20000 operations
    """
    import time
    from collections import defaultdict
    
    start_time = time.time()
    DEADLINE = 80.0  # Safety margin before 90s limit
    
    # Parse the DAG structure - handle multiple possible formats
    if hasattr(dag, 'parents'):
        parents = {k: list(v) for k, v in dag.parents.items()}
        outputs = list(dag.outputs) if hasattr(dag, 'outputs') else []
        inputs = list(dag.inputs) if hasattr(dag, 'inputs') else []
        vertices = list(dag.vertices) if hasattr(dag, 'vertices') else list(parents.keys())
    elif isinstance(dag, dict):
        parents = {k: list(v) for k, v in dag.get('parents', {}).items()}
        outputs = list(dag.get('outputs', []))
        inputs = list(dag.get('inputs', []))
        vertices = list(dag.get('vertices', list(parents.keys())))
    elif isinstance(dag, (list, tuple)):
        parents = {}
        vertices = []
        for item in dag:
            if isinstance(item, (list, tuple)):
                v, ps = item[0], item[1]
                parents[v] = list(ps)
                vertices.append(v)
            else:
                vertices.append(item)
                parents[item] = []
        outputs = []
        inputs = [v for v in vertices if len(parents.get(v, [])) == 0]
    else:
        # Fallback
        parents = {}
        vertices = []
        outputs = []
        inputs = []
    
    # Ensure all vertices are in the parents dict
    for v in vertices:
        if v not in parents:
            parents[v] = []
    
    # If no outputs specified, assume all vertices with no children are outputs
    if not outputs:
        child_set = set()
        for v in parents:
            for p in parents[v]:
                child_set.add(p)
        outputs = [v for v in vertices if v not in child_set]
    
    # If no inputs specified, assume vertices with no parents
    if not inputs:
        inputs = [v for v in vertices if len(parents.get(v, [])) == 0]
    
    # Build adjacency and compute topological order
    children = defaultdict(list)
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
    
    # Compute in-degrees for topological sort
    in_degree = {v: len(parents.get(v, [])) for v in vertices}
    topo_order = []
    queue = [v for v in vertices if in_degree[v] == 0]
    queue.sort()  # Deterministic ordering
    from collections import deque
    queue = deque(queue)
    in_degree_copy = dict(in_degree)
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for c in children.get(u, []):
            in_degree_copy[c] -= 1
            if in_degree_copy[c] == 0:
                queue.append(c)
    
    # If topological sort didn't include all vertices (cycle?), fall back to original order
    if len(topo_order) != len(vertices):
        topo_order = list(vertices)
    
    # Compute out-degree (number of children)
    out_degree = {v: len(children.get(v, [])) for v in vertices}
    
    # State tracking
    state = {}  # 'blue', 'red', 'gone'
    for v in vertices:
        if v in inputs:
            state[v] = 'blue'
        else:
            state[v] = 'gone'
    
    fast_mem = set()  # Vertices currently in fast memory (red)
    operations = []
    
    def do_load(v):
        nonlocal fast_mem
        if state[v] == 'blue':
            operations.append(['load', v])
            state[v] = 'red'
            fast_mem.add(v)
        elif state[v] == 'red':
            pass  # Already in fast memory
        else:
            # Should not happen in valid topological processing
            pass
    
    def do_store(v):
        nonlocal fast_mem
        if state[v] == 'red':
            operations.append(['store', v])
            state[v] = 'blue'
            fast_mem.discard(v)
        elif state[v] == 'blue':
            pass  # Already in slow memory
        else:
            pass
    
    def do_delete(v):
        nonlocal fast_mem
        if state[v] == 'red':
            operations.append(['delete', v])
            state[v] = 'gone'
            fast_mem.discard(v)
        elif state[v] == 'blue':
            pass
        else:
            pass
    
    def do_compute(v):
        nonlocal fast_mem
        # All parents must be red
        for p in parents.get(v, []):
            if state[p] != 'red':
                do_load(p)
        operations.append(['compute', v])
        state[v] = 'red'
        fast_mem.add(v)
    
    # Check time
    def time_left():
        return (start_time + DEADLINE) - time.time()
    
    # Check operation budget
    def ops_left():
        return 20000 - len(operations)
    
    # Compute remaining uses for each vertex (how many more times it's needed as a parent)
    remaining_uses = dict(out_degree)
    
    # Process in topological order with memory management
    # Strategy: 
    # - Load parents as needed
    # - Compute vertex
    # - Immediately store parents that are no longer needed
    # - Store current vertex if it's no longer needed and we're at capacity
    
    for v in topo_order:
        if time_left() < 0:
            break
        if ops_left() <= 0:
            break
        
        # Load all parents that are blue
        for p in parents.get(v, []):
            if state[p] == 'blue':
                # Need to make room if at capacity
                if len(fast_mem) >= fast_memory_budget:
                    # Try to store a vertex that's no longer needed and not a parent of v
                    freed = False
                    for u in list(fast_mem):
                        if u != p and u not in parents.get(v, []) and remaining_uses.get(u, 0) == 0:
                            do_store(u)
                            freed = True
                            break
                    if not freed:
                        # Store any vertex that's not a parent of v
                        for u in list(fast_mem):
                            if u not in parents.get(v, []):
                                do_store(u)
                                freed = True
                                break
                    if not freed and fast_memory_budget > 0:
                        # Emergency: store any vertex
                        for u in list(fast_mem):
                            do_store(u)
                            break
                do_load(p)
            elif state[p] == 'gone':
                # This should not happen in valid topological order
                # If it does, we can't recover easily
                pass
        
        # Compute v
        do_compute(v)
        
        # Update remaining uses for parents
        for p in parents.get(v, []):
            if remaining_uses.get(p, 0) > 0:
                remaining_uses[p] -= 1
        
        # Store parents that are no longer needed (to free up space)
        for p in parents.get(v, []):
            if remaining_uses.get(p, 0) == 0 and state[p] == 'red':
                do_store(p)
        
        # Store v if it's no longer needed and we're at capacity
        if remaining_uses.get(v, 0) == 0 and v not in outputs and len(fast_mem) >= fast_memory_budget:
            do_store(v)
        # Also store v if it's an output but we'll store all outputs at the end
        # For now, keep outputs red
    
    # Ensure all outputs are blue at the end
    for v in outputs:
        if state[v] == 'red':
            do_store(v)
    
    # Clean up any remaining red values
    for v in list(fast_mem):
        if state[v] == 'red':
            do_store(v)
    
    return operations
