def pebble(dag, fast_memory_budget):
    # dag structure: need to figure out what it looks like
    # Assume dag has:
    # - vertices (list or dict)
    # - parents: dict mapping vertex -> list of parent vertices
    # - outputs: list of output vertices
    # - inputs: vertices with no parents
    
    # Let me assume a common structure
    # dag = {
    #   'vertices': list of vertex names/ids,
    #   'parents': {vertex: [parent1, parent2, ...]},
    #   'outputs': [output_vertex1, ...]
    # }
    
    # Or maybe dag is a dict of vertex -> list of parents
    # Let me be flexible
    
    operations = []
    
    # Determine structure
    if isinstance(dag, dict):
        if 'parents' in dag:
            parents_map = dag['parents']
            all_vertices = set(parents_map.keys())
            for v, ps in parents_map.items():
                for p in ps:
                    all_vertices.add(p)
            outputs = dag.get('outputs', [])
        else:
            # Maybe dag is directly {vertex: [parents]}
            parents_map = dag
            all_vertices = set(dag.keys())
            for v, ps in dag.items():
                for p in ps:
                    all_vertices.add(p)
            outputs = []
    else:
        # Unknown structure, fallback
        raise ValueError("Unknown dag structure")
    
    # Topological sort
    from collections import defaultdict, deque
    
    # Build in-degree and adjacency
    children_map = defaultdict(list)
    in_degree = defaultdict(int)
    for v in all_vertices:
        if v not in in_degree:
            in_degree[v] = 0
    for v, ps in parents_map.items():
        for p in ps:
            children_map[p].append(v)
            in_degree[v] += 1
    
    # Kahn's algorithm
    queue = deque([v for v in all_vertices if in_degree[v] == 0])
    topo_order = []
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children_map[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    # State: blue = in fast memory, red = in slow memory
    # Initially, inputs are blue
    state = {}  # vertex -> 'blue' or 'red'
    for v in all_vertices:
        if v not in parents_map or len(parents_map.get(v, [])) == 0:
            state[v] = 'blue'  # inputs start blue
        else:
            state[v] = 'red'
    
    # Track how many more times each vertex will be needed as a parent
    future_needs = defaultdict(int)
    for v, ps in parents_map.items():
        for p in ps:
            future_needs[p] += 1
    
    # Fast memory tracking
    fast_mem = set()
    for v in all_vertices:
        if state[v] == 'blue':
            fast_mem.add(v)
    
    # Process in topological order
    for v in topo_order:
        # Ensure all parents are blue
        ps = parents_map.get(v, [])
        for p in ps:
            if state[p] != 'blue':
                # Need to load p
                operations.append(['load', p])
                state[p] = 'blue'
                fast_mem.add(p)
        
        # Compute v
        operations.append(['compute', v])
        state[v] = 'blue'
        fast_mem.add(v)
        
        # Update future_needs for v's parents (they've been used once more)
        for p in ps:
            future_needs[p] -= 1
        
        # If v is not an output and fast memory is full, we may need to free space
        # But we can only free space by storing or deleting blue values
        # We should store values that won't be needed again
        
        # Check if we're at capacity
        while len(fast_mem) > fast_memory_budget:
            # Find a blue value to store
            # Prefer values with future_needs == 0 (won't be needed again)
            candidates = [x for x in fast_mem if x != v and future_needs.get(x, 0) == 0]
            if not candidates:
                # All have future needs, pick one with least future needs
                candidates = [x for x in fast_mem if x != v]
                candidates.sort(key=lambda x: future_needs.get(x, 0))
            
            if candidates:
                to_store = candidates[0]
                operations.append(['store', to_store])
                state[to_store] = 'red'
                fast_mem.remove(to_store)
            else:
                # Can't free space, this shouldn't happen if budget is sufficient
                break
    
    # Ensure all outputs are blue at the end
    for out in outputs:
        if state.get(out, 'red') != 'blue':
            operations.append(['load', out])
            state[out] = 'blue'
            fast_mem.add(out)
    
    return operations
