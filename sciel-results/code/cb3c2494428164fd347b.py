def pebble(dag, fast_memory_budget):
    vertices = dag['vertices']
    parents = dag['parents']
    outputs = set(dag['outputs'])
    
    # Build child map
    children = {v: [] for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            if p in children:
                children[p].append(v)
    
    # Vertex index for topological ordering
    vertex_index = {v: i for i, v in enumerate(vertices)}
    
    # For each vertex, compute future use metrics
    future_uses = {}
    last_use_idx = {}
    for v in vertices:
        if children[v]:
            future_uses[v] = len(children[v])
            last_use_idx[v] = max(vertex_index[c] for c in children[v])
        else:
            future_uses[v] = 0
            last_use_idx[v] = vertex_index[v]
    
    # State: which vertices are currently red (in fast memory)
    red = set()
    operations = []
    
    for v in vertices:
        # Ensure all parents are red
        for p in parents.get(v, []):
            if p not in red:
                operations.append(['load', p])
                red.add(p)
        
        # Compute v (result is red)
        operations.append(['compute', v])
        red.add(v)
        
        # Enforce budget: if |red| > budget, evict
        while len(red) > fast_memory_budget:
            # Find least useful red vertex to evict
            best_evict = min(red, key=lambda x: (future_uses[x], last_use_idx[x]))
            operations.append(['delete', best_evict])
            red.remove(best_evict)
    
    # Store all outputs that are still red
    for v in outputs:
        if v in red:
            operations.append(['store', v])
            red.remove(v)
    
    return operations
