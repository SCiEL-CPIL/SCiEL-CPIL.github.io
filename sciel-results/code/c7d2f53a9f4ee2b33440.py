def pebble(dag, fast_memory_budget):
    vertices = dag['vertices']
    parents = dag['parents']
    inputs = set(dag['inputs'])
    outputs = set(dag['outputs'])
    
    children = {v: [] for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
    
    # Topological order
    in_degree = {v: len(parents.get(v, [])) for v in vertices}
    from collections import deque
    queue = deque([v for v in vertices if in_degree[v] == 0])
    topo_order = []
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    if len(topo_order) != len(vertices):
        topo_order = list(vertices)
    
    fast_mem = set()
    computed = set()
    operations = []
    remaining_uses = {v: len(children[v]) for v in vertices}
    
    for v in topo_order:
        # Ensure parents are in fast memory
        for p in parents.get(v, []):
            if p not in fast_mem:
                if p in computed or p in inputs:
                    operations.append(['load', p])
                    fast_mem.add(p)
        
        # Compute v
        if v not in inputs:
            operations.append(['compute', v])
        fast_mem.add(v)
        computed.add(v)
        
        for p in parents.get(v, []):
            remaining_uses[p] -= 1
        
        # Evict if over budget
        while len(fast_mem) > fast_memory_budget:
            # Evict value with fewest remaining uses, prefer non-outputs
            candidates = [(remaining_uses[x], 0 if x not in outputs else 1, x) for x in fast_mem]
            candidates.sort()
            if not candidates:
                break
            _, _, evict = candidates[0]
            if evict in outputs:
                # If all candidates are outputs, break
                if all(c[1] == 1 for c in candidates):
                    break
            operations.append(['store', evict])
            fast_mem.remove(evict)
    
    # Ensure outputs are in fast memory
    for v in outputs:
        if v not in fast_mem:
            operations.append(['load', v])
            fast_mem.add(v)
    
    if len(operations) > 20000:
        operations = operations[:20000]
    
    return operations
