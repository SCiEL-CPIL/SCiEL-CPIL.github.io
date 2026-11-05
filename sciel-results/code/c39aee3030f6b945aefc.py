def pebble(dag, fast_memory_budget):
    vertices = dag.get('vertices', [])
    parents = dag.get('parents', {})
    outputs = set(dag.get('outputs', []))
    inputs = set(dag.get('inputs', []))
    
    if not inputs:
        for v in vertices:
            if v not in parents or len(parents[v]) == 0:
                inputs.add(v)
    
    topo_order = list(vertices)
    
    state = {}
    for v in vertices:
        if v in inputs:
            state[v] = 'blue'
        else:
            state[v] = None
    
    red_set = set()
    
    operations = []
    
    # Build children map
    children = {v: [] for v in vertices}
    for v in vertices:
        if v in parents:
            for p in parents[v]:
                children[p].append(v)
    
    # Track remaining uses: for each vertex, how many more times it will be used as a parent
    remaining_uses = {}
    for v in vertices:
        remaining_uses[v] = len(children[v])
    
    for idx, v in enumerate(topo_order):
        # Ensure parents are red
        if v in parents:
            for p in parents[v]:
                if state[p] == 'blue':
                    operations.append(['load', p])
                    state[p] = 'red'
                    red_set.add(p)
        
        # Compute v if not already red
        if v not in inputs:
            if state[v] is None:
                operations.append(['compute', v])
                state[v] = 'red'
                red_set.add(v)
            elif state[v] == 'blue':
                operations.append(['load', v])
                state[v] = 'red'
                red_set.add(v)
        
        # After computing v, decrement remaining uses of parents
        if v in parents:
            for p in parents[v]:
                remaining_uses[p] -= 1
                # If parent is no longer needed and is red, delete it
                if remaining_uses[p] == 0 and p not in outputs and state[p] == 'red':
                    operations.append(['delete', p])
                    state[p] = None
                    red_set.discard(p)
        
        # Manage fast memory: if over budget, evict
        while len(red_set) > fast_memory_budget:
            # Find red vertex with farthest last use
            # Prefer to evict vertices that are no longer needed (but we already deleted those)
            # So all remaining red vertices are still needed
            candidates = list(red_set)
            if not candidates:
                break
            # Evict the one with the farthest next use
            # Use remaining_uses as a proxy: fewer remaining uses = can be evicted sooner?
            # No, we want to evict the one needed farthest in the future
            # Actually, evict the one with the LARGEST index of next use
            # Simplify: evict the one with the fewest remaining uses (least critical)
            victim = min(candidates, key=lambda x: remaining_uses.get(x, 0))
            
            operations.append(['store', victim])
            state[victim] = 'blue'
            red_set.discard(victim)
    
    # Ensure outputs are blue
    for v in outputs:
        if state[v] == 'red':
            operations.append(['store', v])
            state[v] = 'blue'
            red_set.discard(v)
    
    # Delete remaining red values
    for v in list(red_set):
        operations.append(['delete', v])
        state[v] = None
        red_set.discard(v)
    
    return operations
