def pebble(dag, fast_memory_budget):
    import time
    start_time = time.time()
    deadline = start_time + 85
    
    n = len(dag)
    parents = dag
    
    has_children = [False] * n
    for i in range(n):
        for p in parents[i]:
            has_children[p] = True
    
    inputs = [i for i in range(n) if not parents[i]]
    outputs = [i for i in range(n) if not has_children[i]]
    
    layer = [0] * n
    for v in range(n):
        if parents[v]:
            layer[v] = max(layer[p] for p in parents[v]) + 1
    
    max_layer = max(layer) if n > 0 else 0
    
    layers = [[] for _ in range(max_layer + 1)]
    for v in range(n):
        layers[layer[v]].append(v)
    
    # Precompute which vertices are needed in which layers
    needed_in_layer = [set() for _ in range(max_layer + 1)]
    for l in range(1, max_layer + 1):
        for v in layers[l]:
            for p in parents[v]:
                needed_in_layer[layer[p]].add(v)  # p is needed for v
    
    operations = []
    fast_mem = set()
    
    def ensure_space(current_layer):
        while len(fast_mem) >= fast_memory_budget:
            # Find a vertex to delete: not an output, and not needed in current or future layers
            candidates = []
            for v in fast_mem:
                if v in outputs:
                    continue
                # Check if v is needed in current or future layers
                needed_future = False
                for l in range(current_layer, max_layer + 1):
                    if v in needed_in_layer[l]:
                        needed_future = True
                        break
                if not needed_future:
                    candidates.append(v)
            if not candidates:
                # Can't delete anything, break
                break
            # Delete the one from the earliest layer
            d = min(candidates, key=lambda v: layer[v])
            operations.append(['delete', d])
            fast_mem.remove(d)
    
    for l in range(max_layer + 1):
        for v in layers[l]:
            if l == 0:
                ensure_space(l)
                operations.append(['load', v])
                fast_mem.add(v)
            else:
                for p in parents[v]:
                    if p not in fast_mem:
                        ensure_space(l)
                        operations.append(['load', p])
                        fast_mem.add(p)
                ensure_space(l)
                operations.append(['compute', v])
                fast_mem.add(v)
    
    for v in outputs:
        if v in fast_mem:
            operations.append(['store', v])
            fast_mem.remove(v)
    
    return operations
