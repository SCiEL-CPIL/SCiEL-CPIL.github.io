def pebble(dag, fast_memory_budget):
    parents = dag.get('parents', {})
    outputs = set(dag.get('outputs', []))
    
    all_vertices = set(parents.keys())
    for v in parents:
        for p in parents[v]:
            all_vertices.add(p)
    
    inputs = [v for v in all_vertices if not parents.get(v, [])]
    
    children = {v: [] for v in all_vertices}
    for v in all_vertices:
        for p in parents.get(v, []):
            children[p].append(v)
    
    from collections import deque
    layer = {}
    queue = deque(inputs)
    for v in inputs:
        layer[v] = 0
    
    while queue:
        v = queue.popleft()
        for c in children[v]:
            if c not in layer:
                layer[c] = layer[v] + 1
                queue.append(c)
    
    for v in all_vertices:
        if v not in layer:
            layer[v] = max(layer.values()) + 1 if layer else 0
    
    max_layer = max(layer.values()) if layer else 0
    layers = [[] for _ in range(max_layer + 1)]
    for v in all_vertices:
        layers[layer[v]].append(v)
    
    fast_memory = set()
    operations = []
    
    for i in range(max_layer + 1):
        if i == 0:
            for v in layers[0]:
                operations.append(['load', v])
                fast_memory.add(v)
        else:
            for v in layers[i]:
                # Ensure parents in fast memory
                for p in parents.get(v, []):
                    if p not in fast_memory:
                        operations.append(['load', p])
                        fast_memory.add(p)
                
                # Compute v
                operations.append(['compute', v])
                fast_memory.add(v)
                
                # Delete no-longer-needed vertices from previous layers
                for p in list(fast_memory):
                    if layer.get(p, -1) < i:
                        # p is from a previous layer
                        # Check if all children are computed
                        if all(c in fast_memory for c in children[p]):
                            operations.append(['delete', p])
                            fast_memory.discard(p)
                
                # If still over budget, delete more aggressively
                while len(fast_memory) > fast_memory_budget:
                    # Find a vertex to delete
                    deleted = False
                    for candidate in list(fast_memory):
                        if candidate in outputs:
                            continue
                        if all(c in fast_memory for c in children[candidate]):
                            operations.append(['delete', candidate])
                            fast_memory.discard(candidate)
                            deleted = True
                            break
                    if not deleted:
                        # Delete any non-output
                        for candidate in list(fast_memory):
                            if candidate not in outputs:
                                operations.append(['delete', candidate])
                                fast_memory.discard(candidate)
                                deleted = True
                                break
                    if not deleted:
                        # Store an output
                        for candidate in list(fast_memory):
                            if candidate in outputs:
                                operations.append(['store', candidate])
                                fast_memory.discard(candidate)
                                deleted = True
                                break
    
    # Final cleanup
    for v in list(fast_memory):
        if v in outputs:
            operations.append(['store', v])
            fast_memory.discard(v)
        else:
            operations.append(['delete', v])
            fast_memory.discard(v)
    
    return operations
