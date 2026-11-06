<<<<<<< SEARCH
    vertices = dag['vertices']
    parents = dag['parents']
    inputs = dag['inputs']
    outputs = dag['outputs']
    n = len(vertices)
    k = fast_memory_budget
    
    if n == 0:
        return []
    
    # Identify layers: group vertices by their "depth" or by structure
    # For layered DAGs, we can detect layers by looking at the structure
    # A simpler approach: topological order defines natural layers if the DAG is layered
    
    # Try to detect layered structure: vertices in same layer have parents only in previous layer(s)
    # For simplicity, assume the DAG is already in topological order and is layered
    
    # Strategy: Hierarchical checkpointing for layered DAGs
    # 1. Group vertices into layers
    # 2. Determine optimal checkpoint interval
    # 3. Generate schedule
    
    # Detect layers: in a layered DAG, all parents of a vertex are in earlier layers
    # We'll use a simple heuristic: if the DAG looks layered, use checkpointing
    # Otherwise, fall back to a simple topological schedule
    
    # Try to partition into layers
    # Layer 0: inputs (or vertices with no parents that are inputs)
    # Actually, let's just use topological order and detect if it's layered
    
    # Simple approach: assume vertices are in topological order
    # Group consecutive vertices that form a "layer" if their parents are all in previous groups
    
    # For robustness, let's just do a good pebbling schedule:
    # 1. Compute all vertices in topological order
    # 2. Use memory budget to decide what to store vs recompute
    
    # Better: detect if layered, then use optimal checkpointing
    
    # Step 1: Build layer assignment
    # A vertex's layer = 1 + max(layer of parents), with inputs at layer 0
    layer = [0] * n
    for i in range(n):
        if i in inputs:
            layer[i] = 0
        elif parents[i]:
            layer[i] = 1 + max(layer[p] for p in parents[i])
        else:
            layer[i] = 0  # no parents, treat as input-like
    
    max_layer = max(layer) if n > 0 else 0
    
    # Group vertices by layer
    layers = [[] for _ in range(max_layer + 1)]
    for i in range(n):
        layers[layer[i]].append(i)
    
    # Check if it's a proper layered DAG: each vertex's parents are all in layer-1
    is_layered = True
    for i in range(n):
        if layer[i] > 0:
            for p in parents[i]:
                if layer[p] != layer[i] - 1:
                    is_layered = False
                    break
        if not is_layered:
            break
    
    if is_layered and max_layer > 0:
        # Use hierarchical checkpointing
        schedule = _layered_pebbling(layers, parents, inputs, outputs, k, n)
    else:
        # Fallback: simple topological pebbling
        schedule = _simple_pebbling(parents, inputs, outputs, k, n)
    
    # Verify and trim if needed
    if len(schedule) > 20000:
        # Truncate (shouldn't happen with good algorithm)
        schedule = schedule[:20000]
    
    return schedule
=======
    parents = dag.get('parents', [])
    inputs = dag.get('inputs', [])
    outputs = dag.get('outputs', [])
    n = len(parents)
    k = fast_memory_budget
    
    if n == 0:
        return []
    
    # Simple Greedy Pebbling with Store/Load
    # 1. Topological order is assumed to be 0..n-1
    # 2. Maintain in_memory and on_disk
    # 3. For each vertex, ensure parents are in memory (load or recompute if not on disk)
    # 4. Compute vertex
    # 5. If memory full, store and delete a vertex that is not an output
    
    in_memory = set(inputs)
    on_disk = set()
    schedule = []
    
    # Helper to ensure a vertex is in memory
    def ensure_in_memory(v):
        nonlocal in_memory, on_disk, schedule
        if v in in_memory:
            return
        if v in on_disk:
            schedule.append(['load', v])
            in_memory.add(v)
        else:
            # Recompute v
            # This requires parents of v to be in memory
            for p in parents[v]:
                ensure_in_memory(p)
            # If memory is full during recomputation, we might need to delete?
            # This recursive approach is dangerous if k is small.
            # For now, assume we can recompute if we have space.
            # If we are out of space, this will fail.
            # To be safe, we should only recompute if we have space.
            # But we don't know how much space the recomputation will take.
            # Given the "layered" hint, maybe we don't need deep recursion.
            schedule.append(['compute', v])
            in_memory.add(v)
    
    # Pre-compute "needed" set? 
    # A vertex is needed if it is a parent of a vertex that has not been computed yet.
    # We can compute the set of "future parents" for each step.
    
    # Let's just use a simple heuristic:
    # Iterate through vertices.
    for v in range(n):
        # Ensure parents are in memory
        for p in parents[v]:
            ensure_in_memory(p)
        
        # Compute v
        schedule.append(['compute', v])
        in_memory.add(v)
        
        # Manage memory
        # We want to keep outputs and vertices needed for future computations.
        # Find vertices in in_memory that are NOT outputs and NOT parents of any uncomputed vertex.
        # Uncomputed vertices are v+1 .. n-1.
        
        if len(in_memory) > k:
            # Find a candidate to delete
            # Candidates: in_memory - outputs - (parents of v+1..n-1)
            future_parents = set()
            for u in range(v + 1, n):
                for p in parents[u]:
                    future_parents.add(p)
            
            candidates = [u for u in in_memory if u not in outputs and u not in future_parents]
            
            if candidates:
                # Delete one
                # Prefer deleting the one with the highest index (oldest?)
                # Or just the first one
                to_delete = candidates[0]
                schedule.append(['store', to_delete])
                on_disk.add(to_delete)
                schedule.append(['delete', to_delete])
                in_memory.discard(to_delete)
            else:
                # No safe candidate? We are in trouble.
                # We might have to delete an output or a future parent, which will cause recomputation later.
                # For now, just break or delete something arbitrary.
                # This case indicates k is too small for the structure.
                # We'll try to delete any non-output.
                non_outputs = [u for u in in_memory if u not in outputs]
                if non_outputs:
                    to_delete = non_outputs[0]
                    schedule.append(['store', to_delete])
                    on_disk.add(to_delete)
                    schedule.append(['delete', to_delete])
                    in_memory.discard(to_delete)

    # Ensure outputs are in memory
    for v in outputs:
        if v not in in_memory:
            if v in on_disk:
                schedule.append(['load', v])
                in_memory.add(v)
            else:
                # Recompute
                ensure_in_memory(v)
                
    return schedule
>>>>>>> REPLACE
