def pebble(dag, fast_memory_budget):
    """
    Communication-minimizing DAG pebbling.
    
    Args:
        dag: dict with keys 'vertices' (list in topo order), 'parents' (dict), 'outputs' (list)
        fast_memory_budget: int, number of fast memory slots
    
    Returns:
        List of [operation, vertex]
    """
    vertices = list(dag['vertices'])
    parents = dag['parents']
    outputs = set(dag['outputs'])
    
    # Build children map
    children = {v: [] for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
    
    # Identify inputs (no parents)
    inputs = set()
    for v in vertices:
        if not parents.get(v, []):
            inputs.add(v)
    
    # Compute fan-out
    fanout = {v: len(children[v]) for v in vertices}
    
    # Decide which vertices to keep in fast memory
    # Priority: outputs first, then by fanout
    def mem_priority(v):
        is_output = 1 if v in outputs else 0
        return (is_output, fanout[v])
    
    sorted_for_mem = sorted(vertices, key=mem_priority, reverse=True)
    keep_in_memory = set()
    for v in sorted_for_mem:
        if len(keep_in_memory) < fast_memory_budget:
            keep_in_memory.add(v)
    
    # Generate operations
    ops = []
    in_fast = set()
    computed = set()
    
    # Load all inputs
    for v in sorted(inputs, key=str):
        ops.append(['load', v])
        in_fast.add(v)
        computed.add(v)
    
    # Process in topological order
    for v in vertices:
        if v in inputs:
            continue
        
        # Ensure all parents are in fast memory
        for p in sorted(parents.get(v, []), key=str):
            if p not in in_fast:
                ops.append(['load', p])
                in_fast.add(p)
        
        # Compute v
        ops.append(['compute', v])
        computed.add(v)
        
        # Place v in fast memory or store to slow memory
        if v in keep_in_memory:
            in_fast.add(v)
        else:
            ops.append(['store', v])
        
        # Free memory: delete parents that are no longer needed
        for p in sorted(parents.get(v, []), key=str):
            # Check if p is still needed by any uncomputed child
            needed = False
            for c in children[p]:
                if c not in computed:
                    needed = True
                    break
            if not needed and p in in_fast and p not in outputs:
                ops.append(['delete', p])
                in_fast.remove(p)
    
    # Ensure all outputs are in fast memory
    for v in sorted(outputs, key=str):
        if v not in in_fast:
            ops.append(['load', v])
            in_fast.add(v)
    
    return ops
