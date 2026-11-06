def pebble(dag, fast_memory_budget):
    """
    Communication-minimizing DAG pebbling using topological layering and greedy eviction.
    """
    # Extract DAG components
    if isinstance(dag, dict):
        if 'vertices' in dag:
            vertices = dag['vertices']
        elif 'n' in dag:
            vertices = list(range(dag['n']))
        else:
            vertices = list(dag.keys())
            
        if 'parents' in dag:
            parents_map = dag['parents']
        elif 'edges' in dag:
            parents_map = {v: [] for v in vertices}
            for u, v in dag['edges']:
                parents_map[v].append(u)
        else:
            parents_map = {}
            
        if 'outputs' in dag:
            outputs = set(dag['outputs'])
        else:
            # Calculate outputs as sinks
            children_map = {v: [] for v in vertices}
            for v in vertices:
                if v in parents_map:
                    for p in parents_map[v]:
                        if p in children_map:
                            children_map[p].append(v)
            outputs = set(v for v in vertices if not children_map[v])
    else:
        # Assume class with attributes
        vertices = getattr(dag, 'vertices', list(range(getattr(dag, 'n', 0))))
        parents_map = getattr(dag, 'parents', {})
        outputs = set(getattr(dag, 'outputs', []))
        if not outputs:
            children_map = {v: [] for v in vertices}
            for v in vertices:
                if v in parents_map:
                    for p in parents_map[v]:
                        if p in children_map:
                            children_map[p].append(v)
            outputs = set(v for v in vertices if not children_map[v])

    # Build children map if not already done
    if 'children_map' not in locals():
        children_map = {v: [] for v in vertices}
        for v in vertices:
            if v in parents_map:
                for p in parents_map[v]:
                    if p in children_map:
                        children_map[p].append(v)

    # remaining_children: number of children of v that have not been processed yet
    remaining_children = {v: len(children_map[v]) for v in vertices}
    
    fast_memory = set()
    operations = []
    
    # Process vertices in topological order
    for v in vertices:
        # Ensure all parents are in fast_memory
        parents_of_v = parents_map.get(v, [])
        for p in parents_of_v:
            if p not in fast_memory:
                # Need to load p
                if len(fast_memory) >= fast_memory_budget:
                    # Evict a vertex
                    # First try to evict non-parents of v
                    best_candidate = None
                    min_count = float('inf')
                    for e in fast_memory:
                        if e in parents_of_v:
                            continue
                        count = remaining_children.get(e, 0)
                        if count < min_count:
                            min_count = count
                            best_candidate = e
                    
                    # If no non-parents to evict, evict any vertex (including parents)
                    if best_candidate is None:
                        min_count = float('inf')
                        for e in fast_memory:
                            count = remaining_children.get(e, 0)
                            if count < min_count:
                                min_count = count
                                best_candidate = e

                    if best_candidate is not None:
                        operations.append(['store', best_candidate])
                        fast_memory.remove(best_candidate)
                
                operations.append(['load', p])
                fast_memory.add(p)
        
        # Compute v
        operations.append(['compute', v])
        fast_memory.add(v)
        
        # Update remaining children counts for parents
        for p in parents_of_v:
            remaining_children[p] -= 1
        
        # If v is an output, store it immediately to free space
        if v in outputs:
            if v in fast_memory:
                operations.append(['store', v])
                fast_memory.remove(v)
        
        # Safety check for operation limit
        if len(operations) > 20000:
            break

    return operations
