def pebble(dag, fast_memory_budget):
    # Extract DAG structure
    if isinstance(dag, dict):
        n = dag.get('n', dag.get('num_vertices', len(dag.get('parents', []))))
        parents = dag.get('parents', dag.get('edges', []))
        inputs = set(dag.get('inputs', dag.get('input_vertices', [])))
        outputs = set(dag.get('outputs', dag.get('output_vertices', [])))
    else:
        n = getattr(dag, 'n', getattr(dag, 'num_vertices', len(getattr(dag, 'parents', []))))
        parents = getattr(dag, 'parents', getattr(dag, 'edges', []))
        inputs = set(getattr(dag, 'inputs', getattr(dag, 'input_vertices', [])))
        outputs = set(getattr(dag, 'outputs', getattr(dag, 'output_vertices', [])))
    
    children = [[] for _ in range(n)]
    for i in range(n):
        for p in parents[i]:
            children[p].append(i)
    
    last_use = [0] * n
    for i in range(n):
        for c in children[i]:
            last_use[i] = max(last_use[i], c)
    
    fast_memory = set()
    in_slow_computed = set()  # computed values that are in slow memory
    operations = []
    
    for v in range(n):
        # Load parents not in fast memory
        for p in parents[v]:
            if p not in fast_memory:
                operations.append(['load', p])
                fast_memory.add(p)
        
        # Compute v
        operations.append(['compute', v])
        fast_memory.add(v)
        
        # Evict if over budget
        while len(fast_memory) > fast_memory_budget:
            best_evict = None
            best_score = float('inf')
            
            for u in fast_memory:
                if u == v:
                    continue
                if last_use[u] < v:
                    score = 0
                else:
                    remaining_uses = sum(1 for c in children[u] if c > v)
                    score = remaining_uses * 10 + (last_use[u] - v)
                
                if score < best_score:
                    best_score = score
                    best_evict = u
            
            if best_evict is None:
                break
            
            if last_use[best_evict] < v:
                # Not needed again
                if best_evict in inputs:
                    # Input, already in slow, just delete
                    operations.append(['delete', best_evict])
                else:
                    # Computed, if in slow, just delete; else store then delete
                    if best_evict in in_slow_computed:
                        operations.append(['delete', best_evict])
                    else:
                        operations.append(['store', best_evict])
                        in_slow_computed.add(best_evict)
                        operations.append(['delete', best_evict])
                fast_memory.remove(best_evict)
            else:
                # Will be needed again
                if best_evict in inputs:
                    # Input, just delete (stays in slow)
                    operations.append(['delete', best_evict])
                else:
                    # Computed, store then delete
                    operations.append(['store', best_evict])
                    in_slow_computed.add(best_evict)
                    operations.append(['delete', best_evict])
                fast_memory.remove(best_evict)
        
        # If v is an output, ensure it's in slow memory
        if v in outputs:
            if v not in in_slow_computed:
                operations.append(['store', v])
                in_slow_computed.add(v)
            # Keep in fast for now, will delete at end if needed
            # Actually, outputs must finish blue, so they must be in slow
            # They can be in both fast and slow
            # We'll delete from fast at the end if memory is tight
    
    # Final cleanup: ensure all outputs are in slow (they are, we stored them)
    # Delete any remaining fast memory items that are not outputs? 
    # Actually, the problem says outputs must finish blue, but doesn't say other vertices must be deleted
    # So we can leave non-output vertices in fast memory
    # But outputs must be in slow, which they are
    
    return operations
