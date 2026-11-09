def pebble(dag, fast_memory_budget):
    def _parse_dag(dag):
        if isinstance(dag, (list, tuple)):
            if len(dag) > 0 and all(isinstance(x, (list, tuple)) for x in dag):
                n = len(dag)
                parents = [list(p) for p in dag]
                children = [[] for _ in range(n)]
                for i in range(n):
                    for p in parents[i]:
                        children[p].append(i)
                outputs = [i for i in range(n) if len(children[i]) == 0]
                return n, parents, outputs
            if len(dag) == 2:
                p, o = dag
                if isinstance(p, (list, tuple)) and isinstance(o, (list, tuple)):
                    n = len(p)
                    parents = [list(x) for x in p]
                    outputs = list(o)
                    return n, parents, outputs
        if isinstance(dag, dict):
            if 'parents' in dag:
                parents = dag['parents']
                n = len(parents)
                parents = [list(p) for p in parents]
                outputs = dag.get('outputs', [i for i in range(n) if all(i not in p for p in parents)])
                return n, parents, outputs
            if 'n' in dag and 'edges' in dag:
                n = dag['n']
                edges = dag['edges']
                parents = [[] for _ in range(n)]
                for (u, v) in edges:
                    parents[v].append(u)
                outputs = dag.get('outputs', [i for i in range(n) if all(i not in p for p in parents)])
                return n, parents, outputs
        if hasattr(dag, 'parents'):
            parents = dag.parents
            n = len(parents)
            parents = [list(p) for p in parents]
            outputs = getattr(dag, 'outputs', [i for i in range(n) if all(i not in p for p in parents)])
            return n, parents, outputs
        raise ValueError("Cannot parse DAG")
    
    n, parents, outputs = _parse_dag(dag)
    outputs_set = set(outputs)
    
    children = [[] for _ in range(n)]
    for i in range(n):
        for p in parents[i]:
            children[p].append(i)
    
    inputs_set = set(i for i in range(n) if len(parents[i]) == 0)
    
    remaining = {v: len(children[v]) for v in range(n)}
    
    operations = []
    fast_memory = set()
    
    for v in range(n):
        if v in inputs_set:
            continue
        
        # Ensure parents are in fast memory
        for p in parents[v]:
            if p not in fast_memory:
                while len(fast_memory) >= fast_memory_budget:
                    if not fast_memory:
                        break
                    candidates = list(fast_memory)
                    candidates.sort(key=lambda x: (remaining[x], x))
                    evict = candidates[0]
                    operations.append(['store', evict])
                    fast_memory.remove(evict)
                
                operations.append(['load', p])
                fast_memory.add(p)
        
        # Compute v
        operations.append(['compute', v])
        fast_memory.add(v)
        
        # Decrement remaining for parents
        for p in parents[v]:
            remaining[p] -= 1
        
        # Delete parents that are no longer needed
        for p in parents[v]:
            if p in fast_memory and remaining[p] == 0:
                operations.append(['delete', p])
                fast_memory.remove(p)
        
        # Check budget after adding v
        while len(fast_memory) > fast_memory_budget:
            if not fast_memory:
                break
            candidates = list(fast_memory)
            candidates.sort(key=lambda x: (remaining[x], x))
            evict = candidates[0]
            operations.append(['store', evict])
            fast_memory.remove(evict)
    
    # Store outputs that are in fast memory
    for v in sorted(outputs):
        if v in fast_memory:
            operations.append(['store', v])
            fast_memory.remove(v)
    
    return operations
