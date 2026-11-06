def pebble(dag, fast_memory_budget):
    import time
    from collections import deque
    
    start_time = time.time()
    DEADLINE = 85.0
    
    def parse_dag(dag):
        if not isinstance(dag, dict):
            return None
        vertices = None
        parents = None
        inputs = []
        outputs = []
        
        if 'vertices' in dag:
            vertices = list(dag['vertices'])
        elif 'num_vertices' in dag:
            vertices = list(range(dag['num_vertices']))
        
        if vertices is None:
            if 'parents' in dag:
                vertices = sorted(dag['parents'].keys())
            elif 'edges' in dag:
                all_v = set()
                for e in dag['edges']:
                    all_v.add(e[0])
                    all_v.add(e[1])
                vertices = sorted(all_v)
            else:
                return None
        
        if 'parents' in dag:
            parents = {v: list(dag['parents'].get(v, [])) for v in vertices}
        elif 'edges' in dag:
            parents = {v: [] for v in vertices}
            for e in dag['edges']:
                p, c = e[0], e[1]
                if c not in parents:
                    parents[c] = []
                if p not in parents[c]:
                    parents[c].append(p)
        else:
            return None
        
        inputs = list(dag.get('inputs', []))
        outputs = list(dag.get('outputs', []))
        
        return vertices, parents, inputs, outputs
    
    parsed = parse_dag(dag)
    if parsed is None:
        return []
    
    vertices, parents, inputs, outputs = parsed
    
    # Build children
    children = {v: [] for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            if p in children:
                children[p].append(v)
    
    # Topological order
    in_degree = {v: len(parents.get(v, [])) for v in vertices}
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
        # Cycle or error, fallback
        return []
    
    # Compute depth and last_needed
    depth = {v: 0 for v in vertices}
    for v in topo_order:
        for p in parents.get(v, []):
            depth[v] = max(depth[v], depth[p] + 1)
    
    last_needed = {v: depth[v] for v in vertices}
    for v in vertices:
        for c in children[v]:
            last_needed[v] = max(last_needed[v], depth[c])
    
    input_set = set(inputs)
    output_set = set(outputs)
    
    operations = []
    fast_mem = set()
    processed = set()
    
    for v in topo_order:
        if time.time() - start_time > DEADLINE:
            break
        
        if v in input_set:
            if v not in fast_mem:
                operations.append(['load', v])
                fast_mem.add(v)
        else:
            if v not in fast_mem:
                # Ensure parents in fast mem
                for p in parents.get(v, []):
                    if p not in fast_mem:
                        operations.append(['load', p])
                        fast_mem.add(p)
                operations.append(['compute', v])
                fast_mem.add(v)
        
        processed.add(v)
        
        # Enforce memory budget
        while len(fast_mem) > fast_memory_budget:
            # Find best candidate to delete
            best = None
            best_score = float('inf')
            for u in fast_mem:
                if u in output_set:
                    continue
                # Is u needed for any unprocessed vertex?
                needed = False
                for c in children[u]:
                    if c not in processed and c not in input_set:
                        needed = True
                        break
                if needed:
                    continue
                # Score: lower last_needed is better to delete
                score = last_needed[u]
                if score < best_score:
                    best_score = score
                    best = u
            
            if best is None:
                # Delete any non-output
                for u in fast_mem:
                    if u not in output_set:
                        operations.append(['delete', u])
                        fast_mem.discard(u)
                        break
                else:
                    break
            else:
                operations.append(['delete', best])
                fast_mem.discard(best)
    
    # Store outputs
    for v in output_set:
        if v in fast_mem:
            operations.append(['store', v])
            fast_mem.discard(v)
    
    return operations
