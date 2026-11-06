def pebble(dag, fast_memory_budget):
    import bisect
    from collections import deque
    
    # Parse dag structure
    parents = dag.get('parents', dag.get('edges', dag.get('preds', {})))
    outputs = set(dag.get('outputs', dag.get('sinks', dag.get('outs', []))))
    
    # If parents is a list of (parent, child) pairs, convert to dict
    if isinstance(parents, list):
        parents_dict = {}
        for edge in parents:
            if isinstance(edge, (list, tuple)) and len(edge) == 2:
                p, c = edge
                parents_dict.setdefault(c, []).append(p)
        parents = parents_dict
    
    # All vertices
    all_vertices = set(parents.keys())
    for p_list in parents.values():
        all_vertices.update(p_list)
    all_vertices.update(outputs)
    
    # Inputs: vertices with no parents
    inputs = set()
    for v in all_vertices:
        if v not in parents or len(parents[v]) == 0:
            inputs.add(v)
    
    # Topological order (Kahn's algorithm)
    in_degree = {v: len(parents.get(v, [])) for v in all_vertices}
    children = {v: [] for v in all_vertices}
    for v in all_vertices:
        for p in parents.get(v, []):
            children[p].append(v)
    
    queue = deque([v for v in all_vertices if in_degree[v] == 0])
    topo = []
    while queue:
        v = queue.popleft()
        topo.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    for v in all_vertices:
        if v not in topo:
            topo.append(v)
    
    # Build events
    events = []
    ref_positions = {v: [] for v in all_vertices}
    for v in topo:
        for p in parents.get(v, []):
            events.append(('ref', p, v))
            ref_positions[p].append(len(events) - 1)
        if v not in inputs:
            events.append(('compute', v, v))
    
    def next_ref_after(v, event_idx):
        pos = ref_positions[v]
        i = bisect.bisect_right(pos, event_idx)
        if i < len(pos):
            return pos[i]
        return float('inf')
    
    cache = set()
    operations = []
    transfers = 0
    
    def evict_excluding(event_idx, exclude):
        nonlocal transfers
        victim = None
        farthest = -1
        for c in cache:
            if c in exclude:
                continue
            nr = next_ref_after(c, event_idx)
            if nr > farthest:
                farthest = nr
                victim = c
        if victim is None:
            for c in cache:
                nr = next_ref_after(c, event_idx)
                if nr > farthest:
                    farthest = nr
                    victim = c
        if victim is not None:
            operations.append(['store', victim])
            transfers += 1
            cache.remove(victim)
    
    for event_idx, (etype, v, ctx) in enumerate(events):
        if etype == 'ref':
            if v not in cache:
                if len(cache) >= fast_memory_budget:
                    exclude = set(parents.get(ctx, []))
                    evict_excluding(event_idx, exclude)
                operations.append(['load', v])
                transfers += 1
                cache.add(v)
        elif etype == 'compute':
            if len(cache) >= fast_memory_budget:
                exclude = set(parents.get(ctx, []))
                evict_excluding(event_idx, exclude)
            operations.append(['compute', v])
            cache.add(v)
    
    # Ensure outputs are blue
    for v in outputs:
        if v in cache:
            operations.append(['store', v])
            transfers += 1
            cache.remove(v)
    
    if len(operations) > 20000:
        operations = operations[:20000]
    
    return operations
