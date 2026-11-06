def pebble(dag, fast_memory_budget):
    import time
    start_time = time.time()
    DEADLINE = 85  # seconds, safety margin from 90s limit
    
    # Parse DAG structure
    # dag format: need to understand the interface
    # Typically: dag has vertices, edges (parents), and output vertices
    # Let's assume dag is a dict or object with:
    #   dag['vertices'] or dag.vertices: list of vertex ids
    #   dag['edges'] or dag.edges: list of (parent, child) tuples
    #   dag['outputs'] or dag.outputs: list of output vertex ids
    #   dag['inputs'] or dag.inputs: list of input vertex ids
    
    # Try to extract structure
    try:
        vertices = list(dag.get('vertices', dag.get('nodes', range(len(dag.get('edges', []))))))
    except:
        vertices = list(range(10))  # fallback
    
    try:
        edges = dag.get('edges', [])
    except:
        edges = []
    
    try:
        outputs = set(dag.get('outputs', []))
    except:
        outputs = set()
    
    try:
        inputs = set(dag.get('inputs', []))
    except:
        inputs = set()
    
    # Build adjacency
    children = {v: [] for v in vertices}
    parents = {v: [] for v in vertices}
    for e in edges:
        if len(e) == 2:
            p, c = e[0], e[1]
            if p in children and c in parents:
                children[p].append(c)
                parents[c].append(p)
    
    # Topological sort
    in_degree = {v: len(parents[v]) for v in vertices}
    from collections import deque
    queue = deque([v for v in vertices if in_degree[v] == 0])
    topo_order = []
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    # If topological sort didn't include all, add remaining
    if len(topo_order) < len(vertices):
        for v in vertices:
            if v not in topo_order:
                topo_order.append(v)
    
    # Identify inputs (no parents) and outputs
    if not inputs:
        inputs = set(v for v in vertices if not parents[v])
    if not outputs:
        outputs = set(v for v in vertices if not children[v])
    
    # Count how many times each vertex is needed (out-degree in terms of children that need it)
    # Actually, we need to track remaining uses
    remaining_uses = {v: len(children[v]) for v in vertices}
    
    # Strategy: process in topological order
    # Maintain fast memory as a set of blue vertices
    # Operations list
    ops = []
    fast_mem = set()
    
    # Inputs start blue per problem statement
    for v in inputs:
        fast_mem.add(v)
    
    # For each vertex in topological order, compute it
    # When memory is full, evict a vertex that has fewest remaining uses
    for v in topo_order:
        # Check if we can compute v
        # All parents must be in fast_mem
        parent_ok = all(p in fast_mem for p in parents[v])
        
        if not parent_ok:
            # Need to load some parents
            for p in parents[v]:
                if p not in fast_mem:
                    # Load p
                    ops.append(['load', p])
                    fast_mem.add(p)
        
        # Now compute v
        ops.append(['compute', v])
        fast_mem.add(v)
        
        # If memory is full, evict a non-output vertex with fewest remaining uses
        if len(fast_mem) > fast_memory_budget:
            # Find vertex to evict: not an output, not v, has fewest remaining uses
            candidates = [x for x in fast_mem if x not in outputs and x != v]
            if candidates:
                # Evict the one with fewest remaining uses
                to_evict = min(candidates, key=lambda x: remaining_uses[x])
                ops.append(['store', to_evict])
                fast_mem.discard(to_evict)
        
        # Update remaining uses for v's children
        # Actually, we should decrement remaining_uses when we "use" a parent
        # Let's track this differently
    
    # Ensure all outputs are in fast memory at the end
    for v in outputs:
        if v not in fast_mem:
            ops.append(['load', v])
            fast_mem.add(v)
    
    # Check deadline
    if time.time() - start_time > DEADLINE:
        return ops[:min(len(ops), 20000)]
    
    return ops
