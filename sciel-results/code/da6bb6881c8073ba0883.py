import time

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    N = 40
    edges_set = set()
    
    # Base: Hamiltonian cycle on vertices 0..39
    for i in range(N):
        edges_set.add((min(i, (i + 1) % N), max(i, (i + 1) % N)))
    
    # Add spokes from vertex 0 to cover cycle lengths
    # For length k, we need a cycle using k vertices
    # Strategy: use path 0->1->2->...->(k-1) plus edge (0,k-1) for forward cycles
    # And path 0->39->38->...->(40-k) plus edge (0,40-k) for backward cycles
    
    cycles = {}
    cycles[N] = list(range(N))  # The full Hamiltonian cycle
    
    # Forward cycles: 0,1,2,...,k-1 (needs edge (0,k-1))
    # Backward cycles: 0,39,38,...,40-k (needs edge (0,40-k))
    
    needed_edges = set()
    for k in range(3, N):
        # Forward: vertices 0,1,...,k-1
        fwd_edge = (0, k-1)
        fwd_edge = (min(fwd_edge), max(fwd_edge))
        
        # Backward: vertices 0,39,38,...,40-k
        bwd_last = 40 - k
        bwd_edge = (0, bwd_last)
        bwd_edge = (min(bwd_edge), max(bwd_edge))
        
        needed_edges.add(fwd_edge)
        needed_edges.add(bwd_edge)
    
    # Add all needed spoke edges
    for e in needed_edges:
        edges_set.add(e)
    
    # Build cycles
    for k in range(3, N + 1):
        if k == N:
            cycles[k] = list(range(N))
        else:
            # Try forward cycle: 0,1,2,...,k-1
            fwd_cyc = list(range(k))
            # Check if all edges exist
            valid_fwd = True
            for i in range(k):
                a = fwd_cyc[i]
                b = fwd_cyc[(i + 1) % k]
                if (min(a, b), max(a, b)) not in edges_set:
                    valid_fwd = False
                    break
            
            if valid_fwd:
                cycles[k] = fwd_cyc
            else:
                # Try backward cycle: 0,39,38,...,40-k
                bwd_cyc = [0] + list(range(39, 39 - k + 1, -1))
                cycles[k] = bwd_cyc
    
    # Validate
    valid = True
    for k in range(3, N + 1):
        if k not in cycles:
            valid = False
            break
        cyc = cycles[k]
        if len(cyc) != k:
            valid = False
            break
        if len(set(cyc)) != k:
            valid = False
            break
        for i in range(k):
            a = cyc[i]
            b = cyc[(i + 1) % k]
            if (min(a, b), max(a, b)) not in edges_set:
                valid = False
                break
        if not valid:
            break
    
    if valid:
        out = [cycles[k] for k in range(3, N + 1)]
        return {'edges': [list(e) for e in edges_set], 'cycles': out}
    else:
        # Fallback: fan graph
        fan_edges = set()
        for i in range(1, N):
            fan_edges.add((0, i))
        for i in range(1, N - 1):
            fan_edges.add((i, i + 1))
        cycles_fb = []
        for k in range(3, N + 1):
            cycles_fb.append(list(range(k)))
        return {'edges': [list(e) for e in fan_edges], 'cycles': cycles_fb}
