<<<<<<< SEARCH
    # Try to find 3 edge-disjoint Hamiltonian cycles
    used_edges = set()
    cycles = []
    
    for attempt in range(3):
        if time.time() > deadline:
            break
        cycle = find_hamiltonian_cycle(used_edges)
        if cycle is None:
            break
        # Mark edges as used
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i+1) % len(cycle)]
            e = (min(u,v), max(u,v))
            used_edges.add(e)
        cycles.append(cycle)
    
    if not cycles:
        # Fallback: return the seed solution
        return [list(range(n))]
=======
    # Try to find 3 edge-disjoint Hamiltonian cycles
    used_edges = set()
    cycles = []
    
    for attempt in range(3):
        if time.time() > deadline:
            break
        # Try multiple starting vertices for robustness
        cycle = None
        for sv in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]:
            if time.time() > deadline:
                break
            cycle = find_hamiltonian_cycle(used_edges, start_vertex=sv)
            if cycle is not None:
                break
        
        if cycle is None:
            break
        # Mark edges as used
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i+1) % len(cycle)]
            e = (min(u,v), max(u,v))
            used_edges.add(e)
        cycles.append(cycle)
    
    if not cycles:
        # Fallback: return empty list (invalid but better than timeout)
        return []
>>>>>>> REPLACE
