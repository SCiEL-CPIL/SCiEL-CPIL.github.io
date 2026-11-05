N = 40

def construct():
    # Original Ladder
    edges = []
    for i in range(19):
        edges.append((i, i + 1))
    for i in range(20, 39):
        edges.append((i, i + 1))
    for i in range(20):
        edges.append((i, i + 20))
    edges.append((0, 2))
    
    # Identify "chords" (non-Hamiltonian cycle edges)
    # Hamiltonian cycle: 0-1-...-19-39-...-20-0
    # Edges: (0,1)..(18,19), (19,39), (39,38)..(21,20), (20,0)
    ham_edges = set()
    for i in range(19):
        ham_edges.add(frozenset((i, i+1)))
    ham_edges.add(frozenset((19, 39)))
    for i in range(39, 20, -1):
        ham_edges.add(frozenset((i, i-1)))
    ham_edges.add(frozenset((20, 0)))
        
    chords = [e for e in edges if frozenset(e) not in ham_edges]
    
    # Try to remove chords
    # We need to verify if the graph remains pancyclic.
    # To do this fast, we can check if the *specific* cycles from the original construction still exist.
    # If they do, we are safe.
    # If they don't, we might still be safe if *other* cycles exist, but we can't easily check that.
    # So, we only remove a chord if the *original* cycles that depended on it are still valid?
    # No, if we remove a chord, the original cycles using it are broken.
    # We need to find *new* cycles.
    
    # Given the time constraint, I will just return the Ladder.
    # The Greedy Removal is too risky/slow in Python without a fast cycle finder.
    
    cycles = []
    for L in range(3, N + 1):
        if L == 3:
            cyc = [0, 1, 2]
        elif L % 2 == 0:
            k = L // 2
            part1 = list(range(0, k))
            part2 = list(range(20 + k - 1, 19, -1))
            cyc = part1 + part2
        else:
            k = (L - 1) // 2
            part1 = [0, 2] + list(range(3, k + 1))
            part2 = list(range(20 + k, 19, -1))
            cyc = part1 + part2
        cycles.append(cyc)
        
    return {'edges': edges, 'cycles': cycles}
