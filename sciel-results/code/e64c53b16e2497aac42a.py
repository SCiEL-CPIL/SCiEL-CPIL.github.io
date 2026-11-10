import time

N = 40

def construct():
    """
    Constructs a sparse pancyclic graph on 40 vertices.
    Returns a dictionary with 'edges' and 'cycles'.
    
    Strategy:
    1. Create a Hamiltonian cycle on vertices 0..39. (40 edges)
    2. Add chords from vertex 0 to vertices 2..20. (19 edges)
    Total edges: 59. Value: 59 - 40 = 19.
    
    Cycle Coverage:
    - L=40: Hamiltonian cycle 0->1->...->39->0
    - L in 3..21: Use chord (0, L-1), cycle is 0->1->...->(L-1)->0
    - L in 22..39: Use chord (0, k) where k=41-L, cycle is 0->39->38->...->k->0
    All lengths 3..40 covered.
    """
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    # Build edges as a set for uniqueness
    edge_set = set()
    
    # 1. Hamiltonian Cycle: (i, (i+1)%N) for i in 0..39
    for i in range(N):
        u = i
        v = (i + 1) % N
        if u > v:
            u, v = v, u
        edge_set.add((u, v))
    
    # 2. Chords from 0 to k for k in 2..20
    for k in range(2, 21):
        u = 0
        v = k
        if u > v:
            u, v = v, u
        edge_set.add((u, v))
    
    edges_list = list(edge_set)
    
    # Build cycles for lengths 3 to 40
    cycles = []
    for L in range(3, N + 1):
        if L == N:
            # Hamiltonian cycle: 0, 1, 2, ..., 39
            cycle = list(range(N))
        elif 3 <= L <= 21:
            # Use chord (0, L-1). Cycle: 0, 1, 2, ..., L-1
            # Edges: (0,1), (1,2), ..., (L-2, L-1) from Hamiltonian cycle
            #         (L-1, 0) from chord
            cycle = list(range(L))
        else:
            # L in 22..39
            # Use chord (0, k) where k = 41 - L
            # Cycle: 0, 39, 38, ..., k
            # Edges: (0,39) from Hamiltonian cycle, (39,38), ..., (k+1,k) from Hamiltonian
            #         (k, 0) from chord
            k = 41 - L
            cycle = [0] + list(range(N - 1, k - 1, -1))
        
        cycles.append(cycle)
    
    # Validate: check all cycle edges exist in edge_set
    for cycle in cycles:
        n = len(cycle)
        for i in range(n):
            u = cycle[i]
            v = cycle[(i + 1) % n]
            if u > v:
                u, v = v, u
            if (u, v) not in edge_set:
                # This should never happen, but guard against it
                raise ValueError(f"Missing edge ({u},{v}) for cycle length {n}")
        # Check vertex distinctness
        if len(set(cycle)) != n:
            raise ValueError(f"Duplicate vertices in cycle of length {n}")
    
    return {'edges': edges_list, 'cycles': cycles}

if __name__ == "__main__":
    res = construct()
    print(f"Edges: {len(res['edges'])}, Cycles: {len(res['cycles'])}")
    # Verify all cycle lengths
    for i, cyc in enumerate(res['cycles']):
        assert len(cyc) == i + 3, f"Cycle {i} has length {len(cyc)}, expected {i+3}"
    print("All validations passed.")
