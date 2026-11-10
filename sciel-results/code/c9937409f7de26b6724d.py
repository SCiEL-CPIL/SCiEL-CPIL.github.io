"""Sparse pancyclic graphs on 40 vertices.
Strategy: Hamiltonian cycle (40 edges) + 19 chords (0,k) for k=2..20.
Total 59 edges. Each chord (0,k) with k in 2..20 creates cycles of length k+1 and 41-k.
This covers all lengths 3..40 exactly once each.
Value = 59 - 40 = 19.
"""
N = 40

def construct():
    # Base Hamiltonian cycle: 0-1-2-...-39-0
    edges = set()
    for i in range(N):
        edges.add((min(i, (i+1) % N), max(i, (i+1) % N)))
    
    # Add chords (0, k) for k = 2, 3, ..., 20
    for k in range(2, 21):
        edges.add((0, k))
    
    edge_list = sorted(edges)
    
    # Build cycles for each length 3..40
    cycles = []
    for L in range(3, N + 1):
        if L == 40:
            # Full Hamiltonian cycle
            cycles.append(list(range(N)))
        else:
            # We need a cycle of length L.
            # Using chord (0, k) where k+1 = L, so k = L-1.
            # But k must be in 2..20. For L in 3..21, k = L-1 is in 2..20.
            # For L in 22..39, we use the "other side": chord (0, k) where 41-k = L, so k = 41-L.
            # k = 41-L is in 2..19 for L in 22..39.
            
            if L <= 21:
                k = L - 1
                # Cycle: 0, 1, 2, ..., k-1, k, 0  -- wait, that's k+1 vertices? No.
                # Path from 0 to k along the Hamiltonian cycle (shorter arc): 0,1,...,k has k+1 vertices.
                # Plus edge (k,0) closes the cycle. So cycle is [0, 1, 2, ..., k].
                # Length = k+1 = L. Correct.
                cycles.append(list(range(k + 1)))
            else:
                # L in 22..39
                k = 41 - L
                # Cycle: 0, 39, 38, ..., k, 0
                # Path from 0 to k along the longer arc of Hamiltonian cycle:
                # 0, 39, 38, ..., k. Number of vertices: 40 - k + 1 = 41 - k = L.
                # Wait: 0, 39, 38, ..., k. That's 40 - k vertices in the path 0->39->...->k?
                # Vertices: 0, 39, 38, ..., k+1, k. Count: 1 + (39 - k + 1) = 41 - k.
                # Yes, 41 - k = L vertices.
                # The cycle is [0, 39, 38, ..., k].
                path = list(range(0, k, -1))  # 0, -1, -2... no, that's wrong
                # Let me think again. We go from 0 to 39 to 38 to ... to k.
                # So the list is [0, 39, 38, ..., k].
                cyc = [0] + list(range(39, k - 1, -1))
                cycles.append(cyc)
    
    return {'edges': edge_list, 'cycles': cycles}
