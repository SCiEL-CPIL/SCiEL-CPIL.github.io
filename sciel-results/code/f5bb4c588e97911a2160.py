"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 40

# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 85.0
    
    # Base: 40-cycle (edges i-(i+1) mod 40)
    cycle_edges = [(i, (i+1) % N) for i in range(N)]
    
    # We need cycles of every length 3..40
    # The 40-cycle gives us cycle length 40: [0,1,2,...,39]
    # Chord (0,k) creates cycles of length k+1 (path 0->1->...->k->0) 
    # and length N-k+1 (path 0->39->38->...->k->0)
    
    # Strategy: use chords from vertex 0 to cover lengths
    # Chord (0,k) gives cycles of length k+1 and N-k+1 = 41-k
    # So chord (0,k) covers lengths {k+1, 41-k}
    
    # We need lengths 3,4,...,39 (length 40 is the base cycle)
    # Length 3: need triangle. Chord (0,1) gives length 2 (not valid, it's just edge) 
    # Actually chord (0,k) with k>=2: path 0->1->...->k has k edges, plus chord = k+1 cycle
    # Path 0->39->38->...->k has N-k edges, plus chord = N-k+1 cycle
    
    # Chord (0,2): lengths 3 and 39
    # Chord (0,3): lengths 4 and 38
    # Chord (0,4): lengths 5 and 37
    # ...
    # Chord (0,k): lengths k+1 and 41-k
    
    # For k=2: {3, 39}
    # For k=3: {4, 38}
    # For k=4: {5, 37}
    # For k=5: {6, 36}
    # For k=6: {7, 35}
    # For k=7: {8, 34}
    # For k=8: {9, 33}
    # For k=9: {10, 32}
    # For k=10: {11, 31}
    # For k=11: {12, 30}
    # For k=12: {13, 29}
    # For k=13: {14, 28}
    # For k=14: {15, 27}
    # For k=15: {16, 26}
    # For k=16: {17, 25}
    # For k=17: {18, 24}
    # For k=18: {19, 23}
    # For k=19: {20, 22}
    # For k=20: {21, 21} - same length twice
    
    # So chords (0,2) through (0,19) give us lengths:
    # 3,39, 4,38, 5,37, 6,36, 7,35, 8,34, 9,33, 10,32, 11,31, 12,30, 
    # 13,29, 14,28, 15,27, 16,26, 17,25, 18,24, 19,23, 20,22
    # That's lengths 3-20 and 22-39. Missing: 21
    
    # Need length 21. Chord (0,20) gives length 21 (both paths give 21).
    # So add chord (0,20).
    
    # Total chords: (0,2) through (0,20) = 19 chords
    # Total edges: 40 (cycle) + 19 (chords) = 59
    
    chords = [(0, k) for k in range(2, 21)]
    
    all_edges = cycle_edges + chords
    edge_set = set((min(e), max(e)) for e in all_edges)
    
    # Verify all edges are distinct
    assert len(edge_set) == len(all_edges), f"Duplicate edges: {len(all_edges)} vs {len(edge_set)}"
    
    # Build cycles for each length 3..40
    cycles = []
    
    # Helper: get the path from 0 to k going forward (0,1,2,...,k)
    def forward_path(k):
        return list(range(k+1))
    
    # Helper: get the path from 0 to k going backward (0,39,38,...,k)
    def backward_path(k):
        return [0] + list(range(N-1, k-1, -1))
    
    for length in range(3, N+1):
        if length == N:
            # The full 40-cycle
            cycles.append(list(range(N)))
        elif length == 21:
            # Chord (0,20): forward path 0->1->...->20->0 gives 21 vertices
            # Or backward: 0->39->...->20->0 also gives 21 vertices
            # Use forward
            cycles.append(forward_path(20))
        else:
            # Find k such that k+1 == length or 41-k == length
            # k+1 == length => k = length-1
            # 41-k == length => k = 41-length
            
            if length <= 20:
                # Use forward path: k = length-1, cycle length = k+1 = length
                k = length - 1
                cyc = forward_path(k)
                # Verify: 0->1->...->k->0, has k+1 vertices = length
                assert len(cyc) == length, f"Length {length}: got {len(cyc)}"
                cycles.append(cyc)
            else:
                # Use backward path: k = 41-length, cycle length = 41-k = length
                k = 41 - length
                cyc = backward_path(k)
                # Verify: 0->39->...->k->0
                # Number of edges in path: from 0 to 39 is 1 edge, 39 to 38 is 1, ..., k+1 to k is 1
                # That's N-k edges in the path, plus the chord = N-k+1 = 41-k = length
                assert len(cyc) == length, f"Length {length}: got {len(cyc)}"
                cycles.append(cyc)
    
    # Verify all cycles are valid
    for i, cyc in enumerate(cycles):
        length = i + 3
        assert len(cyc) == length, f"Cycle {length}: length {len(cyc)} != {length}"
        # Check distinct vertices
        assert len(set(cyc)) == length, f"Cycle {length}: duplicate vertices"
        # Check all edges exist
        for j in range(len(cyc)):
            u, v = cyc[j], cyc[(j+1) % len(cyc)]
            e = (min(u,v), max(u,v))
            assert e in edge_set, f"Cycle {length}: edge {e} missing"
    
    return {'edges': [list(e) for e in edge_set], 'cycles': cycles}
# EVOLVE-BLOCK-END
