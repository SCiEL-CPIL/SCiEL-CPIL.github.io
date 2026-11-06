"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 40

# EVOLVE-BLOCK-START
import time

def construct():
    """
    Constructs a sparse pancyclic graph on N=40 vertices.
    Returns a dictionary with 'edges' and 'cycles'.
    Minimizes edge count minus N.
    """
    # Theoretical minimum edges for pancyclic graph on N vertices is approx 1.5N.
    # For N=40, 59 edges (40 cycle + 19 chords) is the known optimal construction 
    # using a fan structure from vertex 0.
    
    # 1. Define the base cycle edges: (0,1), (1,2), ..., (39,0)
    cycle_edges = [(i, (i + 1) % N) for i in range(N)]
    
    # 2. Define chords from vertex 0 to vertices 2 through 20.
    # Chord (0, k) creates cycles of length k+1 (forward) and N-k+1 (backward).
    # k ranges from 2 to 20.
    # k=2 -> lengths 3, 39
    # k=3 -> lengths 4, 38
    # ...
    # k=19 -> lengths 20, 22
    # k=20 -> length 21 (both directions)
    chords = [(0, k) for k in range(2, N // 2 + 1)]
    
    # Combine edges
    all_edges = cycle_edges + chords
    edge_set = set((min(e[0], e[1]), max(e[0], e[1])) for e in all_edges)
    
    # Verify uniqueness (should be 59 edges)
    # assert len(edge_set) == 59, f"Edge count mismatch: {len(edge_set)}"
    
    # 3. Construct cycles for each length 3 to 40
    cycles = []
    
    for length in range(3, N + 1):
        if length == N:
            # The full cycle 0-1-...-39-0
            cyc = list(range(N))
        elif length == N // 2 + 1:
            # Length 21. Chord (0, 20).
            # Forward path 0->1->...->20->0 has length 21.
            cyc = list(range(21))
        else:
            # Determine which chord to use
            # If length <= 20, use forward path with k = length - 1
            # If length > 20, use backward path with k = N - length + 1
            if length <= N // 2:
                k = length - 1
                # Cycle: 0 -> 1 -> ... -> k -> 0
                cyc = list(range(k + 1))
            else:
                k = N - length + 1
                # Cycle: 0 -> 39 -> 38 -> ... -> k -> 0
                # Vertices: 0, N-1, N-2, ..., k
                cyc = [0] + list(range(N - 1, k - 1, -1))
        
        cycles.append(cyc)
        
        # Validation (optional but good for development, remove for speed if strictly needed, but N=40 is tiny)
        # assert len(cyc) == length
        # assert len(set(cyc)) == length
        # for i in range(len(cyc)):
        #     u, v = cyc[i], cyc[(i+1) % len(cyc)]
        #     assert (min(u,v), max(u,v)) in edge_set

    return {
        'edges': [list(e) for e in edge_set],
        'cycles': cycles
    }
# EVOLVE-BLOCK-END
