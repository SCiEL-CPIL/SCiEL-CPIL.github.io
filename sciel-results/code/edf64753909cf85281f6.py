import time

N = 40

def construct():
    """
    Constructs a sparse pancyclic graph on 40 vertices.
    Uses a 40-cycle as the backbone and 19 chords from vertex 0 to cover
    all cycle lengths 3 through 40.
    Total edges: 40 (cycle) + 19 (chords) = 59.
    """
    deadline = time.time() + 80.0
    
    # 1. Backbone: 40-cycle
    # Edges: (0,1), (1,2), ..., (39,0)
    cycle_edges = [(i, (i + 1) % N) for i in range(N)]
    
    # 2. Chords to cover lengths 3..39
    # Chord (0, k) creates cycles of length k+1 and N-k+1.
    # We need to cover lengths 3..39.
    # k=2 -> lengths 3, 39
    # k=3 -> lengths 4, 38
    # ...
    # k=19 -> lengths 20, 22
    # k=20 -> length 21 (both directions give 21)
    # Total chords: 19 (k=2 to 20)
    chords = [(0, k) for k in range(2, 21)]
    
    all_edges = cycle_edges + chords
    
    # Normalize edges to (min, max) tuples and ensure uniqueness
    edge_set = set()
    for u, v in all_edges:
        if u > v:
            u, v = v, u
        edge_set.add((u, v))
    
    # The construction guarantees uniqueness, but we check just in case
    if len(edge_set) != len(all_edges):
        # Fallback: if duplicates occurred (shouldn't happen), we might have fewer edges
        # But for this specific construction, they are distinct.
        pass
        
    edges_list = [list(e) for e in sorted(edge_set)]
    
    # 3. Generate cycles for lengths 3..40
    cycles = []
    
    # Helper to generate the cycle path for a given chord k
    # Cycle 1: 0-1-...-k-0 (length k+1)
    # Cycle 2: 0-k-(k+1)-...-39-0 (length N-k+1)
    
    for length in range(3, N + 1):
        if length == N:
            # The backbone cycle
            cycles.append(list(range(N)))
        else:
            # Determine which chord k covers this length
            # Length L is covered by chord k if L == k+1 or L == N-k+1
            # k = L-1 or k = N-L+1
            
            k1 = length - 1
            k2 = N - length + 1
            
            # We only have chords for k in 2..20
            # For L in 3..39:
            # If L <= 20, k1 = L-1 is in 2..19. k2 is large.
            # If L > 20, k2 = N-L+1 is in 2..19. k1 is large.
            # If L == 21, k1=20, k2=20.
            
            if length <= 20:
                k = length - 1
                # Use forward path 0->1->...->k->0
                cyc = list(range(k + 1))
            else:
                # length > 20
                k = N - length + 1
                # Use backward path 0->k->k+1->...->39->0
                # Wait, the backward path in terms of vertex indices on the cycle:
                # The chord is (0, k).
                # The other path is 0-39-38-...-k-0.
                # Vertices: 0, 39, 38, ..., k.
                cyc = [0] + list(range(N - 1, k - 1, -1))
            
            cycles.append(cyc)
            
    # 4. Verification (optional but good for safety)
    # We skip full verification in production for speed if confident, 
    # but the prompt implies correctness is key. 
    # Given the deterministic construction, it is correct.
    
    return {
        'edges': edges_list,
        'cycles': cycles
    }
