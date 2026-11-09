import time

N = 40

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 60.0
    
    # Base Hamiltonian cycle edges: 0-1-2-...-39-0
    edges = set()
    for i in range(N):
        edges.add((min(i, (i+1)%N), max(i, (i+1)%N)))
        
    # Chords from 0 to k for k in 2..20
    # This gives us cycles of length 3..21 (via 0-1-...-k-0) 
    # and cycles of length 20..38 (via 0-39-38-...-k-0)
    # Combined with Hamiltonian cycle (length 40), we cover 3..40
    for k in range(2, 21):
        edges.add((0, k))
        
    # Convert to list for return
    edge_list = sorted(list(edges))
    
    # Generate cycles of every length 3..40
    cycle_by_length = {}
    
    # Hamiltonian cycle: length 40
    cycle_by_length[40] = list(range(N))
    
    for k in range(2, 21):
        # Cycle of length k+1: 0 -> 1 -> ... -> k -> 0
        # Vertices: 0, 1, 2, ..., k (k+1 vertices)
        len1 = k + 1
        cycle_by_length[len1] = list(range(k+1))
        
        # Cycle of length N-k+1: 0 -> 39 -> 38 -> ... -> k -> 0
        # Vertices: 0, 39, 38, ..., k (N-k+1 vertices)
        len2 = N - k + 1
        cycle_by_length[len2] = [0] + list(range(N-1, k-1, -1))
    
    # Verify all lengths 3..40 are covered
    for L in range(3, N+1):
        assert L in cycle_by_length, f"Missing cycle of length {L}"
    
    # Build the ordered list for lengths 3..40
    final_cycles = [cycle_by_length[L] for L in range(3, N+1)]
    
    return {'edges': edge_list, 'cycles': final_cycles}
