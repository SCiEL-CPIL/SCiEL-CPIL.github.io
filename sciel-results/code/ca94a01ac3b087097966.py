import time
import sys

N = 40

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    # 1. Define Edges
    # Hamiltonian Cycle: 0-1-2-...-39-0
    edges = set()
    for i in range(N):
        edges.add((i, (i + 1) % N))
    
    # Chords from 0 to k for k in 2..21
    # These are (0, k)
    for k in range(2, 22):
        edges.add((0, k))
        
    # Convert to list of tuples for output
    # Ensure sorted for consistency if needed, but set handles uniqueness
    edge_list = [tuple(sorted(e)) for e in edges]
    
    # 2. Define Cycles
    cycles = []
    
    # Lengths 3 to 40
    for L in range(3, N + 1):
        if L < 22:
            # Cycle: 0-1-...-(L-1)-0
            # Vertices: 0, 1, ..., L-1
            # Edges: (0,1), (1,2), ..., (L-2, L-1), (L-1, 0)
            # (L-1, 0) is a chord if L-1 >= 2.
            # If L=3, L-1=2. Chord (0,2) exists.
            # If L=2, not required.
            cycle = list(range(L)) # 0, 1, ..., L-1
            # Check if this is a valid cycle in the graph
            # 0-1, 1-2, ..., L-2-L-1, L-1-0
            # All path edges exist in Ham cycle.
            # Edge (L-1, 0) exists as chord (0, L-1) since L-1 is in 2..21 for L in 3..22.
            # Wait, L < 22 means L max 21. L-1 max 20.
            # Chords are 2..21. So 2..20 are covered.
            # For L=3, L-1=2. Chord (0,2) exists.
            # For L=21, L-1=20. Chord (0,20) exists.
            cycles.append(cycle)
        elif L == 22:
            # Use chord k=21
            # Cycle: 0-21-22-...-39-0
            # Vertices: 0, 21, 22, ..., 39
            # Length: 1 (0-21) + (39-21) + 1 (39-0) = 1 + 18 + 1 = 20?
            # Wait. 21 to 39 is 18 steps? 39-21=18.
            # Vertices: 21, 22, ..., 39. Count: 39-21+1 = 19 vertices.
            # Plus 0. Total 20 vertices.
            # Length 20.
            # I made a mistake in length calculation for the "other" cycle.
            
            # Let's re-verify the length of the cycle formed by chord (0, k).
            # Ham cycle: 0-1-...-k-...-39-0.
            # Chord (0, k).
            # Cycle 1: 0-1-...-k-0.
            # Vertices: 0, 1, ..., k. Count: k+1.
            # Length: k+1.
            
            # Cycle 2: 0-k-(k+1)-...-39-0.
            # Vertices: 0, k, k+1, ..., 39.
            # Count: 1 (for 0) + (39 - k + 1) = 41 - k.
            # Length: 41 - k.
            
            # For L=22:
            # We need 41 - k = 22 => k = 19.
            # Or k + 1 = 22 => k = 21.
            
            # If k=19:
            # Cycle 1 length: 20.
            # Cycle 2 length: 22.
            # So k=19 generates length 22.
            
            # If k=21:
            # Cycle 1 length: 22.
            # Cycle 2 length: 20.
            # So k=21 generates length 22.
            
            # So for L=22, we can use k=19 or k=21.
            # Let's use k=19 to be consistent with the "upper" range logic?
            # Or just pick one.
            # Let's use k=19 for L=22.
            # Cycle: 0-19-20-...-39-0.
            cycle = [0] + list(range(19, N))
            cycles.append(cycle)
            
        else: # L > 22
            # Use k = 41 - L
            # k is in range 2..18
            k = 41 - L
            # Cycle: 0-k-(k+1)-...-39-0
            cycle = [0] + list(range(k, N))
            cycles.append(cycle)
            
    # Verify validity (optional but good for safety)
    # Check if all edges in cycles exist
    edge_set = set(edge_list)
    for cycle in cycles:
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i+1) % len(cycle)]
            if (u, v) not in edge_set and (v, u) not in edge_set:
                raise ValueError(f"Missing edge ({u},{v}) in cycle {cycle}")
                
    return {'edges': edge_list, 'cycles': cycles}
