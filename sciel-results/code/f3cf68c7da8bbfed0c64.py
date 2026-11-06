import time
import sys

N = 40

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 60.0  # Seconds
    
    # Construct a cycle with chords from vertex 0
    # Base cycle: 0-1-2-...-39-0
    edges = []
    
    # Add cycle edges
    for i in range(N):
        edges.append((i, (i + 1) % N))
        
    # Add chords (0, j) for j in 2..20
    # This creates cycles of lengths 3..21 (short) and 21..39 (long)
    # Plus the base cycle length 40
    chords = []
    for j in range(2, 21):
        edge = (0, j)
        if edge not in edges:
            edges.append(edge)
            chords.append(j)
            
    # Generate cycles
    cycles = []
    
    # Helper to generate a cycle from a list of vertices
    # The cycle must be a closed loop in the graph
    # We construct them based on the known structure
    
    # Lengths 3 to 21: Cycle 0-1-...-(k-1)-0
    for k in range(3, 22):
        cycle = list(range(k))  # 0, 1, ..., k-1
        # Verify edges exist? 
        # (0,1)..(k-2, k-1) are in base cycle.
        # (k-1, 0) is a chord if k-1 in 2..20.
        # For k=3, k-1=2, chord (0,2) exists.
        # For k=21, k-1=20, chord (0,20) exists.
        cycles.append(cycle)
        
    # Lengths 22 to 39: Cycle 0-j-(j+1)-...-39-0
    # Length L = 41 - j.
    # For L=22, j=19. Cycle 0-19-20-...-39-0.
    # For L=39, j=2. Cycle 0-2-3-...-39-0.
    for L in range(22, 40):
        j = 41 - L
        # Vertices: 0, j, j+1, ..., 39
        cycle = [0] + list(range(j, N))
        cycles.append(cycle)
        
    # Length 40: Base cycle
    cycles.append(list(range(N)))
    
    # Re-order cycles to be in order of length 3..40
    # The list `cycles` is currently:
    # 3..21 (from first loop)
    # 22..39 (from second loop)
    # 40 (last)
    # This is already in order 3..40.
    
    # Sanity check: Ensure all edges in cycles exist
    # For speed, we trust the construction, but a quick check is safe if time permits.
    # Given N=40, a check is trivial.
    
    edge_set = set()
    for e in edges:
        edge_set.add((min(e), max(e)))
        
    valid = True
    for cycle in cycles:
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            if (min(u, v), max(u, v)) not in edge_set:
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        # Fallback to seed if construction failed (should not happen)
        return {'edges': [(0,i) for i in range(1,N)] + [(i,i+1) for i in range(1,N-1)],
                'cycles': [list(range(k)) for k in range(3,N+1)]}
                
    return {'edges': edges, 'cycles': cycles}
# EVOLVE-BLOCK-END
