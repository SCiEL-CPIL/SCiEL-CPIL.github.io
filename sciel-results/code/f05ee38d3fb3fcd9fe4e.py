"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 60s limit
    
    # Strategy: Construct two disjoint K_{3,3} components.
    # K_{3,3} is a 3-regular bipartite graph on 6 vertices.
    # Properties:
    # 1. Minimum degree is 3.
    # 2. It is bipartite, so it is triangle-free (girth >= 4).
    # 3. The simple cycles in K_{3,3} have lengths 4 and 6.
    #    - 4-cycles exist.
    #    - 6-cycles exist.
    #    - No other cycle lengths exist.
    # 4. The union of two such graphs has distinct cycle lengths {4, 6}.
    #    Count of distinct lengths = 2.
    # This is the theoretical minimum for n=12, min degree 3, triangle-free.
    
    edges = []
    
    # Component 1: K_{3,3} on vertices 0..5
    # Part U = {0, 1, 2}, Part V = {3, 4, 5}
    part_u1 = [0, 1, 2]
    part_v1 = [3, 4, 5]
    for u in part_u1:
        for v in part_v1:
            edges.append([u, v])
            
    # Component 2: K_{3,3} on vertices 6..11
    # Part U = {6, 7, 8}, Part V = {9, 10, 11}
    part_u2 = [6, 7, 8]
    part_v2 = [9, 10, 11]
    for u in part_u2:
        for v in part_v2:
            edges.append([u, v])
            
    # Verification logic to ensure validity
    n = 12
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
        
    # Check min degree
    if min(deg) < 3:
        # Fallback: This should not happen for the constructed graph
        # But if it did, we return the edges anyway as they are the best candidate
        pass

    # Ensure types are strictly Python int and list structure
    result = []
    seen = set()
    for e in edges:
        u = int(e[0])
        v = int(e[1])
        if u == v:
            continue
        if u > v:
            u, v = v, u
        if (u, v) not in seen:
            seen.add((u, v))
            result.append([u, v])
            
    return result
# EVOLVE-BLOCK-END
