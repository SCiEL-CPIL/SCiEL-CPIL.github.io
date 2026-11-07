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
    #    - 4-cycles exist (e.g., using 2 vertices from each partition).
    #    - 6-cycles exist (e.g., using all 3 vertices from each partition).
    #    - No cycles of length > 6 exist in K_{3,3} because the partition size is 3.
    #    - No odd cycles exist because it is bipartite.
    # 4. The union of two such graphs has distinct cycle lengths {4, 6}.
    #    Count of distinct lengths = 2.
    # 5. It is not possible to have count = 1 for n=12, min degree 3, triangle-free.
    #    (Any triangle-free graph with min degree 3 on 12 vertices will have at least two distinct cycle lengths, typically 4 and 6).
    
    edges = []
    
    # Component 1: K_{3,3} on vertices 0..5
    # Part U = {0, 1, 2}, Part V = {3, 4, 5}
    for u in [0, 1, 2]:
        for v in [3, 4, 5]:
            edges.append([u, v])
            
    # Component 2: K_{3,3} on vertices 6..11
    # Part U = {6, 7, 8}, Part V = {9, 10, 11}
    for u in [6, 7, 8]:
        for v in [9, 10, 11]:
            edges.append([u, v])
            
    # Verification (Optional but good for robustness)
    # Check for duplicates and loops
    edge_set = set()
    valid_edges = []
    for e in edges:
        u, v = e
        if u == v:
            continue
        if u > v:
            u, v = v, u
        if (u, v) not in edge_set:
            edge_set.add((u, v))
            valid_edges.append([u, v])
            
    # Check degrees
    deg = [0] * 12
    for u, v in valid_edges:
        deg[u] += 1
        deg[v] += 1
        
    if min(deg) < 3:
        # Fallback: This should not happen for the constructed graph
        # But if it did, we return the edges anyway as they are the best candidate
        pass

    # Ensure types are strictly Python int
    result = []
    for e in valid_edges:
        result.append([int(e[0]), int(e[1])])
        
    return result
# EVOLVE-BLOCK-END
