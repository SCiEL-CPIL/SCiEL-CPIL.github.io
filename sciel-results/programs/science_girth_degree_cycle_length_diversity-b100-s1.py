import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 60.0
    
    # Model: Two disjoint copies of K_{3,3}
    # K_{3,3} is bipartite with parts of size 3.
    # Cycle lengths in K_{3,3} are 4 and 6.
    # Thus, the union has distinct cycle lengths {4, 6}.
    # Count = 2.
    
    edges = []
    
    # Component 1: K_{3,3} on vertices 0-5
    # Part A: 0, 1, 2
    # Part B: 3, 4, 5
    part_a1 = [0, 1, 2]
    part_b1 = [3, 4, 5]
    for u in part_a1:
        for v in part_b1:
            edges.append([u, v])
            
    # Component 2: K_{3,3} on vertices 6-11
    # Part A: 6, 7, 8
    # Part B: 9, 10, 11
    part_a2 = [6, 7, 8]
    part_b2 = [9, 10, 11]
    for u in part_a2:
        for v in part_b2:
            edges.append([u, v])
    
    # Verify constraints
    n = 12
    degree = [0] * n
    edge_set = set()
    
    for u, v in edges:
        if u == v:
            return []
        if (u, v) in edge_set or (v, u) in edge_set:
            return []
        edge_set.add((u, v))
        edge_set.add((v, u))
        degree[u] += 1
        degree[v] += 1
        
    if any(d < 3 for d in degree):
        return []
    
    # Bipartite by construction (parts are disjoint, no intra-part edges)
    # K_{3,3} contains 4-cycles and 6-cycles, so cycles exist.
    
    return edges

# EVOLVE-BLOCK-START
# The above code is the entire program.
# EVOLVE-BLOCK-END
