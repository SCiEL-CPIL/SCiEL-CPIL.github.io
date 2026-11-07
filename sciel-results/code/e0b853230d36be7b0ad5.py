import time

def construct():
    """
    Construct a 12-vertex undirected simple graph with:
    - Every vertex has degree at least 3
    - Every simple cycle has length at least 4 (triangle-free)
    - Contains at least one cycle
    - Minimizes the number of distinct simple cycle lengths
    
    Strategy:
    - The theoretical minimum for the number of distinct simple cycle lengths
      in a triangle-free graph with minimum degree 3 is 2.
    - This is achieved by the graph consisting of two disjoint copies of K_{3,3}.
    - K_{3,3} is bipartite with parts of size 3. Its cycles have lengths 4 and 6.
    - Since the components are disjoint, the set of cycle lengths for the whole
      graph is {4, 6}.
    - No graph satisfying the constraints can have only 1 distinct cycle length.
    - We return this optimal graph immediately.
    - A deadline guard is included to satisfy safety constraints, though the
      construction is constant time.
    """
    start_time = time.time()
    deadline = 85.0  # Safety margin before the 90s kill switch
    
    # Two disjoint K_{3,3} components
    # First K_{3,3}: vertices 0,1,2 (part A) and 3,4,5 (part B)
    # Second K_{3,3}: vertices 6,7,8 (part A) and 9,10,11 (part B)
    
    edges = []
    
    # First K_{3,3}
    part_a1 = [0, 1, 2]
    part_b1 = [3, 4, 5]
    for u in part_a1:
        for v in part_b1:
            edges.append([u, v])
    
    # Second K_{3,3}
    part_a2 = [6, 7, 8]
    part_b2 = [9, 10, 11]
    for u in part_a2:
        for v in part_b2:
            edges.append([u, v])
    
    # Check if we are still within time limits (should be, as this is O(1))
    if time.time() - start_time > deadline:
        # Fallback is the same graph, but we return it anyway as it's the best
        pass
        
    # Sort edges for consistent output
    edges.sort()
    
    # Ensure all values are Python int to satisfy type checking requirements
    result = [[int(e[0]), int(e[1])] for e in edges]
    
    return result
