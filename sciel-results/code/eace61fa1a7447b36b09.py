import time

def construct():
    """
    Constructs a graph on 12 vertices (0..11) with minimum degree 3 and girth >= 4.
    The graph consists of two disjoint copies of K_{3,3}.
    
    K_{3,3} properties:
    - Bipartite (triangle-free, girth 4).
    - 3-regular (min degree 3).
    - Simple cycles have lengths 4 and 6 only.
    
    Distinct cycle lengths: {4, 6}. Count = 2.
    This is the theoretical minimum for this problem class (n=12, min_deg=3, girth>=4).
    """
    start_time = time.time()
    deadline = start_time + 59.0  # Safety margin for 90s limit

    edges = []
    
    # Component 1: K_{3,3} on vertices 0, 1, 2 (Left) and 3, 4, 5 (Right)
    left1 = [0, 1, 2]
    right1 = [3, 4, 5]
    for u in left1:
        for v in right1:
            # Ensure types are strictly Python int
            edges.append([int(u), int(v)])
            
    # Component 2: K_{3,3} on vertices 6, 7, 8 (Left) and 9, 10, 11 (Right)
    left2 = [6, 7, 8]
    right2 = [9, 10, 11]
    for u in left2:
        for v in right2:
            # Ensure types are strictly Python int
            edges.append([int(u), int(v)])
            
    # The construction is O(1) and static, so no search or verification loop is needed.
    # The result is valid and optimal.
    
    return edges
