import time

def construct():
    """
    Return edges for a triangle-free graph on 12 vertices with min degree >= 3
    that minimizes the number of distinct simple cycle lengths.
    
    The optimal solution uses two disjoint copies of K_{3,3}.
    K_{3,3} is bipartite (hence triangle-free), has degree 3 for every vertex,
    and contains cycles of length 4 and 6 only.
    Thus the distinct cycle lengths are {4, 6}, giving a raw value of 2.
    
    Deadline guard: track elapsed time and return best result with safety margin.
    """
    t_start = time.time()
    deadline = 85.0  # 90s limit, 5s safety margin
    
    edges = []
    
    # First K_{3,3} component: vertices 0,1,2 in partition A; 3,4,5 in partition B
    for u in range(3):
        for v in range(3, 6):
            edges.append([int(u), int(v)])
    
    # Check deadline (should never trigger for this fixed construction)
    if time.time() - t_start > deadline:
        return edges
    
    # Second K_{3,3} component: vertices 6,7,8 in partition A; 9,10,11 in partition B
    for u in range(6, 9):
        for v in range(9, 12):
            edges.append([int(u), int(v)])
    
    return edges
