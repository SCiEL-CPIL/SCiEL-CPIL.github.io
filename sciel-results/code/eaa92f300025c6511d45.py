import time

def construct():
    """
    Constructs a graph on 12 vertices with minimum degree 3, triangle-free,
    and exactly 2 distinct simple cycle lengths (4 and 6).
    
    The graph consists of two disjoint copies of K_{3,3}.
    """
    start_time = time.time()
    deadline = start_time + 58.0  # Safety margin
    
    edges = []
    
    # Component 1: K_{3,3} on vertices 0, 1, 2 and 3, 4, 5
    for u in [0, 1, 2]:
        for v in [3, 4, 5]:
            edges.append([int(u), int(v)])
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    # Component 2: K_{3,3} on vertices 6, 7, 8 and 9, 10, 11
    for u in [6, 7, 8]:
        for v in [9, 10, 11]:
            edges.append([int(u), int(v)])
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    return edges
