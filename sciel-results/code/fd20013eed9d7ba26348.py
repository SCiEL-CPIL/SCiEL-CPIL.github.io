import time

def construct():
    """
    Constructs a graph on 12 vertices (0..11) with minimum degree 3 and girth 4.
    The graph consists of two disjoint K_{3,3} components.
    This ensures all cycles are of length 4 or 6, minimizing distinct cycle lengths to 2.
    """
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin
    
    edges = []
    
    # First K_{3,3} on vertices 0..5
    # Partition A: 0, 1, 2
    # Partition B: 3, 4, 5
    A1 = [0, 1, 2]
    B1 = [3, 4, 5]
    for u in A1:
        for v in B1:
            edges.append([u, v])
            
    # Second K_{3,3} on vertices 6..11
    # Partition A: 6, 7, 8
    # Partition B: 9, 10, 11
    A2 = [6, 7, 8]
    B2 = [9, 10, 11]
    for u in A2:
        for v in B2:
            edges.append([u, v])
            
    # Deduplicate, normalize, sort, and return as list of [u, v] Python ints
    seen = set()
    out = []
    for u, v in edges:
        if u == v:
            continue
        key = (min(u, v), max(u, v))
        if key in seen:
            continue
        seen.add(key)
        out.append([int(u), int(v)])
    out.sort()
    
    return out
