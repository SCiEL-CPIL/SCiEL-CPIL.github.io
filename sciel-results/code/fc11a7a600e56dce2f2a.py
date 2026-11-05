def construct():
    import time
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    edges = []
    # K_{3,13}: all edges between {0,1,2} and {3,...,15}
    for a in range(3):
        for b in range(3, 16):
            edges.append([a, b])
    
    # Triangle on L = {0,1,2}
    edges.append([0, 1])
    edges.append([0, 2])
    edges.append([1, 2])
    
    # Cycle on R = {3,...,15}: 3-4-5-...-15-3
    for i in range(3, 15):
        edges.append([i, i + 1])
    edges.append([15, 3])
    
    return edges
