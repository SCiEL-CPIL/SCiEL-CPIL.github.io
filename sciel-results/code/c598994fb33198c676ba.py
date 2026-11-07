def construct():
    n = 16
    half = n // 2
    left = list(range(half))
    right = list(range(half, n))
    
    edges = []
    for u in left:
        for v in right:
            edges.append([u, v])
    
    return edges
