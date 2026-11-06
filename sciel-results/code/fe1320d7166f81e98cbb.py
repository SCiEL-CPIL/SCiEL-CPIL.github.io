def construct():
    """
    Returns a C5-free graph with no independent set of size 9.
    Uses 8 disjoint K4s, giving n=32 and alpha=8.
    This is a known lower bound for R(C5, K9) > 32.
    """
    n = 32
    edges = []
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
    
    return {'n': n, 'edges': edges}
