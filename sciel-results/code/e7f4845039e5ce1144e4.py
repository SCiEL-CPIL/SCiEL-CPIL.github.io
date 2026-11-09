import time

def construct():
    """
    Returns a simple graph on n vertices (1 <= n <= 35) that is C5-free 
    and has no independent set of size 9 (alpha <= 8).
    Maximizes n.
    
    Known valid construction: 8 disjoint K4 cliques.
    n = 32.
    alpha = 8.
    C5-free: Yes.
    
    Deadline guard: returns immediately since this is a fixed construction.
    """
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    # 8 disjoint K4s - fixed construction, no search needed
    n = 32
    edges = []
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
    
    # Verify we're within deadline (should be trivially true for fixed construction)
    if time.time() > deadline:
        # This should never happen, but return best we have
        pass
    
    return {'n': n, 'edges': edges}
