import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    def time_left():
        return deadline - time.time()

    # Known valid: 8 copies of K4 (n=32)
    # This graph is C5-free (components are K4, max cycle 4)
    # and has alpha=8 (one vertex from each K4)
    n = 32
    adj = [set() for _ in range(n)]
    
    for b in range(8):
        for i in range(4):
            for j in range(i+1, 4):
                u = 4*b + i
                v = 4*b + j
                adj[u].add(v)
                adj[v].add(u)
                
    # Quick validation (optional, but good for safety)
    # If validation fails, fallback to smaller graph
    def is_valid(n, adj):
        # Check C5
        # Since components are K4, no C5 possible. 
        # But let's be sure.
        for v0 in range(n):
            for v1 in adj[v0]:
                if v1 <= v0: continue
                for v2 in adj[v1]:
                    if v2 in (v0, v1): continue
                    for v3 in adj[v2]:
                        if v3 in (v0, v1, v2): continue
                        for v4 in adj[v3]:
                            if v4 in (v0, v1, v2, v3): continue
                            if v0 in adj[v4]:
                                return False
        # Check alpha
        # Components are K4. Max independent set is 8.
        # So alpha <= 8 is guaranteed.
        return True

    if not is_valid(n, adj):
        # Fallback to 1 vertex
        n = 1
        adj = [set()]

    edges = []
    for i in range(n):
        for j in adj[i]:
            if i < j:
                edges.append([i, j])
                
    return {'n': n, 'edges': edges}
