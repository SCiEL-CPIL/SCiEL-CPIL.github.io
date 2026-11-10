"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 16}

def construct():
    start_time = time.time()
    deadline = start_time + 50
    n = P['n']
    
    # Theoretical maximum for triangle-free graphs is n^2/4 = 64.
    # K_{8,8} is triangle-free, diameter 2, and all edges are critical.
    # We construct K_{8,8} directly.
    
    edges = []
    for i in range(8):
        for j in range(8, 16):
            edges.append([i, j])
            
    # Verify diameter is 2
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    def check_diam2(adj):
        for src in range(n):
            dist = [-1]*n
            dist[src] = 0
            queue = [src]
            for node in queue:
                for nb in adj[node]:
                    if dist[nb] == -1:
                        dist[nb] = dist[node]+1
                        queue.append(nb)
            if -1 in dist:
                return False
            for d in dist:
                if d > 2:
                    return False
        return True

    if not check_diam2(adj):
        # Fallback to search if K_{8,8} somehow fails (it shouldn't)
        # But it will pass.
        pass

    return edges
# EVOLVE-BLOCK-END
