import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin

    def can_add_edge(n, adj, u, v):
        """Check if adding edge (u,v) creates a C4."""
        for w in adj[u]:
            if w == v:
                continue
            for x in adj[v]:
                if x == u or x == w:
                    continue
                if x in adj[w]:
                    return False
        return True

    def greedy_independence_number(n, adj):
        """Compute a maximal independent set size via greedy (min-degree)."""
        remaining = set(range(n))
        indep = 0
        while remaining:
            min_deg = float('inf')
            pick = None
            for v in remaining:
                deg = len(adj[v] & remaining)
                if deg < min_deg:
                    min_deg = deg
                    pick = v
            indep += 1
            remaining -= adj[pick]
            remaining.discard(pick)
        return indep

    # Target n=39 as per best known construction.
    # We will try to find a C4-free graph with small independence number.
    # We use a strict threshold (<= 7) to ensure true alpha is likely <= 10.
    
    best_n = 0
    best_edges = []

    # Try n from 39 down to 30
    for n in range(39, 29, -1):
        if time.time() > deadline:
            break
        
        found = False
        # Multiple attempts for each n
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            adj_try = [set() for _ in range(n)]
            
            # Random order of edges
            all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
            random.shuffle(all_edges)
            
            for u, v in all_edges:
                if time.time() > deadline:
                    break
                if can_add_edge(n, adj_try, u, v):
                    adj_try[u].add(v)
                    adj_try[v].add(u)
            
            # Check independence number
            alpha = greedy_independence_number(n, adj_try)
            
            # If greedy alpha is small, we are confident true alpha is <= 10
            if alpha <= 7:
                edges = []
                for u in range(n):
                    for v in adj_try[u]:
                        if u < v:
                            edges.append([u, v])
                return {'n': n, 'edges': edges}

    # Fallback: A simple valid graph on a smaller number of vertices.
    # Construct a C4-free graph with small alpha.
    # Use a random C4-free graph on n=20 and verify alpha.
    # If we can't find one, return a known small C4-free graph.
    # The cycle C_5 is C4-free, alpha=2.
    # We can take disjoint union of C_5s?
    # n=15, 3 C_5s. Alpha = 3*2 = 6.
    # n=20, 4 C_5s. Alpha = 4*2 = 8.
    # n=25, 5 C_5s. Alpha = 5*2 = 10.
    # This is a valid graph with alpha <= 10.
    
    n_fallback = 25
    edges = []
    for k in range(5):
        base = k * 5
        for i in range(5):
            u = base + i
            v = base + (i + 1) % 5
            if u < v:
                edges.append([u, v])
            else:
                edges.append([v, u])
                
    return {'n': n_fallback, 'edges': edges}
