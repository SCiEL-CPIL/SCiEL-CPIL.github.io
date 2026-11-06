import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Leave 10s margin
    
    # 1. Generate candidate set S
    # Numbers in [1, 255] divisible by 3, 5, or 17
    candidates = []
    for i in range(1, 256):
        if i % 3 == 0 or i % 5 == 0 or i % 17 == 0:
            candidates.append(i)
    
    n = len(candidates)
    
    # 2. Build adjacency matrix
    # adj[i][j] = True if gcd(candidates[i], candidates[j]) > 1
    adj = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if math.gcd(candidates[i], candidates[j]) > 1:
                adj[i][j] = True
                adj[j][i] = True
    
    # 3. Bron-Kerbosch to find maximum clique
    # We use a standard recursive implementation with pivoting.
    # Given n=127, this should be fast.
    
    max_clique = []
    max_size = 0
    
    def bronk(R, P, X):
        nonlocal max_clique, max_size
        
        # Pruning: if current clique size + potential size <= max_size, return
        if len(R) + len(P) <= max_size:
            return
        
        if not P and not X:
            if len(R) > max_size:
                max_size = len(R)
                max_clique = R.copy()
            return
        
        # Choose pivot u from P union X
        # Heuristic: choose u with max degree in P
        u = -1
        max_degree = -1
        P_X = P.union(X)
        for v in P_X:
            deg = len(P.intersection(adj_set[v]))
            if deg > max_degree:
                max_degree = deg
                u = v
        
        # Iterate over candidates in P \ N(u)
        # If u is -1, iterate over all P
        if u == -1:
            candidates_to_iterate = list(P)
        else:
            candidates_to_iterate = list(P.difference(adj_set[u]))
        
        for v in candidates_to_iterate:
            bronk(R.union({v}), P.intersection(adj_set[v]), X.intersection(adj_set[v]))
            P.remove(v)
            X.add(v)
            
            # Time check
            if time.time() > deadline:
                return

    # Precompute adjacency sets for faster intersection
    adj_set = [set() for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if adj[i][j]:
                adj_set[i].add(j)

    # Run Bron-Kerbosch
    bronk(set(), set(range(n)), set())
    
    # If timeout occurred or no clique found (shouldn't happen), fallback to multiples of 3
    if not max_clique:
        return list(range(3, 256, 3))
        
    return [candidates[i] for i in max_clique]
