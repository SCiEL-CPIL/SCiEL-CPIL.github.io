import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    n = 512
    values = list(range(1, n + 1))
    
    # Precompute adjacency as sets of indices
    # adj[i] is a set of j such that lcm(i+1, j+1) <= 512
    adj = [set() for _ in range(n)]
    for i in range(n):
        a = i + 1
        for j in range(i + 1, n):
            b = j + 1
            l = a * b // gcd(a, b)
            if l <= 512:
                adj[i].add(j)
                adj[j].add(i)
                
    # Initial best clique: Divisors of 360 (size 24)
    divs_360 = [d for d in range(1, 361) if 360 % d == 0]
    best_clique = set(divs_360)
    best_size = len(divs_360)
    
    # Try a greedy heuristic to find a larger initial clique
    def greedy_clique():
        P = set(range(n))
        R = set()
        while P:
            # Pick node with max degree in P
            best_node = max(P, key=lambda v: len(P & adj[v]))
            R.add(best_node)
            P &= adj[best_node]
        
        if len(R) > best_size:
            return {v + 1 for v in R}
        return None
    
    g_clique = greedy_clique()
    if g_clique:
        best_clique = g_clique
        best_size = len(best_clique)

    def bronk(R, P, X):
        nonlocal best_clique, best_size
        if time.time() > deadline:
            return
        if not P and not X:
            sz = len(R)
            if sz > best_size:
                best_size = sz
                best_clique = {v + 1 for v in R}
            return
        if not P:
            return
        
        # Upper bound
        if len(R) + len(P) <= best_size:
            return
        
        # Choose pivot u from P ∪ X to maximize |P ∩ N(u)|
        union = P | X
        if not union:
            return
        
        # Optimization: if we find u such that P is subset of N(u), 
        # then P \ N(u) is empty, and we return.
        # This is correct because if X is not empty, R is not maximal.
        # If X is empty, R is maximal, but we would have caught it in `if not P and not X`.
        # Wait, if P is not empty, but P \ N(u) is empty, it means P is a subset of N(u).
        # Then we iterate 0 times.
        
        pivot = max(union, key=lambda u: len(P & adj[u]))
        candidates = P - adj[pivot]
        
        for v in list(candidates):
            bronk(R | {v}, P & adj[v], X & adj[v])
            P.remove(v)
            X.add(v)

    all_nodes = set(range(n))
    bronk(set(), all_nodes, set())
    
    if not best_clique:
        return [1]
    
    return list(best_clique)
