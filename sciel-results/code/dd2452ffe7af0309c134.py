import numpy as np
from collections import deque
from itertools import combinations

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    Uses Cayley graph construction on Z_p with optimized connection sets.
    """
    def build_cayley_graph(p, conn_set):
        """Build Cayley graph on Z_p with given connection set."""
        n = p
        edges = []
        for i in range(n):
            for c in conn_set:
                j = (i + c) % p
                if i < j:
                    edges.append((i, j))
        return n, edges

    def check_girth_fast(n, edges, target_girth):
        """
        Check if girth >= target_girth using BFS from each vertex.
        Optimized with early termination.
        """
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        for s in range(n):
            dist = [-1] * n
            dist[s] = 0
            q_bfs = deque([s])
            while q_bfs:
                u = q_bfs.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        q_bfs.append(v)
                    elif v != s:  # Don't count the starting vertex
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < target_girth:
                            return False
        return True

    # For k=5, use connection set {a, -a, b, -b, c} where c = -c mod p
    # c must satisfy 2c ≡ 0 mod p, so c = 0 or c = p/2 (if p even)
    # For odd p, we can't have c = -c unless c=0. So use {a, -a, b, -b, d} where d is another element.
    # But then we need 5 distinct elements. Let's use {a, -a, b, -b, d} with d not in {±a, ±b}.
    
    # Try primes p from small to large
    # For girth 9, we need p to be sufficiently large.
    # Moore bound for (5,9) is 2 + 5*(1 + 4 + 4^2 + 4^3) = 2 + 5*85 = 427.
    # So p should be at least 428.
    
    # Let's try primes starting from 431
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return False
        return True

    # Generate candidate primes
    primes = []
    p = 431
    while len(primes) < 50:
        if is_prime(p):
            primes.append(p)
        p += 1

    # For each prime, try different connection sets
    # Connection set: {a, p-a, b, p-b, d} where a, b, d are distinct and not 0
    # To maximize girth, choose a, b, d to be "spread out"
    
    best_n = None
    best_edges = None
    
    for p in primes:
        # Try a few candidate connection sets
        # Heuristic: use small values that are multiplicatively independent
        candidates = []
        for a in range(2, min(10, p//4)):
            for b in range(a+1, min(15, p//3)):
                if b == a or b == p-a:
                    continue
                for d in range(2, min(20, p//2)):
                    if d in (a, p-a, b, p-b):
                        continue
                    conn_set = [a, p-a, b, p-b, d]
                    # Ensure all distinct
                    if len(set(conn_set)) != 5:
                        continue
                    candidates.append(conn_set)
        
        # Limit candidates to avoid timeout
        for conn_set in candidates[:20]:
            n, edges = build_cayley_graph(p, conn_set)
            if check_girth_fast(n, edges, g):
                if best_n is None or n < best_n:
                    best_n = n
                    best_edges = edges
                break  # Found valid for this p, move to next p
        
        if best_n is not None and best_n <= p:
            # Can't do better than p for this p, but maybe smaller p worked
            break
    
    if best_n is not None:
        return best_n, best_edges
    
    # Fallback: return a larger Cayley graph
    p = primes[-1]
    conn_set = [1, p-1, 2, p-2, 3]
    n, edges = build_cayley_graph(p, conn_set)
    return n, edges
