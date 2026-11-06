"""Structural change: local search with edge flips to minimize max triangle-free induced subgraph."""
import time
import random
from itertools import combinations

def _is_k4_free(adj):
    """Check if graph is K4-free using adjacency sets."""
    n = len(adj)
    # For each pair of adjacent vertices, check if they share 2 common neighbors
    for u in range(n):
        for v in adj[u]:
            if u < v:
                # Check if u and v have at least 2 common neighbors
                common = adj[u] & adj[v]
                if len(common) >= 2:
                    # Found K4: u, v, and two common neighbors
                    return False
    return True

def _max_triangle_free(adj):
    """Find the maximum number of vertices inducing a triangle-free subgraph."""
    n = len(adj)
    # Try all subsets of size s from largest to smallest
    # But that's exponential. Use a smarter approach.
    # We can use the fact that a triangle-free induced subgraph is one where
    # no three vertices form a triangle.
    
    # For small n=18, we can try a branch and bound or just check all subsets
    # of size >= some threshold.
    
    # Actually, let's use a different approach: 
    # The complement of a triangle-free graph is one where every triple has at least one edge missing... 
    # No, that's not right.
    
    # Let's just do a simple exhaustive search for subsets of size s from 18 down.
    # For each subset, check if it's triangle-free.
    
    # Optimization: if we find a triangle-free set of size s, we can stop.
    # But we want the MAXIMUM, so we go from large to small.
    
    # For n=18, C(18,9) = 48620, C(18,10) = 43758, etc.
    # Let's check from size 9 down (since K6,6,6 gives 12, we hope to do better).
    
    # Actually, let's be smarter. We'll check sizes from 18 down to 1.
    # But C(18,18)=1, C(18,17)=18, C(18,16)=153, C(18,15)=816, C(18,14)=3060, C(18,13)=8568, C(18,12)=18564, C(18,11)=31824, C(18,10)=43758, C(18,9)=48620
    
    # Total for all sizes is 2^18 = 262144, which is manageable.
    
    # Let's check all subsets from size 18 down to 1, and return the first (largest) that is triangle-free.
    
    for size in range(n, 0, -1):
        for subset in combinations(range(n), size):
            if _is_triangle_free(subset, adj):
                return size
    return 0

def _is_triangle_free(subset, adj):
    """Check if the induced subgraph on 'subset' is triangle-free."""
    s = list(subset)
    for i in range(len(s)):
        for j in range(i+1, len(s)):
            for k in range(j+1, len(s)):
                u, v, w = s[i], s[j], s[k]
                if v in adj[u] and w in adj[u] and w in adj[v]:
                    return False
    return True

def _flip_edge(adj, u, v):
    """Flip the edge between u and v. Returns True if flip is valid (K4-free)."""
    if v in adj[u]:
        # Remove edge
        adj[u].remove(v)
        adj[v].remove(u)
        if _is_k4_free(adj):
            return True
        # Revert
        adj[u].add(v)
        adj[v].add(u)
        return False
    else:
        # Add edge
        adj[u].add(v)
        adj[v].add(u)
        if _is_k4_free(adj):
            return True
        # Revert
        adj[u].remove(v)
        adj[v].remove(u)
        return False

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline with 10s margin
    
    n = 18
    
    # Start with K6,6,6
    def _init_adj():
        adj = [set() for _ in range(n)]
        parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
        for i in range(3):
            for j in range(i+1, 3):
                for a in parts[i]:
                    for b in parts[j]:
                        adj[a].add(b)
                        adj[b].add(a)
        return adj
    
    adj = _init_adj()
    best_score = _max_triangle_free(adj)
    best_adj = [set(s) for s in adj]
    
    random.seed(42)
    
    while time.time() < deadline:
        # Try random edge flips
        for _ in range(10000):
            if time.time() >= deadline:
                break
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v:
                continue
            if _flip_edge(adj, u, v):
                score = _max_triangle_free(adj)
                if score < best_score:
                    best_score = score
                    best_adj = [set(s) for s in adj]
                    # Keep improving from here
                    if best_score <= 9:
                        # Good enough, we can try to improve more
                        pass
                else:
                    # Revert if it didn't help
                    pass
        # If we've reached a good score, we can be more aggressive
        if best_score <= 10:
            # Try more targeted flips
            for _ in range(5000):
                if time.time() >= deadline:
                    break
                # Focus on edges within parts (non-edges in K6,6,6) or between parts
                part_of = [i // 6 for i in range(n)]
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u == v:
                    continue
                if _flip_edge(adj, u, v):
                    score = _max_triangle_free(adj)
                    if score < best_score:
                        best_score = score
                        best_adj = [set(s) for s in adj]
    
    # Convert best_adj to edge list
    edges = []
    for u in range(n):
        for v in best_adj[u]:
            if u < v:
                edges.append([u, v])
    
    return edges
