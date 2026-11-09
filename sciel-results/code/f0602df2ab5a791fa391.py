"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def _build_adjacency(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _distinct_degrees(vertices, adj):
    """Count distinct degrees in the induced subgraph on 'vertices'."""
    vset = set(vertices)
    degrees = []
    for v in vertices:
        deg = 0
        for u in adj[v]:
            if u in vset:
                deg += 1
        degrees.append(deg)
    return len(set(degrees))

def construct():
    deadline = time.time() + 80  # 80s safety margin before 90s kill
    n = P['n']
    edges = P['edges']
    min_v = P['minimum_vertices']
    adj = _build_adjacency(n, edges)
    
    best_count = 0
    best_vertices = None
    
    # Precompute adjacency as a list of sets for fast lookup
    # Try all subset sizes from min_v to n
    for k in range(min_v, n + 1):
        if time.time() > deadline:
            break
        for combo in combinations(range(n), k):
            if time.time() > deadline:
                break
            vset = set(combo)
            degrees = []
            for v in combo:
                deg = 0
                for u in adj[v]:
                    if u in vset:
                        deg += 1
                degrees.append(deg)
            dcount = len(set(degrees))
            if dcount > best_count:
                best_count = dcount
                best_vertices = list(combo)
                # Early exit: can't do better than k distinct degrees
                if best_count == k:
                    # Maximum possible for this k is k, but we can still check larger k
                    # Actually max distinct degrees for size k is k (0 to k-1)
                    # So if we hit k, that's the max for this size
                    pass
    
    if best_vertices is None:
        best_vertices = list(range(min_v))
    
    # Ensure return type is list of Python ints
    return [int(v) for v in best_vertices]
# EVOLVE-BLOCK-END
