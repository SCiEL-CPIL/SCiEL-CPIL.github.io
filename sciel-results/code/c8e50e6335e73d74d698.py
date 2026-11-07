"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from itertools import combinations
import time

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}


def _build_adjacency():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _distinct_degrees(vertices, adj):
    """Count distinct degrees in the induced subgraph on 'vertices'."""
    vset = set(vertices)
    degs = set()
    for v in vertices:
        degs.add(len(adj[v] & vset))
    return len(degs)


def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s out of 90s budget

    n = P['n']
    min_v = P['minimum_vertices']
    adj = _build_adjacency()

    best_count = 0
    best_vertices = list(range(min_v))

    # Enumerate all subsets of size >= min_v
    # For a 17-vertex graph, total subsets is 2^17 = 131072
    # Subsets with popcount >= 8: sum_{k=8}^{17} C(17,k) = 131072 - sum_{k=0}^{7} C(17,k)
    # C(17,0)=1, C(17,1)=17, C(17,2)=136, C(17,3)=680, C(17,4)=2380, C(17,5)=6188, C(17,6)=12376, C(17,7)=19448
    # sum = 1+17+136+680+2380+6188+12376+19448 = 41226
    # So ~89846 subsets to check, very fast.

    for size in range(min_v, n + 1):
        if time.time() > deadline:
            break
        for combo in combinations(range(n), size):
            if time.time() > deadline:
                break
            cnt = _distinct_degrees(combo, adj)
            if cnt > best_count:
                best_count = cnt
                best_vertices = list(combo)
            # Early termination: max possible distinct degrees is 'size'
            if best_count == n:  # absolute max
                return best_vertices

    return best_vertices
# EVOLVE-BLOCK-END
