"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def _build_adjacency():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _count_distinct_degrees(vertices, adj):
    """Count distinct degrees in induced subgraph on given vertices."""
    vset = set(vertices)
    degs = []
    for v in vertices:
        d = sum(1 for u in vset if u != v and u in adj[v])
        degs.append(d)
    return len(set(degs))

def _quick_check(vertices, adj):
    """Quick check: compute distinct degrees, return count."""
    vset = set(vertices)
    degs = []
    for v in vertices:
        d = sum(1 for u in vset if u != v and u in adj[v])
        degs.append(d)
    return len(set(degs))

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin
    
    n = P['n']
    adj = _build_adjacency()
    
    best_score = 0
    best_vertices = None
    
    # Try to find a 10-vertex subset with 10 distinct degrees (0-9)
    # This would give score 10/6 = 1.667
    # Also try 9-vertex subsets with 9 distinct degrees (score 9/6 = 1.5)
    # And 8-vertex subsets with 8 distinct degrees (score 8/6 = 1.333)
    
    all_vertices = list(range(n))
    
    # First, try exhaustive search for 10-vertex subsets
    # C(17,10) = 19448 combinations - very manageable
    found_10 = False
    for combo in combinations(all_vertices, 10):
        if time.time() > deadline:
            break
        vset = set(combo)
        degs = []
        for v in combo:
            d = 0
            for u in vset:
                if u != v and u in adj[v]:
                    d += 1
            degs.append(d)
        distinct = len(set(degs))
        if distinct > best_score:
            best_score = distinct
            best_vertices = list(combo)
            if best_score == 10:
                found_10 = True
                break
        if time.time() > deadline:
            break
    
    if found_10:
        return best_vertices
    
    # If we didn't find 10 distinct degrees in 10 vertices,
    # also check 9-vertex subsets (C(17,9) = 24310)
    if not found_10:
        for combo in combinations(all_vertices, 9):
            if time.time() > deadline:
                break
            vset = set(combo)
            degs = []
            for v in combo:
                d = 0
                for u in vset:
                    if u != v and u in adj[v]:
                        d += 1
                degs.append(d)
            distinct = len(set(degs))
            if distinct > best_score:
                best_score = distinct
                best_vertices = list(combo)
                if best_score == 9:
                    break
            if time.time() > deadline:
                break
    
    # Also check 8-vertex subsets if needed
    if best_score < 8:
        for combo in combinations(all_vertices, 8):
            if time.time() > deadline:
                break
            vset = set(combo)
            degs = []
            for v in combo:
                d = 0
                for u in vset:
                    if u != v and u in adj[v]:
                        d += 1
                degs.append(d)
            distinct = len(set(degs))
            if distinct > best_score:
                best_score = distinct
                best_vertices = list(combo)
            if time.time() > deadline:
                break
    
    # Ensure we have at least 8 vertices
    if best_vertices is None or len(best_vertices) < 8:
        best_vertices = list(range(8))
    
    # Ensure all elements are Python ints
    result = [int(v) for v in best_vertices]
    return result
# EVOLVE-BLOCK-END
