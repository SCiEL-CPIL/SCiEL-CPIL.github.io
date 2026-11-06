"""Stronger design: return a known non-2-colorable 5-uniform hypergraph quickly."""

import time
from itertools import combinations

def _check_non_2_colorable_fast(n, edges):
    """Returns True if 2-colorable, False if non-2-colorable."""
    if not edges:
        return True
    
    if n <= 15:
        for mask in range(1 << n):
            valid = True
            for e in edges:
                colors = set((mask >> v) & 1 for v in e)
                if len(colors) == 1:
                    valid = False
                    break
            if valid:
                return True
        return False
    
    # Backtracking for larger n
    color = [-1] * n
    edge_sets = [set(e) for e in edges]
    
    def backtrack(idx):
        if idx == n:
            for e in edge_sets:
                if len(set(color[v] for v in e)) == 1:
                    return False
            return True
        
        for c in [0, 1]:
            color[idx] = c
            contradiction = False
            for e in edge_sets:
                if idx in e:
                    all_assigned = True
                    fc = None
                    for v in e:
                        if color[v] == -1:
                            all_assigned = False
                            break
                        if fc is None:
                            fc = color[v]
                        elif color[v] != fc:
                            break
                    if all_assigned and fc is not None:
                        contradiction = True
                        break
            if not contradiction and backtrack(idx + 1):
                return True
        color[idx] = -1
        return False
    
    return backtrack(0)


def _known_construction():
    """Return a non-2-colorable 5-uniform hypergraph with minimal edges."""
    start_time = time.time()
    deadline = start_time + 105
    
    best_result = None
    
    for n in range(9, 13):
        if time.time() > deadline - 5:
            break
        
        all_edges = [list(e) for e in combinations(range(n), 5)]
        
        if _check_non_2_colorable_fast(n, all_edges):
            continue  # 2-colorable, skip
        
        edges = all_edges[:]
        i = 0
        while i < len(edges):
            if time.time() > deadline:
                break
            
            candidate = edges[:i] + edges[i+1:]
            if not _check_non_2_colorable_fast(n, candidate):
                edges = candidate
            else:
                i += 1
        
        if best_result is None or len(edges) < best_result[1]:
            best_result = (n, len(edges), edges)
    
    if best_result is None:
        # Fallback: complete 5-graph on 9 vertices
        n = 9
        edges = [list(e) for e in combinations(range(n), 5)]
        return {'n': n, 'edges': edges}
    
    n, _, edges = best_result
    return {'n': n, 'edges': edges}


def construct():
    """Return a non-2-colorable 5-uniform hypergraph with minimal edges found."""
    result = _known_construction()
    return result
