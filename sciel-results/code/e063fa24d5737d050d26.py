import time
import random
from itertools import combinations

def _is_not_2_colorable(n, edges):
    """
    Returns True if the hypergraph is NOT 2-colorable.
    """
    if not edges:
        return False
    
    edge_set = [frozenset(e) for e in edges]
    
    def backtrack(vertex, colors):
        if vertex == n:
            # Final check (should be redundant if pruning works, but safe)
            for e in edge_set:
                vals = [colors[u] for u in e]
                if len(set(vals)) == 1:
                    return False
            return True
        
        # Try color 0
        colors[vertex] = 0
        valid = True
        for e in edge_set:
            if vertex in e:
                if all(colors[u] is not None for u in e):
                    vals = [colors[u] for u in e]
                    if len(set(vals)) == 1:
                        valid = False
                        break
        if valid:
            if backtrack(vertex + 1, colors):
                return True
        
        # Try color 1
        colors[vertex] = 1
        valid = True
        for e in edge_set:
            if vertex in e:
                if all(colors[u] is not None for u in e):
                    vals = [colors[u] for u in e]
                    if len(set(vals)) == 1:
                        valid = False
                        break
        if valid:
            if backtrack(vertex + 1, colors):
                return True
        
        colors[vertex] = None
        return False

    colors = [None] * n
    colors[0] = 0
    if backtrack(1, colors):
        return False
    else:
        return True

def _minimize_edges(n, initial_edges, deadline):
    current_edges = list(initial_edges)
    
    while time.time() < deadline:
        removed = False
        indices = list(range(len(current_edges)))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() >= deadline:
                break
                
            e = current_edges[idx]
            candidate = [ed for i, ed in enumerate(current_edges) if i != idx]
            
            if _is_not_2_colorable(n, candidate):
                current_edges = candidate
                removed = True
                break
        
        if not removed:
            break
            
    return current_edges

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    best_edges = None
    best_n = None
    best_count = float('inf')
    
    for n in [9, 10, 11]:
        if time.time() > deadline:
            break
            
        all_edges = [list(e) for e in combinations(range(n), 5)]
        
        # Check if the complete graph is non-2-colorable (it should be for n>=9)
        if not _is_not_2_colorable(n, all_edges):
            continue
            
        min_edges = _minimize_edges(n, all_edges, deadline)
        
        if len(min_edges) < best_count:
            best_count = len(min_edges)
            best_edges = min_edges
            best_n = n
            
    if best_edges is None:
        return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
        
    return {'n': best_n, 'edges': best_edges}
