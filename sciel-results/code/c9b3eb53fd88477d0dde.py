# EVOLVE-BLOCK-START
import random
import time
from itertools import combinations

def _is_2_colorable(edges, n):
    """
    Check if the hypergraph is 2-colorable.
    Returns True if 2-colorable, False otherwise.
    """
    vertex_edges = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            vertex_edges[v].append(i)
    
    colors = [-1] * n
    
    def propagate(v, c):
        if colors[v] != -1:
            return colors[v] == c
        
        colors[v] = c
        
        for e_idx in vertex_edges[v]:
            e = edges[e_idx]
            count_c = 0
            count_1m = 0
            unassigned = -1
            
            for u in e:
                if colors[u] == c:
                    count_c += 1
                elif colors[u] == 1 - c:
                    count_1m += 1
                else:
                    unassigned = u
            
            if count_1m == 4:
                if unassigned != -1:
                    if not propagate(unassigned, c):
                        return False
            elif count_c == 4:
                if unassigned != -1:
                    if not propagate(unassigned, 1 - c):
                        return False
            
            if count_c == 5 or count_1m == 5:
                return False
                
        return True

    def backtrack():
        best_v = -1
        best_deg = -1
        for v in range(n):
            if colors[v] == -1:
                deg = len(vertex_edges[v])
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
        
        if best_v == -1:
            return True
        
        old_colors = colors[:]
        if propagate(best_v, 0):
            if backtrack():
                return True
        colors[:] = old_colors
        
        old_colors = colors[:]
        if propagate(best_v, 1):
            if backtrack():
                return True
        
        return False

    return backtrack()

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_edges = None
    best_count = float('inf')
    best_n = 0
    
    # Try small n first, as they saturate faster
    for n in range(10, 26):
        if time.time() > deadline - 10.0:
            break
            
        # Run a few trials for this n
        for trial in range(20):
            if time.time() > deadline - 5.0:
                break
                
            all_edges = list(combinations(range(n), 5))
            random.shuffle(all_edges)
            
            current_edges = []
            for e in all_edges:
                current_edges.append(e)
                
                if not _is_2_colorable(current_edges, n):
                    count = len(current_edges)
                    if count < best_count:
                        best_count = count
                        best_edges = [list(e) for e in current_edges]
                        best_n = n
                    break

    if best_edges is None:
        return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
        
    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
