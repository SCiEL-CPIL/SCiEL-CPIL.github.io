import random
import time
from itertools import combinations

def is_2_colorable(edges, n):
    """
    Check if the hypergraph is 2-colorable using backtracking with unit propagation.
    Returns True if 2-colorable, False otherwise.
    """
    if not edges:
        return True
    
    m = len(edges)
    
    # Build incidence
    vertex_edges = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            vertex_edges[v].append(i)
    
    colors = [-1] * n
    
    def propagate(v, c):
        """Assign color c to vertex v. Returns False on contradiction."""
        if colors[v] != -1:
            return colors[v] == c
        
        colors[v] = c
        
        for e_idx in vertex_edges[v]:
            e = edges[e_idx]
            c0 = sum(1 for u in e if colors[u] == 0)
            c1 = sum(1 for u in e if colors[u] == 1)
            
            if c0 == 5 or c1 == 5:
                return False
            
            if c1 == 4:
                unassigned = [u for u in e if colors[u] == -1]
                if unassigned and not propagate(unassigned[0], 0):
                    return False
            elif c0 == 4:
                unassigned = [u for u in e if colors[u] == -1]
                if unassigned and not propagate(unassigned[0], 1):
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
        
        old = colors[:]
        if propagate(best_v, 0):
            if backtrack():
                return True
        colors[:] = old
        
        old = colors[:]
        if propagate(best_v, 1):
            if backtrack():
                return True
        
        return False
    
    return backtrack()

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Start with complete 5-uniform on 9 vertices: 126 edges
    best_n = 9
    best_edges = [list(e) for e in combinations(range(9), 5)]
    best_count = len(best_edges)
    
    def try_remove_edges(edges, n, time_limit):
        """Greedy edge removal: remove edge if hypergraph remains non-2-colorable."""
        edges = [list(e) for e in edges]
        changed = True
        while changed and time.time() < time_limit:
            changed = False
            for i in range(len(edges)):
                if time.time() > time_limit:
                    break
                remaining = edges[:i] + edges[i+1:]
                if not is_2_colorable(remaining, n):
                    edges = remaining
                    changed = True
                    break
        return edges
    
    # Phase 1: Reduce the complete 9-vertex hypergraph
    if time.time() < deadline - 40:
        reduced = try_remove_edges(best_edges, best_n, time_limit=20.0)
        if len(reduced) < best_count:
            best_count = len(reduced)
            best_edges = reduced
    
    # Phase 2: Try other n values with random initialization + reduction
    for n in [10, 11, 12]:
        if time.time() > deadline - 20:
            break
        
        all_edges = list(combinations(range(n), 5))
        random.shuffle(all_edges)
        
        # Start with a dense random subset
        subset = [list(e) for e in all_edges[:min(len(all_edges), 80)]]
        
        if not is_2_colorable(subset, n):
            if time.time() < deadline - 15:
                reduced = try_remove_edges(subset, n, time_limit=10.0)
                if len(reduced) < best_count:
                    best_count = len(reduced)
                    best_n = n
                    best_edges = reduced
    
    # Phase 3: More random restarts on n=9
    if time.time() < deadline - 10:
        for trial in range(10):
            if time.time() > deadline - 10:
                break
            
            all_9 = list(combinations(range(9), 5))
            random.shuffle(all_9)
            
            current = []
            for e in all_9:
                if len(current) >= best_count:
                    break
                current.append(list(e))
                if not is_2_colorable(current, 9):
                    break
            
            if len(current) < best_count and not is_2_colorable(current, 9):
                reduced = try_remove_edges(current, 9, time_limit=5.0)
                if len(reduced) < best_count:
                    best_count = len(reduced)
                    best_n = 9
                    best_edges = reduced
    
    return {'n': best_n, 'edges': best_edges}
