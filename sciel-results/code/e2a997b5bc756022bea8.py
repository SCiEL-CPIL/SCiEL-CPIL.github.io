import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    best_edges = None
    best_n = 0
    
    for n in range(9, 16):
        if time.time() > deadline:
            break
            
        # Start with complete graph
        all_edges = [list(e) for e in combinations(range(n), 5)]
        random.shuffle(all_edges)
        current_edges = all_edges[:]
        
        # Minimize with a few passes
        for _ in range(5):
            if time.time() > deadline:
                break
            random.shuffle(current_edges)
            # Iterate in reverse to allow popping
            for i in range(len(current_edges)-1, -1, -1):
                if time.time() > deadline:
                    break
                edge_to_try = current_edges[i]
                current_edges.pop(i)
                
                # Check 2-colorability
                # If 2-colorable, we MUST have this edge to be non-2-colorable
                if is_2_colorable_fast(n, current_edges):
                    current_edges.insert(i, edge_to_try)
                # Else, keep removed
                
        if len(current_edges) < (best_edges is None or len(current_edges) < len(best_edges)):
            best_edges = current_edges
            best_n = n
            
    if best_edges is None:
        # Fallback
        return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
        
    return {'n': best_n, 'edges': best_edges}

def is_2_colorable_fast(n, edges):
    if not edges:
        return True
        
    degree = [0] * n
    for e in edges:
        for v in e:
            degree[v] += 1
            
    order = sorted(range(n), key=lambda v: degree[v], reverse=True)
    
    color = [-1] * n
    
    def propagate():
        changed = True
        while changed:
            changed = False
            for e in edges:
                c0 = 0
                c1 = 0
                unassigned = []
                for v in e:
                    if color[v] == 0: c0 += 1
                    elif color[v] == 1: c1 += 1
                    else: unassigned.append(v)
                    
                if c0 == 5 or c1 == 5:
                    return False
                
                if len(unassigned) == 1:
                    v = unassigned[0]
                    if c0 == 4:
                        if color[v] == 0: return False
                        if color[v] == -1:
                            color[v] = 1
                            changed = True
                    elif c1 == 4:
                        if color[v] == 1: return False
                        if color[v] == -1:
                            color[v] = 0
                            changed = True
        return True

    def backtrack(idx):
        if idx == n:
            for e in edges:
                c0 = 0
                for v in e:
                    if color[v] == 0: c0 += 1
                if c0 == 5 or c0 == 0:
                    return False
            return True
            
        v = order[idx]
        
        if color[v] != -1:
            return backtrack(idx + 1)
            
        color[v] = 0
        if propagate():
            if backtrack(idx + 1):
                return True
                
        color[v] = 1
        if propagate():
            if backtrack(idx + 1):
                return True
                
        color[v] = -1
        return False

    if not propagate():
        return False
        
    return backtrack(0)
