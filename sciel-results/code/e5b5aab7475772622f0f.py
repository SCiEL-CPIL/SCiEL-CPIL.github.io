import time
import random
from itertools import combinations

def is_2_colorable(n, edges):
    colors = [-1] * n
    
    def propagate():
        changed = True
        while changed:
            changed = False
            for e in edges:
                c0 = 0
                c1 = 0
                unassigned = []
                for v in e:
                    if colors[v] == 0:
                        c0 += 1
                    elif colors[v] == 1:
                        c1 += 1
                    else:
                        unassigned.append(v)
                
                if c0 == 4 and unassigned:
                    for v in unassigned:
                        if colors[v] == -1:
                            colors[v] = 1
                            changed = True
                elif c1 == 4 and unassigned:
                    for v in unassigned:
                        if colors[v] == -1:
                            colors[v] = 0
                            changed = True
                            
                if c0 == 5 or c1 == 5:
                    return False
        return True

    def backtrack(idx):
        if idx == n:
            for e in edges:
                c0 = sum(1 for v in e if colors[v] == 0)
                c1 = sum(1 for v in e if colors[v] == 1)
                if c0 == 5 or c1 == 5:
                    return False
            return True
            
        old_color = colors[idx]
        colors[idx] = 0
        if propagate():
            if backtrack(idx + 1):
                return True
        colors[idx] = 1
        if propagate():
            if backtrack(idx + 1):
                return True
        
        colors[idx] = old_color
        return False

    return backtrack(0)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n = 9
    all_edges = list(combinations(range(n), 5))
    
    best_edges = None
    best_count = len(all_edges)
    
    for attempt in range(100):
        if time.time() > deadline:
            break
            
        edges_set = set(all_edges)
        edges_list = list(edges_set)
        random.shuffle(edges_list)
        
        i = 0
        while i < len(edges_list):
            if time.time() > deadline:
                break
            e = edges_list[i]
            edges_set.remove(e)
            current_edges = list(edges_set)
            
            if not is_2_colorable(n, current_edges):
                pass
            else:
                edges_set.add(e)
            
            i += 1
            
        if len(edges_set) < best_count:
            best_count = len(edges_set)
            best_edges = list(edges_set)
            
            if best_count <= 51:
                break
            
    if best_edges is None:
        return {'n': n, 'edges': [list(e) for e in all_edges]}
    else:
        return {'n': n, 'edges': [list(e) for e in best_edges]}
