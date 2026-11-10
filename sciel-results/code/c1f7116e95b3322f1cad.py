import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115

    def is_2_colorable(n, edges):
        if not edges:
            return True
        
        edge_list = edges
        
        def solve(colors):
            changed = True
            while changed:
                changed = False
                for e in edge_list:
                    zeros = 0
                    ones = 0
                    unassigned = -1
                    unassigned_count = 0
                    
                    for v in e:
                        c = colors[v]
                        if c == 0:
                            zeros += 1
                        elif c == 1:
                            ones += 1
                        else:
                            unassigned = v
                            unassigned_count += 1
                    
                    if unassigned_count == 0:
                        if zeros == len(e) or ones == len(e):
                            return False
                    elif unassigned_count == 1:
                        if zeros == len(e) - 1:
                            if colors[unassigned] != -1:
                                if colors[unassigned] != 1:
                                    return False
                            else:
                                colors[unassigned] = 1
                                changed = True
                        elif ones == len(e) - 1:
                            if colors[unassigned] != -1:
                                if colors[unassigned] != 0:
                                    return False
                            else:
                                colors[unassigned] = 0
                                changed = True
            
            var = -1
            for i in range(n):
                if colors[i] == -1:
                    var = i
                    break
            
            if var == -1:
                return True
            
            colors[var] = 0
            if solve(colors):
                return True
            
            colors[var] = 1
            if solve(colors):
                return True
            
            colors[var] = -1
            return False
        
        colors = [-1] * n
        return solve(colors)

    n = 9
    all_edges = [list(e) for e in combinations(range(n), 5)]
    
    best_edges = None
    best_size = float('inf')
    best_n = n

    for attempt in range(50):
        if time.time() > deadline:
            break
            
        edges_set = set(tuple(e) for e in all_edges)
        current_edges = list(edges_set)
        random.shuffle(current_edges)
        
        active_edges = set(current_edges)
        
        for e in current_edges:
            if time.time() > deadline:
                break
                
            active_edges.remove(e)
            remaining = [list(e) for e in active_edges]
            
            if is_2_colorable(n, remaining):
                active_edges.add(e)
            else:
                pass
        
        if len(active_edges) < best_size:
            best_size = len(active_edges)
            best_edges = [list(e) for e in active_edges]
            best_n = n
            
        if best_size <= 51:
            break

    if best_edges is None:
        return {'n': n, 'edges': all_edges}
        
    return {'n': best_n, 'edges': best_edges}
