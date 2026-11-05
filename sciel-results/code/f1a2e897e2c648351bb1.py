import random
import time
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
                unk = None
                for v in e:
                    cv = colors[v]
                    if cv == 0:
                        c0 += 1
                    elif cv == 1:
                        c1 += 1
                    else:
                        if unk is None:
                            unk = v
                if c0 == 5 or c1 == 5:
                    return False
                if unk is not None:
                    if c0 == 4:
                        colors[unk] = 1
                        changed = True
                    elif c1 == 4:
                        colors[unk] = 0
                        changed = True
        return True

    def backtrack(idx):
        if idx == n:
            return True
        if colors[idx] != -1:
            return backtrack(idx + 1)
        
        old = colors[idx]
        colors[idx] = 0
        if propagate():
            if backtrack(idx + 1):
                return True
        colors[idx] = 1
        if propagate():
            if backtrack(idx + 1):
                return True
        colors[idx] = old
        return False

    if not propagate():
        return False
    return backtrack(0)

def minimize_hypergraph(n, edges):
    edges = list(edges)
    improved = True
    while improved:
        improved = False
        random.shuffle(edges)
        for i in range(len(edges)):
            if not edges: break
            candidate = edges[:i] + edges[i+1:]
            if not is_2_colorable(n, candidate):
                edges = candidate
                improved = True
                break
    return edges

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_edges = None
    best_count = float('inf')
    
    all_edges_cache = {}
    for n in range(9, 13):
        all_edges_cache[n] = [list(e) for e in combinations(range(n), 5)]
        
    while time.time() < deadline:
        n = 9
        # Random search
        # Bias towards smaller e
        m = random.randint(25, 45)
        random_edges = random.sample(all_edges_cache[n], m)
        
        if not is_2_colorable(n, random_edges):
            minimized = minimize_hypergraph(n, random_edges)
            if len(minimized) < best_count:
                best_count = len(minimized)
                best_edges = minimized
                
        if best_edges is not None:
             # Local search on best
             if random.random() < 0.2:
                 idx = random.randint(0, len(best_edges) - 1)
                 candidate = best_edges[:idx] + best_edges[idx+1:]
                 if not is_2_colorable(n, candidate):
                     best_edges = candidate
                     best_count = len(best_edges)
                     best_edges = minimize_hypergraph(n, best_edges)
                     best_count = len(best_edges)

    if best_edges is None:
        return {'n': 9, 'edges': all_edges_cache[9]}
    
    return {'n': 9, 'edges': best_edges}
