"""C6-free graph on 40 vertices: maximize edges via Simulated Annealing."""
import random
import time

def _has_c6(adj, n):
    """Check if graph has a 6-cycle. adj is list of sets."""
    for s in range(n):
        for t in adj[s]:
            for u in adj[t]:
                if u == s:
                    continue
                for v in adj[u]:
                    if v == s or v == t:
                        continue
                    for w in adj[v]:
                        if w == s or w == t or w == u:
                            continue
                        for x in adj[w]:
                            if x == s or x == t or x == u or x == v:
                                continue
                            if s in adj[x]:
                                return True
    return False

def construct():
    n = 40
    start_time = time.time()
    deadline = start_time + 145.0
    
    random.seed(42)
    
    # Start with 8 copies of K5 (104 edges)
    adj = [set() for _ in range(n)]
    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                u = 5 * k + a
                v = 5 * k + b
                adj[u].add(v)
                adj[v].add(u)
    
    current_m = sum(len(s) for s in adj) // 2
    best_m = current_m
    best_adj = [set(s) for s in adj]
    
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    T = 50.0
    T_min = 0.05
    alpha = 0.9999
    
    while T > T_min:
        if time.time() > deadline:
            break
        
        i, j = random.choice(all_edges)
        present = j in adj[i]
        
        if present:
            adj[i].discard(j)
            adj[j].discard(i)
            current_m -= 1
        else:
            adj[i].add(j)
            adj[j].add(i)
            current_m += 1
            if _has_c6(adj, n):
                adj[i].discard(j)
                adj[j].discard(i)
                current_m -= 1
                continue
        
        if current_m > best_m:
            best_m = current_m
            best_adj = [set(s) for s in adj]
        
        T *= alpha
    
    result = []
    for i in range(n):
        for j in best_adj[i]:
            if i < j:
                result.append([i, j])
    
    return result
