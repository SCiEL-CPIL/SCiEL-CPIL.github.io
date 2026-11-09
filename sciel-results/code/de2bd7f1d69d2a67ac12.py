import time
import random

def construct():
    start = time.time()
    deadline = 135.0
    
    n = 40
    target = 401
    
    adj = [0] * n
    edges = set()
    
    def add(u, v):
        if u == v: return False
        if u > v: u, v = v, u
        if (u, v) in edges: return False
        edges.add((u, v))
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        return True
    
    def remove(u, v):
        if u == v: return False
        if u > v: u, v = v, u
        if (u, v) not in edges: return False
        edges.remove((u, v))
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        return True
    
    def max_book():
        best = 0
        for u, v in edges:
            cnt = bin(adj[u] & adj[v]).count('1')
            if cnt > best:
                best = cnt
        return best
    
    # Initialize: K20,20 + 1 intra edge
    for a in range(20):
        for b in range(20, 40):
            add(a, b)
    add(0, 1)
    
    current_max = max_book()
    best_max = current_max
    best_edges = set(edges)
    
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    while time.time() < deadline:
        u, v = random.choice(list(edges))
        remove(u, v)
        
        found = False
        for _ in range(200):
            a, b = random.choice(all_pairs)
            if (a, b) not in edges:
                add(a, b)
                nm = max_book()
                if nm < current_max:
                    current_max = nm
                    found = True
                    break
                elif nm == current_max and random.random() < 0.3:
                    found = True
                    break
                remove(a, b)
        
        if not found:
            add(u, v)
        
        if current_max < best_max:
            best_max = current_max
            best_edges = set(edges)
            if best_max <= 7:
                break
    
    edges = best_edges
    return [[u, v] for u, v in edges]
# EVOLVE-BLOCK-END
