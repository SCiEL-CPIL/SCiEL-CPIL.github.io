import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    n = 16
    all_edges = list(combinations(range(n), 2))
    
    def is_triangle_free_masks(masks):
        """Check if graph with given adjacency bitmasks is triangle-free."""
        for u in range(n):
            for v in range(u + 1, n):
                if masks[u] & (1 << v):
                    common = masks[u] & masks[v]
                    common &= ~(1 << u)
                    common &= ~(1 << v)
                    if common:
                        return False
        return True

    def is_3_colorable_masks(masks, max_time=None):
        """Check if graph is 3-colorable using backtracking with bitmasks."""
        neighbors = [[] for _ in range(n)]
        for v in range(n):
            m = masks[v]
            while m:
                lsb = m & -m
                u = lsb.bit_length() - 1
                neighbors[v].append(u)
                m ^= lsb
        
        order = sorted(range(n), key=lambda v: -len(neighbors[v]))
        color = [-1] * n
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            if idx == n:
                return True
            
            v = order[idx]
            forbidden = 0
            for u in neighbors[v]:
                if color[u] != -1:
                    forbidden |= (1 << color[u])
            
            for c in range(3):
                if not (forbidden & (1 << c)):
                    color[v] = c
                    if backtrack(idx + 1):
                        return True
                    color[v] = -1
            return False
        
        return backtrack(0)

    def get_grotzsch_masks():
        edges = []
        for i in range(5):
            edges.append((i, (i + 1) % 5))
        for i in range(5):
            u, v = i, (i + 1) % 5
            edges.append((u, 5 + v))
            edges.append((v, 5 + u))
        for i in range(5):
            edges.append((10, 5 + i))
        
        masks = [0] * n
        for u, v in edges:
            masks[u] |= (1 << v)
            masks[v] |= (1 << u)
        return masks

    best_masks = get_grotzsch_masks()
    best_count = sum(bin(m).count('1') for m in best_masks) // 2
    
    candidates = all_edges[:]
    
    def try_add_edges(current_masks, num_trials=10000):
        nonlocal best_masks, best_count
        current = current_masks[:]
        random.shuffle(candidates)
        
        for edge in candidates:
            if time.time() > deadline:
                break
            u, v = edge
            
            if current[u] & (1 << v):
                continue
            
            current[u] |= (1 << v)
            current[v] |= (1 << u)
            
            common = current[u] & current[v]
            common &= ~(1 << u)
            common &= ~(1 << v)
            if common:
                current[u] &= ~(1 << v)
                current[v] &= ~(1 << u)
                continue
            
            if is_3_colorable_masks(current):
                current[u] &= ~(1 << v)
                current[v] &= ~(1 << u)
                continue
            
            count = sum(bin(m).count('1') for m in current) // 2
            if count > best_count:
                best_masks = current[:]
                best_count = count
        
        return current

    for restart in range(200):
        if time.time() > deadline:
            break
        
        current = best_masks[:]
        
        if restart > 0 and random.random() < 0.5:
            num_remove = random.randint(1, 3)
            edges_to_remove = []
            for u in range(n):
                for v in range(u + 1, n):
                    if current[u] & (1 << v):
                        edges_to_remove.append((u, v))
            
            if len(edges_to_remove) >= num_remove:
                to_remove = random.sample(edges_to_remove, num_remove)
                for u, v in to_remove:
                    current[u] &= ~(1 << v)
                    current[v] &= ~(1 << u)
        
        try_add_edges(current)

    best_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_masks[u] & (1 << v):
                best_edges.append([u, v])
                
    return best_edges
