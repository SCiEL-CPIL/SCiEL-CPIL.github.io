import random
import sys

def construct():
    n = 40
    random.seed(42)
    
    adj_mask = [0] * n
    edges = set()
    
    def add_edge(u, v):
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        edges.add((u, v) if u < v else (v, u))
        
    def remove_edge(u, v):
        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)
        edges.discard((u, v) if u < v else (v, u))

    def creates_c6(u, v):
        # Check if adding (u,v) creates a C6
        # Equivalent to: is there a path of length 4 from u to v in current graph?
        mask_u = adj_mask[u]
        mask_v = adj_mask[v]
        
        # a in N(u)
        for a in range(n):
            if not (mask_u & (1 << a)): continue
            if a == v: continue
            
            mask_a = adj_mask[a]
            for b in range(n):
                if not (mask_a & (1 << b)): continue
                if b == u or b == a: continue
                
                mask_b = adj_mask[b]
                for c in range(n):
                    if not (mask_b & (1 << c)): continue
                    if c == u or c == a or c == b: continue
                    
                    mask_c = adj_mask[c]
                    forbidden = (1 << u) | (1 << a) | (1 << b) | (1 << c) | (1 << v)
                    if (mask_c & mask_v) & ~forbidden:
                        return True
        return False

    # Initialize with random graph
    # p=0.5 gives ~380 edges, too many.
    # p=0.2 gives ~150 edges.
    # Let's try p=0.25
    for u in range(n):
        for v in range(u+1, n):
            if random.random() < 0.25:
                add_edge(u, v)
                
    # Remove C6s
    changed = True
    while changed:
        changed = False
        edge_list = list(edges)
        random.shuffle(edge_list)
        for u, v in edge_list:
            if creates_c6(u, v):
                remove_edge(u, v)
                changed = True

    # Greedy Add
    improved = True
    while improved:
        improved = False
        candidates = [(u, v) for u in range(n) for v in range(u+1, n) if not (adj_mask[u] & (1 << v))]
        random.shuffle(candidates)
        for u, v in candidates:
            if not creates_c6(u, v):
                add_edge(u, v)
                improved = True

    # One restart if time permits? 
    # Let's just do one more pass of swapping?
    # No, let's just return the result.
    # To improve score, maybe do 2 random inits and keep the best.
    
    best_edges = len(edges)
    best_adj_mask = adj_mask[:]
    best_edges_set = set(edges)
    
    # Second run
    adj_mask = [0] * n
    edges = set()
    for u in range(n):
        for v in range(u+1, n):
            if random.random() < 0.25:
                add_edge(u, v)
                
    changed = True
    while changed:
        changed = False
        edge_list = list(edges)
        random.shuffle(edge_list)
        for u, v in edge_list:
            if creates_c6(u, v):
                remove_edge(u, v)
                changed = True

    improved = True
    while improved:
        improved = False
        candidates = [(u, v) for u in range(n) for v in range(u+1, n) if not (adj_mask[u] & (1 << v))]
        random.shuffle(candidates)
        for u, v in candidates:
            if not creates_c6(u, v):
                add_edge(u, v)
                improved = True

    if len(edges) > best_edges:
        best_edges = len(edges)
        best_adj_mask = adj_mask[:]
        best_edges_set = set(edges)

    return [list(e) for e in best_edges_set]
