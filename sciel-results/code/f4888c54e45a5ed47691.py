import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n = 16
    target_edges = 65
    
    # Precompute all triples
    triples = [(i, j, k) for i in range(n) for j in range(i+1, n) for k in range(j+1, n)]
    
    def make_initial_adj():
        adj = np.zeros((n, n), dtype=np.int8)
        for a in range(6):
            for b in range(6, 16):
                adj[a, b] = 1
                adj[b, a] = 1
        matching_pairs = [(6,7), (8,9), (10,11), (12,13), (14,15)]
        for u, v in matching_pairs:
            adj[u, v] = 1
            adj[v, u] = 1
        return adj
    
    def compute_score(adj):
        deg = adj.sum(axis=1)
        max_sum = 0
        for i, j, k in triples:
            if adj[i,j] and adj[j,k] and adj[i,k]:
                s = deg[i] + deg[j] + deg[k]
                if s > max_sum:
                    max_sum = s
        return max_sum
    
    best_score = float('inf')
    best_adj = None
    
    # Try multiple restarts
    for seed in range(300):
        if time.time() > deadline - 2:
            break
        
        random.seed(seed)
        adj = make_initial_adj()
        
        # Build edge_set
        edge_set = set()
        for i in range(n):
            for j in range(i+1, n):
                if adj[i,j]:
                    edge_set.add((i,j))
        
        T = 10.0
        cooling = 0.99995
        
        while time.time() < deadline - 2:
            # Pick a random edge to remove
            u, v = random.choice(list(edge_set))
            
            # Pick a random non-edge to add
            while True:
                w = random.randint(0, n-1)
                x = random.randint(0, n-1)
                if w == x:
                    continue
                if w > x:
                    w, x = x, w
                if (w, x) in edge_set:
                    continue
                if (w, x) == (u, v):
                    continue
                break
            
            # Try the swap
            adj[u, v] = 0
            adj[v, u] = 0
            adj[w, x] = 1
            adj[x, w] = 1
            edge_set.remove((u, v))
            edge_set.add((w, x))
            
            new_score = compute_score(adj)
            
            if new_score < best_score:
                best_score = new_score
                best_adj = adj.copy()
            elif new_score > best_score:
                # Accept with probability exp(-(new_score - best_score) / T)
                if random.random() >= np.exp(-(new_score - best_score) / max(T, 1e-10)):
                    # Reject: undo
                    adj[w, x] = 0
                    adj[x, w] = 0
                    adj[u, v] = 1
                    adj[v, u] = 1
                    edge_set.remove((w, x))
                    edge_set.add((u, v))
            
            T *= cooling
        
        if best_score <= 23:
            break
    
    # Reconstruct edges from best_adj
    if best_adj is None:
        best_adj = make_initial_adj()
    
    final_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i, j]:
                final_edges.append([i, j])
    
    return {'n': n, 'edges': final_edges}
