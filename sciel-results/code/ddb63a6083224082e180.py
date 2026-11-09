import numpy as np
import time
import random

def construct():
    n = 40
    m = 401
    t_start = time.time()
    deadline = t_start + 115.0

    # Precompute indices for edges to speed up masking?
    # Or just use numpy boolean indexing.
    
    def get_max_book(adj):
        # adj is 40x40 numpy array of 0s and 1s
        # common neighbors for all pairs
        common = np.dot(adj, adj.T)
        # Mask diagonal to 0 to avoid picking up degrees
        np.fill_diagonal(common, 0)
        # We only care about edges where adj == 1
        # Boolean mask
        mask = (adj == 1)
        if not np.any(mask):
            return 0
        return np.max(common[mask])

    def random_graph():
        edges = set()
        while len(edges) < m:
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u != v:
                if u > v: u, v = v, u
                edges.add((u, v))
        adj = np.zeros((n, n), dtype=np.uint8)
        for u, v in edges:
            adj[u, v] = 1
            adj[v, u] = 1
        return adj, list(edges)

    def bipartite_plus_k(k):
        adj = np.zeros((n, n), dtype=np.uint8)
        for i in range(20):
            for j in range(20, 40):
                adj[i, j] = 1
                adj[j, i] = 1
        edges = [(i, j) for i in range(20) for j in range(20, 40)]
        
        intra_edges = set()
        while len(intra_edges) < k:
            u = random.randint(0, 19)
            v = random.randint(0, 19)
            if u != v:
                if u > v: u, v = v, u
                if (u, v) not in intra_edges:
                    intra_edges.add((u, v))
        for u, v in intra_edges:
            adj[u, v] = 1
            adj[v, u] = 1
            edges.append((u, v))
        return adj, edges

    best_score = 100
    best_adj = None
    best_edges = None

    # Try a few different initializations
    initializations = [
        bipartite_plus_k(1),
        random_graph(),
        random_graph()
    ]
    
    for init_adj, init_edges in initializations:
        current_adj = init_adj.copy()
        current_edges = init_edges[:]
        current_score = get_max_book(current_adj)
        
        if current_score < best_score:
            best_score = current_score
            best_adj = current_adj.copy()
            best_edges = current_edges[:]
            if best_score == 7:
                return [[u, v] for u, v in best_edges]

        # Simulated Annealing
        T = 10.0
        cooling_rate = 0.999
        run_deadline = time.time() + 30.0 # 30s per run, total ~90s
        
        while time.time() < run_deadline and time.time() < deadline:
            # Pick random edge to remove
            idx = random.randint(0, len(current_edges) - 1)
            out_u, out_v = current_edges[idx]
            
            # Pick random non-edge to add
            while True:
                in_u = random.randint(0, n-1)
                in_v = random.randint(0, n-1)
                if in_u == in_v: continue
                if in_u > in_v: in_u, in_v = in_v, in_u
                if current_adj[in_u, in_v] == 0:
                    break

            # Apply swap
            current_adj[out_u, out_v] = 0
            current_adj[out_v, out_u] = 0
            current_edges[idx] = (in_u, in_v)
            
            current_adj[in_u, in_v] = 1
            current_adj[in_v, in_u] = 1

            new_score = get_max_book(current_adj)
            delta = new_score - current_score
            
            if delta < 0:
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_adj = current_adj.copy()
                    best_edges = current_edges[:]
                    if best_score == 7:
                        return [[u, v] for u, v in best_edges]
            else:
                # Accept with probability exp(-delta / T)
                if random.random() < np.exp(-delta / T):
                    current_score = new_score
                else:
                    # Revert
                    current_adj[in_u, in_v] = 0
                    current_adj[in_v, in_u] = 0
                    current_edges[idx] = (out_u, out_v)
                    
                    current_adj[out_u, out_v] = 1
                    current_adj[out_v, out_u] = 1
            
            T *= cooling_rate
            if T < 0.5:
                T = 0.5

    return [[u, v] for u, v in best_edges]
