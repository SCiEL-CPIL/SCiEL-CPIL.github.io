import time
import random
import math
import numpy as np

def construct():
    n = 40
    target_edges = 401
    start_time = time.time()
    deadline = start_time + 105.0  # Safety margin
    
    # Use NumPy adjacency matrix for fast operations
    # adj[u, v] = 1 if edge exists
    adj = np.zeros((n, n), dtype=np.uint8)
    
    # Random initialization: start with random 401-edge graph
    # This avoids the K20,20 local optimum trap
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.shuffle(all_pairs)
    for u, v in all_pairs[:target_edges]:
        adj[u, v] = 1
        adj[v, u] = 1
    
    def compute_books(adj):
        """Compute book size for each edge using matrix multiplication."""
        # Common neighbors for all edge pairs: (adj @ adj) gives count of common neighbors
        # But we only want edges that exist
        common = adj @ adj  # common[u,v] = number of common neighbors of u and v
        # For each edge (u,v), book size = common[u,v]
        # We need max over all edges
        books = np.zeros((n, n), dtype=np.int32)
        for i in range(n):
            for j in range(i+1, n):
                if adj[i, j]:
                    books[i, j] = common[i, j]
                    books[j, i] = common[i, j]
        return books
    
    def get_score(adj):
        """Get maximum book size."""
        common = adj @ adj
        max_book = 0
        for i in range(n):
            for j in range(i+1, n):
                if adj[i, j]:
                    c = common[i, j]
                    if c > max_book:
                        max_book = c
        return max_book
    
    current_score = get_score(adj)
    best_score = current_score
    best_adj = adj.copy()
    
    # Precompute edge list for faster iteration
    def get_edge_list(adj):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i, j]:
                    edges.append((i, j))
        return edges
    
    # Simulated Annealing with NumPy
    T = 3.0
    cooling = 0.9995
    
    while time.time() < deadline:
        # Pick random edge to remove
        edges = get_edge_list(adj)
        if not edges:
            break
        u, v = random.choice(edges)
        
        # Remove edge
        adj[u, v] = 0
        adj[v, u] = 0
        
        # Pick random non-edge to add
        # Find non-edges efficiently
        non_edges = []
        for i in range(n):
            for j in range(i+1, n):
                if not adj[i, j]:
                    non_edges.append((i, j))
        
        if not non_edges:
            # No non-edges, revert
            adj[u, v] = 1
            adj[v, u] = 1
            break
        
        a, b = random.choice(non_edges)
        
        # Add edge
        adj[a, b] = 1
        adj[b, a] = 1
        
        new_score = get_score(adj)
        
        if new_score < current_score:
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_adj = adj.copy()
        else:
            delta = new_score - current_score
            if delta > 0:
                prob = math.exp(-delta / T)
                if random.random() < prob:
                    current_score = new_score
                else:
                    # Revert
                    adj[a, b] = 0
                    adj[b, a] = 0
                    adj[u, v] = 1
                    adj[v, u] = 1
            else:
                current_score = new_score
        
        T *= cooling
        if T < 0.1:
            T = 0.1
    
    # Return best edges
    best_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i, j]:
                best_edges.append([i, j])
    
    return best_edges
