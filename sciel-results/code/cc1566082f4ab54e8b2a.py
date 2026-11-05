"""Rewritten program with simulated annealing to minimize max triangle degree sum."""
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n = 16
    target_edges = 65
    
    def get_degrees(adj):
        return [adj[i].sum() for i in range(n)]
    
    def compute_max_triangle_degree_sum(adj, degrees):
        """Compute max degree sum over all triangles."""
        max_sum = 0
        for i in range(n):
            for j in range(i+1, n):
                if adj[i, j]:
                    # Find common neighbors
                    common = adj[i] & adj[j]
                    for k in range(j+1, n):
                        if common[k]:
                            s = degrees[i] + degrees[j] + degrees[k]
                            if s > max_sum:
                                max_sum = s
        return max_sum
    
    # Initialize with K_{8,8} + one edge
    def make_initial_adj():
        adj = np.zeros((n, n), dtype=np.int8)
        # Bipartite between 0-7 and 8-15
        for u in range(8):
            for v in range(8, 16):
                adj[u, v] = 1
                adj[v, u] = 1
        # Add one edge within part A
        adj[0, 1] = 1
        adj[1, 0] = 1
        return adj
    
    adj = make_initial_adj()
    degrees = get_degrees(adj)
    best_score = compute_max_triangle_degree_sum(adj, degrees)
    best_adj = adj.copy()
    best_degrees = degrees.copy()
    
    # Simulated annealing
    temp = 5.0
    cooling = 0.997
    min_temp = 0.01
    
    # Precompute all possible edge pairs for moves
    all_edges = []
    for i in range(n):
        for j in range(i+1, n):
            all_edges.append((i, j))
    
    while temp > min_temp and time.time() < deadline:
        # Choose a move: swap an existing edge with a non-existing edge
        # Pick a random existing edge to remove
        existing_edges = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i, j]]
        if not existing_edges:
            break
        
        remove_edge = random.choice(existing_edges)
        
        # Pick a random non-existing edge to add
        non_existing = [(i, j) for i in range(n) for j in range(i+1, n) if not adj[i, j]]
        if not non_existing:
            break
        add_edge = random.choice(non_existing)
        
        # Apply move
        old_adj = adj.copy()
        adj[remove_edge[0], remove_edge[1]] = 0
        adj[remove_edge[1], remove_edge[0]] = 0
        adj[add_edge[0], add_edge[1]] = 1
        adj[add_edge[1], add_edge[0]] = 1
        
        new_degrees = get_degrees(adj)
        new_score = compute_max_triangle_degree_sum(adj, new_degrees)
        
        delta = new_score - best_score
        
        if delta < 0 or random.random() < np.exp(-delta / max(temp, 1e-10)):
            degrees = new_degrees
            if new_score < best_score:
                best_score = new_score
                best_adj = adj.copy()
                best_degrees = degrees.copy()
        else:
            # Revert
            adj = old_adj
            degrees = get_degrees(adj)
        
        temp *= cooling
    
    # Build edge list from best_adj
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i, j]:
                edges.append([i, j])
    
    return {'n': n, 'edges': edges}
