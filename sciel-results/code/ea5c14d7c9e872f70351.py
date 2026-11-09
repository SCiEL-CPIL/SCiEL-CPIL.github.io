"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def construct():
    n = 16
    target_edges = 65
    
    # Strategy: QUBO-like formulation with Simulated Annealing
    # Objective: Minimize sum of squared triangle degree sums + penalty for max triangle sum
    # This encourages uniform low degree sums across all triangles.
    
    def compute_triangles(adj):
        """
        Find all triangles in the graph.
        Returns list of (u, v, w) where u < v < w.
        """
        triangles = []
        for u in range(n):
            for v in range(u+1, n):
                if adj[u, v] == 1:
                    for w in range(v+1, n):
                        if adj[u, w] == 1 and adj[v, w] == 1:
                            triangles.append((u, v, w))
        return triangles

    def compute_objective(adj):
        """
        Compute the QUBO objective:
        - Primary: minimize max triangle degree sum
        - Secondary: minimize sum of squares of triangle degree sums (for uniformity)
        - Tertiary: penalty if edge count != target_edges
        """
        triangles = compute_triangles(adj)
        edge_count = int(adj.sum() // 2)
        
        if edge_count != target_edges:
            # Heavy penalty for wrong edge count
            return 10000 + abs(edge_count - target_edges) * 100
        
        if not triangles:
            # No triangles is actually fine for the "max" part, but we need triangles
            # The task says "so it has triangles", so we should have at least one
            # But minimizing max over empty set... let's return a large value
            return 1000
        
        degree_sums = []
        for (u, v, w) in triangles:
            s = adj[u].sum() + adj[v].sum() + adj[w].sum()
            degree_sums.append(s)
        
        max_sum = max(degree_sums)
        sum_sq = sum(s * s for s in degree_sums)
        
        # Weighted combination: primarily minimize max, secondarily minimize variance
        # Use a large weight for max to ensure it's the primary objective
        objective = 100 * max_sum + 0.1 * sum_sq
        
        return objective

    # Initial state: K_{8,8} + edge (0,1)
    adj = np.zeros((n, n), dtype=int)
    for u in range(8):
        for v in range(8, 16):
            adj[u, v] = 1
            adj[v, u] = 1
    adj[0, 1] = 1
    adj[1, 0] = 1
    
    current_edges = int(adj.sum() // 2)
    assert current_edges == 65, f"Initial edges {current_edges} != 65"
    
    current_obj = compute_objective(adj)
    
    # SA Parameters
    T = 8000  # More steps for better exploration
    t_start = 50.0
    t_end = 0.5
    cooling = (t_end / t_start) ** (1.0 / T)
    
    best_adj = adj.copy()
    best_obj = current_obj
    
    # Precompute all possible edges for faster sampling
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    for step in range(T):
        t = t_start * (cooling ** step)
        
        # Move: Swap one existing edge with one non-existing edge
        # This keeps edge count constant
        
        # Get current edge and non-edge lists
        edge_list = [(i, j) for i, j in all_edges if adj[i, j] == 1]
        non_edge_list = [(i, j) for i, j in all_edges if adj[i, j] == 0]
        
        if not edge_list or not non_edge_list:
            continue
        
        a, b = random.choice(edge_list)
        c, d = random.choice(non_edge_list)
        
        # Apply move
        adj[a, b] = 0
        adj[b, a] = 0
        adj[c, d] = 1
        adj[d, c] = 1
        
        new_obj = compute_objective(adj)
        
        # Acceptance criterion
        delta = new_obj - current_obj
        
        if delta < 0:
            current_obj = new_obj
            if new_obj < best_obj:
                best_obj = new_obj
                best_adj = adj.copy()
        else:
            if delta == 0:
                current_obj = new_obj
            else:
                # Use temperature-scaled acceptance
                # Scale delta by a factor to make it comparable to t
                prob = np.exp(-delta / (t * 10))  # Scale factor 10
                if random.random() < prob:
                    current_obj = new_obj
                else:
                    # Revert
                    adj[a, b] = 1
                    adj[b, a] = 1
                    adj[c, d] = 0
                    adj[d, c] = 0
    
    # Extract edges from best_adj
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if best_adj[u, v] == 1:
                edges.append([u, v])
    
    return {'n': n, 'edges': edges}

# EVOLVE-BLOCK-END
