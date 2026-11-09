import numpy as np
import time
import random

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Heuristic matrix for ant colony optimization of capacitated vehicle routing.
    
    Args:
        distance_matrix: (n, n) matrix of Euclidean distances
        coordinates: (n, 2) array of coordinates in unit square
        demands: (n,) array of demands (node 0 is depot with demand 0)
        capacity: vehicle capacity (50)
    
    Returns:
        (n, n) edge-attractiveness matrix where larger values are more attractive.
        Non-positive entries become 1e-9.
    """
    d = np.asarray(distance_matrix, dtype=float)
    coords = np.asarray(coordinates, dtype=float)
    demands = np.asarray(demands, dtype=float)
    n = d.shape[0]
    
    # Ensure shape
    if d.shape[0] != d.shape[1]:
        return np.ones((n, n))
    
    # Start with base inverse distance heuristic
    heuristic = np.zeros_like(d)
    mask = d > 1e-12
    heuristic[mask] = 1.0 / (d[mask] ** 2.5)
    
    # Demand factor: favor edges to high-demand nodes
    demand_factor = 1.0 + demands / capacity
    heuristic = heuristic * demand_factor[np.newaxis, :]
    
    # Handle non-positive or non-finite values
    heuristic[~np.isfinite(heuristic)] = 1e-9
    heuristic[heuristic <= 0] = 1e-9
    
    # Multi-start Clarke-Wright Savings to identify good edges
    deadline = time.time() + 280  # Safety margin before 330s limit
    
    # Compute savings: s(i,j) = d(0,i) + d(0,j) - d(i,j)
    depot = 0
    
    # Calculate savings for all pairs of customers
    savings = np.zeros((n, n))
    for i in range(1, n):
        for j in range(i+1, n):
            s = d[depot, i] + d[depot, j] - d[i, j]
            savings[i, j] = s
            savings[j, i] = s
    
    # Sort savings in descending order
    pairs = [(savings[i, j], i, j) for i in range(1, n) for j in range(i+1, n) if savings[i, j] > 0]
    pairs.sort(reverse=True)
    
    # Multi-start Clarke-Wright Savings with capacity constraint
    # Run multiple times with different random perturbations to get diverse solutions
    edge_weights = {}  # (i, j) -> count of how many solutions include this edge
    
    num_starts = 10
    rng = random.Random(42)
    
    for start in range(num_starts):
        if time.time() > deadline:
            break
        
        # For each start, slightly perturb the savings to get different solutions
        perturbed_pairs = []
        for s, i, j in pairs:
            # Add small random noise to savings
            noisy_s = s * (1.0 + rng.uniform(-0.05, 0.05))
            perturbed_pairs.append((noisy_s, i, j))
        perturbed_pairs.sort(reverse=True)
        
        # Union-Find with capacity constraint
        parent = list(range(n))
        rank = [0] * n
        route_load = [0.0] * n
        
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        
        def union(x, y):
            rx, ry = find(x), find(y)
            if rx == ry:
                return False
            if rank[rx] < rank[ry]:
                rx, ry = ry, rx
            parent[ry] = rx
            if rank[rx] == rank[ry]:
                rank[rx] += 1
            route_load[rx] = route_load[ry] + demands[rx] + demands[ry]
            return True
        
        # Initialize: each customer is its own route
        for i in range(1, n):
            route_load[i] = demands[i]
        
        # Process savings in order
        for s, i, j in perturbed_pairs:
            if time.time() > deadline:
                break
            
            ri, rj = find(i), find(j)
            if ri == rj:
                continue
            
            # Check capacity constraint
            new_load = route_load[ri] + route_load[rj]
            if new_load > capacity:
                continue
            
            if union(ri, rj):
                key = (min(i, j), max(i, j))
                edge_weights[key] = edge_weights.get(key, 0) + 1
    
    # Boost heuristic for good edges identified by multi-start Savings
    # Use a smooth weighting based on frequency
    max_count = max(edge_weights.values()) if edge_weights else 1
    
    for (i, j), count in edge_weights.items():
        # Smooth boost: edges that appear more often get higher boost
        # Use log-sum-exp style smooth max
        boost = 1.0 + 4.0 * (count / max_count)
        heuristic[i, j] *= boost
        heuristic[j, i] *= boost
    
    # Final cleanup
    heuristic[~np.isfinite(heuristic)] = 1e-9
    heuristic[heuristic <= 0] = 1e-9
    
    return heuristic
