import numpy as np
import time

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Heuristic matrix for ant colony optimization of capacitated vehicle routing.
    Uses a Lagrangian-relaxation-inspired approach: solve a relaxed TSP-like problem
    (ignoring capacity, penalizing via dual variables) to identify attractive edges.
    
    Args:
        distance_matrix: (n, n) matrix of Euclidean distances
        coordinates: (n, 2) array of coordinates in unit square
        demands: (n,) array of demands (node 0 is depot with demand 0)
        capacity: vehicle capacity (50)
    
    Returns:
        (n, n) edge-attractiveness matrix where larger values are more attractive.
        Non-positive entries become 1e-9.
    """
    start_time = time.time()
    deadline = 330.0 - 10.0  # safety margin
    
    d = np.array(distance_matrix, dtype=float)
    coords = np.array(coordinates, dtype=float)
    demands = np.array(demands, dtype=float)
    n = d.shape[0]
    
    if d.shape[0] != d.shape[1]:
        return np.ones((n, n))
    
    # Ensure finite values in distance matrix
    d = np.where(np.isfinite(d), d, 1e9)
    d = np.maximum(d, 1e-12)
    
    # --- Step 1: Compute a base inverse-distance heuristic with tuned exponent ---
    # Use power 2.5 for a balance between exploration and exploitation
    heuristic = np.zeros_like(d)
    mask = d > 1e-12
    heuristic[mask] = 1.0 / (d[mask] ** 2.5)
    
    # --- Step 2: Lagrangian relaxation approach ---
    # Idea: Solve a relaxed problem where we ignore capacity constraints but penalize
    # "expensive" edges (those that would lead to poor load utilization).
    # 
    # We approximate this by:
    # 1. Computing a minimum-spanning-tree-like structure to identify "backbone" edges
    # 2. Using the dual variable (penalty) to adjust edge weights based on demand
    #
    # For CVRP, edges that connect high-demand nodes together are more attractive
    # because they allow filling vehicles efficiently.
    
    # Compute demand-based edge attractiveness:
    # For edge (i, j), the attractiveness should consider:
    # - How well i and j can be served together (demand compatibility)
    # - The geometric proximity
    #
    # A simple proxy: edges between nodes with similar demand profiles are better
    # for routing (they can be grouped in the same vehicle)
    
    # Demand compatibility factor:
    # If demand[i] + demand[j] <= capacity, they can be in the same vehicle
    # Higher compatibility -> higher attractiveness
    demand_sum = demands[np.newaxis, :] + demands[:, np.newaxis]  # (n, n)
    compatible = demand_sum <= capacity
    
    # For compatible pairs, boost the heuristic
    # For incompatible pairs, slightly reduce (but keep positive)
    compat_factor = np.where(compatible, 1.0 + 0.3 * (1.0 - demand_sum / capacity), 0.7)
    
    heuristic = heuristic * compat_factor
    
    # --- Step 3: Identify "backbone" edges using a relaxed tour construction ---
    # Build a minimum-cost cycle (approximation of optimal TSP tour) ignoring capacity
    # This identifies edges that are locally optimal in a relaxed sense
    
    # Use a simple nearest-neighbor heuristic to get a baseline tour
    # This gives us a set of "good" edges
    
    # Nearest neighbor tour from depot
    visited = np.zeros(n, dtype=bool)
    visited[0] = True
    tour = [0]
    current = 0
    
    for _ in range(n - 1):
        # Find nearest unvisited node
        unvisited = ~visited
        unvisited[current] = True  # exclude current
        distances_from_current = d[current].copy()
        distances_from_current[visited] = np.inf
        next_node = np.argmin(distances_from_current)
        tour.append(next_node)
        visited[next_node] = True
        current = next_node
    
    # Create edge indicator for the NN tour
    tour_edges = np.zeros_like(d)
    for i in range(len(tour) - 1):
        u, v = tour[i], tour[i + 1]
        tour_edges[u, v] = 1.0
        tour_edges[v, u] = 1.0
    # Close the tour
    tour_edges[tour[-1], tour[0]] = 1.0
    tour_edges[tour[0], tour[-1]] = 1.0
    
    # Boost edges that are in the NN tour (they are locally attractive)
    heuristic = heuristic * (1.0 + 0.5 * tour_edges)
    
    # --- Step 4: 2-opt improvement to refine the tour and identify better edges ---
    # Perform a few iterations of 2-opt to improve the tour
    improved_tour = tour.copy()
    best_improvement = np.inf
    for iteration in range(10):
        if time.time() > deadline - 5:
            break
        improvement = False
        for i in range(len(improved_tour) - 2):
            if time.time() > deadline - 5:
                break
            for j in range(i + 1, len(improved_tour) - 1):
                # Calculate delta for 2-opt move
                a, b = improved_tour[i], improved_tour[i + 1]
                c, d_node = improved_tour[j], improved_tour[(j + 1) % len(improved_tour)]
                
                old_cost = d[a, b] + d[c, d_node]
                new_cost = d[a, c] + d[b, d_node]
                
                if new_cost < old_cost - 1e-9:
                    # Perform the 2-opt move
                    improved_tour[i + 1:j + 1] = improved_tour[i + 1:j + 1][::-1]
                    improvement = True
        
        if not improvement:
            break
    
    # Update tour edges with improved tour
    tour_edges = np.zeros_like(d)
    for i in range(len(improved_tour) - 1):
        u, v = improved_tour[i], improved_tour[i + 1]
        tour_edges[u, v] = 1.0
        tour_edges[v, u] = 1.0
    tour_edges[improved_tour[-1], improved_tour[0]] = 1.0
    tour_edges[improved_tour[0], improved_tour[-1]] = 1.0
    
    # Re-boost with improved tour edges
    heuristic = heuristic * (1.0 + 0.3 * tour_edges)
    
    # --- Step 5: Add a demand-weighted component ---
    # Favor edges to high-demand nodes (to fill vehicles efficiently)
    demand_factor = 1.0 + 0.2 * (demands / capacity)[np.newaxis, :]
    heuristic = heuristic * demand_factor
    
    # --- Step 6: Handle non-finite and non-positive values ---
    heuristic[~np.isfinite(heuristic)] = 1e-9
    heuristic[heuristic <= 0] = 1e-9
    
    # Ensure diagonal is small (no self-loops)
    np.fill_diagonal(heuristic, 1e-9)
    
    return heuristic
