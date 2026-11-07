import numpy as np
import time

def heuristics(distance_matrix, coordinates, demands, capacity):
    start_time = time.time()
    deadline = start_time + 10.0  # Safety margin, total limit is 330s, this is just the heuristic phase
    
    n = len(demands)
    demands = np.array(demands)
    d_mat = np.array(distance_matrix)
    
    # 1. Calculate Mean Distance for Noise Scaling
    # Exclude zeros (diagonal)
    mask = ~np.eye(n, dtype=bool)
    mean_dist = np.mean(d_mat[mask])
    if mean_dist == 0:
        mean_dist = 1.0
    
    # 2. Simulate Greedy CVRP with Noise to build an "Edge Frequency" Matrix
    # This creates a data-driven heuristic that reflects which edges are part of good feasible routes.
    edge_counts = np.zeros((n, n), dtype=float)
    
    # Number of simulations. 500 is very fast for N=50.
    n_sims = 500
    
    for i in range(n_sims):
        if time.time() > deadline - 1.0:
            break
        
        # Add symmetric noise to distance matrix to explore different greedy choices
        # Noise should be small to keep the structure similar to the true distances
        noise = np.random.normal(0, 0.05 * mean_dist, (n, n))
        # Symmetrize noise
        noise = (noise + noise.T) / 2.0
        # Keep diagonal 0
        np.fill_diagonal(noise, 0)
        
        d_noisy = d_mat + noise
        
        # Run Greedy CVRP on noisy distances
        unvisited = set(range(1, n))
        current = 0
        load = 0.0
        
        while unvisited:
            # Find feasible next node
            best_j = -1
            min_dist = float('inf')
            
            # Optimization: Pre-filter feasible? For N=50, simple loop is fine.
            for j in unvisited:
                if load + demands[j] <= capacity:
                    dist_val = d_noisy[current, j]
                    if dist_val < min_dist:
                        min_dist = dist_val
                        best_j = j
            
            if best_j == -1:
                # Return to depot
                # We record the edge (current, 0)
                edge_counts[current, 0] += 1
                edge_counts[0, current] += 1
                current = 0
                load = 0.0
            else:
                # Move to best_j
                edge_counts[current, best_j] += 1
                edge_counts[best_j, current] += 1
                
                load += demands[best_j]
                unvisited.remove(best_j)
                current = best_j
        
        # Final return to depot
        if current != 0:
            edge_counts[current, 0] += 1
            edge_counts[0, current] += 1

    # 3. Construct the Heuristic Matrix
    # Combine the structural information (edge_counts) with the geometric cost (distance)
    # eta_ij = (counts_ij + epsilon) / (dist_ij ^ alpha)
    
    epsilon = 1e-6
    alpha = 1.5  # Exponent to penalize distance
    
    # Avoid division by zero
    safe_dist = d_mat + 1e-9
    
    # Calculate base heuristic
    # We want higher attractiveness for frequently used, short edges
    result = (edge_counts + epsilon) / (safe_dist ** alpha)
    
    # Handle depot edges: The heuristic for (0, j) and (i, 0) is crucial for route start/end
    # The simulation already accounts for these frequencies.
    
    # Ensure finite and positive
    result[~np.isfinite(result)] = 1e-9
    result[result < 1e-9] = 1e-9
    
    return result
