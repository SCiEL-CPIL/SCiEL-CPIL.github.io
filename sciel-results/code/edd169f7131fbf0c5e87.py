"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _aco_simulate(distance_matrix, coordinates, demands, capacity, heuristic, n_ants=20, n_iters=30, seed=42):
    """Fast simulation of the ACO to evaluate a heuristic's quality."""
    n = len(demands)
    rng = np.random.RandomState(seed)
    pheromone = np.ones_like(distance_matrix)
    best_cost = float('inf')
    demands_arr = demands.astype(float)
    
    for it in range(n_iters):
        iter_best = float('inf')
        for ant in range(n_ants):
            route_cost = 0.0
            current = 0
            remaining_cap = capacity
            visited = set()
            
            while True:
                # Find feasible neighbors
                feasible = []
                for j in range(1, n):
                    if j not in visited and demands_arr[j] <= remaining_cap:
                        feasible.append(j)
                
                if not feasible:
                    # Return to depot
                    route_cost += distance_matrix[current, 0]
                    break
                
                # Compute probabilities
                probs = np.zeros(len(feasible))
                for idx, j in enumerate(feasible):
                    probs[idx] = pheromone[current, j] * heuristic[current, j]
                
                total = probs.sum()
                if total < 1e-30:
                    # Pick random feasible
                    j = feasible[rng.randint(len(feasible))]
                else:
                    probs /= total
                    r = rng.random()
                    cumsum = np.cumsum(probs)
                    idx = np.searchsorted(cumsum, r)
                    if idx >= len(feasible):
                        idx = len(feasible) - 1
                    j = feasible[idx]
                
                route_cost += distance_matrix[current, j]
                visited.add(j)
                remaining_cap -= demands_arr[j]
                current = j
            
            # Add final return to depot
            route_cost += distance_matrix[current, 0]
            if route_cost < iter_best:
                iter_best = route_cost
        
        if iter_best < best_cost:
            best_cost = iter_best
        
        # Update pheromone
        pheromone *= 0.9
        if iter_best > 0:
            pheromone += 1.0 / iter_best
    
    return best_cost


def heuristics(distance_matrix, coordinates, demands, capacity):
    """Compute heuristic matrix using node-potential field with spatial demand density."""
    n = len(demands)
    t_start = time.time()
    deadline = t_start + 250.0  # Safety margin before 300s limit
    
    # Node-potential field strategy:
    # eta[i,j] = 1 / (dist[i,j] * exp(-sigma * dist_j_to_centroid))
    # where centroid is computed from all customers (weighted by demand)
    # This encourages ants to visit nodes close to the demand centroid,
    # which tends to create compact, efficient routes
    
    demands_arr = demands.astype(float)
    
    # Compute weighted centroid of all customers
    # Weights are the demands, so high-demand customers pull the centroid
    weights = demands_arr[1:]  # Exclude depot
    if weights.sum() > 0:
        centroid = np.average(coordinates[1:], axis=0, weights=weights)
    else:
        centroid = coordinates[0].copy()
    
    # Distance from each customer to the centroid
    dist_to_centroid = np.sqrt(np.sum((coordinates[1:] - centroid) ** 2, axis=1))
    
    # Candidate sigma values (decay rate of the potential field)
    # Smaller sigma = broader influence, larger sigma = sharper peak at centroid
    sigmas = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]
    
    # Also try combining with distance power
    alphas = [1.0, 1.2, 1.5, 2.0]
    
    best_heuristic = 1.0 / (distance_matrix + 1e-9)
    best_score = float('inf')
    
    # Evaluate combinations of sigma and alpha
    for alpha in alphas:
        for sigma in sigmas:
            if time.time() > deadline:
                break
            try:
                # Build the potential field
                # potential[j] = exp(-sigma * dist_to_centroid[j])
                potential = np.exp(-sigma * dist_to_centroid)
                
                # Heuristic: eta[i,j] = 1 / (dist[i,j]^alpha * (1 / potential[j]))
                # = potential[j] / dist[i,j]^alpha
                # This makes edges to nodes with high potential (close to centroid) more attractive
                
                dist_power = distance_matrix ** alpha
                # For each column j, multiply by potential[j]
                heuristic = potential[None, 1:] / dist_power[:, 1:]
                
                # Handle depot row separately (potential for depot is 1.0)
                heuristic[0, 1:] = potential / dist_power[0, 1:]
                
                # Ensure positive and finite
                heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
                
                # Evaluate via simulation
                score = _aco_simulate(distance_matrix, coordinates, demands, capacity, heuristic, 
                                       n_ants=10, n_iters=10, seed=42)
                if score < best_score:
                    best_score = score
                    best_heuristic = heuristic.copy()
            except Exception:
                continue
    
    # Try a variant: use unweighted centroid (geometric center)
    centroid_unweighted = coordinates[1:].mean(axis=0)
    dist_to_centroid_unweighted = np.sqrt(np.sum((coordinates[1:] - centroid_unweighted) ** 2, axis=1))
    
    for alpha in [1.0, 1.5, 2.0]:
        for sigma in [1.0, 2.0, 3.0, 5.0]:
            if time.time() > deadline:
                break
            try:
                potential = np.exp(-sigma * dist_to_centroid_unweighted)
                dist_power = distance_matrix ** alpha
                heuristic = potential[None, 1:] / dist_power[:, 1:]
                heuristic[0, 1:] = potential / dist_power[0, 1:]
                
                heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
                
                score = _aco_simulate(distance_matrix, coordinates, demands, capacity, heuristic, 
                                       n_ants=10, n_iters=10, seed=42)
                if score < best_score:
                    best_score = score
                    best_heuristic = heuristic.copy()
            except Exception:
                continue
    
    # Try combining centroid potential with a demand-based boost
    # Nodes with higher demand get additional attractiveness
    demand_boost = 1.0 + 0.5 * (demands_arr[1:] / capacity)
    
    for alpha in [1.0, 1.5]:
        for sigma in [1.0, 2.0, 3.0]:
            if time.time() > deadline:
                break
            try:
                potential = np.exp(-sigma * dist_to_centroid) * demand_boost
                dist_power = distance_matrix ** alpha
                heuristic = potential[None, 1:] / dist_power[:, 1:]
                heuristic[0, 1:] = potential / dist_power[0, 1:]
                
                heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
                
                score = _aco_simulate(distance_matrix, coordinates, demands, capacity, heuristic, 
                                       n_ants=10, n_iters=10, seed=42)
                if score < best_score:
                    best_score = score
                    best_heuristic = heuristic.copy()
            except Exception:
                continue
    
    # Fine-tune: try scaling the best heuristic
    if time.time() < deadline - 30:
        for scale in [0.5, 0.8, 1.2, 1.5, 2.0]:
            if time.time() > deadline:
                break
            try:
                h = best_heuristic * scale
                h = np.where(h > 0, h, 1e-9)
                score = _aco_simulate(distance_matrix, coordinates, demands, capacity, h,
                                       n_ants=10, n_iters=10, seed=42)
                if score < best_score:
                    best_score = score
                    best_heuristic = h.copy()
            except Exception:
                continue
    
    # Final validation
    best_heuristic = np.asarray(best_heuristic, dtype=float)
    if best_heuristic.shape != distance_matrix.shape:
        best_heuristic = 1.0 / (distance_matrix + 1e-9)
    best_heuristic = np.where(np.isfinite(best_heuristic) & (best_heuristic > 0), best_heuristic, 1e-9)
    
    return best_heuristic
# EVOLVE-BLOCK-END
