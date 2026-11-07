import numpy as np
import time

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute a heuristic matrix for ant colony optimization of CVRP.
    
    Strategy: Perform a parameter search over a heuristic family.
    The heuristic is of the form:
    eta[i, j] = (1 / (d[i, j] + eps))^alpha * demand_weight[j] * depot_bias[i, j]
    
    We evaluate each candidate heuristic using a fast Greedy Nearest Neighbor 
    proxy solver to find the parameters that minimize the total route distance.
    """
    start_time = time.time()
    deadline = start_time + 250  # Safety margin
    
    d = distance_matrix.astype(float)
    n = d.shape[0]
    eps = 1e-9
    
    # Precompute demand as float
    dem = demands.astype(float)
    
    def greedy_cost(heuristic):
        """
        Run a greedy construction heuristic and return total distance.
        This serves as a proxy to evaluate the quality of the heuristic.
        """
        total_dist = 0.0
        visited = np.zeros(n, dtype=bool)
        visited[0] = True
        current = 0
        load = 0.0
        remaining = n - 1
        
        while remaining > 0:
            # Find best unvisited customer that fits
            best_j = -1
            best_score = -1.0
            
            # Vectorized selection for speed
            # Create a mask of valid next nodes
            valid_mask = ~visited
            valid_mask[0] = False # Don't go to depot unless forced
            
            # Check capacity
            # We need to check load + dem[j] <= capacity
            # This is hard to vectorize fully without knowing 'current' load interaction, 
            # but dem[j] is independent of current.
            # So valid_mask &= (load + dem <= capacity)
            
            if load + dem.max() > capacity:
                 # Some might not fit, some might
                 pass
                 
            # Filter by capacity
            # We can just loop, n=50 is small
            for j in range(1, n):
                if visited[j]:
                    continue
                if load + dem[j] > capacity:
                    continue
                score = heuristic[current, j]
                if score > best_score:
                    best_score = score
                    best_j = j
            
            if best_j == -1:
                # No customer fits, return to depot
                total_dist += d[current, 0]
                current = 0
                load = 0.0
                continue
            
            total_dist += d[current, best_j]
            load += dem[best_j]
            visited[best_j] = True
            remaining -= 1
            current = best_j
        
        # Return to depot
        total_dist += d[current, 0]
        return total_dist

    # Parameter grid
    # alpha: distance exponent
    # beta: demand penalty strength (higher beta -> stronger penalty for high demand)
    # gamma: depot bias strength
    
    alpha_values = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
    beta_values = [0.0, 0.5, 1.0, 2.0, 4.0]
    gamma_values = [0.0, 0.5, 1.0, 2.0]
    
    best_heuristic = None
    best_cost = float('inf')
    
    # Precompute base inverse distance for each alpha
    base_matrices = {}
    for alpha in alpha_values:
        # 1 / (d + eps)^alpha
        base_matrices[alpha] = (1.0 / (d + eps)) ** alpha
        # Zero out diagonal to prevent self-loops if not careful, though ACO handles it
        # But for greedy, diagonal doesn't matter as visited[0]=True initially
        # and we skip visited.
        
    # Demand weights:
    # We want to penalize high demand.
    # w[j] = 1 / (1 + beta * (dem[j] / capacity))
    # Or w[j] = 1 + beta * (1 - dem[j] / capacity)
    # Let's try the multiplicative penalty form as it's more standard for "cost"
    
    for alpha in alpha_values:
        for beta in beta_values:
            for gamma in gamma_values:
                if time.time() > deadline:
                    break
                
                base = base_matrices[alpha]
                
                # Demand weight
                # If beta=0, weight is 1.
                # If beta>0, high demand -> lower weight.
                # Formula: w_j = 1.0 / (1.0 + beta * (dem[j] / capacity))
                # This keeps weights positive.
                demand_weight = 1.0 / (1.0 + beta * (dem / capacity))
                demand_weight[0] = 1.0 # Depot weight irrelevant for destination, but keep consistent
                
                # Depot bias
                # Boost edges from depot (start) and to depot (end)
                depot_bias = np.ones((n, n))
                depot_bias[0, :] = 1.0 + gamma
                depot_bias[:, 0] = 1.0 + gamma * 0.5
                
                heuristic = base * demand_weight.reshape(1, -1) * depot_bias
                
                # Ensure positive and finite
                mask = ~np.isfinite(heuristic) | (heuristic <= 0)
                heuristic[mask] = 1e-9
                np.fill_diagonal(heuristic, 0.0) # Prevent self-loops in greedy proxy
                
                # Evaluate
                cost = greedy_cost(heuristic)
                
                if cost < best_cost:
                    best_cost = cost
                    best_heuristic = heuristic.copy()
            
            if time.time() > deadline:
                break
        
        if time.time() > deadline:
            break
    
    if best_heuristic is None:
        # Fallback: simple inverse distance
        heuristic = 1.0 / (d + eps)
        np.fill_diagonal(heuristic, 0.0)
        mask = ~np.isfinite(heuristic) | (heuristic <= 0)
        heuristic[mask] = 1e-9
        return heuristic
    
    return best_heuristic
