import numpy as np
import time

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute edge-attractiveness matrix for ant colony optimization of CVRP.
    
    Key Insight: The pheromone update in the task is uniform across all edges 
    (every entry gains the same value). Since pheromone starts at 1, 
    tau[i][j] remains equal for all i,j throughout the run. 
    Thus, the selection probability P[i][j] is proportional to heuristic[i][j].
    The ACO effectively becomes a stochastic greedy algorithm guided by the heuristic.
    
    Therefore, we must design a heuristic that strongly encourages 
    tight, capacity-feasible clusters. A sharp inverse distance heuristic 
    (1/d^alpha) encourages nearest-neighbor behavior, which forms tight clusters.
    We tune alpha and a mild demand penalty using a fast greedy proxy solver.
    """
    dm = np.asarray(distance_matrix, dtype=np.float64)
    coords = np.asarray(coordinates, dtype=np.float64)
    dem = np.asarray(demands, dtype=np.float64)
    n = dm.shape[0]
    
    # Deadline guard
    start_time = time.time()
    DEADLINE = 320.0

    def safe_default():
        # Fallback: simple inverse distance
        base = 1.0 / (dm + 1e-9)
        base[~np.isfinite(base)] = 1e-9
        base[base <= 0] = 1e-9
        return base

    def calculate_heuristic(alpha, gamma):
        """
        eta[i][j] = (1 / d[i][j]^alpha) * (1 / (1 + gamma * d[j]))
        alpha: sharpness of distance preference
        gamma: strength of demand penalty (penalizes high demand nodes)
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            dist_term = 1.0 / (dm ** alpha)
        
        # Demand penalty on destination
        # If gamma is 0, no penalty.
        demand_penalty = 1.0 / (1.0 + gamma * dem)
        
        adjusted = dist_term * demand_penalty[np.newaxis, :]
        
        # Ensure finite and positive
        adjusted[~np.isfinite(adjusted)] = 1e-9
        adjusted[adjusted <= 0] = 1e-9
        
        return adjusted

    def proxy_solve(eta):
        """
        Fast greedy construction:
        At each step, pick the unvisited feasible node j that maximizes eta[current][j].
        This mimics the deterministic limit of the ACO selection.
        """
        visited = np.zeros(n, dtype=bool)
        visited[0] = True
        total_dist = 0.0
        current = 0
        load = 0.0
        
        for _ in range(n - 1):
            best_j = -1
            best_score = -1.0
            
            # Vectorized selection for speed
            # Mask unvisited and feasible
            feasible_mask = (~visited) & (dem <= (capacity - load))
            
            if not np.any(feasible_mask):
                # Return to depot
                total_dist += dm[current, 0]
                current = 0
                load = 0.0
                # Note: The loop continues, but since all are visited? 
                # Wait, if we return to depot, we haven't visited any new node.
                # The loop runs n-1 times to visit n-1 customers.
                # If no feasible node, we just return to depot. 
                # But we must still visit all customers eventually.
                # In the real ACO, if no node fits, it returns to depot.
                # Then from depot, it picks a node that fits.
                # My proxy loop structure is slightly flawed for multiple trips.
                # Let's restructure proxy to be robust.
                pass
            else:
                # Get scores for feasible nodes
                scores = np.where(feasible_mask, eta[current, :], -np.inf)
                best_j = np.argmax(scores)
                best_score = scores[best_j]
                
                if best_score <= 0: # Should not happen if eta is positive
                     # Fallback to nearest neighbor
                     dists = np.where(feasible_mask, dm[current, :], np.inf)
                     best_j = np.argmin(dists)
                
                total_dist += dm[current, best_j]
                load += dem[best_j]
                visited[best_j] = True
                current = best_j
        
        # Final return to depot
        if current != 0:
            total_dist += dm[current, 0]
            
        return total_dist

    # Corrected Proxy Solver for multiple trips
    def proxy_solve_fast(eta):
        visited = np.zeros(n, dtype=bool)
        visited[0] = True
        total_dist = 0.0
        current = 0
        load = 0.0
        remaining = n - 1
        
        while remaining > 0:
            best_j = -1
            best_score = -1.0
            
            # Find feasible unvisited nodes
            feasible_indices = np.where((~visited) & (dem <= (capacity - load)))[0]
            
            if len(feasible_indices) == 0:
                # Return to depot
                total_dist += dm[current, 0]
                current = 0
                load = 0.0
                # Retry selection from depot
                feasible_indices = np.where((~visited) & (dem <= capacity))[0]
                if len(feasible_indices) == 0:
                    break # Should not happen if demands are valid
            
            # Select best from feasible
            if current == 0:
                scores = eta[0, feasible_indices]
            else:
                scores = eta[current, feasible_indices]
            
            best_idx_in_feasible = np.argmax(scores)
            best_j = feasible_indices[best_idx_in_feasible]
            
            total_dist += dm[current, best_j]
            load += dem[best_j]
            visited[best_j] = True
            current = best_j
            remaining -= 1
            
        if current != 0:
            total_dist += dm[current, 0]
            
        return total_dist

    # Search for best alpha and gamma
    # Alpha: 1.0 to 5.0
    # Gamma: 0.0 to 0.5
    best_cost = float('inf')
    best_alpha = 1.0
    best_gamma = 0.0
    
    alphas = np.linspace(1.0, 5.0, 20)
    gammas = np.linspace(0.0, 0.5, 10)
    
    # Coarse search
    for alpha in alphas:
        for gamma in gammas:
            elapsed = time.time() - start_time
            if elapsed > DEADLINE - 30:
                break
                
            eta = calculate_heuristic(alpha, gamma)
            cost = proxy_solve_fast(eta)
            
            if cost < best_cost:
                best_cost = cost
                best_alpha = alpha
                best_gamma = gamma

    # Fine search around best
    if best_cost < float('inf'):
        alphas_fine = np.linspace(best_alpha - 0.5, best_alpha + 0.5, 10)
        gammas_fine = np.linspace(max(0, best_gamma - 0.1), best_gamma + 0.1, 10)
        
        for alpha in alphas_fine:
            for gamma in gammas_fine:
                elapsed = time.time() - start_time
                if elapsed > DEADLINE - 10:
                    break
                    
                if alpha < 0.5 or gamma < 0:
                    continue
                    
                eta = calculate_heuristic(alpha, gamma)
                cost = proxy_solve_fast(eta)
                
                if cost < best_cost:
                    best_cost = cost
                    best_alpha = alpha
                    best_gamma = gamma

    # Final heuristic calculation
    result = calculate_heuristic(best_alpha, best_gamma)
    
    # Safety check
    result[~np.isfinite(result)] = 1e-9
    result[result <= 0] = 1e-9
    
    return result
