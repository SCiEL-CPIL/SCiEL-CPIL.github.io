import numpy as np
import time

def heuristics(distance_matrix, coordinates, demands, capacity):
    start_time = time.time()
    deadline = start_time + 320.0  # Safety margin against 330s limit
    
    D = np.asarray(distance_matrix, dtype=np.float64)
    n = D.shape[0]
    demands = np.asarray(demands, dtype=np.float64)
    
    # Ensure D is safe for operations
    D_safe = np.where(D > 0, D, 1.0)
    
    # Precompute demand penalty factor: exp(-lambda * demand[j] / capacity)
    # We will search for lambda.
    # The heuristic is eta[i, j] = (1 / D[i, j]^2) * exp(-lambda * demands[j] / capacity)
    
    def run_proxy_aco(lam, num_ants=50):
        """
        Simulates the ACO process using the heuristic eta defined by lambda.
        Returns the best cost found.
        """
        # Construct heuristic matrix for this lambda
        # eta[i, j] = D[i, j]^-2 * exp(-lam * demands[j] / capacity)
        # Note: The actual ACO uses pheromone * heuristic. 
        # In the proxy, we assume pheromone starts at 1 and decays, but for 
        # heuristic selection, we primarily care about the shape of the heuristic 
        # guiding the ants. 
        # To be faithful, we should simulate the pheromone dynamics too, or 
        # approximate that the heuristic dominates the initial selection.
        # Given the task says "Heuristic is computed once", and pheromone starts at 1,
        # the initial choice is driven by heuristic. Pheromone updates reinforce good paths.
        # A simple proxy: Run a few iterations of ACO with this heuristic.
        
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            base_inv_dist = D_safe ** (-2)
            demand_penalty = np.exp(-lam * demands / capacity)
            # eta[i, j] = base_inv_dist[i, j] * demand_penalty[j]
            eta = base_inv_dist * demand_penalty[np.newaxis, :]
        
        # Sanitize eta
        eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
        
        # Simulate ACO
        # Pheromone matrix
        pheromone = np.ones((n, n))
        best_cost_iter = float('inf')
        best_overall = float('inf')
        
        rng = np.random.RandomState(42)
        
        # Run a few iterations to let pheromone settle
        # The real task runs 30 iterations. We run fewer to save time but enough to see effect.
        # Let's run 10 iterations in proxy.
        num_iterations = 10
        
        for _ in range(num_iterations):
            iteration_best = float('inf')
            
            for _ in range(num_ants):
                unvisited = np.ones(n, dtype=bool)
                unvisited[0] = False
                current = 0
                load = 0.0
                total_dist = 0.0
                
                while unvisited.any():
                    rem_cap = capacity - load
                    feasible_mask = unvisited & (demands <= rem_cap)
                    
                    if not feasible_mask.any():
                        # Return to depot
                        total_dist += D[current, 0]
                        current = 0
                        load = 0.0
                        continue
                    
                    feasible_indices = np.where(feasible_mask)[0]
                    
                    # Probability proportional to pheromone[i, j] * heuristic[i, j]
                    weights = pheromone[current, feasible_indices] * eta[current, feasible_indices]
                    
                    sum_w = np.sum(weights)
                    if sum_w <= 0 or not np.isfinite(sum_w):
                        next_node = rng.choice(feasible_indices)
                    else:
                        probs = weights / sum_w
                        # Handle numerical issues in probs
                        probs = np.where(probs > 0, probs, 0)
                        sum_p = np.sum(probs)
                        if sum_p == 0:
                            next_node = rng.choice(feasible_indices)
                        else:
                            probs = probs / sum_p
                            next_node = feasible_indices[rng.choice(len(feasible_indices), p=probs)]
                    
                    total_dist += D[current, next_node]
                    load += demands[next_node]
                    current = next_node
                    unvisited[next_node] = False
                    
                    if not unvisited.any():
                        total_dist += D[current, 0]
                        break
                
                if total_dist < iteration_best:
                    iteration_best = total_dist
                if total_dist < best_overall:
                    best_overall = total_dist
            
            # Pheromone update
            # Multiply by 0.9
            pheromone *= 0.9
            # Add 1 / best_cost of that iteration
            if iteration_best < float('inf') and iteration_best > 0:
                pheromone += 1.0 / iteration_best
            
            # Optional: Normalize pheromone to prevent overflow/underflow issues in proxy
            # Not strictly required if values stay reasonable, but good practice.
            # pheromone = pheromone / np.max(pheromone)

        return best_overall

    # Search for optimal lambda
    # Lambda controls the strength of demand penalty.
    # Lambda = 0 -> pure distance heuristic (d^-2)
    # Lambda > 0 -> penalizes high demand nodes
    # We want to find if penalizing high demand helps pack them or if it hurts by forcing ants to avoid them.
    # The inspiration suggests "penalty for high-demand nodes" to "pack heavier customers early".
    # Wait, if we penalize high demand, ants avoid them. This might force ants to visit them when they have no other choice (near capacity) or group them if the distance penalty is strong enough?
    # Actually, if eta is low for high demand, ants are less likely to pick them.
    # This might lead to visiting high demand nodes later or in separate trips if they are "unattractive".
    # Let's test a range of lambda from 0 to 5.
    
    best_lam = 0.0
    best_cost = float('inf')
    
    # Coarse search
    lambdas = np.arange(0.0, 5.1, 0.5)
    
    for lam in lambdas:
        if time.time() > deadline - 10.0:
            break
        cost = run_proxy_aco(lam, num_ants=50)
        if cost < best_cost:
            best_cost = cost
            best_lam = lam

    # Fine search around best_lam
    if time.time() < deadline - 5.0:
        fine_lambdas = np.arange(max(0, best_lam - 0.5), best_lam + 0.5, 0.1)
        for lam in fine_lambdas:
            if time.time() > deadline - 2.0:
                break
            if lam < 0: continue
            cost = run_proxy_aco(lam, num_ants=100)
            if cost < best_cost:
                best_cost = cost
                best_lam = lam

    # Construct final heuristic with best lambda
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        base_inv_dist = D_safe ** (-2)
        demand_penalty = np.exp(-best_lam * demands / capacity)
        final_heuristic = base_inv_dist * demand_penalty[np.newaxis, :]
    
    # Sanitize
    final_heuristic = np.where(np.isfinite(final_heuristic) & (final_heuristic > 0), final_heuristic, 1e-9)
    
    return final_heuristic
