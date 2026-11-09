"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _run_aco(distance_matrix, coordinates, demands, capacity, heuristic, n_ants=20, n_iterations=30, seed=42):
    """Run ACO and return the best (lowest) total route distance found by any ant."""
    rng = np.random.RandomState(seed)
    n = distance_matrix.shape[0]
    pheromone = np.ones_like(heuristic, dtype=float)
    # Ensure heuristic is positive
    heuristic = np.where(heuristic > 0, heuristic, 1e-9)
    
    best_overall = float('inf')
    
    for it in range(n_iterations):
        best_iter_cost = float('inf')
        
        for ant in range(n_ants):
            # Ant state
            current = 0
            remaining_cap = capacity
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            total_dist = 0.0
            
            while True:
                # Find eligible customers
                # Unvisited, not depot, demand fits remaining
                eligible_mask = (~visited) & (np.arange(n) > 0)
                eligible_mask &= (demands[:n] <= remaining_cap)
                eligible_indices = np.where(eligible_mask)[0]
                
                if len(eligible_indices) == 0:
                    # Return to depot
                    total_dist += distance_matrix[current, 0]
                    current = 0
                    remaining_cap = capacity
                    # Check if all customers visited
                    if visited[1:].all():
                        break
                    continue
                
                # Compute probabilities
                probs = pheromone[current, eligible_indices] * heuristic[current, eligible_indices]
                probs = np.where(probs > 0, probs, 1e-9)
                probs /= probs.sum()
                
                # Pick next
                idx = rng.choice(len(eligible_indices), p=probs)
                next_node = eligible_indices[idx]
                
                total_dist += distance_matrix[current, next_node]
                remaining_cap -= demands[next_node]
                visited[next_node] = True
                current = next_node
                
                # If all visited, go back to depot
                if visited[1:].all():
                    total_dist += distance_matrix[current, 0]
                    break
        
        # Track best per iteration
        if total_dist < best_iter_cost:
            best_iter_cost = total_dist
        
        # Track overall best
        if total_dist < best_overall:
            best_overall = total_dist
        
        # Pheromone update
        pheromone *= 0.9
        pheromone += 1.0 / best_iter_cost
    
    return best_overall


def _generate_heuristic(distance_matrix, coordinates, demands, capacity, params):
    """
    Generate heuristic matrix based on parameters.
    params: dict with keys 'alpha' (distance power), 'beta' (demand penalty), 'gamma' (depot bias)
    """
    n = distance_matrix.shape[0]
    alpha, beta, gamma = params['alpha'], params['beta'], params['gamma']
    
    # Base: inverse distance power
    # Avoid division by zero
    dm_safe = distance_matrix + 1e-9
    base = 1.0 / (dm_safe ** alpha)
    
    # Demand penalty: penalize edges to high-demand nodes if current load is low?
    # Since heuristic is static, we can't know current load. 
    # Instead, we can penalize edges between high-demand nodes or towards high-demand nodes 
    # to encourage spreading them out? Or maybe just a simple static penalty.
    # Let's try: Penalize edges to nodes with high demand relative to capacity.
    # This might encourage picking smaller demands first? 
    # Actually, a common heuristic is 1/d. 
    # Let's try modifying with demand.
    # demand_factor[i,j] = 1 / (1 + beta * demands[j])
    # This makes edges to high-demand nodes less attractive.
    
    demand_col = demands[np.newaxis, :] # Shape (1, n)
    demand_factor = 1.0 / (1.0 + beta * demand_col)
    
    # Depot bias: Make edges from/to depot more attractive?
    # If gamma > 0, multiply depot edges by gamma.
    # This encourages shorter trips (more returns to depot).
    # Or maybe discourage?
    # Let's make it a multiplicative factor for depot edges.
    
    heuristic = base * demand_factor
    
    # Apply depot bias
    # Edges involving node 0
    depot_mask = (np.arange(n) == 0)
    # heuristic[:, 0] *= gamma  # Edges to depot
    # heuristic[0, :] *= gamma  # Edges from depot
    # Let's apply to both
    heuristic[:, 0] *= gamma
    heuristic[0, :] *= gamma
    
    # Ensure positive
    heuristic = np.where(heuristic > 0, heuristic, 1e-9)
    
    return heuristic

def heuristics(distance_matrix, coordinates, demands, capacity):
    start_time = time.time()
    deadline = start_time + 280  # Leave 50s buffer for safety
    
    # Parameter space
    # alpha: distance exponent (0.5 to 2.0)
    # beta: demand penalty (0.0 to 1.0)
    # gamma: depot bias (0.5 to 2.0)
    
    # Initial grid or random points
    # Let's use a coarse grid and then refine best region
    
    # Grid search
    alphas = [0.5, 1.0, 1.5, 2.0]
    betas = [0.0, 0.2, 0.5, 1.0]
    gammas = [0.5, 1.0, 1.5, 2.0]
    
    best_score = float('inf')
    best_params = {'alpha': 1.0, 'beta': 0.0, 'gamma': 1.0}
    
    # Evaluate grid
    for a in alphas:
        for b in betas:
            for g in gammas:
                if time.time() > deadline:
                    break
                params = {'alpha': a, 'beta': b, 'gamma': g}
                try:
                    h = _generate_heuristic(distance_matrix, coordinates, demands, capacity, params)
                    if not np.all(np.isfinite(h)):
                        continue
                    score = _run_aco(distance_matrix, coordinates, demands, capacity, h, seed=42)
                    if score < best_score:
                        best_score = score
                        best_params = params
                except Exception:
                    continue
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # Local refinement around best params
    if time.time() < deadline - 10:
        # Define small steps
        steps = [0.1, 0.2, 0.5]
        keys = ['alpha', 'beta', 'gamma']
        
        current_params = best_params.copy()
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for key in keys:
                for step in steps:
                    for direction in [-1, 1]:
                        new_params = current_params.copy()
                        new_val = current_params[key] + direction * step
                        # Bounds
                        if key == 'alpha' and (new_val < 0.1 or new_val > 3.0):
                            continue
                        if key == 'beta' and (new_val < 0.0 or new_val > 2.0):
                            continue
                        if key == 'gamma' and (new_val < 0.1 or new_val > 3.0):
                            continue
                        
                        new_params[key] = new_val
                        try:
                            h = _generate_heuristic(distance_matrix, coordinates, demands, capacity, new_params)
                            if not np.all(np.isfinite(h)):
                                continue
                            score = _run_aco(distance_matrix, coordinates, demands, capacity, h, seed=42)
                            if score < best_score:
                                best_score = score
                                best_params = new_params
                                current_params = new_params
                                improved = True
                        except Exception:
                            continue
                        if time.time() > deadline - 5:
                            break
                    if time.time() > deadline - 5:
                        break
                if time.time() > deadline - 5:
                    break
            if time.time() > deadline - 5:
                break
    
    # Generate final heuristic
    final_heuristic = _generate_heuristic(distance_matrix, coordinates, demands, capacity, best_params)
    
    # Final safety checks
    if not np.all(np.isfinite(final_heuristic)):
        # Fallback to simple 1/d
        final_heuristic = 1.0 / (distance_matrix + 1e-9)
    
    final_heuristic = np.where(final_heuristic > 0, final_heuristic, 1e-9)
    
    return final_heuristic
# EVOLVE-BLOCK-END
