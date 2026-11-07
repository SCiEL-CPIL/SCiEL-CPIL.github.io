import numpy as np
import time

def _run_aco(distance_matrix, coordinates, demands, capacity, heuristic_matrix, seed, n_ants=20, n_iterations=30):
    """
    Runs the specific ACO algorithm described in the task.
    Returns the best cost found.
    """
    np.random.seed(seed)
    n = distance_matrix.shape[0]
    # Pheromone starts at 1
    pheromone = np.ones((n, n))
    
    best_cost_overall = float('inf')
    
    for _ in range(n_iterations):
        iteration_best_cost = float('inf')
        
        for _ in range(n_ants):
            current_node = 0
            current_load = 0.0
            unvisited = set(range(1, n))
            route_cost = 0.0
            
            while unvisited:
                # Find feasible neighbors
                feasible = []
                for j in unvisited:
                    if demands[j] <= capacity - current_load:
                        feasible.append(j)
                
                if not feasible:
                    break
                
                # Calculate probabilities
                # p_ij proportional to tau_ij * eta_ij
                probs = np.zeros(len(feasible))
                for idx, j in enumerate(feasible):
                    probs[idx] = pheromone[current_node, j] * heuristic_matrix[current_node, j]
                
                total_prob = np.sum(probs)
                if total_prob == 0 or not np.isfinite(total_prob):
                    # Fallback: pick random feasible
                    next_node = np.random.choice(feasible)
                else:
                    probs = probs / total_prob
                    next_node = np.random.choice(feasible, p=probs)
                
                # Move
                route_cost += distance_matrix[current_node, next_node]
                current_load += demands[next_node]
                unvisited.remove(next_node)
                current_node = next_node
            
            # Return to depot
            route_cost += distance_matrix[current_node, 0]
            
            if route_cost < iteration_best_cost:
                iteration_best_cost = route_cost
        
        if iteration_best_cost < best_cost_overall:
            best_cost_overall = iteration_best_cost
            
        # Pheromone Update: global
        # tau = 0.9 * tau + 1 / best_cost
        # Note: If best_cost is inf, 1/inf is 0.
        if iteration_best_cost > 0 and np.isfinite(iteration_best_cost):
            pheromone = 0.9 * pheromone + (1.0 / iteration_best_cost)
        else:
            pheromone = 0.9 * pheromone

    return best_cost_overall

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Computes the heuristic matrix by tuning the gamma parameter.
    """
    try:
        start_time = time.time()
        deadline = start_time + 290.0 # Safety margin
        
        dm = np.asarray(distance_matrix, dtype=np.float64)
        n = dm.shape[0]
        
        # Ensure diagonal is 0 to avoid 1/0 in heuristic
        dm = dm.copy()
        np.fill_diagonal(dm, 0.0)
        
        # Candidate gammas
        gammas = np.arange(0.5, 3.5, 0.1)
        
        best_gamma = 1.0
        best_score = float('inf')
        
        # Use a few seeds for robustness
        seeds = [1, 2, 3]
        
        for gamma in gammas:
            if time.time() > deadline:
                break
                
            # Compute heuristic
            # eta = 1 / d^gamma
            with np.errstate(divide='ignore', invalid='ignore'):
                eta = 1.0 / (dm ** gamma)
            
            # Set diagonal to 0
            np.fill_diagonal(eta, 0.0)
            
            # Replace non-finite or non-positive with 1e-9 as per task
            eta[~np.isfinite(eta)] = 1e-9
            eta[eta <= 0] = 1e-9
            
            # Evaluate this heuristic
            scores = []
            for seed in seeds:
                cost = _run_aco(dm, coordinates, demands, capacity, eta, seed)
                scores.append(cost)
            
            mean_score = np.mean(scores)
            
            if mean_score < best_score:
                best_score = mean_score
                best_gamma = gamma
                
        # Final heuristic with best gamma
        with np.errstate(divide='ignore', invalid='ignore'):
            final_eta = 1.0 / (dm ** best_gamma)
        
        np.fill_diagonal(final_eta, 0.0)
        final_eta[~np.isfinite(final_eta)] = 1e-9
        final_eta[final_eta <= 0] = 1e-9
        
        return final_eta

    except Exception:
        # Fallback to standard heuristic if anything fails
        dm = np.asarray(distance_matrix, dtype=np.float64)
        dm = dm.copy()
        np.fill_diagonal(dm, 0.0)
        with np.errstate(divide='ignore', invalid='ignore'):
            eta = 1.0 / (dm + 1e-9)
        np.fill_diagonal(eta, 0.0)
        eta[~np.isfinite(eta)] = 1e-9
        eta[eta <= 0] = 1e-9
        return eta
