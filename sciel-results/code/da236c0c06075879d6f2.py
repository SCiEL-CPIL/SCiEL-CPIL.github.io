# EVOLVE-BLOCK-START
import numpy as np
import time


def _proxy_aco(distance_matrix, coordinates, demands, capacity, heuristic, n_ants=20, n_iters=30, seed=42, deadline=None):
    """Run a proxy ACO to evaluate a heuristic's quality."""
    rng = np.random.default_rng(seed)
    n = distance_matrix.shape[0]
    dm = distance_matrix
    eta = heuristic
    tau = np.ones((n, n))
    best_cost = np.inf
    demands_arr = np.asarray(demands, dtype=np.float64)
    
    for it in range(n_iters):
        if deadline is not None and time.time() > deadline:
            break
        iter_best = np.inf
        for _ in range(n_ants):
            current = 0
            load = 0.0
            visited = set()
            route_cost = 0.0
            while True:
                # Find feasible unvisited customers
                feasible = []
                for j in range(1, n):
                    if j not in visited and demands_arr[j] <= capacity - load:
                        feasible.append(j)
                if not feasible:
                    # Return to depot
                    route_cost += dm[current, 0]
                    break
                # Compute probabilities
                probs = np.zeros(len(feasible))
                for idx, j in enumerate(feasible):
                    probs[idx] = tau[current, j] * eta[current, j]
                total = probs.sum()
                if total <= 0 or not np.isfinite(total):
                    probs = np.ones(len(feasible))
                    total = probs.sum()
                probs /= total
                # Select next node
                r = rng.random()
                cumsum = np.cumsum(probs)
                idx = np.searchsorted(cumsum, r)
                if idx >= len(feasible):
                    idx = len(feasible) - 1
                next_node = feasible[idx]
                route_cost += dm[current, next_node]
                load += demands_arr[next_node]
                visited.add(next_node)
                current = next_node
            if route_cost < iter_best:
                iter_best = route_cost
        if iter_best < best_cost:
            best_cost = iter_best
        # Update pheromone
        tau *= 0.9
        tau += 1.0 / max(iter_best, 1e-9)
    
    return best_cost


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute heuristic matrix for ACO by finding optimal gamma via proxy ACO.
    Uses a focused range of gamma values with full-strength proxy ACO for accurate evaluation.
    """
    start_time = time.time()
    deadline = start_time + 270.0  # Safety margin before 300s limit
    
    dm = np.asarray(distance_matrix, dtype=np.float64)
    n = dm.shape[0]
    
    # Focused gamma range: search around 2.0 where 1/d^2 scored well,
    # with a denser grid for finer tuning
    gammas = np.array([1.5, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.5, 3.0])
    
    best_gamma = 2.0
    best_score = np.inf
    
    # Use a fixed seed for reproducibility in heuristic selection
    seed = 42
    
    for gamma in gammas:
        if time.time() > deadline:
            break
        
        # Compute heuristic for this gamma
        with np.errstate(divide='ignore', invalid='ignore'):
            eta = 1.0 / np.power(dm + 1e-9, gamma)
        
        # Ensure positive and finite values
        eta[~np.isfinite(eta)] = 1e-9
        eta[eta <= 0] = 1e-9
        
        # Run proxy ACO with this heuristic - use full strength for accuracy
        # but with deadline guard
        score = _proxy_aco(dm, coordinates, demands, capacity, eta, 
                           n_ants=20, n_iters=30, seed=seed, 
                           deadline=deadline - 5.0)
        
        if score < best_score:
            best_score = score
            best_gamma = gamma
    
    # Compute final heuristic with best gamma
    with np.errstate(divide='ignore', invalid='ignore'):
        heuristic = 1.0 / np.power(dm + 1e-9, best_gamma)
    
    # Ensure positive and finite
    heuristic[~np.isfinite(heuristic)] = 1e-9
    heuristic[heuristic <= 0] = 1e-9
    
    # Set diagonal to small value (no self-loops)
    np.fill_diagonal(heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
