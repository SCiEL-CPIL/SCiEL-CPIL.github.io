"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _fast_aco_eval(distance_matrix, demands, capacity, heuristic, n_ants=5, n_iters=5, seed=42):
    """
    A very fast, simplified ACO evaluation to approximate the true ACO performance.
    This is used to filter the top candidates from the greedy pre-screening.
    """
    n = len(demands)
    rng = np.random.RandomState(seed)
    pheromone = np.ones_like(distance_matrix)
    best_overall = np.inf
    
    for it in range(n_iters):
        best_iter_cost = np.inf
        
        for ant in range(n_ants):
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            current = 0
            remaining_cap = capacity
            total_dist = 0.0
            
            while not np.all(visited):
                scores = pheromone[current] * heuristic[current]
                
                # Mask visited and infeasible
                feasible_mask = (~visited) & (demands <= remaining_cap)
                scores[~feasible_mask] = 0.0
                scores[0] = 0.0
                
                if not np.any(scores > 0):
                    total_dist += distance_matrix[current, 0]
                    current = 0
                    remaining_cap = capacity
                    continue
                
                total_score = np.sum(scores)
                if total_score <= 0:
                    feasible_indices = np.where(feasible_mask)[0]
                    if len(feasible_indices) == 0:
                        total_dist += distance_matrix[current, 0]
                        current = 0
                        remaining_cap = capacity
                        continue
                    next_node = feasible_indices[rng.randint(len(feasible_indices))]
                else:
                    probs = scores / total_score
                    next_node = rng.choice(n, p=probs)
                
                total_dist += distance_matrix[current, next_node]
                remaining_cap -= demands[next_node]
                visited[next_node] = True
                current = next_node
            
            total_dist += distance_matrix[current, 0]
            
            if total_dist < best_iter_cost:
                best_iter_cost = total_dist
        
        # Update pheromone
        pheromone *= 0.9
        pheromone += 1.0 / best_iter_cost
        
        if best_iter_cost < best_overall:
            best_overall = best_iter_cost
    
    return best_overall

def _greedy_eval_fast(distance_matrix, demands, capacity, heuristic):
    """
    Fast greedy evaluation for initial screening.
    """
    n = len(demands)
    visited = np.zeros(n, dtype=bool)
    visited[0] = True
    current = 0
    remaining_cap = capacity
    total_dist = 0.0
    
    for _ in range(n - 1):
        scores = heuristic[current].copy()
        feasible_mask = (~visited) & (demands <= remaining_cap)
        scores[~feasible_mask] = -np.inf
        scores[0] = -np.inf
        
        if not np.any(np.isfinite(scores)):
            total_dist += distance_matrix[current, 0]
            current = 0
            remaining_cap = capacity
            continue
        
        best_j = np.argmax(scores)
        total_dist += distance_matrix[current, best_j]
        remaining_cap -= demands[best_j]
        visited[best_j] = True
        current = best_j
    
    total_dist += distance_matrix[current, 0]
    return total_dist

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Returns an edge-attractiveness matrix using a depot-pressure field model.
    
    The heuristic is structured as:
    H[i,j] = (1/d[i,j]) * depot_pressure[j] * capacity_efficiency[i,j]
    
    Where:
    - depot_pressure[j]: inversely proportional to demand[j], so high-demand nodes 
      are prioritized early (when truck is full)
    - capacity_efficiency: scales by the ratio of remaining capacity potential 
      at departure to the demand at arrival, encouraging routes that cluster
      compatible demands together
    """
    n = len(demands)
    d_safe = distance_matrix + 1e-9
    start_time = time.time()
    deadline = start_time + 295.0  # Safety margin
    
    # Precompute demand statistics
    demand_ratio = demands / capacity
    # Depot pressure: high demand nodes should be visited early
    # pressure[j] = 1 / (1 + gamma * demand[j])
    # Higher gamma means stronger pressure to visit high-demand nodes early
    
    def make_depot_pressure_heuristic(gamma, delta, epsilon):
        """
        gamma: controls depot pressure strength (higher = more urgent for high demand)
        delta: controls distance decay (higher = more local)
        epsilon: controls capacity efficiency coupling
        """
        # Depot pressure: nodes with higher demand get higher pressure
        # This encourages visiting them early when the truck is full
        pressure = 1.0 / (1.0 + gamma * demands)
        
        # Distance component: closer is better, with adjustable decay
        dist_component = 1.0 / (d_safe ** delta)
        
        # Capacity efficiency: for edge i->j, scale by how well the demands fit
        # When at node i, if remaining_cap is high, we can take larger demands
        # Approximate: efficiency[i,j] = 1 / (1 + epsilon * |demand[i] - demand[j]|)
        # This encourages clustering similar demands together
        demand_diff = np.abs(demands[:, None] - demands[None, :])
        capacity_eff = 1.0 / (1.0 + epsilon * demand_diff)
        
        # Combine: H[i,j] = dist_component[i,j] * pressure[j] * capacity_eff[i,j]
        h = dist_component * pressure[None, :] * capacity_eff
        
        # Ensure positivity and valid shape
        h = np.maximum(h, 1e-9)
        np.fill_diagonal(h, 1e-9)
        
        return h

    # Generate diverse candidates in the new parameter space
    candidate_params = []
    
    # Systematic exploration
    for gamma in [0.5, 1.0, 2.0, 4.0]:
        for delta in [0.8, 1.0, 1.5]:
            for epsilon in [0.0, 0.5, 1.0]:
                candidate_params.append((gamma, delta, epsilon))
    
    # Random sampling
    rng = np.random.RandomState(42)
    for _ in range(30):
        g = rng.uniform(0.3, 5.0)
        d = rng.uniform(0.5, 2.0)
        e = rng.uniform(0.0, 1.5)
        candidate_params.append((g, d, e))
    
    # Phase 1: Greedy Screening
    greedy_scores = []
    heuristics_cache = []
    
    for g, d, e in candidate_params:
        if time.time() > deadline - 30.0:
            break
        h = make_depot_pressure_heuristic(g, d, e)
        score = _greedy_eval_fast(distance_matrix, demands, capacity, h)
        greedy_scores.append(score)
        heuristics_cache.append((score, h, g, d, e))
    
    # Sort by greedy score
    heuristics_cache.sort(key=lambda x: x[0])
    
    # Phase 2: Fast ACO Verification
    top_k = min(6, len(heuristics_cache))
    top_candidates = heuristics_cache[:top_k]
    
    best_aco_score = np.inf
    best_heuristic = top_candidates[0][1] if top_candidates else make_depot_pressure_heuristic(1.0, 1.0, 0.5)
    
    for score, h, g, d, e in top_candidates:
        if time.time() > deadline - 10.0:
            break
            
        aco_score = _fast_aco_eval(distance_matrix, demands, capacity, h)
        
        if aco_score < best_aco_score:
            best_aco_score = aco_score
            best_heuristic = h

    # Final Safety Checks
    if not np.all(np.isfinite(best_heuristic)):
        return 1.0 / d_safe
    if np.any(best_heuristic <= 0):
        best_heuristic = np.clip(best_heuristic, 1e-9, None)
    if best_heuristic.shape != distance_matrix.shape:
        return 1.0 / d_safe
        
    return best_heuristic
# EVOLVE-BLOCK-END
