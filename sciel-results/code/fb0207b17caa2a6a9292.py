"""Heuristic matrix for ant colony optimization of orienteering with regret-based flexibility."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _simulate_aco(prizes, distance_matrix, max_len, heuristic, n_ants=20, n_iters=30, seed=42):
    """Run the fixed ACO and return the best total prize collected by any ant."""
    rng = np.random.RandomState(seed)
    n = len(prizes)
    pheromone = np.ones((n, n))
    best_total = 0.0
    dist_to_home = distance_matrix[:, 0].copy()
    
    for it in range(n_iters):
        iter_best = 0.0
        for ant in range(n_ants):
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            current = 0
            length_so_far = 0.0
            total_prize = 0.0
            
            while True:
                remaining = max_len - length_so_far
                # Vectorized feasibility check
                feasible = (~visited) & ((distance_matrix[current, :] + dist_to_home) <= remaining)
                feasible[0] = False
                
                if not np.any(feasible):
                    break
                
                candidates = np.where(feasible)[0]
                weights = pheromone[current, candidates] * heuristic[current, candidates]
                weights = np.maximum(weights, 1e-9)
                probs = weights / weights.sum()
                
                idx = rng.choice(len(candidates), p=probs)
                next_node = candidates[idx]
                
                visited[next_node] = True
                length_so_far += distance_matrix[current, next_node]
                total_prize += prizes[next_node]
                current = next_node
            
            iter_best = max(iter_best, total_prize)
        
        best_total = max(best_total, iter_best)
        
        # Update pheromone
        pheromone *= 0.9
        pheromone += iter_best / 10.0
    
    return best_total


def _compute_regret_flexibility(distance_matrix, max_len, n):
    """
    Compute a flexibility/regret score for each node j.
    H_flex[j] is inversely proportional to the number of nodes k that can be visited
    after going to j from node 0, given the remaining budget.
    This rewards moves that preserve future routing flexibility.
    """
    dist_from_0 = distance_matrix[0, :]
    flex_scores = np.zeros(n)
    
    for j in range(n):
        if j == 0:
            flex_scores[j] = 1.0
            continue
        
        # Cost to go from 0 to j and back to 0
        cost_0_j_0 = dist_from_0[j] + distance_matrix[j, 0]
        
        if cost_0_j_0 > max_len:
            flex_scores[j] = 0.0
            continue
        
        remaining_after_j = max_len - dist_from_0[j]
        
        # Count how many other nodes k can be visited after j
        # We need: dist[j, k] + dist[k, 0] <= remaining_after_j
        # Note: k != 0, k != j
        num_continuations = 0
        for k in range(n):
            if k == 0 or k == j:
                continue
            if distance_matrix[j, k] + distance_matrix[k, 0] <= remaining_after_j:
                num_continuations += 1
        
        # Flexibility score: more continuations = more flexible
        # Use 1 + num to avoid division by zero and give some base value
        flex_scores[j] = 1.0 + num_continuations
    
    # Normalize to [0, 1] range for use as a multiplier
    max_flex = np.max(flex_scores)
    if max_flex > 0:
        flex_scores = flex_scores / max_flex
    
    # Ensure positive
    flex_scores = np.maximum(flex_scores, 1e-6)
    
    return flex_scores


def _make_heuristic(prizes, distance_matrix, max_len, params):
    """
    Generate heuristic matrix.
    params: (prize_exp, dist_exp, origin_exp, prize_w, dist_w, origin_w, flex_w)
    """
    prize_exp, dist_exp, origin_exp, prize_w, dist_w, origin_w, flex_w = params
    
    n = len(prizes)
    d_to_origin = distance_matrix[0, :]
    
    # Compute regret-based flexibility scores
    flex_scores = _compute_regret_flexibility(distance_matrix, max_len, n)
    
    # Base terms
    prize_term = (prizes[None, :] ** prize_exp + 1e-9) * prize_w
    dist_penalty = (1.0 / (distance_matrix + 1e-9) ** dist_exp) * dist_w
    origin_bonus = (1.0 / (d_to_origin[None, :] + 1e-9) ** origin_exp) * origin_w
    
    # Flexibility term: reward nodes that allow many future moves
    # flex_scores[j] is high if j allows many continuations
    # We want to penalize edges leading to low-flexibility nodes
    # So we multiply by (1 + flex_w * flex_scores[j])
    flex_term = (1.0 + flex_w * flex_scores[None, :])
    
    heuristic = prize_term * dist_penalty * origin_bonus * flex_term
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic


def _optimize_parameters(prizes, distance_matrix, max_len, initial_params, n_evals=40, time_budget=20.0):
    """Use Nelder-Mead to optimize heuristic parameters."""
    from scipy.optimize import minimize
    
    start_time = time.time()
    
    def objective(params):
        if time.time() - start_time > time_budget:
            return 1e10
        try:
            heuristic = _make_heuristic(prizes, distance_matrix, max_len, params)
            if not np.all(np.isfinite(heuristic)):
                return 1e10
            # Use fewer ants/iters for speed during optimization
            score = _simulate_aco(prizes, distance_matrix, max_len, heuristic, n_ants=10, n_iters=10, seed=42)
            return -score
        except Exception:
            return 1e10
    
    # Nelder-Mead optimization
    try:
        result = minimize(
            objective,
            initial_params,
            method='Nelder-Mead',
            options={'maxiter': n_evals, 'xatol': 1e-3, 'fatol': 1e-4, 'disp': False}
        )
        return result.x
    except Exception:
        return initial_params


def heuristics(prizes, distance_matrix, max_len):
    """
    Compute an edge-attractiveness matrix for ACO orienteering using regret-based flexibility.
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    n = len(prizes)
    
    start_time = time.time()
    deadline = start_time + 280.0  # 20 second safety margin before 300s limit
    
    # Initial parameters: (prize_exp, dist_exp, origin_exp, prize_w, dist_w, origin_w, flex_w)
    # flex_w controls the strength of the regret-based flexibility bonus
    initial_params = np.array([1.0, 2.0, 1.5, 1.0, 1.0, 1.0, 0.5])
    
    best_params = initial_params.copy()
    best_score = -np.inf
    
    # Try multiple starting points with varying flex_w
    starting_points = [
        np.array([1.0, 2.0, 1.5, 1.0, 1.0, 1.0, 0.5]),   # Moderate flexibility
        np.array([1.0, 2.5, 1.0, 1.0, 1.0, 1.0, 1.0]),   # Strong flexibility
        np.array([1.5, 2.0, 1.0, 1.0, 1.0, 1.0, 0.3]),   # High prize weight, low flex
        np.array([1.0, 3.0, 1.5, 1.0, 1.0, 1.0, 0.8]),   # Strong distance penalty, high flex
        np.array([1.2, 2.0, 1.2, 1.0, 1.0, 1.0, 0.6]),   # Balanced
        np.array([0.8, 2.5, 1.0, 1.0, 1.0, 1.0, 1.2]),   # Low prize weight, very high flex
    ]
    
    for sp in starting_points:
        elapsed = time.time() - start_time
        if elapsed > deadline - 15.0:
            break
        
        try:
            # Optimize parameters
            remaining_budget = min(20.0, deadline - (time.time() - start_time) - 5.0)
            if remaining_budget < 5.0:
                break
            
            optimized_params = _optimize_parameters(
                prizes, distance_matrix, max_len, sp,
                n_evals=30, time_budget=remaining_budget
            )
            
            elapsed = time.time() - start_time
            if elapsed > deadline - 10.0:
                break
            
            # Evaluate the optimized parameters with full simulation
            heuristic = _make_heuristic(prizes, distance_matrix, max_len, optimized_params)
            if not np.all(np.isfinite(heuristic)):
                continue
            
            score = _simulate_aco(prizes, distance_matrix, max_len, heuristic, n_ants=20, n_iters=30, seed=42)
            
            if score > best_score:
                best_score = score
                best_params = optimized_params.copy()
                
        except Exception:
            continue
    
    # If optimization failed or timed out, use a reasonable default
    if best_score == -np.inf:
        best_params = initial_params
    
    # Generate final heuristic
    heuristic = _make_heuristic(prizes, distance_matrix, max_len, best_params)
    
    # Ensure validity
    if not np.all(np.isfinite(heuristic)):
        # Fallback to simple heuristic with flexibility
        d_to_origin = distance_matrix[0, :]
        flex_scores = _compute_regret_flexibility(distance_matrix, max_len, n)
        prize_term = np.sqrt(prizes[None, :]) + 1e-9
        dist_penalty = 1.0 / (distance_matrix + 1e-9) ** 2.5
        origin_bonus = 1.0 / (d_to_origin[None, :] + 1e-9) ** 1.5
        flex_term = 1.0 + 0.5 * flex_scores[None, :]
        heuristic = prize_term * dist_penalty * origin_bonus * flex_term
    
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
