import numpy as np
import time
import scipy.optimize

def heuristics(prizes, distance_matrix, max_len):
    start_time = time.time()
    # Evaluator kills at 330s. We should return well before that.
    # Let's aim to finish in < 200s to be safe.
    deadline = start_time + 180.0 

    prizes = np.asarray(prizes, dtype=float)
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    n = len(prizes)

    if time.time() > deadline:
        return np.ones((n, n)) * 1e-9

    # Precompute constants
    dist_to_origin = distance_matrix[:, 0]
    # total_cost[i, j] = d[i, j] + d[j, 0] is the cost to go i->j->0
    # This is used for slack calculation
    total_cost = distance_matrix + dist_to_origin[None, :]
    max_cost = np.max(total_cost)
    if max_cost <= 0:
        max_cost = 1.0

    eps = 1e-9

    # Fast deterministic ant simulation
    # Returns total prize collected
    def get_prize(eta):
        visited = np.zeros(n, dtype=bool)
        visited[0] = True
        curr = 0
        total_len = 0.0
        total_prize = 0.0

        # Unvisited nodes mask for fast filtering
        unvisited = np.arange(1, n)
        
        for _ in range(n):
            if len(unvisited) == 0:
                break
            
            # Calculate new lengths for all unvisited nodes
            # new_len[j] = total_len + d[curr, j] + d[j, 0]
            # We need to check if new_len <= max_len
            # d[curr, unvisited] gives distances from curr to all unvisited
            d_curr_unv = distance_matrix[curr, unvisited]
            d_unv_0 = dist_to_origin[unvisited]
            
            new_lens = total_len + d_curr_unv + d_unv_0
            valid_mask = new_lens <= max_len
            
            if not np.any(valid_mask):
                break
            
            # Get valid unvisited nodes
            valid_unv = unvisited[valid_mask]
            
            # Get heuristic values for these valid nodes
            # eta[curr, valid_unv]
            eta_vals = eta[curr, valid_unv]
            
            # Find the node with the highest heuristic value
            # If there are ties, np.argmax picks the first one
            best_idx_in_valid = np.argmax(eta_vals)
            best_j = valid_unv[best_idx_in_valid]
            
            # Update state
            step_len = distance_matrix[curr, best_j]
            total_len += step_len
            total_prize += prizes[best_j]
            
            visited[best_j] = True
            curr = best_j
            
            # Remove best_j from unvisited list
            # This is O(n) but n=50, so it's fine. 
            # Alternatively, maintain a boolean mask.
            # Let's stick to list removal for clarity, it's fast enough for n=50.
            unvisited = np.setdiff1d(unvisited, [best_j], assume_unique=True)

        return total_prize

    def compute_heuristic(params):
        a, b, g, s = params
        
        # Ensure parameters are within reasonable bounds to avoid numerical issues
        # We rely on the optimizer's bounds, but add safety clips
        a = np.clip(a, 0.1, 10.0)
        b = np.clip(b, 0.0, 5.0)
        g = np.clip(g, 0.0, 5.0)
        s = np.clip(s, 0.0, 2.0)

        prize_term = (prizes + eps) ** a
        dist_ij_term = (distance_matrix + eps) ** b
        dist_j0_term = (dist_to_origin + eps) ** g
        
        # eta[i, j] depends on j's prize and distance to 0, and i's distance to j
        # shape: (n, n)
        # prize_term is (n,), dist_j0_term is (n,) -> (n, n) via broadcasting
        # dist_ij_term is (n, n)
        
        numerator = (prize_term / dist_j0_term)[None, :]
        denominator = dist_ij_term
        
        eta = numerator / denominator
        
        if s > 1e-6:
            # Slack factor: prefer edges that leave more room
            # slack = 1 - (d[i,j] + d[j,0]) / max_cost
            # If total cost is high, slack is low.
            slack_factor = 1.0 - s * (total_cost / max_cost)
            slack_factor = np.clip(slack_factor, 0.1, None)
            eta = eta * slack_factor
        
        eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
        return eta

    def objective(params):
        if time.time() > deadline - 5:
            return 0.0 # Penalize timeout
        eta = compute_heuristic(params)
        prize = get_prize(eta)
        return -prize # Minimize negative prize

    # Initial guesses for parameters (a, b, g, s)
    # Based on the "prize-to-radius" idea: a~1, g~1, b~1
    initial_guesses = [
        [1.0, 1.0, 1.0, 0.0],
        [2.0, 1.5, 1.0, 0.5],
        [1.5, 1.0, 2.0, 0.0],
        [3.0, 2.0, 1.0, 1.0],
        [1.0, 0.5, 1.0, 0.5],
        [2.0, 1.0, 0.5, 0.0],
    ]
    
    bounds = [(0.1, 10.0), (0.0, 5.0), (0.0, 5.0), (0.0, 2.0)]
    
    best_params = None
    best_score = -np.inf

    # Run Nelder-Mead from multiple starting points
    n_restarts = 5
    for i in range(n_restarts):
        if time.time() > deadline - 20:
            break
            
        x0 = initial_guesses[i % len(initial_guesses)]
        
        try:
            res = scipy.optimize.minimize(
                objective, 
                x0, 
                method='Nelder-Mead',
                bounds=bounds,
                options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-6, 'adaptive': True}
            )
            
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            continue

    # If optimization failed or wasn't run, fallback to a simple grid search
    if best_params is None:
        # Simple grid search as fallback
        best_params = [1.0, 1.0, 1.0, 0.0]
        for a in [1.0, 2.0, 3.0]:
            for b in [1.0, 2.0]:
                for g in [1.0, 2.0]:
                    for s in [0.0, 0.5]:
                        p = [a, b, g, s]
                        score = -objective(p)
                        if score > best_score:
                            best_score = score
                            best_params = p

    # Final polish: ensure we have a valid heuristic
    # Sometimes Nelder-Mead can end up at a boundary or slightly outside due to float errors
    # Clip to bounds again
    best_params = np.clip(best_params, [b[0] for b in bounds], [b[1] for b in bounds])
    
    final_heuristic = compute_heuristic(best_params)
    
    # Ensure output is valid
    final_heuristic = np.where(np.isfinite(final_heuristic) & (final_heuristic > 0), final_heuristic, 1e-9)
    
    return final_heuristic
