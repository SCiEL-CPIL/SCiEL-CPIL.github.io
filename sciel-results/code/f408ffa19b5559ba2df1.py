# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def heuristics(prizes, distance_matrix, max_len):
    start_time = time.time()
    time_limit = 320.0  # Safety margin before 330s kill
    
    N = len(prizes)
    d_j0 = distance_matrix[:, 0]  # distance from node 0 to each node (symmetric)
    
    prizes_arr = np.asarray(prizes, dtype=np.float64)
    dm = np.asarray(distance_matrix, dtype=np.float64)
    
    # Run a single ACO instance with given heuristic matrix and seed
    def run_aco(eta, seed):
        np.random.seed(seed)
        tau = np.ones((N, N))
        best_prize_global = 0.0
        
        for it in range(30):
            best_prize_iter = 0.0
            
            for ant in range(20):
                current = 0
                path_len = 0.0
                prize_total = 0.0
                visited = [0]
                
                while True:
                    eligible = []
                    for j in range(N):
                        if j in visited:
                            continue
                        if path_len + dm[current, j] + dm[j, 0] <= max_len:
                            eligible.append(j)
                    
                    if not eligible:
                        break
                    
                    total_p = 0.0
                    probs = np.zeros(len(eligible))
                    for k, j in enumerate(eligible):
                        p = tau[current, j] * eta[current, j]
                        probs[k] = p
                        total_p += p
                    
                    if total_p <= 0:
                        break
                    
                    probs /= total_p
                    r = np.random.rand()
                    cumsum = np.cumsum(probs)
                    idx = np.searchsorted(cumsum, r)
                    if idx >= len(eligible):
                        idx = len(eligible) - 1
                    
                    next_node = eligible[idx]
                    path_len += dm[current, next_node]
                    prize_total += prizes_arr[next_node]
                    visited.append(next_node)
                    current = next_node
            
            if prize_total > best_prize_iter:
                best_prize_iter = prize_total
            
            if best_prize_iter > best_prize_global:
                best_prize_global = best_prize_iter
            
            tau = tau * 0.9
            tau += best_prize_iter / 10.0
        
        return best_prize_global
    
    # Regret-based heuristic: entry (i, j) represents the loss in expected total prize
    # if the ant at node i fails to visit node j next.
    # Approximation: prize[j] * probability that j is the best next step in a greedy tour from i,
    # weighted by how much remaining path length would be consumed.
    
    def make_regret_eta(params):
        """
        Regret-based heuristic with parameters:
          - alpha: exponent on prize
          - beta: exponent on distance penalty
          - gamma: weight on return-to-origin distance
          - delta: weight on regret (opportunity cost of not visiting j)
          - form: 0 = standard, 1 = normalized regret
        """
        alpha = params.get('alpha', 1.0)
        beta = params.get('beta', 1.0)
        gamma = params.get('gamma', 1.0)
        delta = params.get('delta', 0.5)
        form = params.get('form', 0)
        
        dm_safe = np.maximum(dm, 1e-9)
        prizes_safe = np.maximum(prizes_arr[None, :], 1e-9)
        
        # Base attractiveness: prize / distance
        base = (prizes_safe ** alpha) / (dm_safe ** beta)
        
        # Penalty for nodes far from origin (harder to return)
        return_penalty = (1.0 + gamma * d_j0[None, :]) ** beta
        
        base /= return_penalty
        
        # Regret component: for each i, j, how much "value" is lost if we skip j?
        # Approximate: prize[j] * (1 - d[i,j]/max_len) * (1 - d[j,0]/max_len)
        # This captures: high prize, close to current, close to origin = high regret if skipped
        
        if form == 0:
            # Standard regret
            regret = (prizes_safe ** alpha) * \
                     np.maximum(0, 1.0 - dm_safe / max_len) ** beta * \
                     np.maximum(0, 1.0 - d_j0[None, :] / max_len) ** beta
        else:
            # Normalized regret: rank-based
            # For each row i, normalize prize[j] * (1 - d[i,j]/max_len) * (1 - d[j,0]/max_len)
            raw_regret = (prizes_safe ** alpha) * \
                         np.maximum(0, 1.0 - dm_safe / max_len) ** beta * \
                         np.maximum(0, 1.0 - d_j0[None, :] / max_len) ** beta
            row_max = np.max(raw_regret, axis=1, keepdims=True)
            row_max = np.maximum(row_max, 1e-9)
            regret = raw_regret / row_max
        
        eta = base + delta * regret
        eta[eta <= 0] = 1e-9
        eta[~np.isfinite(eta)] = 1e-9
        return eta
    
    # Alternative: pure regret-based (no base term)
    def make_pure_regret_eta(params):
        alpha = params.get('alpha', 1.0)
        beta = params.get('beta', 1.0)
        gamma = params.get('gamma', 1.0)
        
        dm_safe = np.maximum(dm, 1e-9)
        prizes_safe = np.maximum(prizes_arr[None, :], 1e-9)
        
        # Regret: prize * feasibility * return_feasibility
        feasibility = np.maximum(0, 1.0 - dm_safe / max_len) ** beta
        return_feasibility = np.maximum(0, 1.0 - d_j0[None, :] / max_len) ** beta
        
        eta = (prizes_safe ** alpha) * feasibility * return_feasibility
        eta[eta <= 0] = 1e-9
        eta[~np.isfinite(eta)] = 1e-9
        return eta
    
    # Combined: blend multiple heuristic families
    def make_blend_eta(params):
        w_regret = params.get('w_regret', 0.5)
        w_prize_dist = params.get('w_prize_dist', 0.3)
        w_return = params.get('w_return', 0.2)
        alpha = params.get('alpha', 1.0)
        beta = params.get('beta', 1.0)
        
        dm_safe = np.maximum(dm, 1e-9)
        prizes_safe = np.maximum(prizes_arr[None, :], 1e-9)
        
        # Component 1: prize / distance
        c1 = (prizes_safe ** alpha) / (dm_safe ** beta)
        
        # Component 2: regret-based
        feasibility = np.maximum(0, 1.0 - dm_safe / max_len) ** beta
        return_feasibility = np.maximum(0, 1.0 - d_j0[None, :] / max_len) ** beta
        c2 = (prizes_safe ** alpha) * feasibility * return_feasibility
        
        # Component 3: prize / (distance + return_distance)
        c3 = (prizes_safe ** alpha) / (dm_safe + d_j0[None, :])
        
        # Normalize each component
        for c in [c1, c2, c3]:
            row_max = np.max(c, axis=1, keepdims=True)
            row_max = np.maximum(row_max, 1e-9)
            c /= row_max
        
        eta = w_regret * c2 + w_prize_dist * c1 + w_return * c3
        eta[eta <= 0] = 1e-9
        eta[~np.isfinite(eta)] = 1e-9
        return eta
    
    # Candidate parameter sets
    candidate_params = [
        # Regret-based
        {'alpha': 1.0, 'beta': 1.0, 'gamma': 1.0, 'delta': 0.5, 'form': 0},
        {'alpha': 1.0, 'beta': 1.0, 'gamma': 1.0, 'delta': 1.0, 'form': 0},
        {'alpha': 1.0, 'beta': 1.0, 'gamma': 1.0, 'delta': 0.3, 'form': 0},
        {'alpha': 1.5, 'beta': 1.0, 'gamma': 1.0, 'delta': 0.5, 'form': 0},
        {'alpha': 0.5, 'beta': 1.0, 'gamma': 1.0, 'delta': 0.5, 'form': 0},
        {'alpha': 1.0, 'beta': 0.5, 'gamma': 1.0, 'delta': 0.5, 'form': 0},
        {'alpha': 1.0, 'beta': 1.5, 'gamma': 1.0, 'delta': 0.5, 'form': 0},
        {'alpha': 1.0, 'beta': 1.0, 'gamma': 2.0, 'delta': 0.5, 'form': 0},
        {'alpha': 1.0, 'beta': 1.0, 'gamma': 0.5, 'delta': 0.5, 'form': 0},
        {'alpha': 1.0, 'beta': 1.0, 'gamma': 1.0, 'delta': 0.5, 'form': 1},
        {'alpha': 1.0, 'beta': 1.0, 'gamma': 1.0, 'delta': 1.0, 'form': 1},
        {'alpha': 1.5, 'beta': 1.5, 'gamma': 1.0, 'delta': 0.5, 'form': 0},
        {'alpha': 2.0, 'beta': 1.0, 'gamma': 1.0, 'delta': 0.5, 'form': 0},
        
        # Pure regret
        {'alpha': 1.0, 'beta': 1.0, 'gamma': 1.0},
        {'alpha': 1.5, 'beta': 1.0, 'gamma': 1.0},
        {'alpha': 1.0, 'beta': 0.5, 'gamma': 1.0},
        {'alpha': 1.0, 'beta': 1.5, 'gamma': 1.0},
        {'alpha': 2.0, 'beta': 1.0, 'gamma': 1.0},
        {'alpha': 0.5, 'beta': 1.0, 'gamma': 1.0},
        
        # Blend
        {'w_regret': 0.5, 'w_prize_dist': 0.3, 'w_return': 0.2, 'alpha': 1.0, 'beta': 1.0},
        {'w_regret': 0.7, 'w_prize_dist': 0.2, 'w_return': 0.1, 'alpha': 1.0, 'beta': 1.0},
        {'w_regret': 0.3, 'w_prize_dist': 0.5, 'w_return': 0.2, 'alpha': 1.0, 'beta': 1.0},
        {'w_regret': 0.5, 'w_prize_dist': 0.2, 'w_return': 0.3, 'alpha': 1.0, 'beta': 1.0},
        {'w_regret': 0.6, 'w_prize_dist': 0.3, 'w_return': 0.1, 'alpha': 1.5, 'beta': 1.0},
        {'w_regret': 0.4, 'w_prize_dist': 0.4, 'w_return': 0.2, 'alpha': 1.0, 'beta': 0.5},
    ]
    
    eval_seeds = [0, 1, 2]
    
    best_score = -1
    best_eta = None
    best_params = None
    best_func = None
    
    for params in candidate_params:
        if time.time() - start_time > time_limit - 20:
            break
        
        # Determine which function to use based on params
        if 'delta' in params:
            eta = make_regret_eta(params)
            func_name = 'regret'
        elif 'w_regret' in params:
            eta = make_blend_eta(params)
            func_name = 'blend'
        else:
            eta = make_pure_regret_eta(params)
            func_name = 'pure'
        
        scores = []
        for seed in eval_seeds:
            if time.time() - start_time > time_limit - 15:
                break
            s = run_aco(eta, seed)
            scores.append(s)
        
        if scores:
            avg_score = np.mean(scores)
            if avg_score > best_score:
                best_score = avg_score
                best_eta = eta.copy()
                best_params = params.copy()
                best_func = func_name
    
    # Local search around best params
    if best_eta is not None and time.time() - start_time < time_limit - 30:
        np.random.seed(42)
        
        for iteration in range(30):
            if time.time() - start_time > time_limit - 20:
                break
            
            # Perturb parameters
            if best_func == 'regret':
                new_params = {
                    'alpha': max(0.1, best_params.get('alpha', 1.0) + np.random.normal(0, 0.2)),
                    'beta': max(0.1, best_params.get('beta', 1.0) + np.random.normal(0, 0.2)),
                    'gamma': max(0.0, best_params.get('gamma', 1.0) + np.random.normal(0, 0.3)),
                    'delta': max(0.0, best_params.get('delta', 0.5) + np.random.normal(0, 0.2)),
                    'form': best_params.get('form', 0) if np.random.rand() > 0.2 else (1 - best_params.get('form', 0))
                }
                eta = make_regret_eta(new_params)
            elif best_func == 'blend':
                new_params = {
                    'w_regret': max(0.0, min(1.0, best_params.get('w_regret', 0.5) + np.random.normal(0, 0.1))),
                    'w_prize_dist': max(0.0, min(1.0, best_params.get('w_prize_dist', 0.3) + np.random.normal(0, 0.1))),
                    'w_return': max(0.0, min(1.0, best_params.get('w_return', 0.2) + np.random.normal(0, 0.1))),
                    'alpha': max(0.1, best_params.get('alpha', 1.0) + np.random.normal(0, 0.2)),
                    'beta': max(0.1, best_params.get('beta', 1.0) + np.random.normal(0, 0.2))
                }
                # Normalize weights
                total_w = new_params['w_regret'] + new_params['w_prize_dist'] + new_params['w_return']
                if total_w > 0:
                    new_params['w_regret'] /= total_w
                    new_params['w_prize_dist'] /= total_w
                    new_params['w_return'] /= total_w
                eta = make_blend_eta(new_params)
            else:  # pure
                new_params = {
                    'alpha': max(0.1, best_params.get('alpha', 1.0) + np.random.normal(0, 0.2)),
                    'beta': max(0.1, best_params.get('beta', 1.0) + np.random.normal(0, 0.2)),
                    'gamma': max(0.0, best_params.get('gamma', 1.0) + np.random.normal(0, 0.3))
                }
                eta = make_pure_regret_eta(new_params)
            
            scores = []
            for seed in eval_seeds:
                if time.time() - start_time > time_limit - 15:
                    break
                s = run_aco(eta, seed)
                scores.append(s)
            
            if scores:
                avg_score = np.mean(scores)
                if avg_score > best_score:
                    best_score = avg_score
                    best_eta = eta.copy()
                    best_params = new_params
    
    # Final safety check
    if best_eta is None or best_eta.shape != dm.shape or not np.all(np.isfinite(best_eta)):
        # Fallback: simple prize/distance heuristic
        dm_safe = np.maximum(dm, 1e-9)
        prizes_safe = np.maximum(prizes_arr[None, :], 1e-9)
        best_eta = prizes_safe / dm_safe
        best_eta[best_eta <= 0] = 1e-9
        best_eta[~np.isfinite(best_eta)] = 1e-9
    
    return best_eta

# EVOLVE-BLOCK-END
