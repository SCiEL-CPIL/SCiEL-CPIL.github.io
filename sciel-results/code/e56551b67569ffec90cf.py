import numpy as np
import time
from scipy.optimize import minimize

# Precompute the value arrays for the four slopes
_N = 10
_y_idx, _x_idx = np.indices((_N, _N))
_x_flat = _x_idx.flatten()
_y_flat = _y_idx.flatten()

# For each slope, precompute the bin index for each cell
# D: x - y, range -9..9 -> shift by 9 to get 0..18
_d_vals = (_x_flat - _y_flat + _N - 1).astype(np.int64)
# H0: x, range 0..9
_h0_vals = _x_flat.astype(np.int64)
# H1: x + y, range 0..18
_h1_vals = (_x_flat + _y_flat).astype(np.int64)
# H2: x + 2y, range 0..27
_h2_vals = (_x_flat + 2 * _y_flat).astype(np.int64)

# Number of bins for each
_N_D = _d_vals.max() + 1
_N_H0 = _h0_vals.max() + 1
_N_H1 = _h1_vals.max() + 1
_N_H2 = _h2_vals.max() + 1

# Precompute masks for fast entropy computation
_d_mask = _d_vals  # bin indices
_h0_mask = _h0_vals
_h1_mask = _h1_vals
_h2_mask = _h2_vals

def _entropy_from_counts(counts):
    """Compute Shannon entropy in bits from a count array."""
    probs = counts[counts > 1e-15]
    if len(probs) == 0:
        return 0.0
    return -np.sum(probs * np.log2(probs))

def _score_fast_impl(w):
    """Fast score computation assuming w is a flat array of non-negative weights summing to 1."""
    # Use add.at for pooled probabilities - faster than bincount for small arrays
    d_probs = np.zeros(_N_D)
    np.add.at(d_probs, _d_vals, w)
    h0_probs = np.zeros(_N_H0)
    np.add.at(h0_probs, _h0_vals, w)
    h1_probs = np.zeros(_N_H1)
    np.add.at(h1_probs, _h1_vals, w)
    h2_probs = np.zeros(_N_H2)
    np.add.at(h2_probs, _h2_vals, w)
    
    D = _entropy_from_counts(d_probs)
    H0 = _entropy_from_counts(h0_probs)
    H1 = _entropy_from_counts(h1_probs)
    H2 = _entropy_from_counts(h2_probs)
    
    M = max(H0, H1, H2)
    
    if M <= 1e-9:
        return 0.0
        
    return D / M

def _compute_ratio(P):
    P = np.clip(P, 0, None)
    s = P.sum()
    if s <= 1e-15:
        return 0.0
    w = P.flatten() / s
    return _score_fast_impl(w)

def arithmetic_kakeya_distribution() -> np.ndarray:
    n = _N
    P = np.ones((n, n), dtype=np.float64) / (n * n)
    
    start_time = time.time()
    timeout = 630
    safety_margin = 15
    
    best_score = _compute_ratio(P)
    best_P = P.copy()
    
    np.random.seed(42)
    
    def _normalize(P):
        P = np.clip(P, 0, None)
        s = P.sum()
        if s <= 1e-15:
            return None
        return P / s
    
    def _score_fast(P_flat):
        """Fast score computation assuming P_flat is already normalized and non-negative."""
        return _score_fast_impl(P_flat)
    
    def _log_softmax_params_to_prob(params):
        """Convert unconstrained params to probability via log-sum-exp."""
        log_p = params
        max_log_p = np.max(log_p)
        log_probs = log_p - max_log_p
        exp_probs = np.exp(log_probs)
        probs = exp_probs / np.sum(exp_probs)
        return probs
    
    def _negative_ratio_from_params(params):
        """Objective for scipy: minimize negative ratio."""
        probs = _log_softmax_params_to_prob(params)
        return -_score_fast(probs)
    
    def _smooth_max_entropy(H0, H1, H2, temperature=1.0):
        """Soft max using log-sum-exp for smoothness."""
        max_h = np.maximum(np.maximum(H0, H1), H2)
        # Use soft-max: log(exp(H0/T) + exp(H1/T) + exp(H2/T)) * T
        scaled = np.array([H0, H1, H2]) / temperature
        max_scaled = np.max(scaled)
        soft_max = (max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))) * temperature
        return soft_max
    
    def _negative_ratio_smooth(params, temperature=0.5):
        """Smooth objective using soft-max for M."""
        probs = _log_softmax_params_to_prob(params)
        d_probs = np.zeros(_N_D)
        np.add.at(d_probs, _d_vals, probs)
        h0_probs = np.zeros(_N_H0)
        np.add.at(h0_probs, _h0_vals, probs)
        h1_probs = np.zeros(_N_H1)
        np.add.at(h1_probs, _h1_vals, probs)
        h2_probs = np.zeros(_N_H2)
        np.add.at(h2_probs, _h2_vals, probs)
        
        D = _entropy_from_counts(d_probs)
        H0 = _entropy_from_counts(h0_probs)
        H1 = _entropy_from_counts(h1_probs)
        H2 = _entropy_from_counts(h2_probs)
        
        M = _smooth_max_entropy(H0, H1, H2, temperature)
        
        if M <= 1e-9:
            return 0.0
            
        return -D / M
    
    # Multi-start with different initializations
    num_restarts = 30
    time_per_restart = (timeout - safety_margin - 60) / num_restarts  # reserve 60s for polish
    
    for r in range(num_restarts):
        if time.time() - start_time > timeout - safety_margin:
            break
            
        # Initialize current P
        if r == 0:
            current_P = P.copy()
        elif r == 1:
            # Concentrate on x=0
            current_P = np.zeros((n, n))
            current_P[0, :] = 1.0 / n
        elif r == 2:
            # Concentrate on a few x values
            current_P = np.zeros((n, n))
            current_P[0, :] = 0.5 / n
            current_P[9, :] = 0.5 / n
        elif r == 3:
            # Diagonal pattern
            current_P = np.zeros((n, n))
            np.fill_diagonal(current_P, 1.0 / n)
        elif r == 4:
            # Anti-diagonal
            current_P = np.zeros((n, n))
            for i in range(n):
                current_P[i, n-1-i] = 1.0 / n
        else:
            # Perturb best found
            current_P = best_P + np.random.normal(0, 0.1, (n, n))
            current_P = _normalize(current_P)
            if current_P is None:
                current_P = P.copy()
                
        current_score = _compute_ratio(current_P)
        
        # Phase 1: Hill climbing with adaptive step size
        step_size = 0.3
        local_time_limit = time_per_restart * 0.5
        local_start = time.time()
        no_improve_count = 0
        max_no_improve = 200
        
        while time.time() - local_start < local_time_limit:
            # Try a few random moves per iteration
            for _ in range(5):
                if time.time() - local_start >= local_time_limit:
                    break
                    
                i, j = np.random.randint(0, n, 2)
                delta = np.random.uniform(-step_size, step_size)
                
                new_P = current_P.copy()
                new_P[i, j] += delta
                
                if new_P[i, j] < -1e-9:
                    continue
                    
                new_P = np.clip(new_P, 0, None)
                s = new_P.sum()
                if s <= 1e-9:
                    continue
                new_P = new_P / s
                
                new_score = _score_fast(new_P.flatten())
                
                if new_score > current_score:
                    current_P = new_P
                    current_score = new_score
                    no_improve_count = 0
                    if current_score > best_score:
                        best_score = current_score
                        best_P = current_P.copy()
                    step_size *= 0.97
                else:
                    no_improve_count += 1
                    if no_improve_count > max_no_improve:
                        # Restart local search from a perturbation of best
                        current_P = best_P + np.random.normal(0, 0.15, (n, n))
                        current_P = _normalize(current_P)
                        if current_P is None:
                            current_P = P.copy()
                        current_score = _compute_ratio(current_P)
                        no_improve_count = 0
                        step_size = 0.3
                        break
                        
        # Phase 2: Smooth optimization using scipy L-BFGS-B on log-softmax params
        if time.time() - start_time < timeout - safety_margin - 30:
            # Convert current_P to log-softmax parameters
            current_P_clipped = np.clip(current_P, 1e-15, None)
            current_P_norm = current_P_clipped / current_P_clipped.sum()
            log_params = np.log(current_P_norm.flatten())
            
            # Try smooth objective with decreasing temperature
            for temp in [1.0, 0.5, 0.2, 0.1]:
                if time.time() - start_time > timeout - safety_margin - 20:
                    break
                    
                try:
                    result = minimize(
                        _negative_ratio_smooth,
                        log_params,
                        args=(temp,),
                        method='L-BFGS-B',
                        options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
                    )
                    
                    if result.success or result.fun < -current_score:
                        opt_params = result.x
                        opt_P = _log_softmax_params_to_prob(opt_params).reshape(n, n)
                        opt_score = _compute_ratio(opt_P)
                        
                        if opt_score > current_score:
                            current_P = opt_P
                            current_score = opt_score
                            if current_score > best_score:
                                best_score = current_score
                                best_P = current_P.copy()
                except Exception:
                    pass
        
        # Phase 3: Fine-grained hill climbing polish
        if time.time() - start_time < timeout - safety_margin - 10:
            step_size = 0.05
            polish_time_limit = min(15, time_per_restart * 0.3)
            polish_start = time.time()
            
            while time.time() - polish_start < polish_time_limit:
                i, j = np.random.randint(0, n, 2)
                delta = np.random.uniform(-step_size, step_size)
                
                new_P = current_P.copy()
                new_P[i, j] += delta
                
                if new_P[i, j] < -1e-9:
                    continue
                    
                new_P = np.clip(new_P, 0, None)
                s = new_P.sum()
                if s <= 1e-9:
                    continue
                new_P = new_P / s
                
                new_score = _score_fast(new_P.flatten())
                
                if new_score > current_score:
                    current_P = new_P
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_P = current_P.copy()
                    step_size *= 0.95
                else:
                    step_size *= 0.98
                    if step_size < 0.001:
                        break
                        
    # Final polish: very fine-grained search on best_P
    if time.time() - start_time < timeout - safety_margin:
        final_P = best_P.copy()
        final_score = best_score
        step_size = 0.02
        final_start = time.time()
        final_limit = timeout - safety_margin - (time.time() - start_time)
        
        if final_limit > 5:
            no_improve = 0
            while time.time() - final_start < final_limit:
                i, j = np.random.randint(0, n, 2)
                delta = np.random.uniform(-step_size, step_size)
                
                new_P = final_P.copy()
                new_P[i, j] += delta
                
                if new_P[i, j] < -1e-9:
                    continue
                    
                new_P = np.clip(new_P, 0, None)
                s = new_P.sum()
                if s <= 1e-9:
                    continue
                new_P = new_P / s
                
                new_score = _score_fast(new_P.flatten())
                
                if new_score > final_score:
                    final_P = new_P
                    final_score = new_score
                    no_improve = 0
                    step_size *= 0.98
                else:
                    no_improve += 1
                    step_size *= 0.97
                    if no_improve > 100 or step_size < 0.0005:
                        break
            
            if final_score > best_score:
                best_P = final_P
                best_score = final_score
    
    return best_P
