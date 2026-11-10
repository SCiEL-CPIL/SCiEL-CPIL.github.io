import numpy as np
import time
from scipy.optimize import minimize

# Precompute the value arrays for the four slopes
_N = 10
_y_idx, _x_idx = np.indices((_N, _N))
_x_flat = _x_idx.flatten()
_y_flat = _y_idx.flatten()

# For each slope, precompute the bin index for each cell
_d_vals = (_x_flat - _y_flat + _N - 1).astype(np.int64)
_h0_vals = _x_flat.astype(np.int64)
_h1_vals = (_x_flat + _y_flat).astype(np.int64)
_h2_vals = (_x_flat + 2 * _y_flat).astype(np.int64)

_N_D = _d_vals.max() + 1
_N_H0 = _h0_vals.max() + 1
_N_H1 = _h1_vals.max() + 1
_N_H2 = _h2_vals.max() + 1

def _entropy_from_counts(counts):
    """Compute Shannon entropy in bits from a count array."""
    probs = counts[counts > 1e-15]
    if len(probs) == 0:
        return 0.0
    return -np.sum(probs * np.log2(probs))

def _score_fast_impl(w):
    """Fast score computation assuming w is a flat array of non-negative weights summing to 1."""
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
    safety_margin = 20
    
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
        
        # Soft max using log-sum-exp
        scaled = np.array([H0, H1, H2]) / temperature
        max_scaled = np.max(scaled)
        soft_max = (max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))) * temperature
        
        if soft_max <= 1e-9:
            return 0.0
            
        return -D / soft_max
    
    # Multi-start with different initializations
    num_restarts = 40
    time_per_restart = (timeout - safety_margin - 30) / num_restarts
    
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
        elif r == 5:
            # Concentrate on x+2y=0 line and x+2y=27
            current_P = np.zeros((n, n))
            current_P[0, 0] = 0.5
            current_P[9, 9] = 0.5
        elif r == 6:
            # Spread over x+2y=const lines with boost at extremes
            current_P = np.zeros((n, n))
            for y in range(n):
                for x in range(n):
                    current_P[y, x] = 1.0 / (n*n)
            current_P[0, 0] *= 3
            current_P[9, 9] *= 3
            current_P = _normalize(current_P)
        elif r == 7:
            # Concentrate on corners
            current_P = np.zeros((n, n))
            current_P[0, 0] = 0.25
            current_P[0, n-1] = 0.25
            current_P[n-1, 0] = 0.25
            current_P[n-1, n-1] = 0.25
        elif r == 8:
            # Row 0 and column 0
            current_P = np.zeros((n, n))
            current_P[0, :] = 0.5 / n
            current_P[:, 0] = 0.5 / n
            current_P[0, 0] = 0.5/n + 0.5/n - 1.0/(n*n)  # avoid double counting
            current_P = _normalize(current_P)
        elif r == 9:
            # Concentrate on x-y = 0 (main diagonal) and x-y = ±9
            current_P = np.zeros((n, n))
            np.fill_diagonal(current_P, 1.0/(3*n))
            current_P[0, n-1] = 1.0/3
            current_P[n-1, 0] = 1.0/3
            current_P = _normalize(current_P)
        else:
            # Perturb best found with various scales
            scale = 0.05 + 0.1 * (r % 5)
            current_P = best_P + np.random.normal(0, scale, (n, n))
            current_P = _normalize(current_P)
            if current_P is None:
                current_P = P.copy()
                
        current_score = _compute_ratio(current_P)
        
        # Phase 1: Hill climbing with adaptive step size
        step_size = 0.3
        local_time_limit = time_per_restart * 0.35
        local_start = time.time()
        no_improve_count = 0
        max_no_improve = 200
        
        while time.time() - local_start < local_time_limit:
            for _ in range(10):
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
                        current_P = best_P + np.random.normal(0, 0.15, (n, n))
                        current_P = _normalize(current_P)
                        if current_P is None:
                            current_P = P.copy()
                        current_score = _compute_ratio(current_P)
                        no_improve_count = 0
                        step_size = 0.3
                        break
                        
        # Phase 2: Smooth optimization using scipy L-BFGS-B on log-softmax params
        if time.time() - start_time < timeout - safety_margin - 20:
            current_P_clipped = np.clip(current_P, 1e-15, None)
            current_P_norm = current_P_clipped / current_P_clipped.sum()
            log_params = np.log(current_P_norm.flatten())
            
            # Try smooth objective with decreasing temperature
            for temp in [1.0, 0.5, 0.3, 0.15, 0.08, 0.04, 0.02]:
                if time.time() - start_time > timeout - safety_margin - 15:
                    break
                    
                try:
                    result = minimize(
                        _negative_ratio_smooth,
                        log_params,
                        args=(temp,),
                        method='L-BFGS-B',
                        options={'maxiter': 1500, 'ftol': 1e-14, 'gtol': 1e-10}
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
                            current_P_clipped = np.clip(current_P, 1e-15, None)
                            current_P_norm = current_P_clipped / current_P_clipped.sum()
                            log_params = np.log(current_P_norm.flatten())
                except Exception:
                    pass
        
        # Phase 3: Coordinate descent polish
        if time.time() - start_time < timeout - safety_margin - 10:
            coord_time_limit = min(8, time_per_restart * 0.25)
            coord_start = time.time()
            step_size = 0.05
            
            improved = True
            while improved and time.time() - coord_start < coord_time_limit:
                improved = False
                for i in range(n):
                    for j in range(n):
                        if time.time() - coord_start >= coord_time_limit:
                            break
                        for delta in [step_size, -step_size, step_size*0.5, -step_size*0.5]:
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
                                improved = True
                                if current_score > best_score:
                                    best_score = current_score
                                    best_P = current_P.copy()
                step_size *= 0.7
                if step_size < 0.001:
                    break
            
        # Phase 4: Fine-grained random hill climbing polish
        if time.time() - start_time < timeout - safety_margin - 5:
            step_size = 0.03
            polish_time_limit = min(8, time_per_restart * 0.2)
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
                    step_size *= 0.96
                else:
                    step_size *= 0.98
                    if step_size < 0.001:
                        break
                        
    # Final polish: very fine-grained search on best_P
    if time.time() - start_time < timeout - safety_margin:
        final_P = best_P.copy()
        final_score = best_score
        step_size = 0.015
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
                    if no_improve > 200 or step_size < 0.0003:
                        break
            
            if final_score > best_score:
                best_P = final_P
                best_score = final_score
        
        # Final scipy polish with very tight tolerances
        if time.time() - final_start < final_limit - 5:
            try:
                final_P_clipped = np.clip(final_P, 1e-15, None)
                final_P_norm = final_P_clipped / final_P_clipped.sum()
                log_params = np.log(final_P_norm.flatten())
                
                result = minimize(
                    _negative_ratio_smooth,
                    log_params,
                    args=(0.01,),
                    method='L-BFGS-B',
                    options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-12}
                )
                
                if result.success or result.fun < -final_score:
                    opt_params = result.x
                    opt_P = _log_softmax_params_to_prob(opt_params).reshape(n, n)
                    opt_score = _compute_ratio(opt_P)
                    
                    if opt_score > final_score:
                        best_P = opt_P
                        best_score = opt_score
            except Exception:
                pass
    
    return best_P
