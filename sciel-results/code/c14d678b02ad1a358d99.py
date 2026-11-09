# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def _compute_P_exact(support, weights):
    """
    Compute P[X1 + X2 + X3 < 2*X4] exactly for a discrete measure.
    Returns the probability P.
    """
    n = len(support)
    if n == 0:
        return 0.0
    
    # Normalize weights
    w_sum = np.sum(weights)
    if w_sum <= 1e-9:
        return 0.0
    w = weights / w_sum
    
    # We need P = sum_{k,l} w_k w_l * (sum_{i,j: s_i + s_j < 2*s_l - s_k} w_i w_j)
    
    # Precompute pair sums and pair weights
    # pair_sums[i, j] = s_i + s_j
    # pair_w[i, j] = w_i * w_j
    pair_sums = support[:, None] + support[None, :]
    pair_w = w[:, None] * w[None, :]
    
    # Flatten for sorting
    flat_sums = pair_sums.ravel()
    flat_w = pair_w.ravel()
    
    # Sort pair sums
    order = np.argsort(flat_sums, kind='mergesort')
    sorted_sums = flat_sums[order]
    sorted_w = flat_w[order]
    
    # Cumulative sum of pair weights
    # cumsum_w[i] = sum(sorted_w[0..i-1])
    cumsum_w = np.cumsum(sorted_w)
    padded_cumsum = np.concatenate([[0.0], cumsum_w])
    
    # For each (k, l), threshold T_{k,l} = 2*s_l - s_k
    # We want sum of pair_w where pair_sum < T_{k,l}
    # This is padded_cumsum[ searchsorted(sorted_sums, T_{k,l}, side='left') ]
    
    # Thresholds matrix: T[k, l] = 2 * s_l - s_k
    # s_k is support[k], s_l is support[l]
    # T = 2 * support[None, :] - support[:, None]
    thresholds = 2.0 * support[None, :] - support[:, None]
    
    # Get indices into sorted_sums
    # side='left' ensures strict inequality <
    indices = np.searchsorted(sorted_sums, thresholds, side='left')
    
    # Get the cumulative probabilities
    inner_probs = padded_cumsum[indices]
    
    # Final P = sum_{k,l} w_k w_l inner_probs[k,l]
    P = np.sum(w[:, None] * w[None, :] * inner_probs)
    
    return P

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Optimize a discrete measure to maximize P[X1 + X2 + X3 < 2*X4].
    Strategy: Jointly optimize support and weights of a small number of atoms.
    Uses a heavy-tailed initialization to encourage large X4 values.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    # Number of atoms. 
    # Too few: can't represent complex distributions.
    # Too many: optimization landscape is high-dimensional and noisy.
    # n=30 is a good balance.
    n_atoms = 30
    
    # Objective function for the optimizer
    # Params: [s_0, ..., s_{n-1}, log_w_0, ..., log_w_{n-1}]
    # We optimize in log-space for weights to keep them positive.
    # We optimize support directly. To ensure support >= 0, we can use exp(s) or just bound s.
    # Let's use a transformation for support to ensure non-negativity and scale.
    # s_i = exp(raw_s_i) * scale + offset? 
    # Simpler: just optimize raw support values with bounds [0, 100].
    
    def objective(params):
        # Check time
        if time.time() > deadline:
            return 1e10 # Large penalty
        
        s_raw = params[:n_atoms]
        log_w_raw = params[n_atoms:]
        
        # Support: ensure non-negative. 
        # We allow support to be anywhere >= 0.
        # The optimizer might push them negative, so we clip or transform.
        # Let's use s = np.abs(s_raw) to be safe, or better, optimize in log-space for support too?
        # Log-space support: s_i = exp(raw_s_i). This ensures s_i > 0.
        # But we might want s_i close to 0. exp(-large) is close to 0.
        # Let's use s = np.exp(s_raw). 
        # If s_raw is -10, s ~ 0. If s_raw is 5, s ~ 150.
        # This covers a wide range.
        
        try:
            support = np.exp(s_raw)
            # Clip very large values to avoid overflow in sums, though exp(50) is ~5e21, 
            # float64 handles it, but pairwise sums might get huge. 
            # The inequality is scale-invariant? 
            # P[X1+X2+X3 < 2X4] is invariant under scaling X -> cX.
            # So we can fix the scale by normalizing support, e.g., max(support) = 1.
            # This reduces the dimensionality effectively.
            
            max_s = np.max(support)
            if max_s > 0:
                support = support / max_s
            else:
                support = np.ones(n_atoms)
                
            weights = np.exp(log_w_raw)
            # Normalize weights
            w_sum = np.sum(weights)
            if w_sum < 1e-9:
                return 1e10
            weights = weights / w_sum
            
            # Compute P
            P = _compute_P_exact(support, weights)
            return -P
        except Exception:
            return 1e10

    # Initialization
    # We want a mix of small and large support values.
    # Small values: s ~ 0.01 -> log(s) ~ -4.6
    # Large values: s ~ 1.0 -> log(s) ~ 0.0
    # Let's initialize log_s uniformly in [-5, 0]
    
    # Weights: start uniform
    log_w_init = np.zeros(n_atoms)
    
    # Support init: log-spaced from 0.001 to 1.0
    # log(0.001) = -6.9, log(1) = 0
    s_init = np.linspace(-7.0, 0.0, n_atoms)
    
    # Add some noise
    s_init += np.random.uniform(-0.5, 0.5, n_atoms)
    
    x0 = np.concatenate([s_init, log_w_init])
    
    best_P = -np.inf
    best_support = None
    best_weights = None
    
    # Run multiple restarts with different random initializations
    n_restarts = 5
    
    for r in range(n_restarts):
        if time.time() > deadline - 60:
            break
            
        # Perturb initial conditions
        if r == 0:
            x_start = x0
        else:
            # Random restart
            # Mix of very small and large values
            # Half the atoms small, half large
            small_logs = np.random.uniform(-10, -2, n_atoms // 2)
            large_logs = np.random.uniform(-2, 1, n_atoms - n_atoms // 2)
            s_r = np.concatenate([small_logs, large_logs])
            np.random.shuffle(s_r)
            
            log_w_r = np.random.uniform(-2, 2, n_atoms)
            
            x_start = np.concatenate([s_r, log_w_r])
            
        # Optimize
        # Nelder-Mead is good for small dimensions and non-smooth landscapes
        # Powell is also good
        # Let's try Nelder-Mead first, then Powell if time permits
        
        try:
            # Method 1: Nelder-Mead
            # maxiter needs to be high enough to converge
            res_nm = minimize(objective, x_start, method='Nelder-Mead', 
                              options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
            
            if res_nm.success and res_nm.fun < -best_P:
                best_P = -res_nm.fun
                best_params = res_nm.x
                # Decode best params
                s_raw = best_params[:n_atoms]
                log_w_raw = best_params[n_atoms:]
                support = np.exp(s_raw)
                max_s = np.max(support)
                if max_s > 0:
                    support = support / max_s
                weights = np.exp(log_w_raw)
                weights = weights / np.sum(weights)
                best_support = support
                best_weights = weights
                
        except Exception:
            pass
        
        if time.time() > deadline - 120:
            continue
            
        try:
            # Method 2: Powell (more robust for this kind of problem?)
            # Use the best point found so far as starting point
            if best_params is not None:
                x_start_pw = best_params + np.random.normal(0, 0.1, len(best_params))
            else:
                x_start_pw = x_start + np.random.normal(0, 0.1, len(x_start))
                
            res_pw = minimize(objective, x_start_pw, method='Powell', 
                              options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
            
            if res_pw.success and res_pw.fun < -best_P:
                best_P = -res_pw.fun
                best_params = res_pw.x
                s_raw = best_params[:n_atoms]
                log_w_raw = best_params[n_atoms:]
                support = np.exp(s_raw)
                max_s = np.max(support)
                if max_s > 0:
                    support = support / max_s
                weights = np.exp(log_w_raw)
                weights = weights / np.sum(weights)
                best_support = support
                best_weights = weights
                
        except Exception:
            pass
        
    # If optimization failed or didn't find anything, use a fallback
    if best_support is None:
        # Fallback: Uniform distribution on [0, 1] with n_atoms points
        best_support = np.linspace(0.01, 1.0, n_atoms)
        best_weights = np.ones(n_atoms) / n_atoms
        
    # Final validation and cleanup
    # Ensure support >= 0
    best_support = np.maximum(best_support, 0.0)
    # Ensure weights >= 0
    best_weights = np.maximum(best_weights, 0.0)
    
    # Normalize weights to sum to 1
    w_sum = np.sum(best_weights)
    if w_sum > 0:
        best_weights = best_weights / w_sum
    else:
        # Absolute fallback
        best_support = np.array([0.0, 1.0])
        best_weights = np.array([0.5, 0.5])
        
    # Check for NaN/Inf
    if not np.all(np.isfinite(best_support)) or not np.all(np.isfinite(best_weights)):
        best_support = np.array([0.0, 1.0])
        best_weights = np.array([0.5, 0.5])
        
    return best_support, best_weights

# EVOLVE-BLOCK-END
