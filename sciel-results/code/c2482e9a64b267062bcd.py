import numpy as np
from scipy.optimize import minimize
import time

def _compute_entropies(P):
    """
    Computes D, H0, H1, H2 for a given 10x10 probability distribution P.
    Returns D, H0, H1, H2.
    """
    y_idx, x_idx = np.indices((10, 10))
    x_flat = x_idx.flatten()
    y_flat = y_idx.flatten()
    w = P.flatten()
    
    # Precomputed indices for binning
    # D: x - y. Range: -9 to 9. Shift by 9 to 0..18.
    d_vals = (x_flat - y_flat + 9).astype(np.int32)
    # H0: x. Range: 0 to 9.
    h0_vals = x_flat.astype(np.int32)
    # H1: x + y. Range: 0 to 18.
    h1_vals = (x_flat + y_flat).astype(np.int32)
    # H2: x + 2y. Range: 0 to 27.
    h2_vals = (x_flat + 2 * y_flat).astype(np.int32)
    
    def entropy_from_bins(bins, max_bin):
        counts = np.bincount(bins, weights=w, minlength=max_bin)
        # Ignore zero probabilities
        probs = counts[counts > 1e-15]
        if len(probs) == 0:
            return 0.0
        return -np.sum(probs * np.log2(probs))
    
    D = entropy_from_bins(d_vals, 19)
    H0 = entropy_from_bins(h0_vals, 10)
    H1 = entropy_from_bins(h1_vals, 19)
    H2 = entropy_from_bins(h2_vals, 28)
    
    return D, H0, H1, H2

def _score_smooth(z, tau=10.0):
    """
    Objective function for minimization (returns negative score).
    z is a 100-dim vector. P is softmax(z).
    Uses soft-max for the max(H0, H1, H2) term to ensure differentiability.
    """
    # Softmax parameterization to ensure P is a valid probability distribution
    z_max = np.max(z)
    exp_z = np.exp(z - z_max)
    P = exp_z / np.sum(exp_z)
    
    D, H0, H1, H2 = _compute_entropies(P)
    
    # Smooth max approximation
    # M = (1/tau) * log(sum(exp(tau * H)))
    # If tau is large, this approaches max(H)
    # We need to be careful with numerical stability
    H_vals = np.array([H0, H1, H2])
    H_max = np.max(H_vals)
    # If H_max is very small, log(sum(exp(tau*H))) might be unstable or 0
    if H_max < 1e-9:
        return 0.0
        
    # Shift to prevent overflow/underflow in exp
    shifted_H = H_vals - H_max
    exp_H = np.exp(tau * shifted_H)
    log_sum_exp = np.log(np.sum(exp_H)) + tau * H_max
    M_smooth = log_sum_exp / tau
    
    if M_smooth <= 1e-9:
        return 0.0
        
    ratio = D / M_smooth
    return -ratio

def _score_hard(P):
    """
    Computes the actual score D/M using hard max.
    """
    D, H0, H1, H2 = _compute_entropies(P)
    M = max(H0, H1, H2)
    if M <= 1e-9:
        return 0.0
    return D / M

def arithmetic_kakeya_distribution() -> np.ndarray:
    start_time = time.time()
    timeout = 600
    safety_margin = 10
    
    best_score = 0.0
    best_P = np.ones((10, 10), dtype=np.float64) / 100.0
    best_z = np.zeros(100)
    
    np.random.seed(42)
    
    num_restarts = 50
    # We will run a coarse pass with lower tau, then a fine pass with higher tau
    # Or just run L-BFGS-B with a moderate tau and evaluate hard score at the end.
    
    # Let's try a few restarts with different tau values
    tau_values = [5.0, 10.0, 20.0, 50.0]
    
    restarts_per_tau = 10
    
    for tau in tau_values:
        for r in range(restarts_per_tau):
            if time.time() - start_time > timeout - safety_margin:
                break
                
            # Initialize z randomly
            # Uniform P corresponds to z = 0
            # We can perturb around 0 or start from previous best
            if r == 0:
                z0 = np.random.normal(0, 1.0, 100)
            else:
                # Perturb the best z found so far for this tau or global best
                # Using global best z helps if we are refining
                z0 = best_z + np.random.normal(0, 0.5, 100)
            
            try:
                res = minimize(
                    _score_smooth, 
                    z0, 
                    method='L-BFGS-B', 
                    args=(tau,),
                    options={'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                z_opt = res.x
                score_smooth = -res.fun
                
                # Convert back to P
                z_max = np.max(z_opt)
                exp_z = np.exp(z_opt - z_max)
                P_opt = exp_z.reshape(10, 10) / np.sum(exp_z)
                
                # Evaluate with hard max to get true score
                true_score = _score_hard(P_opt)
                
                if true_score > best_score:
                    best_score = true_score
                    best_P = P_opt.copy()
                    best_z = z_opt.copy()
                    
            except Exception as e:
                pass

    # Final polishing: Use the best P found as a starting point for a high-tau optimization
    # Or just return best_P
    
    # One more check: ensure P is valid
    P_final = np.clip(best_P, 0, None)
    s = P_final.sum()
    if s > 0:
        P_final = P_final / s
    else:
        P_final = np.ones((10, 10), dtype=np.float64) / 100.0

    return P_final
