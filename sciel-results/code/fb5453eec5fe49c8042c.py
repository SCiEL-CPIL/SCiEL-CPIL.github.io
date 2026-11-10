import numpy as np
import time
from scipy.optimize import minimize

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya conjecture (entropy formulation). Construct a joint probability
    distribution P[y, x] of two discrete random variables X, Y supported on an n x n grid
    (here n = 10) so as to maximize the entropy projection ratio

        H(pi_{r_inf}) / max_i H(pi_{r_i}),

    where pi_r(x, y) = x + r * y projects the grid along slope r, H is the Shannon entropy
    (in bits) of the resulting projected distribution, the finite slopes are
    r_1, ..., r_k = 0.0, 1.0, 2.0 and the distinguished slope is r_inf = -1.0.

    The baseline is the uniform distribution over the grid.

    Returns:
        P: np.ndarray of shape (10, 10), non-negative and summing to 1, where P[y, x] is the
            joint probability that X = x and Y = y.
    """
    n = 10
    start_time = time.time()
    time_limit = 600.0 # Safety margin
    
    # Precompute indices
    xs = np.arange(n)
    ys = np.arange(n)
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    # P[y, x] corresponds to Y=y, X=x
    # Values:
    # x - y
    diff_vals = X - Y
    # x
    x_vals = X
    # x + y
    sum_vals = X + Y
    # x + 2y
    sum2_vals = X + 2 * Y

    # Unique values for binning
    # We need to sum probabilities for each unique value
    # Using np.unique with return_inverse or just iterating since range is small
    
    def compute_entropies(P):
        # P is shape (10, 10)
        # Flatten and get values
        p_flat = P.flatten()
        
        def entropy_from_values(vals, p):
            # vals: array of same shape as p
            # Map each unique val to a bin
            unique_vals, inverse = np.unique(vals, return_inverse=True)
            # Sum probabilities for each bin
            bin_probs = np.zeros(len(unique_vals))
            np.add.at(bin_probs, inverse, p)
            # Filter out zero probabilities
            bin_probs = bin_probs[bin_probs > 1e-15]
            if len(bin_probs) == 0:
                return 0.0
            return -np.sum(bin_probs * np.log2(bin_probs))
        
        H_diff = entropy_from_values(diff_vals.flatten(), p_flat)
        H_x = entropy_from_values(x_vals.flatten(), p_flat)
        H_sum = entropy_from_values(sum_vals.flatten(), p_flat)
        H_sum2 = entropy_from_values(sum2_vals.flatten(), p_flat)
        
        return H_diff, H_x, H_sum, H_sum2

    def objective(theta):
        # theta: 100 params, log-probs
        # Check time
        if time.time() - start_time > time_limit - 5:
            return 1e10 # Penalty to stop
        
        # Softmax to get probabilities
        # Subtract max for numerical stability
        theta_max = np.max(theta)
        exp_theta = np.exp(theta - theta_max)
        P = exp_theta / np.sum(exp_theta)
        
        H_diff, H_x, H_sum, H_sum2 = compute_entropies(P)
        
        # We want to maximize H_diff / max(H_x, H_sum, H_sum2)
        # Let M = max(...)
        # If M is very small, ratio is large.
        # Use smooth max: tau * log(sum(exp(H_i / tau)))
        tau = 0.1
        Hs = np.array([H_x, H_sum, H_sum2])
        # Avoid div by zero or log(0)
        # If Hs are all 0, M=0.
        # If H_diff is 0, ratio 0.
        
        # Soft max
        Hs_scaled = Hs / tau
        max_H = np.max(Hs_scaled)
        log_sum_exp = max_H + np.log(np.sum(np.exp(Hs_scaled - max_H)))
        M_smooth = tau * log_sum_exp
        
        # Add small epsilon to denominator to avoid 0/0
        M_safe = M_smooth + 1e-9
        
        # We want to maximize H_diff / M_safe
        # So minimize - H_diff / M_safe
        return -H_diff / M_safe

    # Initial guess: Uniform
    # log(1/100) = -log(100)
    theta0 = np.full((100,), -np.log(100))
    
    best_score = -1e10
    best_theta = theta0.copy()
    
    # Try a few random restarts
    num_restarts = 10
    for i in range(num_restarts):
        if time.time() - start_time > time_limit - 10:
            break
            
        if i == 0:
            theta_init = theta0
        else:
            # Random perturbation
            theta_init = theta0 + np.random.randn(100) * 0.1
            # Or structured init: e.g. concentrated on a diagonal
            # Let's try a few structured inits
            if i == 1:
                # Diagonal x=y
                P_init = np.zeros((10,10))
                for k in range(10):
                    P_init[k,k] = 1.0
                P_init /= 10.0
                theta_init = np.log(P_init + 1e-10).flatten()
            elif i == 2:
                # Anti-diagonal x+y=9
                P_init = np.zeros((10,10))
                for k in range(10):
                    P_init[k, 9-k] = 1.0
                P_init /= 10.0
                theta_init = np.log(P_init + 1e-10).flatten()
            elif i == 3:
                # x=0
                P_init = np.zeros((10,10))
                P_init[:, 0] = 1.0
                P_init /= 10.0
                theta_init = np.log(P_init + 1e-10).flatten()
        
        try:
            res = minimize(objective, theta_init, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
            
            if res.fun < best_score:
                best_score = res.fun
                best_theta = res.x
        except Exception:
            pass

    # Final check on best_theta
    theta_best = best_theta
    # Normalize to get P
    theta_max = np.max(theta_best)
    P = np.exp(theta_best - theta_max)
    P /= np.sum(P)
    
    # Ensure validity
    P = np.clip(P, 0, None)
    P /= np.sum(P)
    
    return P

# EVOLVE-BLOCK-END
