# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya conjecture (entropy formulation). Construct a joint probability
    distribution P[y, x] of two discrete random variables X, Y supported on an n x n grid
    (here n = 10) so as to maximize the entropy projection ratio

        H(pi_{-1}) / max(H(pi_0), H(pi_1), H(pi_2)),

    where pi_r(x, y) = x + r * y projects the grid along slope r, H is the Shannon entropy
    (in bits) of the resulting projected distribution.

    Returns:
        P: np.ndarray of shape (10, 10), non-negative and summing to 1.
    """
    n = 10
    # Start time for deadline guard
    start_time = time.time()
    deadline = 600.0
    safety_margin = 10.0

    # Precompute indices for slopes
    # P is [y, x]
    ys, xs = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
    
    # Values for each slope
    # r = -1: x - y
    vals_m1 = xs - ys
    # r = 0: x
    vals_0 = xs
    # r = 1: x + y
    vals_1 = xs + ys
    # r = 2: x + 2y
    vals_2 = xs + 2 * ys

    # Unique values and inverse indices for pooling
    # We need to sum P[y,x] for all (y,x) with the same value.
    # This can be done using np.bincount if we flatten the value arrays and P.
    
    def get_entropy(P_flat, vals, max_val):
        """
        Calculate Shannon entropy of the distribution of vals given probabilities P_flat.
        vals: 1D array of integer values.
        P_flat: 1D array of probabilities corresponding to vals.
        """
        # Sum probabilities for each unique value
        # max_val is the maximum possible value + 1
        sums = np.bincount(vals, weights=P_flat, minlength=max_val + 1)
        # Ignore zero probabilities
        sums = sums[sums > 1e-12]
        if len(sums) == 0:
            return 0.0
        # Normalize just in case (though P_flat sums to 1)
        sums = sums / sums.sum()
        return -np.sum(sums * np.log2(sums))

    def objective(Z):
        """
        Minimize the negative of the smoothed ratio.
        Z: 1D array of log-weights.
        """
        # Convert Z to probabilities P
        # Use log-sum-exp for stability
        Z_max = np.max(Z)
        P_flat = np.exp(Z - Z_max)
        P_flat /= np.sum(P_flat)
        
        # Reshape to 2D for consistency, though not strictly needed for bincount
        # P = P_flat.reshape((n, n))
        
        # Calculate entropies
        H_m1 = get_entropy(P_flat, vals_m1.ravel(), 2*n - 1) # range -9 to 9, shift? 
        # bincount handles negative indices poorly if not shifted.
        # Let's shift vals_m1 by n to make them non-negative.
        # vals_m1_shifted = vals_m1 + n
        # Actually, let's just use unique and inverse to be safe and generic.
        
        # Re-calculate using unique/inverse for robustness
        def calc_H(P_flat, vals):
            u_vals, inv = np.unique(vals, return_inverse=True)
            sums = np.bincount(inv, weights=P_flat, minlength=len(u_vals))
            sums = sums[sums > 1e-12]
            if len(sums) == 0:
                return 0.0
            sums = sums / sums.sum()
            return -np.sum(sums * np.log2(sums))
            
        H_m1 = calc_H(P_flat, vals_m1.ravel())
        H_0 = calc_H(P_flat, vals_0.ravel())
        H_1 = calc_H(P_flat, vals_1.ravel())
        H_2 = calc_H(P_flat, vals_2.ravel())
        
        # Soft max to make it differentiable
        # Using temperature 0.1
        temp = 0.1
        Hs = np.array([H_0, H_1, H_2])
        soft_max = temp * np.log(np.sum(np.exp(Hs / temp)))
        
        # We want to maximize H_m1 / soft_max
        # Equivalent to minimizing -H_m1 + soft_max (if we treat ratio as difference for gradient)
        # Or minimize -log(H_m1) + log(soft_max)
        # Let's stick to the difference for stability, as the ratio is monotonic with the difference if we scale.
        # Actually, maximizing H_m1 - soft_max is a good surrogate.
        return -(H_m1 - soft_max)

    # Initialize Z
    # Start with uniform distribution (Z = 0)
    Z0 = np.zeros(n * n)
    
    # Try a few random restarts to escape local minima
    best_Z = Z0.copy()
    best_score = -np.inf
    
    num_restarts = 5
    max_time_per_restart = (deadline - safety_margin - (time.time() - start_time)) / num_restarts
    
    for i in range(num_restarts):
        if time.time() > deadline - safety_margin:
            break
            
        # Random initialization
        if i == 0:
            Z_init = Z0
        else:
            # Small random perturbation around uniform or previous best
            Z_init = best_Z + np.random.randn(n * n) * 0.1
            
        try:
            res = minimize(objective, Z_init, method='L-BFGS-B', 
                           options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-8})
            
            # Evaluate final score
            Z_final = res.x
            Z_max = np.max(Z_final)
            P_flat = np.exp(Z_final - Z_max)
            P_flat /= np.sum(P_flat)
            
            # Calculate actual hard max ratio for comparison
            def calc_H(P_flat, vals):
                u_vals, inv = np.unique(vals, return_inverse=True)
                sums = np.bincount(inv, weights=P_flat, minlength=len(u_vals))
                sums = sums[sums > 1e-12]
                if len(sums) == 0:
                    return 0.0
                sums = sums / sums.sum()
                return -np.sum(sums * np.log2(sums))
                
            H_m1 = calc_H(P_flat, vals_m1.ravel())
            H_0 = calc_H(P_flat, vals_0.ravel())
            H_1 = calc_H(P_flat, vals_1.ravel())
            H_2 = calc_H(P_flat, vals_2.ravel())
            M = max(H_0, H_1, H_2)
            
            if M > 1e-9:
                score = H_m1 / M
            else:
                score = 0.0
                
            if score > best_score:
                best_score = score
                best_Z = Z_final
                
        except Exception:
            pass

    # Construct final P from best_Z
    Z_final = best_Z
    Z_max = np.max(Z_final)
    P_flat = np.exp(Z_final - Z_max)
    P_flat /= np.sum(P_flat)
    P = P_flat.reshape((n, n))
    
    # Final validation and clipping
    P = np.clip(P, 0, None)
    P_sum = np.sum(P)
    if P_sum < 1e-9:
        P = np.ones((n, n)) / (n * n)
    else:
        P /= P_sum
        
    return P

# EVOLVE-BLOCK-END
