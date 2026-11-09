# EVOLVE-BLOCK-START
import numpy as np
import time
import sys
from scipy.optimize import linprog


def _compute_P_for_support(support):
    """
    Compute the coefficient matrix A where A[k,l] = P[X1+X2 < 2*X3 - X4 | X4=support[k], X3=support[l]]
    Actually, we need: P = sum_k sum_l w_k * w_l * (sum_i sum_j w_i * w_j * 1[s_i + s_j < 2*s_l - s_k])
    
    Let's reformulate. Let support = s_0, ..., s_{n-1}.
    P = sum_{k,l,i,j} w_k w_l w_i w_j * 1[s_i + s_j < 2*s_l - s_k]
    
    This is a quartic form in w. We can't directly LP this.
    
    Alternative approach: Fix the support, and note that for given (k,l), the inner sum
    S(k,l) = sum_{i,j} w_i w_j * 1[s_i + s_j < 2*s_l - s_k]
    is a quadratic form in w.
    
    So P = sum_{k,l} w_k w_l * S(k,l) = sum_{k,l} w_k w_l * (w^T M_{kl} w)
    
    This is still quartic. Not directly LP-able.
    
    Let me reconsider. Maybe I should try a different structural approach.
    
    Actually, let's try: use a small support size (like 4-8 points) and optimize the support
    positions and weights jointly using scipy.optimize.minimize.
    """
    pass


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    
    Strategy: Use a small number of support points (n=4) and optimize both support
    positions and weights using scipy.optimize.minimize.
    
    Returns:
        support: np.ndarray of shape (n,), non-negative support points of the measure.
        weights: np.ndarray of shape (n,), non-negative weights (need not be normalized;
            the evaluator normalizes them to sum to 1).
    """
    start_time = time.time()
    deadline = start_time + 580  # 580s safety margin before 630s kill
    
    def compute_P(support, weights):
        """Compute P[X1+X2+X3 < 2*X4] for given support and weights."""
        n = len(support)
        # Normalize weights
        w = weights / np.sum(weights)
        
        # Compute all pairwise sums s_i + s_j
        s_i = support[:, np.newaxis]
        s_j = support[np.newaxis, :]
        pair_sums = s_i + s_j  # shape (n, n)
        
        # For each (k, l) where k is index of X4, l is index of X3:
        # We need P[X1 + X2 < 2*s_l - s_k]
        # This is sum_{i,j} w_i * w_j * 1[s_i + s_j < 2*s_l - s_k]
        
        P = 0.0
        for k in range(n):
            for l in range(n):
                threshold = 2.0 * support[l] - support[k]
                # Count pairs (i,j) with s_i + s_j < threshold
                mask = pair_sums < threshold
                inner = np.sum(w[np.newaxis, :] * w[:, np.newaxis] * mask)
                P += w[k] * w[l] * inner
        
        return P
    
    def neg_P(params):
        """Negative of P for minimization. params = [s0, s1, s2, s3, w0, w1, w2, w3]"""
        support = params[:4]
        weights = params[4:]
        
        # Ensure support is non-negative
        support = np.maximum(support, 0.0)
        # Ensure weights are non-negative
        weights = np.maximum(weights, 0.0)
        
        # If all weights are zero, return large penalty
        if np.sum(weights) < 1e-12:
            return 1e10
        
        return -compute_P(support, weights)
    
    # Initial guess: try a few different starting points
    best_params = None
    best_P = -1.0
    
    # Starting point 1: uniform on {0, 1, 2, 3}
    init1 = np.array([0.0, 1.0, 2.0, 3.0, 0.25, 0.25, 0.25, 0.25])
    
    # Starting point 2: concentrated at 0 and a larger value
    init2 = np.array([0.0, 10.0, 20.0, 30.0, 0.7, 0.1, 0.1, 0.1])
    
    # Starting point 3: geometric-like
    init3 = np.array([0.0, 1.0, 3.0, 9.0, 0.5, 0.3, 0.15, 0.05])
    
    inits = [init1, init2, init3]
    
    for init in inits:
        if time.time() > deadline - 100:
            break
        
        try:
            result = linprog(
                c=np.zeros(8),  # We'll use a different approach
                A_ub=np.zeros((0, 8)),
                b_ub=[],
                bounds=[(0, None)] * 4 + [(0, 1)] * 4,
                method='highs'
            )
            # Actually, let's use scipy.optimize.minimize instead
            from scipy.optimize import minimize
            
            res = minimize(
                neg_P,
                init,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            if res.success or res.fun < best_P:
                params = res.x
                support = np.maximum(params[:4], 0.0)
                weights = np.maximum(params[4:], 0.0)
                
                if np.sum(weights) > 1e-12:
                    P_val = -res.fun
                    if P_val > best_P:
                        best_P = P_val
                        best_params = params.copy()
        except Exception as e:
            pass
    
    # If we found a good solution, use it
    if best_params is not None:
        support = np.maximum(best_params[:4], 0.0)
        weights = np.maximum(best_params[4:], 0.0)
        
        # Ensure validity
        if np.sum(weights) < 1e-9:
            weights = np.ones(4) / 4
            support = np.arange(4, dtype=np.float64)
        
        return support.astype(np.float64), weights.astype(np.float64)
    else:
        # Fallback to baseline
        n = 32
        support = np.arange(n, dtype=np.float64)
        weights = np.ones(n, dtype=np.float64) / n
        return support, weights


# EVOLVE-BLOCK-END
