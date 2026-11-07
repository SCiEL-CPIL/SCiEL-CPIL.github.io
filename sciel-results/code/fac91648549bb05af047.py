import numpy as np
import time
from scipy.optimize import minimize

N = 100
TARGET_SUM = N / 2.0
DEADLINE = 140.0
START = time.time()

def remaining():
    return DEADLINE - (time.time() - START)

def compute_correlation(h):
    """Compute full correlation of h and 1-h."""
    return np.correlate(h, 1.0 - h, mode="full")

def objective_and_grad(x, beta=100.0):
    """
    x: array of N-1 elements (h[0] to h[N-2])
    h[N-1] is determined by sum constraint.
    """
    h = np.empty(N)
    h[:N-1] = x
    h[N-1] = TARGET_SUM - np.sum(x)
    
    # Check bounds for h[N-1]
    # If h[N-1] is out of [0,1], we are infeasible.
    # SLSQP handles this, but for gradient computation, we assume feasible.
    
    c = compute_correlation(h)
    
    # Soft max
    c_max = np.max(c)
    c_shifted = c - c_max
    exp_c = np.exp(beta * c_shifted)
    sum_exp = np.sum(exp_c)
    val = (c_max + np.log(sum_exp) / beta) / N * 2.0
    
    # Gradient
    # d val / d x = (2/N) * d soft_max / d h * dh/dx
    # d soft_max / d h = sum_k w_k * dc_k/dh
    # w_k = exp_c / sum_exp
    w = exp_c / sum_exp
    
    # dc_k / dh
    # c_k = sum_i h_i (1 - h_{i - (N-1-k)})
    # Let d = N-1-k.
    # c_k = sum_i h_i - sum_i h_i h_{i-d}
    # dc_k/dh_j = 1 - (h_{j-d} + h_{j+d}) ? 
    # Let's derive carefully.
    # c_k = sum_i h_i (1 - h_{i-d}) = sum_i h_i - sum_i h_i h_{i-d}
    # d/dh_j (sum_i h_i) = 1
    # d/dh_j (sum_i h_i h_{i-d})
    # Term i=j: h_{j-d}
    # Term i such that i-d=j => i=j+d: h_{j+d}
    # So derivative is h_{j-d} + h_{j+d} (with 0 for out of bounds).
    # So dc_k/dh_j = 1 - (h_{j-d} + h_{j+d})
    
    grad_h = np.zeros(N)
    for k in range(2*N - 1):
        if w[k] < 1e-15:
            continue
        d = N - 1 - k
        for j in range(N):
            jd = j - d
            jp = j + d
            term1 = h[jd] if 0 <= jd < N else 0.0
            term2 = h[jp] if 0 <= jp < N else 0.0
            grad_h[j] += w[k] * (1.0 - term1 - term2)
            
    grad_h *= 2.0 / N
    
    # Map grad_h to grad_x
    # x = h[:N-1]
    # h[N-1] = TARGET - sum(x)
    # d val / d x_i = d val / d h_i + d val / d h[N-1] * d h[N-1] / d x_i
    # d h[N-1] / d x_i = -1
    # So grad_x = grad_h[:N-1] - grad_h[N-1]
    
    grad_x = grad_h[:N-1] - grad_h[N-1]
    
    return val, grad_x

def construct():
    global _CACHED
    try:
        if '_CACHED' in globals():
            return _CACHED
            
        # Initial guess: uniform
        x0 = np.full(N-1, TARGET_SUM / N) # 0.5
        
        # Bounds for x
        bounds = [(0.0, 1.0)] * (N-1)
        
        # We use SLSQP to handle the implicit constraint on h[N-1] via the objective/gradient?
        # No, SLSQP minimizes f(x). We need to ensure h[N-1] is in [0,1].
        # We can add this as a constraint or just rely on the fact that if we start feasible and move slowly, we stay feasible?
        # Better: use 'trust-constr' which can handle bounds on the derived variable? No.
        # Let's just use SLSQP and hope it doesn't go too far, or add a penalty.
        # Actually, if we start at 0.5, h[N-1]=0.5.
        # The gradient will push towards lower max correlation.
        
        # Let's try a few random restarts to be safe.
        best_val = np.inf
        best_x = x0
        
        rng = np.random.default_rng(100)
        
        for trial in range(5):
            if remaining() < 20.0:
                break
                
            if trial == 0:
                x_start = x0
            else:
                # Random perturbation
                x_start = x0 + rng.normal(0, 0.1, N-1)
                x_start = np.clip(x_start, 0, 1)
                # Ensure sum allows h[N-1] in [0,1]
                s = np.sum(x_start)
                if s > TARGET_SUM:
                    x_start *= (TARGET_SUM - 0.5) / s # rough adjustment
                    x_start = np.clip(x_start, 0, 1)
                elif s < TARGET_SUM - 1:
                    x_start += (TARGET_SUM - 0.5 - s) / (N-1)
                    x_start = np.clip(x_start, 0, 1)
            
            res = minimize(objective_and_grad, x_start, jac=True, method='SLSQP',
                           bounds=bounds, options={'maxiter': 500, 'ftol': 1e-10})
            
            if res.success:
                val = res.fun
                if val < best_val:
                    best_val = val
                    best_x = res.x
            
            # Also try L-BFGS-B on the full vector with projection?
            # SLSQP might be slow. Let's check time.
            
        # Reconstruct h
        h = np.empty(N)
        h[:N-1] = best_x
        h[N-1] = TARGET_SUM - np.sum(best_x)
        
        # Final projection to ensure strict validity
        h = np.clip(h, 0.0, 1.0)
        s = h.sum()
        h += (TARGET_SUM - s) / N
        h = np.clip(h, 0.0, 1.0)
        
        _CACHED = h
        return h
        
    except Exception:
        return np.full(N, 0.5)

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END
