"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _max_overlap(h):
    """
    Compute the maximum over shifts s in [0, 2] of the overlap integral
    integral h(x) * (1 - h(x+s)) dx over the valid domain.
    
    For a step function with N steps on [0, 2], we consider shifts that are
    multiples of dw = 2/N. The overlap for shift k*dw is:
    sum_{i=0}^{N-k-1} h[i] * (1 - h[i+k]) * dw
    
    We take the maximum over all k = 0, 1, ..., N-1.
    """
    dw = 2.0 / N
    n = len(h)
    max_ov = 0.0
    
    for k in range(n):
        # For shift k, the valid range is i from 0 to n-k-1
        # overlap_k = sum_{i=0}^{n-k-1} h[i] * (1 - h[i+k]) * dw
        if k == 0:
            # h[i] * (1 - h[i]) summed over all i, times dw
            ov = np.sum(h * (1.0 - h)) * dw
        else:
            valid_n = n - k
            ov = np.sum(h[:valid_n] * (1.0 - h[k:])) * dw
        
        if ov > max_ov:
            max_ov = ov
    
    return max_ov


def _max_overlap_grad(h):
    """
    Compute gradient of the maximum overlap with respect to h.
    We use the subgradient of the max: the gradient of the overlap at the
    shift that achieves the maximum.
    """
    dw = 2.0 / N
    n = len(h)
    
    # Find which shift gives the maximum overlap
    best_k = 0
    best_ov = -1.0
    
    ovs = np.zeros(n)
    for k in range(n):
        if k == 0:
            ovs[k] = np.sum(h * (1.0 - h)) * dw
        else:
            valid_n = n - k
            ovs[k] = np.sum(h[:valid_n] * (1.0 - h[k:])) * dw
        if ovs[k] > best_ov:
            best_ov = ovs[k]
            best_k = k
    
    k = best_k
    grad = np.zeros(n)
    
    # For shift k, overlap = sum_{i=0}^{n-k-1} h[i] * (1 - h[i+k]) * dw
    # d/dh[j] of this:
    # If j < n-k: contribution from term i=j: (1 - h[j+k]) * dw
    # If j >= k and j-k < n-k: i.e., j in [k, n-1] and j-k in [0, n-k-1]:
    #   contribution from term i=j-k: -h[j-k] * dw
    
    if k == 0:
        # overlap = sum h[i]*(1-h[i])*dw
        # d/dh[j] = (1 - 2*h[j]) * dw
        grad[:] = (1.0 - 2.0 * h) * dw
    else:
        valid_n = n - k
        # Positive terms: h[i] for i in [0, valid_n-1]
        # d/dh[j] for j in [0, valid_n-1]: += (1 - h[j+k]) * dw
        for j in range(valid_n):
            grad[j] += (1.0 - h[j + k]) * dw
        # Negative terms: -h[i+k] for i in [0, valid_n-1]
        # d/dh[j] for j in [k, n-1]: -= h[j-k] * dw
        for j in range(k, n):
            if j - k < valid_n:
                grad[j] -= h[j - k] * dw
    
    return grad


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Initial guess: uniform 0.5
    h0 = np.full(N, 0.5)
    
    best_h = h0.copy()
    best_score = _max_overlap(h0)
    
    # Try multiple random initializations with local optimization
    np.random.seed(42)
    
    n_restarts = 0
    max_restarts = 20
    
    for restart in range(max_restarts):
        elapsed = time.time() - start_time
        if elapsed > deadline - 15.0:
            break
        
        n_restarts = restart + 1
        
        if restart == 0:
            h_init = h0.copy()
        else:
            # Random initialization with sum constraint
            h_init = np.random.uniform(0.0, 1.0, size=N)
            # Adjust to satisfy sum = N/2
            current_sum = np.sum(h_init)
            target_sum = N / 2.0
            h_init = h_init + (target_sum - current_sum) / N
            h_init = np.clip(h_init, 0.0, 1.0)
            # Re-adjust
            current_sum = np.sum(h_init)
            if abs(current_sum - target_sum) > 1e-10:
                h_init[0] += (target_sum - current_sum)
                h_init[0] = np.clip(h_init[0], 0.0, 1.0)
        
        def objective(h):
            return _max_overlap(h)
        
        def obj_grad(h):
            return _max_overlap_grad(h)
        
        bounds = [(0.0, 1.0)] * N
        constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h) - N/2.0}]
        
        try:
            result = minimize(
                objective,
                h_init,
                jac=obj_grad,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-9, 'disp': False}
            )
            
            if result.success or (hasattr(result, 'fun') and result.fun < best_score):
                h_candidate = result.x
                # Ensure sum constraint
                current_sum = np.sum(h_candidate)
                if abs(current_sum - N/2.0) > 1e-8:
                    diff = N/2.0 - current_sum
                    h_candidate = h_candidate + diff / N
                    h_candidate = np.clip(h_candidate, 0.0, 1.0)
                    # Final adjustment
                    current_sum = np.sum(h_candidate)
                    if abs(current_sum - N/2.0) > 1e-10:
                        h_candidate[0] += (N/2.0 - current_sum)
                        h_candidate[0] = np.clip(h_candidate[0], 0.0, 1.0)
                
                score = _max_overlap(h_candidate)
                if score < best_score:
                    best_score = score
                    best_h = h_candidate.copy()
        except Exception:
            pass
    
    # Final constraint enforcement
    current_sum = np.sum(best_h)
    if abs(current_sum - N/2.0) > 1e-10:
        diff = N/2.0 - current_sum
        best_h = best_h + diff / N
        best_h = np.clip(best_h, 0.0, 1.0)
        current_sum = np.sum(best_h)
        if abs(current_sum - N/2.0) > 1e-10:
            best_h[0] += (N/2.0 - current_sum)
            best_h[0] = np.clip(best_h[0], 0.0, 1.0)
    
    return best_h
# EVOLVE-BLOCK-END
