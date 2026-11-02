import numpy as np
import time
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _time_remaining(deadline):
    return max(0, deadline - time.time())

def _max_overlap(h):
    """Compute maximum overlap of h with shifted 1-h."""
    n = len(h)
    overlaps = np.zeros(n - 1)
    for k in range(1, n):
        h_shifted = np.roll(h, k)
        overlaps[k-1] = np.sum(h * (1 - h_shifted))
    return np.max(overlaps)

def _soft_min_obj_and_grad(h, beta):
    """Soft-min surrogate using log-sum-exp for smooth optimization."""
    n = len(h)
    # Compute all overlaps
    overlaps = np.zeros(n - 1)
    for k in range(1, n):
        h_shifted = np.roll(h, k)
        overlaps[k-1] = np.sum(h * (1 - h_shifted))
    
    # Log-sum-exp for numerical stability
    max_ov = np.max(overlaps)
    log_sum_exp = max_ov + np.log(np.sum(np.exp(overlaps - max_ov)))
    obj = log_sum_exp / beta
    
    # Gradient: d/dh_i [sum_k h_i*(1-h_{i-k})] for each k
    # For overlap at shift k: sum_i h_i * (1 - h_{i-k})
    # d/dh_i = (1 - h_{i-k}) for the term where this h_i appears
    # Also -h_{j} for terms where h_i appears in (1-h_{i-k}) as h_{i-k}
    
    grad = np.zeros(n)
    for k in range(1, n):
        h_shifted = np.roll(h, k)
        # Contribution to overlap k from each position
        # d/dh_i [h_i * (1 - h_{i-k})] = (1 - h_{i-k})
        # d/dh_i [h_j * (1 - h_{j-k})] where j-k = i, so j = i+k
        # = -h_{i+k} (circular)
        
        # First term: (1 - h_{i-k}) for all i
        h_minus_k = np.roll(h, -k)  # h_{i-k}
        grad += (1 - h_minus_k)
        
        # Second term: -h_{i+k} for all i
        h_plus_k = np.roll(h, k)  # h_{i+k}
        grad += -h_plus_k
    
    # Weight by softmax probabilities
    exp_overlaps = np.exp(overlaps - max_ov)
    weights = exp_overlaps / np.sum(exp_overlaps)
    
    # Rescale gradient by weights and beta
    # The gradient of log-sum-exp is weighted sum of gradients
    # But we need the gradient of the full objective
    
    # Recompute more carefully
    grad = np.zeros(n)
    for k in range(1, n):
        h_shifted = np.roll(h, k)
        # For shift k, overlap_k = sum_i h_i * (1 - h_{i-k})
        # d overlap_k / d h_i = (1 - h_{i-k}) - h_{i+k}
        h_minus_k = np.roll(h, -k)
        h_plus_k = np.roll(h, k)
        grad_k = (1 - h_minus_k) - h_plus_k
        
        # Weight by softmax
        w_k = weights[k-1]
        grad += w_k * grad_k
    
    grad /= beta
    
    return obj, grad

def _project(h):
    """Project h to [0,1]^N with sum = N/2."""
    h = np.clip(h, 0, 1)
    # Adjust to maintain sum = N/2
    diff = N/2 - np.sum(h)
    if abs(diff) > 1e-10:
        # Distribute difference proportionally
        if diff > 0:
            # Need to increase some values
            mask = h < 1
            if np.sum(mask) == 0:
                return h
            h[mask] += diff / np.sum(mask)
        else:
            # Need to decrease some values
            mask = h > 0
            if np.sum(mask) == 0:
                return h
            h[mask] += diff / np.sum(mask)
        h = np.clip(h, 0, 1)
    return h

def _spectral_init():
    """Initialize using spectral method - try to make autocorrelation flat."""
    # Start with alternating pattern and add noise
    h = np.zeros(N)
    h[::2] = 1.0
    h[1::2] = 0.0
    # Add small random perturbation
    h += np.random.uniform(-0.1, 0.1, N)
    h = _project(h)
    return h

def _anneal_optimize(h_init, deadline):
    """Simulated annealing with soft-min surrogate."""
    best_h = h_init.copy()
    best_score = _max_overlap(best_h)
    
    current_h = h_init.copy()
    current_score = _max_overlap(current_h)
    
    T = 0.5  # Initial temperature
    cooling = 0.995
    min_T = 0.001
    
    while _time_remaining(deadline) > 2:
        # Perturbation: swap two random positions
        i, j = np.random.choice(N, 2, replace=False)
        new_h = current_h.copy()
        new_h[i], new_h[j] = current_h[j], current_h[i]
        new_h = _project(new_h)
        
        new_score = _max_overlap(new_h)
        delta = new_score - current_score
        
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            current_h = new_h
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_h = new_h.copy()
        
        T = max(T * cooling, min_T)
    
    return best_h, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    best_h = None
    best_score = np.inf
    
    # Multiple restarts with different initializations
    n_restarts = 5
    
    for restart in range(n_restarts):
        if _time_remaining(deadline) < 5:
            break
        
        if restart == 0:
            h_init = _spectral_init()
        else:
            # Random binary initialization
            h_init = np.zeros(N)
            indices = np.random.permutation(N)
            h_init[indices[:N//2]] = 1.0
            h_init += np.random.uniform(-0.05, 0.05, N)
            h_init = _project(h_init)
        
        # Use L-BFGS-B with soft-min surrogate
        beta_values = [10, 50, 100]
        
        for beta in beta_values:
            if _time_remaining(deadline) < 3:
                break
            
            def objective(x):
                x = _project(x)
                obj, _ = _soft_min_obj_and_grad(x, beta)
                return obj
            
            def gradient(x):
                x = _project(x)
                _, grad = _soft_min_obj_and_grad(x, beta)
                return grad
            
            try:
                result = minimize(
                    objective,
                    h_init,
                    jac=gradient,
                    method='L-BFGS-B',
                    bounds=[(0, 1)] * N,
                    options={'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-6}
                )
                
                h_opt = _project(result.x)
                score = _max_overlap(h_opt)
                
                if score < best_score:
                    best_score = score
                    best_h = h_opt.copy()
                
                # Use as new initialization for next beta
                h_init = h_opt.copy()
            except Exception:
                continue
        
        # Polish with annealing
        if _time_remaining(deadline) > 10:
            h_polished, score_polished = _anneal_optimize(h_init, deadline)
            if score_polished < best_score:
                best_score = score_polished
                best_h = h_polished.copy()
    
    if best_h is None:
        # Fallback
        h = np.zeros(N)
        h[::2] = 1.0
        best_h = h
    
    return best_h
# EVOLVE-BLOCK-END
