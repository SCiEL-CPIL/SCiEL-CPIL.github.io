"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def _max_overlap(h):
    """Compute the maximum overlap over all shifts."""
    dw = 2.0 / N
    max_overlap = 0.0
    for k in range(N):
        valid = N - k
        if valid <= 0:
            continue
        overlap = np.sum(h[:valid] * (1.0 - h[k:])) * dw
        if overlap > max_overlap:
            max_overlap = overlap
    return max_overlap

def _compute_all_overlaps(h):
    """Compute overlap for all shifts k = 0, ..., N-1."""
    dw = 2.0 / N
    overlaps = np.zeros(N)
    for k in range(N):
        valid = N - k
        if valid > 0:
            overlaps[k] = np.sum(h[:valid] * (1.0 - h[k:])) * dw
    return overlaps

def _soft_min_overlaps(h, beta):
    """
    Compute a smooth approximation to the max overlap using log-sum-exp.
    soft_min(x) = -1/beta * log(sum(exp(-beta * x)))
    This approximates min(x) for large beta, but we want to minimize max, so:
    soft_max(x) = 1/beta * log(sum(exp(beta * x)))
    We minimize soft_max(overlaps) as a smooth proxy for max(overlaps).
    """
    overlaps = _compute_all_overlaps(h)
    # Use soft-max to approximate max
    max_val = np.max(overlaps)
    # Numerically stable log-sum-exp
    exp_terms = np.exp(beta * (overlaps - max_val))
    log_sum = np.log(np.sum(exp_terms)) + beta * max_val
    soft_max_val = log_sum / beta
    return soft_max_val

def _soft_max_grad(h, beta):
    """
    Compute gradient of soft_max(overlaps) with respect to h.
    soft_max(x) = (1/beta) * log(sum_k exp(beta * x_k))
    d(soft_max)/d(h_i) = (1/beta) * sum_k [beta * exp(beta*x_k) / sum_j exp(beta*x_j)] * (d x_k / d h_i)
    = sum_k [p_k * (d x_k / d h_i)] where p_k = exp(beta*x_k) / sum_j exp(beta*x_j)
    
    x_k = sum_{i=0}^{N-k-1} h[i] * (1 - h[i+k]) * dw
    d x_k / d h_i = dw * [(1 - h[i+k]) if i < N-k else 0] + dw * [-h[i] if i+k < N else 0]
    More carefully: for a fixed k, x_k = dw * sum_{j=0}^{N-k-1} h[j] * (1 - h[j+k])
    d x_k / d h_i = dw * [(1 - h[i+k]) if i < N-k] + dw * [-h[i] if i+k < N]
    But we need to be careful: if i is in [0, N-k-1], the first term applies.
    If i is in [k, N-1], then j = i-k is in [0, N-k-1], so the second term applies with j=i-k.
    """
    dw = 2.0 / N
    overlaps = _compute_all_overlaps(h)
    
    # Compute softmax weights
    max_val = np.max(overlaps)
    exp_terms = np.exp(beta * (overlaps - max_val))
    sum_exp = np.sum(exp_terms)
    p = exp_terms / sum_exp  # p[k] = softmax weight for shift k
    
    grad = np.zeros(N)
    
    for k in range(N):
        if p[k] < 1e-15:
            continue
        valid = N - k
        # For shift k: x_k = dw * sum_{j=0}^{valid-1} h[j] * (1 - h[j+k])
        # d x_k / d h_i:
        # If i < valid: contributes dw * (1 - h[i+k])
        # If i >= k and i < N: let j = i-k, if j < valid (i.e., i < k + valid = N): contributes dw * (-h[i-k])
        # Note: i < valid means i < N-k. i >= k and i < N means the second term.
        
        # First part: i in [0, valid-1]
        if valid > 0:
            grad[:valid] += p[k] * dw * (1.0 - h[k:valid+k])
        
        # Second part: i in [k, N-1], j = i-k in [0, valid-1]
        if k < N:
            grad[k:] += p[k] * dw * (-h[:valid])
    
    return grad

def _project_to_constraint(h):
    """
    Project h onto the constraint set: sum(h) = N/2, 0 <= h_i <= 1.
    Use iterative water-filling / bisection.
    """
    h = h.copy()
    target = N / 2.0
    
    # First, clip to [0, 1]
    h = np.clip(h, 0.0, 1.0)
    
    # Then adjust sum using bisection on a shift
    # We want to find lambda such that sum(clip(h + lambda, 0, 1)) = target
    lo, hi = -1.0, 1.0
    
    for _ in range(60):
        mid = (lo + hi) / 2.0
        h_test = np.clip(h + mid, 0.0, 1.0)
        s = np.sum(h_test)
        if s > target:
            hi = mid
        else:
            lo = mid
    
    h_proj = np.clip(h + (lo + hi) / 2.0, 0.0, 1.0)
    
    # Final correction: distribute remaining error
    residual = target - np.sum(h_proj)
    if abs(residual) > 1e-12:
        # Add to elements that have room
        if residual > 0:
            room = 1.0 - h_proj
            idx = np.where(room > 1e-10)[0]
            if len(idx) > 0:
                # Distribute proportionally to room
                weights = room[idx]
                weights = weights / weights.sum()
                h_proj[idx] += residual * weights
        else:
            room = h_proj
            idx = np.where(room > 1e-10)[0]
            if len(idx) > 0:
                weights = room[idx]
                weights = weights / weights.sum()
                h_proj[idx] += residual * weights
    
    h_proj = np.clip(h_proj, 0.0, 1.0)
    
    # Ensure exact sum
    final_residual = target - np.sum(h_proj)
    if abs(final_residual) > 1e-14:
        # Find the element closest to 0.5 and adjust
        idx = np.argmin(np.abs(h_proj - 0.5))
        h_proj[idx] += final_residual
        h_proj[idx] = np.clip(h_proj[idx], 0.0, 1.0)
    
    return h_proj

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_h = np.full(N, 0.5)
    best_score = _max_overlap(best_h)
    
    rng = np.random.default_rng(42)
    
    # Multi-restart gradient descent with soft-min proxy
    for restart in range(10):
        if time.time() > deadline - 15.0:
            break
        
        if restart == 0:
            h_curr = np.full(N, 0.5)
        else:
            # Random initialization with sum = N/2
            h_curr = rng.random(N)
            h_curr = h_curr / h_curr.sum() * (N / 2.0)
            h_curr = np.clip(h_curr, 0.0, 1.0)
            h_curr = _project_to_constraint(h_curr)
        
        current_score = _max_overlap(h_curr)
        if current_score < best_score:
            best_score = current_score
            best_h = h_curr.copy()
        
        # Gradient descent with annealing beta
        # Start with small beta (smooth), increase over time (sharper approximation to max)
        for it in range(200):
            if time.time() > deadline - 5.0:
                break
            
            # Anneal beta: start at 10, increase to 100
            beta = 10.0 + 90.0 * (it / 200.0)
            
            # Compute gradient of soft_max
            grad = _soft_max_grad(h_curr, beta)
            
            # Learning rate: decrease over iterations
            lr = 0.01 / (1.0 + it * 0.01)
            
            # Gradient step
            h_new = h_curr - lr * grad
            
            # Project onto constraint set
            h_new = _project_to_constraint(h_new)
            
            # Evaluate
            new_score = _max_overlap(h_new)
            if new_score < current_score:
                current_score = new_score
                h_curr = h_new
                if current_score < best_score:
                    best_score = current_score
                    best_h = h_curr.copy()
            else:
                # Revert and reduce learning rate
                lr *= 0.5
        
        # Occasional random perturbation to escape local minima
        if restart > 0 and time.time() < deadline - 20.0:
            for _ in range(50):
                if time.time() > deadline - 10.0:
                    break
                i, j = rng.choice(N, size=2, replace=False)
                delta = rng.uniform(-0.05, 0.05)
                h_test = h_curr.copy()
                h_test[i] += delta
                h_test[j] -= delta
                h_test = _project_to_constraint(h_test)
                test_score = _max_overlap(h_test)
                if test_score < current_score:
                    current_score = test_score
                    h_curr = h_test
                    if current_score < best_score:
                        best_score = current_score
                        best_h = h_curr.copy()
    
    # Final projection
    best_h = _project_to_constraint(best_h)
    
    return best_h
# EVOLVE-BLOCK-END
