"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_START_TIME = time.time()
_TIME_LIMIT = 135.0  # Safety margin before 150s kill

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _compute_overlaps(h):
    """Compute R_k for k=1..N-1. R_k = (2/N) sum h[i]h[i+k]."""
    n = len(h)
    corrs = np.zeros(n - 1)
    for k in range(1, n):
        corrs[k-1] = np.dot(h[:n-k], h[k:]) * (2.0 / n)
    return corrs

def _max_overlap(h):
    """Compute the maximum overlap (worst case)."""
    return np.max(_compute_overlaps(h))

def _project_to_simplex(h, target_sum):
    """Project h to satisfy sum(h) = target_sum and 0<=h<=1."""
    h = np.clip(h, 0, 1).copy()
    for _ in range(50):
        diff = target_sum - np.sum(h)
        if abs(diff) < 1e-14:
            break
        mask = (h > 1e-12) & (h < 1 - 1e-12)
        if np.sum(mask) == 0:
            mask = (h > 0) & (h < 1)
            if np.sum(mask) == 0:
                break
        h[mask] += diff / np.sum(mask)
        h = np.clip(h, 0, 1)
    # Final exact adjustment
    diff = target_sum - np.sum(h)
    if abs(diff) > 1e-10:
        # Distribute residual proportionally to available room
        rooms = np.where(h < 1 - 1e-12, 1 - h, h)
        total_room = np.sum(rooms)
        if total_room > 1e-10:
            h += diff * rooms / total_room
            h = np.clip(h, 0, 1)
    return h

def _soft_min_obj_and_grad(params, beta):
    """
    Objective and gradient for symmetric h.
    params: length N/2, represents h[0:N/2]
    h is constructed symmetrically: h[i] = params[i] for i < N/2, h[N-1-i] = params[i]
    """
    n_half = len(params)
    n = 2 * n_half
    h = np.zeros(n)
    h[:n_half] = params
    h[n_half:] = params[::-1]
    
    corrs = _compute_overlaps(h)
    
    # Soft min of corrs
    min_val = np.min(corrs)
    shifted = corrs - min_val
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    soft_min = min_val - log_sum / beta
    
    obj = 1.0 - soft_min
    
    # Gradient
    exp_neg_beta = np.exp(-beta * corrs)
    sum_exp_neg = np.sum(exp_neg_beta)
    weights = exp_neg_beta / sum_exp_neg
    
    grad_h = np.zeros(n)
    for k in range(1, n):
        w = weights[k-1] * (2.0 / n)
        if w < 1e-15:
            continue
        grad_h[:n-k] += w * h[k:]
        grad_h[k:] += w * h[:n-k]
    
    grad_h = -grad_h
    
    # Map grad_h back to params (symmetric)
    grad_params = grad_h[:n_half] + grad_h[n-1:n_half-1:-1]
    
    return obj, grad_params

def _construct_symmetric(params):
    n_half = len(params)
    n = 2 * n_half
    h = np.zeros(n)
    h[:n_half] = params
    h[n_half:] = params[::-1]
    return h

def _generate_structural_candidates(n):
    """
    Generate binary sequences of length n with sum n/2 by searching over periodic patterns.
    We consider periods P that divide n. For each P, we consider all balanced binary sequences
    of length P (with P/2 ones) and repeat them to fill length n.
    """
    candidates = []
    # Divisors of 100: 1, 2, 4, 5, 10, 20, 25, 50, 100
    # We skip P=1,2,4,5 as they are too regular and likely have high overlap.
    # P=10: C(10,5)=252
    # P=20: C(20,10)=184756
    # P=25: C(25,12)=5200300 (too many, skip or sample)
    # P=50: C(50,25)=1.26e14 (too many)
    
    # Let's try P=10 and P=20 fully, and maybe sample P=25.
    periods_to_check = [10, 20]
    
    for P in periods_to_check:
        if n % P != 0:
            continue
        num_repeats = n // P
        half_P = P // 2
        # Generate all combinations of positions for ones in a block of length P
        from itertools import combinations
        for ones_positions in combinations(range(P), half_P):
            block = np.zeros(P)
            for pos in ones_positions:
                block[pos] = 1.0
            # Repeat block
            seq = np.tile(block, num_repeats)
            # Check sum (should be n/2)
            if np.sum(seq) == n/2:
                candidates.append(seq)
    
    # Also add some random balanced binary sequences
    rng = np.random.RandomState(42)
    for _ in range(100):
        seq = np.zeros(n)
        indices = rng.choice(n, size=n//2, replace=False)
        seq[indices] = 1.0
        candidates.append(seq)
    
    return candidates

def _max_overlap_subgrad(h):
    """Compute subgradient of max_k R_k w.r.t. h.
    Returns (max_val, subgradient) where subgradient points in direction to decrease max overlap.
    """
    n = len(h)
    corrs = _compute_overlaps(h)
    max_val = np.max(corrs)
    
    # Find which k achieves max (or near-max for smoothing)
    threshold = max_val - 1e-6
    active = corrs >= threshold
    
    subgrad = np.zeros(n)
    for k in range(1, n):
        if not active[k-1]:
            continue
        # Weight by how close to max this correlation is
        w = (2.0 / n) * corrs[k-1] / max_val if max_val > 1e-15 else (2.0/n)
        subgrad[:n-k] += w * h[k:]
        subgrad[k:] += w * h[:n-k]
    
    # Normalize subgradient
    norm = np.linalg.norm(subgrad)
    if norm > 1e-15:
        subgrad /= norm
    
    return max_val, subgrad

def construct():
    n = N
    n_half = n // 2
    target_sum = n / 2.0
    target_param_sum = n / 4.0
    
    best_params = np.full(n_half, 0.5)
    best_obj = float('inf')
    
    # Initial evaluation
    h_init = _construct_symmetric(best_params)
    best_obj = _max_overlap(h_init)
    
    np.random.seed(42)
    
    restarts = []
    
    # 1. Structural candidates from periodic binary sequences
    structural_candidates = _generate_structural_candidates(n)
    # Evaluate and sort by max overlap
    scored_structural = []
    for seq in structural_candidates:
        # Symmetrize the binary sequence to fit our parameterization
        # h[i] = params[i] for i < n/2, h[n-1-i] = params[i]
        # So we need to map the full binary sequence to params
        # If the sequence is not symmetric, we can't directly use it.
        # Instead, we can take the first half and use that as params, then symmetrize.
        # Or, we can just use the full sequence as an initial point for a non-symmetric optimization?
        # The current optimizer uses symmetric params. Let's project the binary sequence to symmetric form.
        # h_sym[i] = (seq[i] + seq[n-1-i]) / 2
        h_sym = (seq + seq[::-1]) / 2.0
        # h_sym should have sum n/2 if seq does, and is symmetric.
        # Extract params: first half of h_sym
        params = h_sym[:n_half]
        # Project to ensure sum(params) = n/4
        params = _project_to_simplex(params, target_param_sum)
        
        # Also keep the raw max overlap for potential direct use if we switch to full-vector opt later
        raw_obj = _max_overlap(seq)
        scored_structural.append((raw_obj, params))
    
    # Sort structural candidates by raw overlap and take top 10
    scored_structural.sort(key=lambda x: x[0])
    top_structural = [s[1] for s in scored_structural[:10]]
    
    # Add top structural candidates as additional restarts (already added above, but let's ensure they are prioritized)
    # The loop above added all of them. Let's remove the lower ones if we have too many restarts.
    # Actually, let's just use the top 10 structural ones and some random/sinusoidal.
    restarts = []
    for params in top_structural:
        restarts.append(params)
        
    # 2. Other initializations
    restarts.append(np.full(n_half, 0.5))
    restarts.append(np.linspace(0, 1, n_half))
    restarts.append(np.linspace(1, 0, n_half))
    
    x = np.linspace(0, 2*np.pi, n_half)
    for freq in [1, 2, 3, 4, 5]:
        restarts.append(0.5 + 0.45*np.sin(freq*x))
        restarts.append(0.5 + 0.45*np.cos(freq*x))
        
    for _ in range(20):
        restarts.append(np.random.uniform(0, 1, n_half))

    # Optimize each restart
    for x0 in restarts:
        if _time_remaining() < 3.0:
            break
        
        # Project to feasible region
        params = _project_to_simplex(x0, target_param_sum)
        
        # Multi-stage beta optimization
        for beta in [30.0, 80.0, 150.0, 300.0]:
            if _time_remaining() < 2.0:
                break
            try:
                res = minimize(
                    _soft_min_obj_and_grad,
                    params,
                    args=(beta,),
                    method='L-BFGS-B',
                    jac=True,
                    bounds=[(0, 1)] * n_half,
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'disp': False}
                )
                if res.fun < best_obj:
                    params_new = _project_to_simplex(res.x, target_param_sum)
                    h_new = _construct_symmetric(params_new)
                    obj_new = _max_overlap(h_new)
                    if obj_new < best_obj:
                        best_obj = obj_new
                        best_params = params_new
                        params = params_new
            except Exception:
                pass
    
    # Final polish with full vector optimization if time permits
    if _time_remaining() > 15.0:
        h_full = _construct_symmetric(best_params)
        
        def full_obj_and_grad(h):
            corrs = _compute_overlaps(h)
            min_val = np.min(corrs)
            shifted = corrs - min_val
            beta = 500.0
            log_sum = np.log(np.sum(np.exp(-beta * shifted)))
            soft_min = min_val - log_sum / beta
            obj = 1.0 - soft_min
            
            exp_neg_beta = np.exp(-beta * corrs)
            sum_exp_neg = np.sum(exp_neg_beta)
            weights = exp_neg_beta / sum_exp_neg
            
            grad_h = np.zeros(n)
            for k in range(1, n):
                w = weights[k-1] * (2.0 / n)
                if w < 1e-15:
                    continue
                grad_h[:n-k] += w * h[k:]
                grad_h[k:] += w * h[:n-k]
            
            return obj, -grad_h

        try:
            res = minimize(
                full_obj_and_grad,
                h_full,
                method='L-BFGS-B',
                jac=True,
                bounds=[(0, 1)] * n,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'disp': False}
            )
            h_new = np.clip(res.x, 0, 1)
            h_new = _project_to_simplex(h_new, target_sum)
            
            corrs_new = _compute_overlaps(h_new)
            obj_new = np.max(corrs_new)
            if obj_new < best_obj:
                best_obj = obj_new
                final_h = h_new
            else:
                final_h = _construct_symmetric(best_params)
        except Exception:
            final_h = _construct_symmetric(best_params)
    else:
        final_h = _construct_symmetric(best_params)
    
    # Final adjustment for exact sum
    final_h = _project_to_simplex(final_h, target_sum)
    
    return final_h
# EVOLVE-BLOCK-END
