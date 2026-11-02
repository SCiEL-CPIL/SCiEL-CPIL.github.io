"""Erdos minimum overlap: choose equal-width step heights h on [0,2] with sum=len/2,
minimize the maximum overlap of h with shifted 1-h.

The objective is:
  C(h) = max_{k=1..N-1} sum_{i=0}^{N-1-k} h[i] * (1 - h[i+k])
       = max_k [ (N-k)*mean(h[0:N-k]) - sum h[i]*h[i+k] ]

We want to minimize this. The seed returns 0.5 everywhere, giving overlap = (N-k)*0.25,
max at k=1: (N-1)*0.25.

Strategy:
- Use scipy.optimize.minimize with L-BFGS-B on a smooth surrogate (soft-max via log-sum-exp)
- Multiple random restarts with diverse initial conditions
- Final polish with tighter tolerance
- Deadline guard at ~110s to stay safely under 120s
"""
import numpy as np
import time
import math

N = 100

def _compute_overlap_matrix(h):
    """Compute all pairwise shift overlaps: overlap[k] = sum_i h[i]*(1-h[i+k]) for k=1..N-1"""
    N = len(h)
    overlaps = np.zeros(N - 1)
    for k in range(1, N):
        # overlap[k] = sum_{i=0}^{N-1-k} h[i] * (1 - h[i+k])
        overlaps[k - 1] = np.dot(h[:N - k], 1.0 - h[k:])
    return overlaps


def _max_overlap(h):
    """Hard max of overlaps."""
    return float(np.max(_compute_overlap_matrix(h)))


def _soft_max_overlap(h, tau=0.01):
    """Smooth surrogate: log-sum-exp of overlaps / tau (scaled)."""
    ov = _compute_overlap_matrix(h)
    # soft max = tau * log(sum(exp(ov/tau)))
    # To avoid overflow, subtract max
    m = np.max(ov)
    if tau < 1e-12:
        return m
    return tau * (m / tau + np.log(np.sum(np.exp((ov - m) / tau))))


def _grad_soft_max_overlap(h, tau=0.01):
    """Gradient of soft-max overlap w.r.t. h."""
    ov = _compute_overlap_matrix(h)
    m = np.max(ov)
    exps = np.exp((ov - m) / tau)
    weights = exps / np.sum(exps)  # softmax weights over shifts
    
    N = len(h)
    grad = np.zeros(N)
    for k in range(1, N):
        w = weights[k - 1]
        # d/dh[i] of overlap[k] for i in [0, N-1-k):
        # overlap[k] = sum_{i=0}^{N-1-k} h[i]*(1-h[i+k])
        # d/dh[i] = (1 - h[i+k]) for i in [0, N-1-k)
        # d/dh[i+k] = -h[i] for i in [0, N-1-k)
        # So for index j:
        #   if j < N-k: contribution from term h[j]*(1-h[j+k]) -> (1-h[j+k])
        #   if j >= k: contribution from term h[j-k]*(1-h[j]) -> -h[j-k]
        for i in range(N - k):
            grad[i] += w * (1.0 - h[i + k])
            grad[i + k] += w * (-h[i])
    return grad


def _project_to_box_sum(h, target_sum):
    """Project h onto [0,1]^N with sum = target_sum. Simple clipping + redistribution."""
    h = np.clip(h, 0.0, 1.0)
    diff = target_sum - np.sum(h)
    if abs(diff) < 1e-12:
        return h
    # Distribute diff to elements not at bounds
    if diff > 0:
        mask = h < 1.0 - 1e-12
        count = np.sum(mask)
        if count == 0:
            return h
        h[mask] += diff / count
    else:
        mask = h > 1e-12
        count = np.sum(mask)
        if count == 0:
            return h
        h[mask] += diff / count
    return np.clip(h, 0.0, 1.0)


def _refine_with_lbgfsb(h0, target_sum, max_iter=500, tol=1e-10, tau=0.005):
    """Polish with L-BFGS-B using soft-max surrogate."""
    from scipy.optimize import minimize
    
    target_sum = N / 2.0
    
    def objective(x):
        return _soft_max_overlap(x, tau)
    
    def jac(x):
        return _grad_soft_max_overlap(x, tau)
    
    bounds = [(0.0, 1.0)] * N
    # Add equality constraint for sum
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - target_sum,
                   'jac': lambda x: np.ones(N)}]
    
    try:
        res = minimize(objective, h0, method='L-BFGS-B', jac=jac,
                       bounds=bounds, constraints=constraints,
                       options={'maxiter': max_iter, 'ftol': tol, 'gtol': tol,
                                'maxfun': 20000, 'maxls': 50})
        return res.x
    except Exception:
        return h0


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin
    
    target_sum = N / 2.0
    
    best_h = np.full(N, 0.5)
    best_score = _max_overlap(best_h)
    
    # Generate diverse initial conditions
    rng = np.random.RandomState(42)
    
    inits = []
    
    # 1. Bang-bang patterns (alternating 0/1)
    for period in [2, 3, 4, 5, 6, 7, 8, 10, 12, 15]:
        h_bang = np.zeros(N)
        for i in range(N):
            h_bang[i] = 1.0 if (i % period) < period // 2 else 0.0
        h_bang = _project_to_box_sum(h_bang, target_sum)
        inits.append(h_bang)
    
    # 2. Sine-wave patterns
    for freq in [1, 2, 3, 5, 8, 13]:
        for phase in [0, 0.5, 1.0]:
            t = np.arange(N)
            h_sine = 0.5 + 0.5 * np.sin(2 * np.pi * freq * t / N + phase)
            h_sine = _project_to_box_sum(h_sine, target_sum)
            inits.append(h_sine)
    
    # 3. Random inits
    for _ in range(20):
        h_rand = rng.uniform(0, 1, N)
        h_rand = _project_to_box_sum(h_rand, target_sum)
        inits.append(h_rand)
    
    # 4. Block patterns
    for block_size in [2, 3, 4, 5]:
        h_block = np.zeros(N)
        for i in range(N):
            h_block[i] = 1.0 if (i // block_size) % 2 == 0 else 0.0
        h_block = _project_to_box_sum(h_block, target_sum)
        inits.append(h_block)
    
    # Evaluate all inits and keep top few
    scored_inits = []
    for h_init in inits:
        s = _max_overlap(h_init)
        scored_inits.append((s, h_init))
    
    scored_inits.sort(key=lambda x: x[0])
    
    # Take top 8 for polishing
    top_k = 8
    candidates = [h for _, h in scored_inits[:top_k]]
    
    # Phase 1: Quick polish with moderate tau
    tau1 = 0.02
    for h_cand in candidates:
        if time.time() > deadline - 20:
            break
        h_polished = _refine_with_lbgfsb(h_cand, target_sum, max_iter=200, tol=1e-9, tau=tau1)
        h_polished = _project_to_box_sum(h_polished, target_sum)
        s = _max_overlap(h_polished)
        if s < best_score:
            best_score = s
            best_h = h_polished.copy()
    
    # Phase 2: Finer polish with smaller tau
    tau2 = 0.005
    for h_cand in candidates:
        if time.time() > deadline - 10:
            break
        h_polished = _refine_with_lbgfsb(h_cand, target_sum, max_iter=300, tol=1e-11, tau=tau2)
        h_polished = _project_to_box_sum(h_polished, target_sum)
        s = _max_overlap(h_polished)
        if s < best_score:
            best_score = s
            best_h = h_polished.copy()
    
    # Phase 3: Very fine polish on the best
    if time.time() < deadline - 5:
        tau3 = 0.001
        h_polished = _refine_with_lbgfsb(best_h, target_sum, max_iter=500, tol=1e-12, tau=tau3)
        h_polished = _project_to_box_sum(h_polished, target_sum)
        s = _max_overlap(h_polished)
        if s < best_score:
            best_score = s
            best_h = h_polished.copy()
    
    # Final: ensure sum is exactly N/2 and values in [0,1]
    best_h = np.clip(best_h, 0.0, 1.0)
    # Adjust sum precisely
    diff = target_sum - np.sum(best_h)
    if abs(diff) > 1e-15:
        # Distribute to interior points
        interior = (best_h > 1e-10) & (best_h < 1.0 - 1e-10)
        if np.sum(interior) > 0:
            best_h[interior] += diff / np.sum(interior)
            best_h = np.clip(best_h, 0.0, 1.0)
        else:
            # Fallback: uniform tiny adjustment
            best_h += diff / N
            best_h = np.clip(best_h, 0.0, 1.0)
    
    return best_h
