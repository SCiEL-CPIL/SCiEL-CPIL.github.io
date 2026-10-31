"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h (adjacent shift)."""
import numpy as np
from scipy.optimize import minimize
import time

N = 100

def _overlap_max(h):
    """Compute max of h[i]*(1-h[i+1]) with zero-padding at the end."""
    overlaps = h[:-1] * (1.0 - h[1:])
    overlaps = np.append(overlaps, h[-1])
    return overlaps.max()

def _compute_all_overlaps(h):
    """Compute all adjacent overlaps as a vector."""
    overlaps = h[:-1] * (1.0 - h[1:])
    overlaps = np.append(overlaps, h[-1])
    return overlaps

def _smooth_max(values, alpha):
    """Smooth approximation of max using log-sum-exp."""
    m = np.max(values)
    if m > 1e10:
        return m
    shifted = alpha * (values - m)
    shifted = np.clip(shifted, -50, 50)
    return m + np.log(np.sum(np.exp(shifted))) / alpha

def _objective(h, alpha):
    """Smooth surrogate objective: smooth max of adjacent overlaps."""
    overlaps = _compute_all_overlaps(h)
    return _smooth_max(overlaps, alpha)

def _objective_grad(h, alpha):
    """Gradient of the smooth max objective."""
    overlaps = _compute_all_overlaps(h)
    m = np.max(overlaps)
    shifted = alpha * (overlaps - m)
    shifted = np.clip(shifted, -50, 50)
    exp_vals = np.exp(shifted)
    weights = exp_vals / np.sum(exp_vals)
    
    grad = np.zeros(N)
    for i in range(N - 1):
        grad[i] += weights[i] * (1.0 - h[i+1])
        grad[i+1] += weights[i] * (-h[i])
    grad[N-1] += weights[N-1] * 1.0
    
    return grad

def _project_to_polytope(h, S):
    """Project h onto { x : sum(x) = S, 0 <= x_i <= 1 } using binary search."""
    if S < 0 or S > N:
        return np.clip(h, 0, 1)
        
    lo = np.min(h) - 1.0
    hi = np.max(h) + 1.0
    
    for _ in range(60):
        mid = (lo + hi) / 2.0
        val = np.sum(np.clip(h - mid, 0.0, 1.0))
        if val > S:
            lo = mid
        else:
            hi = mid
                
    lambda_star = (lo + hi) / 2.0
    return np.clip(h - lambda_star, 0.0, 1.0)

def _enforce_constraints(h):
    """Project h onto feasible set: [0,1]^N with sum = N/2."""
    return _project_to_polytope(h, N / 2.0)

def _optimize_from_start(h0, alpha, max_iter=200):
    """Optimize from a starting point using SLSQP."""
    n = len(h0)
    target = n / 2.0
    
    bounds = [(0.0, 1.0)] * n
    constraints = [{'type': 'eq', 'fun': lambda x: x.sum() - target, 
                    'jac': lambda x: np.ones(n)}]
    
    def obj(x):
        return _objective(x, alpha)
    
    def obj_grad(x):
        return _objective_grad(x, alpha)
    
    try:
        res = minimize(obj, h0, jac=obj_grad, method='SLSQP', bounds=bounds, 
                      constraints=constraints,
                      options={'maxiter': max_iter, 'ftol': 1e-14, 'disp': False})
        if res.success or (hasattr(res, 'fun') and res.fun < _objective(h0, alpha)):
            return res.x
    except Exception:
        pass
    return h0

def _coordinate_descent(h, max_iters=500, tol=1e-12, deadline=None):
    """
    Coordinate descent local search: minimize max overlap by adjusting individual h_i.
    For each coordinate i, the overlap contributions are:
      - overlap[i-1] = h[i-1]*(1-h[i]) if i > 0
      - overlap[i] = h[i]*(1-h[i+1]) if i < N-1
      - overlap[i] = h[i] if i == N-1
    We try to find the best value of h[i] in [0,1] that minimizes the max of these.
    """
    h = h.copy()
    target = N / 2.0
    
    for iteration in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
            
        improved = False
        for i in range(N):
            # Save original value
            orig = h[i]
            
            # Determine which overlaps are affected by h[i]
            # overlap[i-1] involves h[i] as the second factor: h[i-1]*(1-h[i])
            # overlap[i] involves h[i] as the first factor: h[i]*(1-h[i+1]) or h[i] if i==N-1
            
            # We want to find h[i] that minimizes max of affected overlaps
            # Try a few candidate values: current, and values that equalize the affected overlaps
            
            # Collect current overlap values for affected indices
            affected_idx = []
            if i > 0:
                affected_idx.append(i - 1)
            if i < N - 1:
                affected_idx.append(i)
            else:
                affected_idx.append(i)  # last element
            
            # Current max overlap
            current_max = _overlap_max(h)
            
            # Try setting h[i] to make affected overlaps equal
            # For interior i (0 < i < N-1):
            #   overlap[i-1] = h[i-1]*(1-h[i])
            #   overlap[i] = h[i]*(1-h[i+1])
            # Set equal: h[i-1]*(1-h[i]) = h[i]*(1-h[i+1])
            # h[i-1] - h[i-1]*h[i] = h[i] - h[i]*h[i+1]
            # h[i-1] = h[i] - h[i]*h[i+1] + h[i-1]*h[i]
            # h[i-1] = h[i]*(1 + h[i-1] - h[i+1])
            # h[i] = h[i-1] / (1 + h[i-1] - h[i+1])
            
            candidates = [orig]
            
            if 0 < i < N - 1:
                denom = 1.0 + h[i-1] - h[i+1]
                if abs(denom) > 1e-12:
                    candidate = h[i-1] / denom
                    if 0.0 <= candidate <= 1.0:
                        candidates.append(candidate)
                
                # Also try values that make one overlap zero
                # overlap[i-1] = 0 => h[i] = 1
                # overlap[i] = 0 => h[i] = 0 or h[i+1] = 1
                candidates.append(0.0)
                candidates.append(1.0)
            elif i == 0:
                # Only overlap[0] = h[0]*(1-h[1]) affected
                # Minimize by setting h[0]=0 or h[1]=1, but we're only changing h[0]
                candidates.append(0.0)
            elif i == N - 1:
                # overlap[N-2] = h[N-2]*(1-h[N-1]) and overlap[N-1] = h[N-1]
                # Set equal: h[N-2]*(1-h[N-1]) = h[N-1]
                # h[N-2] - h[N-2]*h[N-1] = h[N-1]
                # h[N-2] = h[N-1]*(1 + h[N-2])
                # h[N-1] = h[N-2] / (1 + h[N-2])
                denom = 1.0 + h[N-2]
                if denom > 1e-12:
                    candidate = h[N-2] / denom
                    if 0.0 <= candidate <= 1.0:
                        candidates.append(candidate)
                candidates.append(0.0)
                candidates.append(1.0)
            
            # Try each candidate and see if it improves
            best_candidate = orig
            best_score = current_max
            
            for cand in candidates:
                h[i] = cand
                # Adjust other coordinates to maintain sum constraint approximately
                # For now, just check the overlap without enforcing sum
                score = _overlap_max(h)
                if score < best_score - tol:
                    best_score = score
                    best_candidate = cand
            
            h[i] = best_candidate
            
            if best_candidate != orig:
                improved = True
                # Re-enforce constraints after each improvement
                h = _enforce_constraints(h)
        
        if not improved:
            break
    
    return h

def _try_ramp_pattern(deadline):
    """Try a ramp/plateau pattern: values rise from 0 to 1 and fall back."""
    best_h = None
    best_score = float('inf')
    
    for width in range(5, N+1, 2):
        for offset in range(0, N - width + 1, 2):
            h = np.zeros(N)
            ramp_len = min(width // 2, N - offset)
            if ramp_len > 0:
                h[offset:offset+ramp_len] = np.linspace(0, 1, ramp_len)
            plateau_start = offset + ramp_len
            plateau_end = offset + width
            if plateau_start < plateau_end and plateau_end <= N:
                h[plateau_start:plateau_end] = 1.0
            if plateau_end < N:
                ramp_down_len = min(width // 2, N - plateau_end)
                if ramp_down_len > 0:
                    h[plateau_end:plateau_end+ramp_down_len] = np.linspace(1, 0, ramp_down_len)
            
            h = _enforce_constraints(h)
            score = _overlap_max(h)
            if score < best_score:
                best_score = score
                best_h = h.copy()
        
        if time.time() > deadline - 20:
            break
    
    return best_h, best_score

def _try_block_pattern(deadline):
    """Try block patterns with smooth transitions."""
    best_h = None
    best_score = float('inf')
    
    for n_blocks in [2, 3, 4, 5]:
        for block_width in range(5, N // n_blocks + 1, 3):
            h = np.zeros(N)
            for i in range(n_blocks):
                start = i * (N // n_blocks)
                end = start + block_width
                if end <= N:
                    trans = min(3, block_width // 2)
                    if trans > 0:
                        h[start:start+trans] = np.linspace(0, 1, trans)
                        h[end-trans:end] = np.linspace(1, 0, trans)
                        if block_width > 2 * trans:
                            h[start+trans:end-trans] = 1.0
                    else:
                        h[start:end] = 1.0
            
            h = _enforce_constraints(h)
            score = _overlap_max(h)
            if score < best_score:
                best_score = score
                best_h = h.copy()
        
        if time.time() > deadline - 20:
            break
    
    return best_h, best_score

def construct():
    deadline = time.time() + 135.0  # Safety margin
    best_h = np.full(N, 0.5)
    best_score = _overlap_max(best_h)
    
    # Try structured patterns first
    h1, s1 = _try_ramp_pattern(deadline)
    if h1 is not None and s1 < best_score:
        best_h = h1
        best_score = s1
    
    h2, s2 = _try_block_pattern(deadline)
    if h2 is not None and s2 < best_score:
        best_h = h2
        best_score = s2
    
    # Multi-start optimization
    rng = np.random.default_rng(42)
    
    starts = []
    starts.append(best_h.copy())
    starts.append(np.full(N, 0.5))
    alt = np.zeros(N)
    alt[::2] = 1.0
    starts.append(_enforce_constraints(alt))
    for freq in [1, 2, 3, 5, 8]:
        x = np.arange(N)
        h = 0.5 + 0.4 * np.sin(2 * np.pi * freq * x / N)
        starts.append(_enforce_constraints(h))
    for i in range(6):
        h = rng.uniform(0, 1, N)
        starts.append(_enforce_constraints(h))
    
    # Annealing with increasing alpha
    alphas = [20.0, 100.0, 500.0, 2000.0, 10000.0]
    
    for start_idx, h0 in enumerate(starts):
        if time.time() > deadline - 15:
            break
        
        h = h0.copy()
        
        for alpha in alphas:
            if time.time() > deadline - 10:
                break
            
            h = _optimize_from_start(h, alpha, max_iter=100)
            h = _enforce_constraints(h)
            
            score = _overlap_max(h)
            if score < best_score:
                best_score = score
                best_h = h.copy()
        
        # Coordinate descent polish
        if time.time() < deadline - 5:
            h = _coordinate_descent(h, max_iters=200, deadline=deadline - 5)
            h = _enforce_constraints(h)
            score = _overlap_max(h)
            if score < best_score:
                best_score = score
                best_h = h.copy()
        
        # Final high-alpha polish
        if time.time() < deadline - 5:
            h = _optimize_from_start(h, 20000.0, max_iter=200)
            h = _enforce_constraints(h)
            score = _overlap_max(h)
            if score < best_score:
                best_score = score
                best_h = h.copy()
    
    # Final global polish on best solution
    if time.time() < deadline - 2:
        h = _optimize_from_start(best_h, 50000.0, max_iter=300)
        h = _enforce_constraints(h)
        score = _overlap_max(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()
    
    # Final coordinate descent polish
    if time.time() < deadline - 1:
        h = _coordinate_descent(best_h, max_iters=500, deadline=deadline - 1)
        h = _enforce_constraints(h)
        score = _overlap_max(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()
    
    # Ensure exact sum
    best_h = _enforce_constraints(best_h)
    
    return best_h
