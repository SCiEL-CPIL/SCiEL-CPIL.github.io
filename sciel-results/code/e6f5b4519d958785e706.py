"""Erdos minimum overlap: minimize max autocorrelation of h vs 1-h."""
import time
import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds

N = 100
TARGET_SUM = N / 2.0  # 50.0
TOL = 1e-9 * N        # 1e-7

# Precompute correlation structure for fast evaluation
# value = max(np.correlate(h, 1-h, mode="full")) / N * 2
# Let g = 1 - h, so we compute correlate(h, g, mode="full")
# correlate(h, g, k) for lag k: sum_i h[i+k] * g[i] for appropriate ranges
# For mode="full", lags go from -(N-1) to N-1
# c[k] = sum_{i} h[i] * g[i+k] where indices are valid
# Actually np.correlate(a, v, mode='full')[i] = sum_j a[j] * v[i-j] ... let me think carefully.
# np.correlate(h, g, mode='full') has length 2N-1
# Index i corresponds to lag = i - (N-1)
# c[i] = sum_j h[j] * g[j + (N-1-i)] ... this is getting complex.
# Let's just use the direct formula:
# For lag d (where d = i - (N-1), so i = N-1+d, d in [-(N-1), N-1]):
# c[d] = sum_{j} h[j] * g[j+d] where 0 <= j < N and 0 <= j+d < N
#       = sum_{j=max(0,-d)}^{min(N-1, N-1-d)} h[j] * (1 - h[j+d])

def _value(h):
    """Compute the objective value (lower is better)."""
    g = 1.0 - h
    corr = np.correlate(h, g, mode="full")
    return np.max(corr) / N * 2.0

def _objective(h):
    """Objective: maximize the minimum margin to make max correlation small.
    We minimize max(corr) subject to constraints."""
    g = 1.0 - h
    corr = np.correlate(h, g, mode="full")
    return np.max(corr)

def _grad(h):
    """Gradient of max(corr) w.r.t. h. Approximate via subgradient of max."""
    g = 1.0 - h
    corr = np.correlate(h, g, mode="full")
    idx = np.argmax(corr)
    # d(corr[idx])/d(h[j])
    # corr[idx] = sum_j h[j] * g[j + d] where d = idx - (N-1)
    d = idx - (N - 1)
    grad = np.zeros(N)
    # corr[idx] = sum_j h[j] * (1 - h[j+d]) for valid j
    # = sum_j h[j] - sum_j h[j]*h[j+d]
    # d/dh[j] = 1 - h[j+d] - h[j] (if j+d valid) ... but also h[j] appears in other terms
    # Let me be more careful.
    # corr[idx] = sum_{j: 0<=j<N, 0<=j+d<N} h[j] * (1 - h[j+d])
    # = sum_{j: 0<=j<N, 0<=j+d<N} h[j] - sum_{j: 0<=j<N, 0<=j+d<N} h[j]*h[j+d]
    
    valid_j = np.arange(max(0, -d), min(N, N - d))
    # Term 1: sum h[j] -> d/dh[j] = 1 for j in valid_j
    grad[valid_j] += 1.0
    # Term 2: sum h[j]*h[j+d]
    # d/dh[j] = h[j+d] if j in valid_j
    # d/dh[j] = h[j-d] if j-d in valid_j (i.e., j in valid_j + d)
    # For j in valid_j: contribution from h[j]*h[j+d] is h[j+d]
    grad[valid_j] -= h[valid_j + d]
    # For j such that j-d in valid_j, i.e., j in valid_j + d:
    valid_j2 = valid_j + d
    valid_j2 = valid_j2[(valid_j2 >= 0) & (valid_j2 < N)]
    grad[valid_j2] -= h[valid_j2 - d]
    
    return grad

def construct():
    """Construct 100 step heights minimizing the Erdos overlap value."""
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    # Start with seed
    h0 = np.full(N, 0.5)
    best_h = h0.copy()
    best_val = _value(best_h)
    
    # Constraints: 0 <= h[i] <= 1, sum(h) = N/2
    bounds = Bounds(0.0, 1.0)
    A_sum = np.ones((1, N))
    sum_constraint = LinearConstraint(A_sum, TARGET_SUM, TARGET_SUM)
    constraints = [sum_constraint]
    
    # Multi-start with random perturbations
    rng = np.random.RandomState(42)
    
    # Try several initial points
    initial_points = [h0.copy()]
    for _ in range(5):
        h_rand = rng.uniform(0, 1, N)
        # Project to satisfy sum constraint
        h_rand -= (h_rand.sum() - TARGET_SUM) / N
        h_rand = np.clip(h_rand, 0, 1)
        # Re-project to get exact sum
        diff = TARGET_SUM - h_rand.sum()
        h_rand += diff / N
        h_rand = np.clip(h_rand, 0, 1)
        initial_points.append(h_rand)
    
    for h_init in initial_points:
        if time.time() > deadline:
            break
        
        try:
            res = minimize(
                _objective,
                h_init,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            if res.success and np.all(np.isfinite(res.x)):
                h_candidate = res.x
                # Verify constraints
                if (abs(h_candidate.sum() - TARGET_SUM) < 1e-6 and
                    np.all(h_candidate >= -1e-9) and np.all(h_candidate <= 1 + 1e-9)):
                    h_candidate = np.clip(h_candidate, 0, 1)
                    # Re-project for exact sum
                    diff = TARGET_SUM - h_candidate.sum()
                    h_candidate += diff / N
                    h_candidate = np.clip(h_candidate, 0, 1)
                    val = _value(h_candidate)
                    if val < best_val:
                        best_val = val
                        best_h = h_candidate.copy()
        except Exception:
            pass
    
    # Final polishing with tighter tolerance
    if time.time() < deadline - 10:
        try:
            res = minimize(
                _objective,
                best_h,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-15}
            )
            if res.success and np.all(np.isfinite(res.x)):
                h_candidate = res.x
                if (abs(h_candidate.sum() - TARGET_SUM) < 1e-6 and
                    np.all(h_candidate >= -1e-9) and np.all(h_candidate <= 1 + 1e-9)):
                    h_candidate = np.clip(h_candidate, 0, 1)
                    diff = TARGET_SUM - h_candidate.sum()
                    h_candidate += diff / N
                    h_candidate = np.clip(h_candidate, 0, 1)
                    val = _value(h_candidate)
                    if val < best_val:
                        best_val = val
                        best_h = h_candidate.copy()
        except Exception:
            pass
    
    # Ensure final validity
    best_h = np.clip(best_h, 0, 1)
    diff = TARGET_SUM - best_h.sum()
    best_h += diff / N
    best_h = np.clip(best_h, 0, 1)
    
    return best_h
