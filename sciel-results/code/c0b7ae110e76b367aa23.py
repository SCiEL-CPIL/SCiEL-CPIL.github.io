"""Second autocorrelation inequality: spectral water-filling approach.

Maximize  ||h||_2^2 / (||h||_1 * ||h*h||_inf)  over 50 nonnegative heights.
Scale-invariant, so fix ||h||_1 = 1 and minimize  ||h||_2^2 * ||h*h||_inf.

New structural approach: treat the problem in the Fourier domain.
  h*h at frequency omega is |H(omega)|^2 where H is the DFT of h.
  ||h*h||_inf = max over spatial lag k of (h*h)[k] = max_k (1/2pi) int |H|^2 e^{ikx} dx
  but discretely, (h*h)[k] = sum_i h[i] h[i-k].

We use iterative water-filling:
  1. Fix a peak lag p (the lag where autoconvolution is largest).
  2. The constraint (h*h)[p] <= t  is a quadratic constraint on h.
  3. Minimize ||h||_2^2 subject to h>=0, sum h = 1, and (h*h)[p] <= t.
  4. Use Lagrange multipliers / projected gradient in the quadratic-constraint space.

Actually, a cleaner spectral approach:
  Let A_p be the N x N matrix for (h*h)[p] = h^T A_p h (symmetric Toeplitz).
  We want to minimize h^T h * (h^T A_p h) subject to h>=0, 1^T h = 1.
  
  This is a ratio of quadratics. By setting t = h^T A_p h, we minimize h^T h subject to
  h^T A_p h <= t, h >= 0, 1^T h = 1. For a fixed t, this is a QP.
  We can use bisection on t combined with QP, or use the fact that the optimal h
  satisfies the KKT conditions.

  Alternative: use the substitution. Let f(h) = (h^T h)(h^T A_p h). 
  Gradient: 2 h (h^T A_p h) + 2 (h^T h) A_p h.
  At optimum with h>0: gradient is parallel to 1 (the constraint normal).
  
  So: 2 h (h^T A_p h) + 2 (h^T h) A_p h = lambda * 1  (for h_i > 0)
  
  This is a nonlinear system. We can solve it iteratively.

Let me implement a practical version:
  - For each peak index p, solve the nonlinear system via fixed-point / Newton-like iteration.
  - Use multiple peak indices and diverse starts.
"""

import time
import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint

N = 50
TIME_LIMIT = 145.0  # safety margin before 150s hard kill
START = time.time()

def _elapsed():
    return time.time() - START

def _deadline():
    return TIME_LIMIT - 3.0

# Build the autoconvolution matrix for each lag
# (h*h)[k] for k in [-(N-1), N-1]
# We'll build A_p for each possible peak position p in [0, 2N-2]
# where p corresponds to lag k = p - (N-1)

def _build_A_for_lag(k):
    """Build symmetric N x N matrix A such that h^T A h = sum_i h[i] h[i-k]."""
    A = np.zeros((N, N))
    for i in range(N):
        j = i - k
        if 0 <= j < N:
            A[i, j] = 0.5
            A[j, i] = 0.5
    # For k=0, A[i,i] should be 1.0, not 0.5+0.5=1.0, so it's fine.
    # Wait: for k=0, j=i, so A[i,i] gets 0.5 from A[i,j] and 0.5 from A[j,i], total 1.0. Good.
    # For k != 0, A[i,j] = 0.5 and A[j,i] = 0.5, so h^T A h = sum_{i,j: j=i-k} 0.5 h[i] h[j] + 0.5 h[j] h[i]
    # = sum_i h[i] h[i-k]. Good.
    return A

# Precompute A for all lags
L = 2 * N - 1
A_lags = {}
for p in range(L):
    k = p - (N - 1)
    A_lags[p] = _build_A_for_lag(k)

def _ratio(h):
    h = np.asarray(h, dtype=float)
    l1 = h.sum()
    if l1 <= 0:
        return 0.0
    l2sq = float(h @ h)
    # Compute full autoconvolution
    # (h*h)[k] for k in [-(N-1), N-1]
    best_pinf = 0.0
    for p in range(L):
        k = p - (N - 1)
        val = float(h @ A_lags[p] @ h)
        if val > best_pinf:
            best_pinf = val
    if best_pinf <= 0:
        return 0.0
    return l2sq / (l1 * best_pinf)

def _cost(h, p):
    """Cost for fixed peak p: ||h||_2^2 * (h*h)[p] with ||h||_1 = 1."""
    h = np.asarray(h, dtype=float)
    l1 = h.sum()
    if l1 <= 0:
        return np.inf
    hn = h / l1
    l2sq = float(hn @ hn)
    conv_p = float(hn @ A_lags[p] @ hn)
    if conv_p <= 0:
        return np.inf
    return l2sq * conv_p

def _full_cost(h):
    """Full cost: ||h||_2^2 * ||h*h||_inf with ||h||_1 = 1."""
    h = np.asarray(h, dtype=float)
    l1 = h.sum()
    if l1 <= 0:
        return np.inf
    hn = h / l1
    l2sq = float(hn @ hn)
    best_conv = 0.0
    for p in range(L):
        val = float(hn @ A_lags[p] @ hn)
        if val > best_conv:
            best_conv = val
    if best_conv <= 0:
        return np.inf
    return l2sq * best_conv

def _solve_kkt(p, h0, max_iter=500, tol=1e-10):
    """Solve KKT conditions for fixed peak p.
    
    Minimize f(h) = (h^T h)(h^T A h) subject to h >= 0, 1^T h = 1.
    KKT: 2 h (h^T A h) + 2 (h^T h) A h = lambda * 1  for h_i > 0
         h_i >= 0, h_i * (lambda - [2 h (h^T A h) + 2 (h^T h) A h]_i) = 0
    """
    A = A_lags[p]
    h = np.array(h0, dtype=float)
    h = np.maximum(h, 0.0)
    if h.sum() <= 0:
        h = np.ones(N) / N
    h = h / h.sum()
    
    for it in range(max_iter):
        l2sq = float(h @ h)
        conv = float(h @ A @ h)
        grad = 2.0 * h * conv + 2.0 * l2sq * (A @ h)
        
        # Find lambda as the median of grad over positive coordinates
        pos = h > 1e-15
        if pos.sum() < 2:
            # Too many zeros, perturb
            h = np.maximum(h, 1e-10)
            h = h / h.sum()
            continue
        lambda_val = np.mean(grad[pos])
        
        # Update: for each i, if grad_i < lambda, set h_i to a value that makes grad_i = lambda
        # This is implicit. Use a damped update.
        # h_new_i = h_i * exp(-eta * (grad_i - lambda))
        eta = 0.3
        diff = grad - lambda_val
        h_new = h * np.exp(-eta * diff / (np.max(np.abs(diff)) + 1e-15))
        h_new = np.maximum(h_new, 0.0)
        s = h_new.sum()
        if s <= 0:
            break
        h_new = h_new / s
        
        change = np.max(np.abs(h_new - h))
        h = h_new
        if change < tol:
            break
    
    return h

def _waterfill(p, h0, max_iter=300):
    """Iterative water-filling: allocate h inversely proportional to local spectral pressure.
    
    The idea: at the peak lag p, the autoconvolution is sum_i h[i] h[i-p].
    To suppress this, we want h[i] h[i-p] to be small for all i.
    This suggests h[i] should be small where h[i-p] is large.
    
    We iterate:
      h_new[i] proportional to 1 / (1 + alpha * h[i-p])  (cyclic shift)
    This is a form of water-filling that suppresses correlated pairs.
    """
    A = A_lags[p]
    h = np.array(h0, dtype=float)
    h = np.maximum(h, 0.0)
    if h.sum() <= 0:
        h = np.ones(N) / N
    h = h / h.sum()
    
    # Determine the shift k for lag p
    k = p - (N - 1)
    
    for it in range(max_iter):
        # Compute the "pressure" at each position: how much it contributes to the peak
        # h[i] h[i-k] is the contribution. We want to reduce h where the partner is large.
        # pressure[i] = h[(i-k) % N]  (cyclic, but we should handle boundaries)
        pressure = np.zeros(N)
        for i in range(N):
            j = i - k
            if 0 <= j < N:
                pressure[i] = h[j]
        
        # Water-filling: h_new[i] = 1 / (1 + alpha * pressure[i])
        alpha = 1.0
        h_new = 1.0 / (1.0 + alpha * pressure)
        # Also incorporate the L2 penalty: we want to spread mass to reduce h^T h
        # Add a small uniform component
        h_new = h_new + 0.1
        h_new = np.maximum(h_new, 0.0)
        s = h_new.sum()
        if s <= 0:
            break
        h_new = h_new / s
        
        change = np.max(np.abs(h_new - h))
        h = h_new
        if change < 1e-12:
            break
    
    return h

def _lbfgs_refine(p, h0):
    """Use scipy L-BFGS-B to refine h for fixed peak p."""
    A = A_lags[p]
    
    def obj(x):
        l2sq = float(x @ x)
        conv = float(x @ A @ x)
        return l2sq * conv
    
    def grad(x):
        l2sq = float(x @ x)
        conv = float(x @ A @ x)
        return 2.0 * x * conv + 2.0 * l2sq * (A @ x)
    
    bounds = [(0.0, 1.0)] * N
    constraints = [{'type': 'eq', 'fun': lambda x: x.sum() - 1.0,
                    'jac': lambda x: np.ones(N)}]
    
    try:
        res = minimize(obj, h0, jac=grad, method='L-BFGS-B', bounds=bounds,
                       constraints=constraints, options={'maxiter': 200, 'ftol': 1e-14})
        if res.success or res.fun < _cost(h0, p):
            h = np.maximum(res.x, 0.0)
            if h.sum() > 0:
                h = h / h.sum()
                return h
    except Exception:
        pass
    return h0

def _try_start(h0, p, t_end):
    """Try multiple refinement methods from start h0 with peak p."""
    h = h0.copy()
    
    # Water-filling
    if _elapsed() < t_end:
        h_wf = _waterfill(p, h, max_iter=200)
        if _cost(h_wf, p) < _cost(h, p):
            h = h_wf
    
    # KKT solver
    if _elapsed() < t_end:
        h_kkt = _solve_kkt(p, h, max_iter=300)
        if _cost(h_kkt, p) < _cost(h, p):
            h = h_kkt
    
    # L-BFGS refinement
    if _elapsed() < t_end:
        h_lb = _lbfgs_refine(p, h)
        if _cost(h_lb, p) < _cost(h, p):
            h = h_lb
    
    # Re-detect peak and iterate once more
    if _elapsed() < t_end:
        # Find actual peak
        best_p = 0
        best_val = 0.0
        for pp in range(L):
            val = float(h @ A_lags[pp] @ h)
            if val > best_val:
                best_val = val
                best_p = pp
        if best_p != p:
            h2 = _try_start(h, best_p, t_end)
            if _full_cost(h2) < _full_cost(h):
                h = h2
    
    return h

def _optimize():
    t_end = START + _deadline()
    rng = np.random.default_rng(42)
    
    best = np.ones(N) / N
    best_score = _ratio(best)
    
    # Generate diverse starting points
    starts = []
    
    # Uniform
    starts.append(np.ones(N) / N)
    
    # Single spikes
    for s in range(N):
        v = np.zeros(N)
        v[s] = 1.0
        starts.append(v)
    
    # Small blocks
    for w in (2, 3, 4, 5):
        for s in range(0, N - w + 1):
            v = np.zeros(N)
            v[s:s+w] = 1.0 / w
            starts.append(v)
    
    # Alternating patterns
    for period in (2, 3, 4, 5, 7):
        v = np.zeros(N)
        for i in range(N):
            if i % period == 0:
                v[i] = 1.0
        if v.sum() > 0:
            v = v / v.sum()
            starts.append(v)
    
    # Random sparse
    for _ in range(60):
        v = rng.random(N)
        threshold = rng.uniform(0.3, 0.8)
        v[v < threshold] = 0.0
        if v.sum() > 0:
            v = v / v.sum()
            starts.append(v)
    
    # Random dense
    for _ in range(40):
        v = rng.random(N)
        v = v / v.sum()
        starts.append(v)
    
    # Gaussian-like
    for center in (10, 20, 25, 30, 40):
        for width in (3, 5, 8):
            x = np.arange(N)
            v = np.exp(-0.5 * ((x - center) / width) ** 2)
            v = v / v.sum()
            starts.append(v)
    
    # Deduplicate
    seen = set()
    uniq = []
    for v in starts:
        key = tuple(np.round(v, 4))
        if key not in seen:
            seen.add(key)
            uniq.append(v)
    starts = uniq
    
    # For each start, try multiple peak indices
    for idx, v in enumerate(starts):
        if _elapsed() > t_end:
            break
        
        # Find the peak of this start
        conv_vals = np.array([float(v @ A_lags[p] @ v) for p in range(L)])
        peak = int(np.argmax(conv_vals))
        
        # Try the detected peak and a few alternatives
        peaks_to_try = [peak]
        # Add peaks with high conv values
        top_peaks = np.argsort(conv_vals)[::-1][:5]
        for tp in top_peaks:
            if tp not in peaks_to_try:
                peaks_to_try.append(int(tp))
        
        for p in peaks_to_try:
            if _elapsed() > t_end:
                break
            h = _try_start(v, p, t_end)
            sc = _ratio(h)
            if sc > best_score:
                best_score = sc
                best = h.copy()
    
    # Final polish: coordinate descent on the best solution
    if _elapsed() < t_end:
        h = best.copy()
        for round_ in range(5):
            if _elapsed() > t_end:
                break
            improved = False
            for i in range(N):
                base = h[i]
                mean_val = h.sum() / N
                cands = [0.0, base, base * 0.5, base * 1.5, mean_val, 2 * mean_val, 4 * mean_val]
                cands = [c for c in cands if c >= 0 and c <= 1.0]
                cur_cost = _full_cost(h)
                for c in cands:
                    if abs(c - base) < 1e-12:
                        continue
                    h2 = h.copy()
                    h2[i] = c
                    if h2.sum() <= 0:
                        continue
                    h2 = h2 / h2.sum()
                    cc = _full_cost(h2)
                    if cc < cur_cost - 1e-12:
                        cur_cost = cc
                        h = h2
                        improved = True
            best = h
            if not improved:
                break
        sc = _ratio(best)
        if sc > best_score:
            best_score = sc
    
    return best, best_score

_cache = None

def construct():
    global _cache
    if _cache is None:
        _cache = _optimize()
    best, _ = _cache
    return [float(x) for x in best]
