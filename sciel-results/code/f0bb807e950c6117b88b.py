import numpy as np
import time
from scipy.optimize import minimize, LinearConstraint

N = 50
DELTA = 1.0 / N

# Precompute the autoconvolution overlap matrix
# A[k, i, j] = overlap length of interval [i, i+1] and [k-j, k-j+1] in units of DELTA
# We'll use a 2D representation: for each lag k, the contribution is sum_{i,j} h[i]*h[j]*A[k,i,j]
# Actually, let's precompute it more efficiently.

def _build_overlap_matrices():
    """Precompute overlap matrices for each lag k.
    
    For lag k, (f*f)(k*DELTA) = DELTA * sum_{i,j} h[i]*h[j] * overlap(i, j, k)
    where overlap(i,j,k) = max(0, min(i+1, k-j+1) - max(i, k-j))
    
    We can write this as h @ M_k @ h where M_k is a matrix.
    Actually it's sum_i sum_j h[i] h[j] A[k,i,j] which is a quadratic form.
    """
    # A[k, i, j] for k in [0, 2N-2], i in [0, N-1], j in [0, N-1]
    # This is large: 99 * 50 * 50 = 247500 entries, manageable
    num_lags = 2 * N - 1
    # We'll store as a 3D array or use the fact that for each k, it's a matrix
    # Let's precompute M_k matrices
    M = np.zeros((num_lags, N, N))
    for k in range(num_lags):
        for i in range(N):
            for j in range(N):
                # overlap of [i, i+1] and [k-j, k-j+1]
                lo = max(i, k - j)
                hi = min(i + 1, k - j + 1)
                M[k, i, j] = max(0.0, hi - lo)
    return M

_M_matrices = _build_overlap_matrices()
_NUM_LAGS = 2 * N - 1

def _autoconv_sup(h):
    """Compute sup norm of autoconvolution using precomputed matrices."""
    h = np.maximum(h, 0)
    # ff_vals[k] = DELTA * h @ M[k] @ h
    ff_vals = np.array([DELTA * (h @ _M_matrices[k] @ h) for k in range(_NUM_LAGS)])
    return np.max(ff_vals)

def _autoconv_all(h):
    """Compute all autoconvolution values."""
    h = np.maximum(h, 0)
    ff_vals = np.array([DELTA * (h @ _M_matrices[k] @ h) for k in range(_NUM_LAGS)])
    return ff_vals

def _ratio(h):
    """Compute the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf)."""
    h = np.maximum(h, 0)
    sum_h2 = np.sum(h**2)
    sum_h = np.sum(h)
    if sum_h < 1e-15:
        return 0.0
    sup_ff = _autoconv_sup(h)
    if sup_ff < 1e-15:
        return 0.0
    return sum_h2 / (sum_h * sup_ff)

def _smooth_autoconv_sup(h, temperature=0.01):
    """Smooth approximation of max using log-sum-exp."""
    h = np.maximum(h, 0)
    ff_vals = np.array([DELTA * (h @ _M_matrices[k] @ h) for k in range(_NUM_LAGS)])
    # log-sum-exp for numerical stability
    max_val = np.max(ff_vals)
    if max_val < 1e-30:
        return 0.0
    shifted = ff_vals - max_val
    log_sum = max_val + temperature * np.log(np.sum(np.exp(shifted / temperature)))
    return log_sum

def _smooth_ratio(h, temperature=0.01):
    """Smooth version of the ratio for gradient-based optimization."""
    h = np.maximum(h, 0)
    sum_h2 = np.sum(h**2)
    sum_h = np.sum(h)
    if sum_h < 1e-15:
        return 0.0
    sup_ff = _smooth_autoconv_sup(h, temperature)
    if sup_ff < 1e-15:
        return 0.0
    return sum_h2 / (sum_h * sup_ff)

def generate_mian_chowla(n, max_size=None):
    """Generate a Mian-Chowla (Sidon) sequence."""
    if max_size is None:
        max_size = n
    seq = [1]
    used_sums = {2}
    for candidate in range(2, n + 1):
        valid = True
        for i in seq:
            if i + candidate in used_sums:
                valid = False
                break
        if valid:
            seq.append(candidate)
            used_sums.add(candidate + candidate)
            for i in seq[:-1]:
                used_sums.add(i + candidate)
        if len(seq) >= max_size:
            break
    # Convert to 0-indexed
    return [x - 1 for x in seq]

def _multiplicative_update(h, max_iter=500, tol=1e-8, temperature=0.05):
    """Sinkhorn-Knopp style multiplicative updates for the min-max game.
    
    The attacker chooses lag k to maximize the autoconvolution peak.
    The defender allocates mass to minimize the ratio denominator.
    
    We formulate: minimize_h max_k [DELTA * h @ M_k @ h] / (sum_h * sum_h2)
    Equivalently, for fixed normalization sum_h = 1:
    minimize_h max_k [DELTA * h @ M_k @ h] / sum_h2
    
    The multiplicative update for the defender (minimizing max_k) is:
    h_i <- h_i / (max_k sum_j M[k,i,j] * h_j)^(1/2)  ... but this isn't quite right.
    
    Let's think of it as: for each lag k, the "pressure" on h[i] is sum_j M[k,i,j] h[j].
    The worst-case lag is the one with the largest quadratic form.
    
    A better formulation: use the dual. Let p_k be the attacker's distribution over lags.
    The defender wants to minimize max_k Q_k(h) where Q_k(h) = h @ M_k @ h.
    The saddle point satisfies: for the optimal p, the gradient of sum_k p_k Q_k(h) = 0
    in the direction of h, and p_k is concentrated on the maximizing k's.
    
    Iterative scheme:
    1. Given h, compute Q_k = h @ M_k @ h for all k
    2. Update p_k proportional to exp(Q_k / temperature) (softmax over lags)
    3. Given p, the effective matrix is M_eff = sum_k p_k M_k
    4. Update h to minimize h @ M_eff @ h subject to sum h = 1, h >= 0
       Using multiplicative: h_i <- h_i / (M_eff @ h)_i (normalized)
    
    Actually, for minimizing h @ M @ h with sum h = 1, the multiplicative update is:
    h_i <- h_i / (M @ h)_i, then normalize.
    """
    h = np.maximum(h, 0).copy()
    s = np.sum(h)
    if s < 1e-15:
        return h
    h = h / s
    
    p = np.ones(_NUM_LAGS) / _NUM_LAGS
    
    for iteration in range(max_iter):
        # Compute Q_k = h @ M_k @ h
        Q = np.array([h @ _M_matrices[k] @ h for k in range(_NUM_LAGS)])
        
        # Update p (attacker): softmax over Q
        max_Q = np.max(Q)
        exp_Q = np.exp((Q - max_Q) / temperature)
        p = exp_Q / np.sum(exp_Q)
        
        # Effective matrix
        M_eff = np.sum([p[k] * _M_matrices[k] for k in range(_NUM_LAGS)], axis=0)
        
        # Multiplicative update for h (defender)
        Mh = M_eff @ h
        Mh = np.maximum(Mh, 1e-15)
        h_new = h / np.sqrt(Mh)  # geometric mean update
        s_new = np.sum(h_new)
        if s_new < 1e-15:
            break
        h_new = h_new / s_new
        
        # Check convergence
        if np.max(np.abs(h_new - h)) < tol:
            h = h_new
            break
        h = h_new
    
    return h

def _epigraph_optimize(h0, max_iter=300, temperature=0.01):
    """Epigraph formulation: minimize t subject to h @ M_k @ h <= t for all k, sum h = 1, h >= 0.
    Then maximize sum_h2 / t.
    
    Actually, let's directly optimize the ratio using the smooth surrogate.
    """
    h = np.maximum(h0, 0).copy()
    s = np.sum(h)
    if s < 1e-15:
        return h
    h = h / s
    
    # Use scipy minimize with SLSQP on the smooth ratio
    def neg_smooth_ratio(x):
        x = np.maximum(x, 0)
        s = np.sum(x)
        if s < 1e-15:
            return 0.0
        return -_smooth_ratio(x / s, temperature)
    
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(np.maximum(x, 0)) - 1.0}]
    bounds = [(0, None)] * N
    
    try:
        result = minimize(neg_smooth_ratio, h.copy(), method='SLSQP',
                         bounds=bounds, constraints=constraints,
                         options={'maxiter': max_iter, 'ftol': 1e-12})
        h_candidate = np.maximum(result.x, 0)
        s = np.sum(h_candidate)
        if s > 1e-15:
            h_candidate = h_candidate / s
        return h_candidate
    except Exception:
        return h

def _coordinate_descent_polish(h, max_iter=1000, tol=1e-10):
    """Coordinate descent with exact line search for each coordinate."""
    h = np.maximum(h, 0).copy()
    s = np.sum(h)
    if s < 1e-15:
        return h
    h = h / s
    
    for iteration in range(max_iter):
        improved = False
        for i in range(N):
            # Optimize h[i] while keeping others fixed
            # The ratio is h2 / (h1 * sup_ff)
            # For a fixed h[i], let's find the optimal value
            # This is complex because sup_ff depends on h[i]
            # Use a small grid search or golden section
            h_old_i = h[i]
            # Try a range of values
            best_val = h_old_i
            best_ratio = _ratio(h)
            
            # Golden section search over h[i] in [0, max(1, 3*h_old_i)]
            lo = 0.0
            hi = max(1.0, 3.0 * h_old_i)
            gr = (np.sqrt(5) - 1) / 2
            
            for _ in range(20):
                x1 = hi - gr * (hi - lo)
                x2 = lo + gr * (hi - lo)
                
                h[i] = x1
                r1 = _ratio(h)
                h[i] = x2
                r2 = _ratio(h)
                
                if r1 > r2:
                    hi = x2
                else:
                    lo = x1
            
            best_h_i = (lo + hi) / 2
            h[i] = best_h_i
            r_new = _ratio(h)
            
            if r_new > best_ratio + 1e-12:
                improved = True
            else:
                h[i] = h_old_i
        
        if not improved:
            break
    
    s = np.sum(h)
    if s > 1e-15:
        h = h / s
    return h

def _iterative_reweighting(h, max_iter=200, temperature=0.05):
    """Iterative reweighting: alternate between computing the worst lag and adjusting heights."""
    h = np.maximum(h, 0).copy()
    s = np.sum(h)
    if s < 1e-15:
        return h
    h = h / s
    
    for iteration in range(max_iter):
        # Compute autoconvolution values
        ff_vals = np.array([DELTA * (h @ _M_matrices[k] @ h) for k in range(_NUM_LAGS)])
        
        # Find the worst lag(s)
        max_ff = np.max(ff_vals)
        if max_ff < 1e-15:
            break
        
        # Compute the "pressure" from the worst lag
        # For the worst lag k*, the gradient of Q_k* w.r.t. h is 2 * M_k* @ h
        worst_k = np.argmax(ff_vals)
        M_worst = _M_matrices[worst_k]
        pressure = M_worst @ h
        pressure = np.maximum(pressure, 1e-15)
        
        # Update: reduce heights where pressure is high
        h_new = h / np.sqrt(pressure)
        s_new = np.sum(h_new)
        if s_new < 1e-15:
            break
        h_new = h_new / s_new
        
        if np.max(np.abs(h_new - h)) < 1e-10:
            h = h_new
            break
        h = h_new
    
    return h

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s kill
    
    best_h = np.ones(N)
    best_score = _ratio(best_h)
    
    np.random.seed(42)
    
    # Generate initializations
    initializations = []
    
    # 1. Uniform
    initializations.append(np.ones(N))
    
    # 2. Mian-Chowla / Sidon sets
    for size in [8, 10, 12, 14, 16]:
        mc = generate_mian_chowla(N, max_size=size)
        h_mc = np.zeros(N)
        for idx in mc:
            if 0 <= idx < N:
                h_mc[idx] = 1.0
        if np.sum(h_mc) > 0:
            initializations.append(h_mc)
    
    # 3. Sparse random
    for trial in range(15):
        h_sparse = np.zeros(N)
        num_ones = np.random.randint(5, 18)
        indices = np.random.choice(N, num_ones, replace=False)
        h_sparse[indices] = np.random.uniform(0.3, 2.0, num_ones)
        initializations.append(h_sparse)
    
    # 4. Structured patterns
    t = np.linspace(-1, 1, N)
    initializations.append(np.maximum(1 - t**2, 0.01))
    initializations.append(np.maximum(t**2 + 0.1, 0.01))
    
    h_alt = np.zeros(N)
    h_alt[::2] = 1.0
    initializations.append(h_alt)
    
    h_sp3 = np.zeros(N)
    h_sp3[::3] = 1.0
    initializations.append(h_sp3)
    
    initializations.append(np.exp(-t**2 / 0.3))
    initializations.append(np.maximum(1.0 + 2.0 * np.exp(-((t - 0.6)**2) / 0.1) + 2.0 * np.exp(-((t + 0.6)**2) / 0.1), 0.05))
    
    # 5. Single spike
    h_spike = np.zeros(N)
    h_spike[N // 2] = 1.0
    initializations.append(h_spike)
    
    # 6. Two spikes
    for offset in [5, 10, 15]:
        h_two = np.zeros(N)
        h_two[N // 2 - offset] = 1.0
        h_two[N // 2 + offset] = 1.0
        initializations.append(h_two)
    
    # Optimize each initialization
    for h0 in initializations:
        if time.time() - start_time > DEADLINE - 10:
            break
        
        h0 = np.maximum(h0, 0)
        s0 = np.sum(h0)
        if s0 < 1e-15:
            continue
        h0 = h0 / s0
        
        # Step 1: Multiplicative updates (Sinkhorn-Knopp style)
        h_mu = _multiplicative_update(h0.copy(), max_iter=300, tol=1e-9, temperature=0.05)
        sc = _ratio(h_mu)
        if sc > best_score:
            best_score = sc
            best_h = h_mu.copy()
        
        if time.time() - start_time > DEADLINE - 10:
            continue
        
        # Step 2: Iterative reweighting
        h_ir = _iterative_reweighting(h_mu.copy(), max_iter=200)
        sc = _ratio(h_ir)
        if sc > best_score:
            best_score = sc
            best_h = h_ir.copy()
        
        if time.time() - start_time > DEADLINE - 10:
            continue
        
        # Step 3: Smooth epigraph optimization
        h_ep = _epigraph_optimize(h_ir.copy(), max_iter=200, temperature=0.02)
        sc = _ratio(h_ep)
        if sc > best_score:
            best_score = sc
            best_h = h_ep.copy()
        
        if time.time() - start_time > DEADLINE - 10:
            continue
        
        # Step 4: Coordinate descent polish
        h_cd = _coordinate_descent_polish(h_ep.copy(), max_iter=500, tol=1e-10)
        sc = _ratio(h_cd)
        if sc > best_score:
            best_score = sc
            best_h = h_cd.copy()
    
    # Final high-precision polish
    if time.time() - start_time < DEADLINE - 15:
        try:
            h_final = _coordinate_descent_polish(best_h.copy(), max_iter=2000, tol=1e-12)
            sc = _ratio(h_final)
            if sc > best_score:
                best_score = sc
                best_h = h_final.copy()
        except Exception:
            pass
    
    # One more multiplicative update pass on the best
    if time.time() - start_time < DEADLINE - 10:
        try:
            h_mu2 = _multiplicative_update(best_h.copy(), max_iter=500, tol=1e-10, temperature=0.03)
            sc = _ratio(h_mu2)
            if sc > best_score:
                best_score = sc
                best_h = h_mu2.copy()
            
            h_cd2 = _coordinate_descent_polish(best_h.copy(), max_iter=1000, tol=1e-12)
            sc = _ratio(h_cd2)
            if sc > best_score:
                best_score = sc
                best_h = h_cd2.copy()
        except Exception:
            pass
    
    return best_h.tolist()
