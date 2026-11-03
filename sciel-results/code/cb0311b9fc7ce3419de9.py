import numpy as np
import time
from scipy.optimize import minimize

N = 100
DEADLINE = 145.0  # Safety margin before 150s kill

def _linear_overlap_values(h):
    """
    Compute the linear overlap values C_k for k=0..N-1.
    The task defines overlap for shift s as:
    C_s = sum_{i=0}^{N-s-1} h_i * (1 - h_{i+s})
    Using FFT:
    r_s = sum_{i=s}^{N-1} h_i h_{i-s} (linear autocorrelation)
    P_s = sum_{i=0}^{N-s-1} h_i
    C_s = P_s - r_s
    """
    n = len(h)
    if n == 0:
        return np.array([])
    
    n_fft = 2 * n
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad, n=n_fft)
    # Power spectrum
    power = H * np.conj(H)
    # Inverse FFT gives circular autocorrelation of the padded signal
    corr = np.fft.irfft(power, n=n_fft)
    # corr[k] for k=0..n-1 is the linear autocorrelation r_k = sum_{i=k}^{n-1} h[i]h[i-k]
    r = corr[:n]
    
    # Compute prefix sums P_s = sum_{i=0}^{N-s-1} h_i
    # Let cumsum = [0, h0, h0+h1, ..., sum(h)]
    # P_s = sum(h[0:N-s])
    # For s=0, P_0 = sum(h)
    # For s=1, P_1 = sum(h) - h[N-1]
    # ...
    # P_s = total_sum - sum(h[N-s:])
    
    total_sum = np.sum(h)
    # suffix_sums[s] = sum(h[N-s:])
    # We can compute this via reverse cumsum
    rev_h = h[::-1]
    cumsum_rev = np.cumsum(rev_h)
    # cumsum_rev[j] = sum(h[N-1-j : N-1] + h[N-1]) = sum of last j+1 elements
    # We need sum of last s elements for s=1..N
    # suffix_sums[s] = cumsum_rev[s-1]
    # P_s = total_sum - suffix_sums[s]
    
    # We want C_s for s=1..N-1
    # C_s = P_s - r_s
    # P_s = total_sum - cumsum_rev[s-1]
    
    cs = np.zeros(n)
    if n > 1:
        # s ranges from 1 to n-1
        s_indices = np.arange(1, n)
        # P_s for these s
        # cumsum_rev indices: s-1
        suffix_sums = cumsum_rev[s_indices - 1]
        P_s = total_sum - suffix_sums
        r_s = r[s_indices]
        cs[s_indices] = P_s - r_s
        
    return cs

def _score(h):
    """Compute the maximum linear overlap for s=1..N-1."""
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    if len(cs) == 0:
        return 0.0
    return float(np.max(cs[1:])) # Exclude s=0 as per task definition (shifted)

def _smooth_max_grad(h, beta):
    """
    Gradient of the smooth maximum objective (LogSumExp) for overlaps.
    Returns (value, gradient)
    """
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    cs = _linear_overlap_values(h)
    
    # We care about max of cs[1:]
    if n <= 1:
        return 0.0, np.zeros(n)
        
    active_cs = cs[1:]
    m = np.max(active_cs)
    exp_terms = np.exp(beta * (active_cs - m))
    sum_exp = np.sum(exp_terms)
    w = exp_terms / sum_exp  # Weights for the soft max
    
    val = m + np.log(sum_exp) / beta
    
    grad = np.zeros(n)
    
    # C_s = P_s - r_s
    # P_s = sum_{i=0}^{N-s-1} h_i
    # r_s = sum_{i=s}^{N-1} h_i h_{i-s}
    
    # dC_s/dh_j:
    # 1. From P_s: 1 if j < N-s, else 0
    # 2. From -r_s: - (h_{j-s} if j>=s else 0) - (h_{j+s} if j+s<N else 0)
    #    Note: r_s = sum_{i=s}^{N-1} h_i h_{i-s}
    #    d(r_s)/d(h_j):
    #    If j >= s: term h_j h_{j-s} -> h_{j-s}
    #    If j+s < N: term h_{j+s} h_j -> h_{j+s}
    #    (If both apply, i.e., s <= j < N-s, then both terms contribute)
    
    for idx, w_s in enumerate(w):
        if w_s < 1e-10:
            continue
        s = idx + 1
        
        # Contribution from P_s
        # grad[0:N-s] += w_s * 1
        
        # Contribution from -r_s
        # grad[s:] += -w_s * h[:N-s]
        # grad[:N-s] += -w_s * h[s:]
        
        # Combine
        # grad[0:N-s] gets +w_s (from P) -w_s*h[s:] (from r)
        # grad[s:] gets -w_s*h[:N-s] (from r)
        
        # Note: indices [s : N-s] receive both P and r contributions if s < N-s
        # Let's just add them up.
        
        # P part
        grad[:N-s] += w_s
        
        # r part
        if s > 0:
            grad[s:] += -w_s * h[:N-s]
        if s < N:
            grad[:N-s] += -w_s * h[s:]
            
    return val, grad

def _project_sum_box(y, target_sum):
    """
    Project y onto {x : 0 <= x_i <= 1, sum(x) = target_sum}
    Using binary search on the shift parameter tau.
    """
    lo = np.min(y) - 1.0
    hi = np.max(y) + 1.0
    
    def g(tau):
        x = np.clip(y + tau, 0.0, 1.0)
        return np.sum(x) - target_sum
        
    if g(lo) > 0:
        return np.clip(y + lo, 0.0, 1.0)
    if g(hi) < 0:
        return np.clip(y + hi, 0.0, 1.0)
        
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if g(mid) > 0:
            hi = mid
        else:
            lo = mid
            
    tau = (lo + hi) / 2.0
    return np.clip(y + tau, 0.0, 1.0)

def _nesterov_pgd(h0, beta, steps, lr, deadline):
    """
    Nesterov-accelerated Projected Gradient Descent.
    """
    h = h0.copy()
    y = h0.copy()
    target_sum = N / 2.0
    prev_val, _ = _smooth_max_grad(h, beta)
    t = 1.0
    
    for i in range(steps):
        if time.time() > deadline:
            break
            
        # Nesterov step
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        y = h + (t / t_new) * (h - y)
        
        val, grad = _smooth_max_grad(y, beta)
        
        # Gradient step from y
        new_h = y - lr * grad
        new_h_proj = _project_sum_box(new_h, target_sum)
        new_val, _ = _smooth_max_grad(new_h_proj, beta)
        
        # Backtracking if needed
        step_lr = lr
        while new_val > prev_val + 1e-12 and step_lr > 1e-10:
            step_lr /= 2.0
            new_h = y - step_lr * grad
            new_h_proj = _project_sum_box(new_h, target_sum)
            new_val, _ = _smooth_max_grad(new_h_proj, beta)
            
        if new_val < prev_val:
            h = new_h_proj
            prev_val = new_val
            t = t_new
        else:
            # Reset momentum if no progress
            y = h.copy()
            t = 1.0
            if step_lr < 1e-10:
                break
                
    return h

def _coordinate_descent_polish(h0, deadline, max_iter=200):
    """
    Coordinate descent polish on the active set of constraints.
    For each variable, try small perturbations that reduce the max overlap.
    """
    h = h0.copy()
    target_sum = N / 2.0
    current_score = _score(h)
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
            
        cs = _linear_overlap_values(h)
        max_val = np.max(cs[1:])
        # active = np.where(cs[1:] > max_val - 0.02)[0]
        
        improved = False
        for j in range(N):
            # Try small perturbations in each direction
            for delta in [0.001, -0.001, 0.0005, -0.0005]:
                h_trial = h.copy()
                h_trial[j] += delta
                h_trial = _project_sum_box(h_trial, target_sum)
                s = _score(h_trial)
                if s < current_score - 1e-12:
                    h = h_trial
                    current_score = s
                    improved = True
                    break
            if time.time() > deadline:
                break
        if not improved:
            break
            
    return h

def _refine_with_lbfgs(h0, deadline):
    """
    Refine a solution using L-BFGS-B with the smooth objective.
    """
    h = h0.copy()
    target_sum = N / 2.0
    
    def obj_and_grad(x):
        val, grad = _smooth_max_grad(x, beta=2000.0)
        return val, grad
        
    try:
        res = minimize(
            obj_and_grad,
            h,
            jac=True,
            method='L-BFGS-B',
            bounds=[(0, 1)] * N,
            options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        if res.fun < _score(h):
            h_new = _project_sum_box(res.x, target_sum)
            return h_new
    except Exception:
        pass
        
    return h

def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    t0 = time.time()
    np.random.seed(42)
    
    best_h = None
    best_score = float('inf')
    
    def _update_best(h):
        nonlocal best_h, best_score
        h = _project_sum_box(h, N / 2.0)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
            
    # 1. Initial Candidates
    i = np.arange(N)
    
    # Chirp signal (quadratic phase) - often has low autocorrelation
    h_chirp = 0.5 + 0.5 * np.sin(2.0 * np.pi * (i**2) / N)
    h_chirp = _project_sum_box(h_chirp, N / 2.0)
    _update_best(h_chirp)
    
    # Cosine wave
    h_cos = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
    h_cos = _project_sum_box(h_cos, N / 2.0)
    _update_best(h_cos)
    
    # Sine wave
    h_sin = 0.5 + 0.5 * np.sin(2.0 * np.pi * i / N)
    h_sin = _project_sum_box(h_sin, N / 2.0)
    _update_best(h_sin)
    
    # Thue-Morse
    h_thue = np.zeros(N)
    for k in range(N):
        h_thue[k] = 1.0 if bin(k).count('1') % 2 == 1 else 0.0
    h_thue = _project_sum_box(h_thue, N / 2.0)
    _update_best(h_thue)
    
    # Random starts
    for _ in range(3):
        h_rand = np.random.uniform(0, 1, N)
        h_rand = _project_sum_box(h_rand, N / 2.0)
        _update_best(h_rand)
        
    # 2. Optimization Phases
    # Phase 1: Coarse Nesterov PGD with lower beta
    if best_h is not None and time.time() - t0 < DEADLINE - 50:
        h_opt = _nesterov_pgd(best_h, beta=10.0, steps=500, lr=0.05, deadline=DEADLINE - 50)
        _update_best(h_opt)
        
    # Phase 2: Medium beta Nesterov
    if best_h is not None and time.time() - t0 < DEADLINE - 25:
        h_opt = _nesterov_pgd(best_h, beta=50.0, steps=800, lr=0.01, deadline=DEADLINE - 25)
        _update_best(h_opt)
        
    # Phase 3: High beta Nesterov
    if best_h is not None and time.time() - t0 < DEADLINE - 10:
        h_opt = _nesterov_pgd(best_h, beta=200.0, steps=1000, lr=0.005, deadline=DEADLINE - 10)
        _update_best(h_opt)
        
    # Phase 4: L-BFGS-B Polish
    if best_h is not None and time.time() - t0 < DEADLINE - 5:
        h_opt = _refine_with_lbfgs(best_h, DEADLINE - 5)
        _update_best(h_opt)
        
    # Phase 5: Coordinate descent polish for final precision
    if best_h is not None and time.time() - t0 < DEADLINE - 1:
        h_opt = _coordinate_descent_polish(best_h, DEADLINE - 1, max_iter=500)
        _update_best(h_opt)

    if best_h is None:
        return np.full(N, 0.5)
        
    return _project_sum_box(best_h, N / 2.0)
