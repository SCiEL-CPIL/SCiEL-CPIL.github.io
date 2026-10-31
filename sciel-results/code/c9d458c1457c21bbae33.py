import numpy as np
import time
from scipy.optimize import minimize, differential_evolution, NelderMead

N = 100
_DEADLINE = time.time() + 110  # Safety margin before 120s kill

def _linear_overlap_values(h):
    """
    Compute the linear overlap values C_k for k=0..N-1.
    C_k = (2/N) * (N/2 - sum_{i=k}^{N-1} h_i h_{i-k})
    Vectorized using FFT for speed.
    """
    n = len(h)
    n_fft = 2 * n
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad, n=n_fft)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    r = corr[:n]
    cs = (2.0 / n) * (n / 2.0 - r)
    return cs

def _score(h):
    """
    Compute the maximum linear overlap.
    """
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))

def _smooth_max_grad(h, beta):
    """
    Gradient of the smooth maximum objective.
    Returns (value, gradient)
    """
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    cs = _linear_overlap_values(h)
    m = np.max(cs)
    exp_terms = np.exp(beta * (cs - m))
    sum_exp = np.sum(exp_terms)
    w = exp_terms / sum_exp
    
    val = m + np.log(sum_exp) / beta
    
    grad = np.zeros(n)
    
    for k in range(n):
        if w[k] < 1e-15:
            continue
        factor = -(2.0 / n) * w[k]
        
        if n - k > 0:
            grad[k:] += factor * h[:n-k]
        if n - k > 0:
            grad[:n-k] += factor * h[k:]
            
    return val, grad

def _fix_sum(h):
    """
    Adjust h to have sum exactly N/2, keeping values in [0, 1].
    """
    h = np.array(h, dtype=np.float64, copy=True)
    target = N / 2.0
    diff = target - np.sum(h)
    h += diff / N
    h = np.clip(h, 0.0, 1.0)
    
    diff = target - np.sum(h)
    if abs(diff) > 1e-12:
        mask = (h > 0) & (h < 1)
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
            
    diff = target - np.sum(h)
    if abs(diff) > 1e-15:
        if diff > 0:
            idx = np.argmin(h)
        else:
            idx = np.argmax(h)
        h[idx] += diff
        h = np.clip(h, 0.0, 1.0)
        
    return h

def _binomial_cdf_profile(i, center, width, order):
    """
    Generate a binomial CDF transition profile.
    i: array of indices
    center: center of transition
    width: width of transition window
    order: binomial order (higher = smoother)
    Returns values in [0, 1]
    """
    if width <= 0:
        # Hard step
        return np.where(i < center, 0.0, 1.0)
    
    # Normalize i to be in terms of width
    t = (i - center) / width
    
    # Use a logistic-like function as a smooth approximation to binomial CDF
    # For large order, binomial CDF approaches a smooth step
    # We use: 0.5 * (1 + tanh(t * order / 4))
    # This gives a smooth transition centered at t=0 with width ~ 4/order
    # Scale by order to control sharpness
    sharpness = order / 4.0
    if sharpness < 0.1:
        return np.full_like(i, 0.5)
    
    # Clip to avoid overflow
    t_clipped = np.clip(t, -10, 10)
    profile = 0.5 * (1.0 + np.tanh(sharpness * t_clipped))
    
    return profile

def _parametric_h(params):
    """
    Generate h from parameters: [center, width, order, amplitude_scale]
    """
    center, width, order, amp_scale = params
    i = np.arange(N, dtype=np.float64)
    
    # Base profile: transition from 0 to 1
    profile = _binomial_cdf_profile(i, center, width, order)
    
    # Scale to have correct mean
    # We want sum(h) = N/2, so mean = 0.5
    # If profile has mean m, we need to scale: h = amp_scale * profile + offset
    # Let's compute the mean of profile
    m = np.mean(profile)
    
    # We want: amp_scale * m + offset = 0.5
    # And we want h in [0,1]
    # Simple approach: h = profile, then fix sum
    # But let's try to center it properly
    # h = profile - m + 0.5 ensures mean is 0.5 before clipping
    h = profile - m + 0.5
    
    # Apply amplitude scale (should be 1.0 for mean preservation)
    h = amp_scale * h
    
    # Fix sum and bounds
    h = _fix_sum(h)
    
    return h

def _parametric_score(params):
    """
    Score for parametric representation.
    """
    h = _parametric_h(params)
    return _score(h)

def _run_slsqp_from(h0, beta, maxiter, ftol, deadline):
    """
    Run SLSQP optimization from h0.
    Returns the best h found.
    """
    cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N/2.0,
             'jac': lambda x: np.ones(N)}]
    bounds = [(0.0, 1.0)] * N
    
    def obj_func(x):
        val, _ = _smooth_max_grad(x, beta)
        return val
        
    def jac_func(x):
        _, grad = _smooth_max_grad(x, beta)
        return grad
        
    try:
        res = minimize(obj_func, h0.copy(), jac=jac_func, method='SLSQP',
                       bounds=bounds, constraints=cons,
                       options={'maxiter': maxiter, 'ftol': ftol})
        if res.success or res.fun < _score(h0):
            return _fix_sum(res.x)
    except Exception:
        pass
    
    return h0

def construct():
    t0 = time.time()
    np.random.seed(42)
    
    best_h = None
    best_score = float('inf')
    
    def _update_best(h):
        nonlocal best_h, best_score
        h = _fix_sum(h)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
    
    # Phase 1: Parametric search using differential evolution
    # Parameters: [center, width, order, amp_scale]
    # center in [0, N], width in [1, N/2], order in [1, 20], amp_scale in [0.5, 1.5]
    
    if time.time() - t0 < _DEADLINE - 40:
        # Use differential evolution for global search over parametric space
        bounds = [(0, N), (1, N/2), (1, 20), (0.5, 1.5)]
        
        def _de_objective(params):
            return _parametric_score(params)
        
        try:
            de_result = differential_evolution(
                _de_objective, bounds, 
                maxiter=50, popsize=15, 
                seed=42, tol=1e-6,
                polish=False,  # We'll do our own polish
                update_interval=1,
                workers=1
            )
            
            if de_result.success:
                best_params = de_result.x
                best_h_param = _parametric_h(best_params)
                _update_best(best_h_param)
                
                # Also try the negated version (by symmetry)
                h_neg = 1.0 - best_h_param
                h_neg = _fix_sum(h_neg)
                _update_best(h_neg)
                
        except Exception:
            pass
    
    # Phase 2: Local refinement with SLSQP from best parametric solution
    if best_h is not None and time.time() - t0 < _DEADLINE - 20:
        # Refine with moderate beta
        h_refined = _run_slsqp_from(best_h, beta=50.0, maxiter=100, ftol=1e-8, deadline=_DEADLINE)
        _update_best(h_refined)
        
        # Also try from the negated version
        h_neg_refined = _run_slsqp_from(1.0 - best_h, beta=50.0, maxiter=100, ftol=1e-8, deadline=_DEADLINE)
        _update_best(h_neg_refined)
    
    # Phase 3: Try cosine start as a fallback
    if time.time() - t0 < _DEADLINE - 15:
        i = np.arange(N)
        h_cos = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
        h_cos = _fix_sum(h_cos)
        _update_best(h_cos)
        
        # Refine cosine
        if time.time() - t0 < _DEADLINE - 10:
            h_cos_refined = _run_slsqp_from(h_cos, beta=50.0, maxiter=80, ftol=1e-8, deadline=_DEADLINE)
            _update_best(h_cos_refined)
    
    # Phase 4: Final polish with higher beta if time permits
    if best_h is not None and time.time() - t0 < _DEADLINE - 5:
        h_polished = _run_slsqp_from(best_h, beta=100.0, maxiter=150, ftol=1e-10, deadline=_DEADLINE)
        _update_best(h_polished)
    
    if best_h is None:
        best_h = np.full(N, 0.5)
    
    best_h = _fix_sum(best_h)
    return best_h
