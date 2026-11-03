import numpy as np
import time
from scipy.optimize import minimize

N = 100
_DEADLINE = time.time() + 130  # Safety margin before 150s kill


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
    Vectorized gradient computation using FFT.
    """
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    cs = _linear_overlap_values(h)
    m = np.max(cs)
    exp_terms = np.exp(beta * (cs - m))
    sum_exp = np.sum(exp_terms)
    w = exp_terms / sum_exp  # Weights for the soft max
    
    val = m + np.log(sum_exp) / beta
    
    # Compute gradient using FFT-based convolution
    n_fft = 2 * n
    W = np.fft.rfft(w, n=n_fft)
    H = np.fft.rfft(h, n=n_fft)
    
    # conv1[j] = sum_k w[k] * h[j-k] (linear convolution)
    conv1 = np.fft.irfft(W * H, n=n_fft)[:n]
    
    # conv2[j] = sum_k w[k] * h[j+k]
    v = h[::-1]
    V = np.fft.rfft(v, n=n_fft)
    conv2_rev = np.fft.irfft(W * V, n=n_fft)[:n]
    conv2 = conv2_rev[n-1::-1]
    
    # Combine
    grad = -(2.0 / n) * (conv1 + conv2)
    
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


def _nesterov_step(h, beta, lr, momentum, t, deadline, max_iter, tol=1e-10):
    """
    Nesterov accelerated gradient descent with projection.
    Uses the smooth max surrogate.
    """
    t0 = time.time()
    h_curr = np.array(h, dtype=np.float64, copy=True)
    h_prev = np.array(h, dtype=np.float64, copy=True)
    v = np.zeros(N)
    best_h = h_curr.copy()
    best_score = _score(h_curr)
    
    for i in range(max_iter):
        if time.time() > deadline:
            break
        
        # Nesterov lookahead
        h_look = h_curr + momentum * (h_curr - h_prev)
        
        # Project lookahead to feasible set
        h_look = _fix_sum(h_look)
        
        # Compute gradient at lookahead
        _, grad = _smooth_max_grad(h_look, beta)
        
        # Update velocity
        v = momentum * v + grad
        h_new = h_curr - lr * v
        
        # Project to feasible set
        h_new = _fix_sum(h_new)
        
        # Evaluate
        s_new = _score(h_new)
        if s_new < best_score:
            best_score = s_new
            best_h = h_new.copy()
        
        # Check convergence
        if np.max(np.abs(h_new - h_curr)) < tol:
            break
        
        h_prev = h_curr
        h_curr = h_new
        
    return best_h, best_score


def _project_onto_simplex(v, t):
    """
    Project v onto {x: sum(x) = t, 0 <= x_i <= 1}.
    """
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - t))[0][-1]
    theta = (cssv[rho] - t) / (rho + 1.0)
    w = np.maximum(v - theta, 0.0)
    # Clip to [0, 1] and fix sum again if needed
    w = np.clip(w, 0.0, 1.0)
    return _fix_sum(w)


def _coordinate_descent_polish(h, beta, deadline, max_iter=500):
    """
    Coordinate descent polish: optimize one coordinate at a time.
    For fixed other coordinates, find optimal h_j in [0,1] satisfying sum constraint.
    """
    h = np.array(h, dtype=np.float64, copy=True)
    best_score = _score(h)
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        improved = False
        for j in range(N):
            if time.time() > deadline:
                break
            
            # Try small perturbations around current value
            h_j = h[j]
            best_hj = h_j
            best_s = best_score
            
            # Grid search around current value
            for delta in [-0.02, -0.01, -0.005, 0.005, 0.01, 0.02, -0.001, 0.001]:
                new_hj = h_j + delta
                if new_hj < 0 or new_hj > 1:
                    continue
                
                h[j] = new_hj
                h = _fix_sum(h)
                s = _score(h)
                
                if s < best_s:
                    best_s = s
                    best_hj = new_hj
            
            h[j] = best_hj
            h = _fix_sum(h)
            
            if best_s < best_score - 1e-10:
                best_score = best_s
                improved = True
        
        if not improved:
            break
    
    return h, best_score


def _structured_initial_guess(freq, amp, phase):
    """
    Generate a structured initial guess: sinusoidal wave with harmonics.
    h_i = 0.5 + amp * cos(2*pi*freq*i/N + phase) + 0.1*amp*cos(4*pi*freq*i/N + 2*phase)
    """
    i = np.arange(N)
    h = 0.5 + amp * np.cos(2.0 * np.pi * freq * i / N + phase)
    h += 0.1 * amp * np.cos(4.0 * np.pi * freq * i / N + 2.0 * phase)
    h = np.clip(h, 0.0, 1.0)
    h = _fix_sum(h)
    return h


def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
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
    
    # 1. Analytical Start: Cosine Wave
    i = np.arange(N)
    h_cos = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
    h_cos = _fix_sum(h_cos)
    _update_best(h_cos)
    
    # 1b. Try structured low-dimensional parameterizations
    # Search over frequency, amplitude, phase
    for freq in [1, 2, 3, 5, 7, 10, 13]:
        for amp in [0.3, 0.4, 0.5, 0.6, 0.7]:
            for phase in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
                if time.time() - t0 > _DEADLINE - 80:
                    break
                h_struct = _structured_initial_guess(freq, amp, phase)
                _update_best(h_struct)
    
    # 2. Multi-stage optimization with adaptive beta
    def _run_optimization(h0, beta, maxiter, ftol, method='SLSQP'):
        nonlocal best_h, best_score
        
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
            res = minimize(obj_func, h0.copy(), jac=jac_func, method=method,
                           bounds=bounds, constraints=cons,
                           options={'maxiter': maxiter, 'ftol': ftol})
            if res.success or res.fun < best_score:
                _update_best(res.x)
        except Exception:
            pass

    # Phase 1: Coarse optimization with low beta (SLSQP)
    if time.time() - t0 < _DEADLINE - 60:
        _run_optimization(best_h, beta=20.0, maxiter=100, ftol=1e-6)
        
    # Phase 2: Nesterov Accelerated Gradient Descent
    # This is the key improvement: NAG with projection escapes local minima better
    if best_h is not None and time.time() - t0 < _DEADLINE - 40:
        # Adaptive learning rate and momentum
        lr = 0.01
        momentum = 0.9
        max_iter_nesterov = 500
        deadline_nesterov = _DEADLINE - 30
        
        h_nest, s_nest = _nesterov_step(best_h, beta=50.0, lr=lr, momentum=momentum, 
                                        t=0, deadline=deadline_nesterov, max_iter=max_iter_nesterov)
        if s_nest < best_score:
            best_h = h_nest.copy()
            best_score = s_nest
        
        # Try with different beta
        if time.time() - t0 < _DEADLINE - 20:
            h_nest2, s_nest2 = _nesterov_step(best_h, beta=100.0, lr=lr*0.5, momentum=momentum,
                                               t=0, deadline=_DEADLINE - 15, max_iter=300)
            if s_nest2 < best_score:
                best_h = h_nest2.copy()
                best_score = s_nest2
        
    # Phase 3: Medium optimization (SLSQP)
    if best_h is not None and time.time() - t0 < _DEADLINE - 25:
        _run_optimization(best_h, beta=50.0, maxiter=150, ftol=1e-8)
        
    # Phase 4: Fine optimization with high beta
    if best_h is not None and time.time() - t0 < _DEADLINE - 15:
        _run_optimization(best_h, beta=100.0, maxiter=200, ftol=1e-10)
        
    # Phase 5: Coordinate descent polish
    if best_h is not None and time.time() - t0 < _DEADLINE - 10:
        h_cd, s_cd = _coordinate_descent_polish(best_h, beta=100.0, deadline=_DEADLINE - 5, max_iter=300)
        if s_cd < best_score:
            best_h = h_cd.copy()
            best_score = s_cd
    
    # Phase 6: Final polish with very high beta
    if best_h is not None and time.time() - t0 < _DEADLINE - 5:
        _run_optimization(best_h, beta=200.0, maxiter=300, ftol=1e-12)
        
    # Phase 7: Try a few random restarts if time permits
    if time.time() - t0 < _DEADLINE - 20:
        for _ in range(2):
            if time.time() - t0 > _DEADLINE - 10:
                break
            h_rand = np.random.uniform(0, 1, N)
            h_rand = _fix_sum(h_rand)
            _update_best(h_rand)
            _run_optimization(h_rand, beta=50.0, maxiter=80, ftol=1e-8)
            if best_h is not None and time.time() - t0 < _DEADLINE - 10:
                _run_optimization(best_h, beta=100.0, maxiter=150, ftol=1e-10)
                # Add coordinate descent polish for random restarts too
                if time.time() - t0 < _DEADLINE - 8:
                    h_cd, s_cd = _coordinate_descent_polish(best_h, beta=100.0, deadline=_DEADLINE - 5, max_iter=200)
                    if s_cd < best_score:
                        best_h = h_cd.copy()
                        best_score = s_cd
        
    if best_h is None:
        best_h = np.full(N, 0.5)
    
    best_h = _fix_sum(best_h)
    return best_h
