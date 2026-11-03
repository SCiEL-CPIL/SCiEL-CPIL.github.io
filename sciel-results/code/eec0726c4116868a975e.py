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


def _spectral_decomposition(h):
    """
    Perform spectral decomposition of h in the cosine basis.
    Returns coefficients c where h = sum_k c_k * cos(2*pi*k*i/N) for i=0..N-1.
    The DC component c_0 is fixed to enforce sum(h) = N/2.
    """
    n = len(h)
    # Compute DCT-II coefficients
    c = np.fft.rfft(h)
    # Scale to get proper cosine coefficients
    c[0] /= n  # DC component
    for k in range(1, n):
        c[k] /= (n/2.0) if k < n//2 else n
    return c


def _spectral_to_time(c):
    """
    Convert spectral coefficients back to time domain.
    """
    n = len(c)
    h = np.zeros(n)
    # Reconstruct from cosine series
    H = np.zeros(n, dtype=np.complex128)
    H[:len(c)] = c
    # Handle conjugate symmetry for real signal
    if n % 2 == 0:
        H[n - len(c)+1:n] = np.conj(c[1:len(c)][::-1])
    else:
        H[n - len(c)+1:n] = np.conj(c[1:len(c)][::-1])
    h = np.fft.irfft(H, n=n)
    return h


def _spectral_nesterov(h, beta, lr, momentum, deadline, max_iter, tol=1e-10):
    """
    Nesterov accelerated gradient descent in spectral space.
    Optimizes in the cosine basis, fixing DC component to enforce sum constraint.
    """
    t0 = time.time()
    n = len(h)
    
    # Convert to spectral domain
    c_curr = _spectral_decomposition(h)
    c_prev = c_curr.copy()
    v = np.zeros(n)
    
    # Fix DC component: c[0] should give sum = n/2
    # sum(h) = n * c[0], so c[0] = (n/2)/n = 1/2
    c_curr[0] = 0.5
    c_prev[0] = 0.5
    
    best_c = c_curr.copy()
    best_h = _fix_sum(_spectral_to_time(best_c))
    best_score = _score(best_h)
    
    for i in range(max_iter):
        if time.time() > deadline:
            break
        
        # Nesterov lookahead in spectral space
        c_look = c_curr + momentum * (c_curr - c_prev)
        c_look[0] = 0.5  # Keep DC fixed
        
        # Convert to time domain
        h_look = _spectral_to_time(c_look)
        h_look = _fix_sum(h_look)
        
        # Compute gradient in time domain
        _, grad_time = _smooth_max_grad(h_look, beta)
        
        # Project gradient to spectral space
        grad_spec = _spectral_decomposition(grad_time)
        grad_spec[0] = 0.0  # No gradient for DC component
        
        # Update velocity
        v = momentum * v + grad_spec
        v[0] = 0.0  # Keep DC velocity at 0
        
        # Update spectral coefficients
        c_new = c_curr - lr * v
        c_new[0] = 0.5  # Keep DC fixed
        
        # Convert to time domain
        h_new = _spectral_to_time(c_new)
        h_new = _fix_sum(h_new)
        
        # Evaluate
        s_new = _score(h_new)
        if s_new < best_score:
            best_score = s_new
            best_h = h_new.copy()
            best_c = c_new.copy()
        
        # Check convergence
        if np.max(np.abs(c_new - c_curr)) < tol:
            break
        
        c_prev = c_curr
        c_curr = c_new
    
    return best_h, best_score


def _epigraph_objective(h, t, beta):
    """
    Epigraph formulation: minimize t subject to C_k <= t for all k.
    Uses smooth constraints via soft-max.
    """
    cs = _linear_overlap_values(h)
    # Smooth constraint violation: sum of exp(beta*(C_k - t))
    violation = np.sum(np.exp(beta * (cs - t)))
    return t + 0.1 * violation


def _epigraph_gradient(h, t, beta):
    """
    Gradient of epigraph objective.
    """
    cs = _linear_overlap_values(h)
    exp_terms = np.exp(beta * (cs - t))
    
    # d/dt = 1 - beta * sum(exp_terms)
    dt = 1.0 - beta * np.sum(exp_terms)
    
    # d/dh_k = 0.1 * beta * sum_j exp_terms[j] * dC_j/dh_k
    _, grad_h = _smooth_max_grad(h, beta)
    dh = 0.1 * beta * np.dot(exp_terms, np.eye(len(cs)) @ grad_h)
    
    return dt, dh


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
        _run_optimization(h_cos, beta=20.0, maxiter=100, ftol=1e-6)
        
    # Phase 2: Spectral-space Nesterov Accelerated Gradient Descent
    # This is the key improvement: optimize in cosine basis, fixing DC component
    if best_h is not None and time.time() - t0 < _DEADLINE - 50:
        # Adaptive learning rate and momentum
        lr = 0.005
        momentum = 0.9
        max_iter_nesterov = 800
        deadline_nesterov = _DEADLINE - 40
        
        h_nest, s_nest = _spectral_nesterov(best_h, beta=50.0, lr=lr, momentum=momentum, 
                                            deadline=deadline_nesterov, max_iter=max_iter_nesterov)
        if s_nest < best_score:
            best_h = h_nest.copy()
            best_score = s_nest
        
        # Try with different beta
        if time.time() - t0 < _DEADLINE - 25:
            h_nest2, s_nest2 = _spectral_nesterov(best_h, beta=100.0, lr=lr*0.5, momentum=momentum,
                                                  deadline=_DEADLINE - 20, max_iter=500)
            if s_nest2 < best_score:
                best_h = h_nest2.copy()
                best_score = s_nest2
                
        # Try with even higher beta for fine tuning
        if time.time() - t0 < _DEADLINE - 15:
            h_nest3, s_nest3 = _spectral_nesterov(best_h, beta=200.0, lr=lr*0.3, momentum=momentum,
                                                  deadline=_DEADLINE - 10, max_iter=300)
            if s_nest3 < best_score:
                best_h = h_nest3.copy()
                best_score = s_nest3
        
    # Phase 3: Medium optimization (SLSQP)
    if best_h is not None and time.time() - t0 < _DEADLINE - 20:
        _run_optimization(best_h, beta=50.0, maxiter=150, ftol=1e-8)
        
    # Phase 4: Fine optimization with high beta
    if best_h is not None and time.time() - t0 < _DEADLINE - 12:
        _run_optimization(best_h, beta=100.0, maxiter=200, ftol=1e-10)
        
    # Phase 5: Final polish with very high beta
    if best_h is not None and time.time() - t0 < _DEADLINE - 5:
        _run_optimization(best_h, beta=200.0, maxiter=300, ftol=1e-12)
        
    # Phase 6: Try a few random restarts if time permits
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
                
    # Phase 7: Final spectral polish
    if best_h is not None and time.time() - t0 < _DEADLINE - 3:
        h_final, s_final = _spectral_nesterov(best_h, beta=300.0, lr=0.001, momentum=0.9,
                                              deadline=_DEADLINE - 1, max_iter=200)
        if s_final < best_score:
            best_h = h_final.copy()
            best_score = s_final
        
    if best_h is None:
        best_h = np.full(N, 0.5)
    
    best_h = _fix_sum(best_h)
    return best_h

