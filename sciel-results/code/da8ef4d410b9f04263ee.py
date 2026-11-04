import numpy as np
import time
import scipy.optimize

N = 400

def _autoconv_lin(x):
    """Compute linear autoconvolution using FFT."""
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    fx = np.fft.rfft(x, size)
    return np.fft.irfft(fx * fx, size)[:2 * n - 1]

def _metric_offdiag(x):
    """Metric: max |c_k| for k != 0 / S^2"""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return float('inf')
    c = _autoconv_lin(x)
    if len(c) > 1:
        return float(np.max(np.abs(c[1:]))) / (s * s)
    return 0.0

def _soft_max_abs(c, beta):
    """Smooth approximation to max|c| using log-sum-exp."""
    abs_c = np.abs(c)
    m = np.max(abs_c)
    if m < 1e-15:
        return 0.0
    return m + np.log(np.sum(np.exp(beta * (abs_c - m)))) / beta

def _chirp_sequence(n, alpha, phi, dc_offset=0.0):
    """Generate a real-valued chirp sequence with DC offset."""
    t = np.arange(n, dtype=float)
    x = np.cos(2.0 * np.pi * alpha * t * t / n + phi) + dc_offset
    return x

def _two_chirp_sequence(n, alpha1, phi1, a1, alpha2, phi2, a2, dc_offset=0.0):
    """Generate a sum of two chirps with DC offset."""
    t = np.arange(n, dtype=float)
    x = a1 * np.cos(2.0 * np.pi * alpha1 * t * t / n + phi1) + \
        a2 * np.cos(2.0 * np.pi * alpha2 * t * t / n + phi2) + dc_offset
    return x

def _objective_chirp(params, n=N, beta=50.0):
    """Objective for chirp parameter optimization."""
    alpha, phi, dc = params
    x = _chirp_sequence(n, alpha, phi, dc)
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    c = _autoconv_lin(x)
    if len(c) <= 1:
        return 0.0
    abs_c = np.abs(c[1:])
    m = np.max(abs_c)
    if m < 1e-15:
        return 0.0
    soft_max = m + np.log(np.sum(np.exp(beta * (abs_c - m)))) / beta
    return soft_max / (s * s)

def _objective_two_chirp(params, n=N, beta=50.0):
    """Objective for two-chirp parameter optimization."""
    alpha1, phi1, a1, alpha2, phi2, a2, dc = params
    x = _two_chirp_sequence(n, alpha1, phi1, a1, alpha2, phi2, a2, dc)
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    c = _autoconv_lin(x)
    if len(c) <= 1:
        return 0.0
    abs_c = np.abs(c[1:])
    m = np.max(abs_c)
    if m < 1e-15:
        return 0.0
    soft_max = m + np.log(np.sum(np.exp(beta * (abs_c - m)))) / beta
    return soft_max / (s * s)

def _optimize_chirp(n=N, time_budget=20.0):
    """
    Optimize a single chirp representation using L-BFGS-B.
    """
    start = time.time()
    
    # Initial guesses: alpha in (0.1, 2.0), phi in (-pi, pi), dc in (-0.5, 0.5)
    # We can try multiple starting points
    best_x = None
    best_score = float('inf')
    
    rng = np.random.default_rng(42)
    n_trials = 10
    for i in range(n_trials):
        if time.time() - start > time_budget:
            break
        alpha0 = rng.uniform(0.1, 2.0)
        phi0 = rng.uniform(-np.pi, np.pi)
        dc0 = rng.uniform(-0.5, 0.5)
        
        try:
            result = scipy.optimize.minimize(
                _objective_chirp,
                [alpha0, phi0, dc0],
                args=(n,),
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if time.time() - start > time_budget:
                break
            alpha, phi, dc = result.x
            x = _chirp_sequence(n, alpha, phi, dc)
            score = _metric_offdiag(x)
            if score < best_score:
                best_score = score
                best_x = x
        except Exception:
            continue
            
    return best_x, best_score

def _optimize_two_chirp(n=N, time_budget=30.0):
    """
    Optimize a two-chirp representation using L-BFGS-B.
    """
    start = time.time()
    
    best_x = None
    best_score = float('inf')
    
    rng = np.random.default_rng(42)
    n_trials = 5
    for i in range(n_trials):
        if time.time() - start > time_budget:
            break
        alpha1 = rng.uniform(0.1, 1.0)
        phi1 = rng.uniform(-np.pi, np.pi)
        a1 = rng.uniform(0.5, 1.5)
        alpha2 = rng.uniform(1.0, 3.0)
        phi2 = rng.uniform(-np.pi, np.pi)
        a2 = rng.uniform(0.5, 1.5)
        dc = rng.uniform(-0.5, 0.5)
        
        try:
            result = scipy.optimize.minimize(
                _objective_two_chirp,
                [alpha1, phi1, a1, alpha2, phi2, a2, dc],
                args=(n,),
                method='L-BFGS-B',
                options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if time.time() - start > time_budget:
                break
            alpha1, phi1, a1, alpha2, phi2, a2, dc = result.x
            x = _two_chirp_sequence(n, alpha1, phi1, a1, alpha2, phi2, a2, dc)
            score = _metric_offdiag(x)
            if score < best_score:
                best_score = score
                best_x = x
        except Exception:
            continue
            
    return best_x, best_score

def _refine_with_lbfgs(x_init, time_budget=30.0, beta=50.0):
    """Refine a candidate using L-BFGS-B on the soft-max objective."""
    start = time.time()
    
    def obj(x):
        if time.time() - start > time_budget:
            return 1e10
        s = np.sum(x)
        if abs(s) < 1e-10:
            return 1e10
        c = _autoconv_lin(x)
        if len(c) <= 1:
            return 0.0
        abs_c = np.abs(c[1:])
        m = np.max(abs_c)
        if m < 1e-15:
            return 0.0
        soft_max = m + np.log(np.sum(np.exp(beta * (abs_c - m)))) / beta
        return soft_max / (s * s)
    
    try:
        result = scipy.optimize.minimize(
            obj,
            x_init,
            method='L-BFGS-B',
            options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 30},
            tol=1e-12
        )
        if time.time() - start < time_budget:
            x_opt = result.x
            score = _metric_offdiag(x_opt)
            return x_opt, score
    except Exception:
        pass
    
    return None, float('inf')

def _gradient_polish(x, max_iters=100, time_budget=20.0):
    """
    Gradient descent polishing using the subgradient of max|c_k|.
    Uses the active set (all lags near the maximum) for a smoother gradient.
    """
    start = time.time()
    x = x.copy()
    best_x = x.copy()
    best_score = _metric_offdiag(x)
    
    for it in range(max_iters):
        if time.time() - start > time_budget:
            break
        
        s = np.sum(x)
        if abs(s) < 1e-12:
            break
        
        c = _autoconv_lin(x)
        abs_c = np.abs(c[1:])
        n_lags = len(abs_c)
        
        # Use top 5% of lags for a smoother gradient
        threshold = np.percentile(abs_c, 95)
        active = abs_c > threshold
        if not np.any(active):
            active = abs_c == np.max(abs_c)
        
        # Gradient: sum over active lags of sign(c_k) * d(c_k)/dx
        grad = np.zeros_like(x)
        n = len(x)
        
        for k_idx in np.where(active)[0]:
            k = k_idx + 1
            sign_c = np.sign(c[k])
            if sign_c == 0:
                sign_c = 1.0
            
            # d(c_k)/dx_j = x_{j+k} + x_{j-k} (with bounds)
            for j in range(n):
                if j + k < n:
                    grad[j] += sign_c * x[j + k]
                if j - k >= 0:
                    grad[j] += sign_c * x[j - k]
        
        gnorm = np.linalg.norm(grad)
        if gnorm < 1e-15:
            break
        
        # Adaptive step size
        step = 0.05 * (1.0 - it / max_iters) + 0.005
        x_new = x - step * grad / gnorm
        
        s_new = np.sum(x_new)
        if abs(s_new) < 1.0:
            x_new += (1.0 - s_new) / n
        
        sc_new = _metric_offdiag(x_new)
        if sc_new < best_score:
            best_score = sc_new
            best_x = x_new.copy()
            x = x_new
        else:
            step *= 0.5
            x_new = x - step * grad / gnorm
            s_new = np.sum(x_new)
            if abs(s_new) < 1.0:
                x_new += (1.0 - s_new) / n
            sc_new = _metric_offdiag(x_new)
            if sc_new < best_score:
                best_score = sc_new
                best_x = x_new.copy()
                x = x_new
    
    return best_x, best_score

def construct():
    """
    Construct a signed step function minimizing max|autoconv| / (sum(x))^2.
    Structural approach: Chirp design + refinement.
    """
    start_time = time.time()
    deadline = start_time + 140.0

    best_x = None
    best_score = float('inf')
    
    def update_best(x, score):
        nonlocal best_x, best_score
        if x is not None and score < best_score:
            best_score = score
            best_x = x.copy()
    
    # Phase 1: Try Chirp designs
    if time.time() < deadline - 60.0:
        x_c, sc_c = _optimize_chirp(N, time_budget=20.0)
        update_best(x_c, sc_c)
        
    if time.time() < deadline - 40.0:
        x_tc, sc_tc = _optimize_two_chirp(N, time_budget=30.0)
        update_best(x_tc, sc_tc)
        
    # Phase 2: Try some hand-crafted candidates
    candidates = []
    
    # All ones
    candidates.append(np.ones(N))
    
    # Hanning
    candidates.append(np.hanning(N))
    
    # Raised cosine
    t = np.linspace(0, np.pi, N)
    candidates.append(np.cos(t) ** 2)
    
    # Triangle wave
    tri = 2 * np.abs(2 * (np.arange(N) / N - 0.5)) - 1
    candidates.append(tri)
    
    # Chirp (real part)
    t = np.arange(N, dtype=float)
    candidates.append(np.cos(2 * np.pi * 0.5 * t * t / N))
    
    # Quadratic phase with DC
    candidates.append(np.cos(2 * np.pi * 0.3 * t * t / N) + 0.3)
    
    # Sum of two incommensurate sines
    candidates.append(np.sin(2 * np.pi * 7 * t / N) + np.sin(2 * np.pi * 13 * t / N) + 0.5)
    
    for x in candidates:
        s = np.sum(x)
        if abs(s) < 1e-10:
            continue
        sc = _metric_offdiag(x)
        update_best(x, sc)
    
    if best_x is None:
        return np.ones(N)
    
    # Phase 3: L-BFGS-B refinement of the best candidate
    if time.time() < deadline - 30.0:
        x_ref, sc_ref = _refine_with_lbfgs(best_x, time_budget=25.0, beta=50.0)
        update_best(x_ref, sc_ref)
    
    # Phase 4: Gradient polish
    if time.time() < deadline - 15.0:
        x_pol, sc_pol = _gradient_polish(best_x, max_iters=100, time_budget=12.0)
        update_best(x_pol, sc_pol)
    
    # Phase 5: Final L-BFGS-B with higher beta for sharper max approximation
    if time.time() < deadline - 5.0:
        x_ref2, sc_ref2 = _refine_with_lbfgs(best_x, time_budget=8.0, beta=100.0)
        update_best(x_ref2, sc_ref2)
    
    return best_x
