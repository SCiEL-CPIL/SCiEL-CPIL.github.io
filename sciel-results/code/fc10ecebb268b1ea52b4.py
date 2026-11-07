import numpy as np
from scipy.optimize import minimize
import time

N = 100
TARGET_SUM = N / 2.0
DEADLINE = 135.0  # safety margin before 150s kill

def _correlation_max(h):
    """Compute the raw value: max correlation / N * 2."""
    corr = np.correlate(h, 1.0 - h, mode='full')
    return np.max(corr) / N * 2.0

def _corr_vector(h):
    """Return the full correlation vector (length 2N-1)."""
    return np.correlate(h, 1.0 - h, mode='full')

def _project_feasible(h):
    """Project h to satisfy 0 <= h <= 1 and sum(h) = N/2."""
    h = np.asarray(h, dtype=float).copy()
    h = np.clip(h, 0.0, 1.0)
    for _ in range(1000):
        diff = TARGET_SUM - np.sum(h)
        if abs(diff) < 1e-15:
            break
        mask = (h > 1e-12) & (h < 1.0 - 1e-12)
        n_free = int(np.sum(mask))
        if n_free == 0:
            break
        h[mask] += diff / n_free
        h = np.clip(h, 0.0, 1.0)
    return h

def _smooth_objective(x, beta):
    """
    x = [h (N), t (1)]
    Objective: t + (1/beta) * log(sum_lag exp(beta * (corr_lag - t)))
    This is a smooth surrogate for max(t, max_lag corr_lag).
    """
    h = x[:N]
    t = x[N]
    corr = _corr_vector(h)
    # corr_lag - t
    shifted = beta * (corr - t)
    m = np.max(shifted)
    if m > 700:
        shifted = shifted - m
        logsum = m + np.log(np.sum(np.exp(shifted)))
    else:
        logsum = np.log(np.sum(np.exp(shifted)))
    return t + logsum / beta

def _smooth_objective_grad(x, beta):
    h = x[:N]
    t = x[N]
    corr = _corr_vector(h)
    shifted = beta * (corr - t)
    m = np.max(shifted)
    if m > 700:
        shifted = shifted - m
        logsum = m + np.log(np.sum(np.exp(shifted)))
    else:
        logsum = np.log(np.sum(np.exp(shifted)))
    # weights: p_lag = exp(beta*(corr_lag - t)) / sum_lag exp(...)
    p = np.exp(shifted)
    p_sum = np.sum(p)
    p = p / p_sum  # softmax weights over lags

    # Gradient w.r.t t is 0 (as derived)
    grad_t = 0.0

    # Gradient w.r.t h: sum_lag p_lag * d(corr_lag)/dh
    grad_h = np.zeros(N)
    for k_idx in range(len(p)):
        pk = p[k_idx]
        if pk < 1e-15:
            continue
        k = k_idx - (N - 1)  # actual lag
        for i in range(N):
            j = i + k
            if 0 <= j < N:
                grad_h[i] += pk * (1.0 - h[j])
            j2 = i - k
            if 0 <= j2 < N:
                grad_h[i] -= pk * h[j2]
    return np.concatenate([grad_h, [grad_t]])

def _parametric_h(params):
    """
    Generate h from 12 parameters: [a1, p1, f1, a2, p2, f2, a3, p3, f3, a4, p4, f4]
    h_raw[i] = 0.5 + sum of 4 sinusoidal components
    Then project to feasible set.
    """
    a1, p1, f1, a2, p2, f2, a3, p3, f3, a4, p4, f4 = params
    i = np.arange(N)
    h_raw = 0.5
    for a, p, f in [(a1, p1, f1), (a2, p2, f2), (a3, p3, f3), (a4, p4, f4)]:
        h_raw += a * np.sin(2 * np.pi * f * i / N + p)
    return _project_feasible(h_raw)

def _poisson_h(sources, smoothing_kernel):
    """
    Generate h from sparse source vector via discrete Poisson solve.
    sources: sparse vector of length N (most entries zero)
    smoothing_kernel: 1D kernel for additional smoothing
    """
    # Discrete Laplacian: L[i] = h[i-1] - 2*h[i] + h[i+1]
    # We want L*h = -sources (with periodic BC)
    # In Fourier space: (-4*sin^2(k*pi/N)) * H[k] = -S[k]
    # So H[k] = S[k] / (4*sin^2(k*pi/N)) for k != 0
    # For k=0, the mean is free (set by projection)
    
    S = np.fft.rfft(sources)
    k = np.arange(len(S))
    denom = 4.0 * np.sin(np.pi * k / N) ** 2
    denom[0] = 1.0  # avoid division by zero
    H = S / denom
    h_raw = np.fft.irfft(H, n=N)
    
    # Apply additional smoothing
    if len(smoothing_kernel) > 1:
        h_raw = np.convolve(h_raw, smoothing_kernel, mode='same')
    
    return _project_feasible(h_raw)

def _poisson_parametric_h(params):
    """
    Generate h from parametric sparse sources.
    params: [s1, pos1, s2, pos2, s3, pos3, s4, pos4, s5, pos5]
    where s_i are source strengths and pos_i are positions (0 to N-1)
    """
    N_sources = len(params) // 2
    sources = np.zeros(N)
    for i in range(N_sources):
        s = params[2*i]
        pos = int(params[2*i+1]) % N
        # Spread source over a few points for smoothness
        for d in [-1, 0, 1]:
            idx = (pos + d) % N
            weight = 1.0 if d == 0 else 0.5
            sources[idx] += s * weight
    
    # Use a small smoothing kernel
    kernel = np.array([0.2, 0.6, 0.2])
    return _poisson_h(sources, kernel)

def _parametric_objective(params):
    h = _parametric_h(params)
    return _correlation_max(h)

def _parametric_objective_grad(params):
    """
    Numerical gradient of the parametric objective using central differences.
    """
    eps = 1e-7
    grad = np.zeros(len(params))
    for i in range(len(params)):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[i] += eps
        params_minus[i] -= eps
        grad[i] = (_parametric_objective(params_plus) - _parametric_objective(params_minus)) / (2 * eps)
    return grad

def _poisson_objective(params):
    h = _poisson_parametric_h(params)
    return _correlation_max(h)

def _poisson_objective_grad(params):
    """
    Numerical gradient of the Poisson parametric objective.
    """
    eps = 1e-5
    grad = np.zeros(len(params))
    for i in range(len(params)):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[i] += eps
        params_minus[i] -= eps
        grad[i] = (_poisson_objective(params_plus) - _poisson_objective(params_minus)) / (2 * eps)
    return grad

def _coordinate_polish(h, max_iter=300):
    """
    Coordinate descent with adaptive step sizes.
    For each i, try multiple step sizes and pick the best improvement.
    """
    h = _project_feasible(h).copy()
    best_val = _correlation_max(h)
    step_sizes = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3]
    
    for it in range(max_iter):
        improved = False
        for i in range(N):
            h_old = h[i]
            best_x = h.copy()
            best_v = best_val
            
            # Try boundaries
            for x in [0.0, 1.0]:
                h_test = h.copy()
                h_test[i] = x
                h_test = _project_feasible(h_test)
                v = _correlation_max(h_test)
                if v < best_v - 1e-12:
                    best_v = v
                    best_x = h_test.copy()
            
            # Try adaptive steps
            for ds in step_sizes:
                for sign in [-1, 1]:
                    x = max(0.0, min(1.0, h_old + sign * ds))
                    if abs(x - h_old) < 1e-10:
                        continue
                    h_test = h.copy()
                    h_test[i] = x
                    h_test = _project_feasible(h_test)
                    v = _correlation_max(h_test)
                    if v < best_v - 1e-12:
                        best_v = v
                        best_x = h_test.copy()
            
            if best_v < best_val - 1e-12:
                h = best_x
                best_val = best_v
                improved = True
        if not improved:
            break
    return h, best_val

def construct():
    start_time = time.time()

    best_h = np.full(N, 0.5)
    best_val = _correlation_max(best_h)

    # Phase 1: Structured initializations
    starts = [np.full(N, 0.5)]

    # Alternating patterns
    for base in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]:
        alt = np.where(np.arange(N) % 2 == 0, base, 1.0 - base)
        starts.append(_project_feasible(alt))

    # Sine waves with more frequencies and amplitudes
    for freq in [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30]:
        for amp in [0.15, 0.25, 0.35, 0.42, 0.47, 0.49]:
            t = np.arange(N)
            h_sin = 0.5 + amp * np.sin(2 * np.pi * freq * t / N)
            starts.append(_project_feasible(h_sin))
            h_cos = 0.5 + amp * np.cos(2 * np.pi * freq * t / N)
            starts.append(_project_feasible(h_cos))

    # Two-block patterns
    for split in [15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85]:
        h_block = np.zeros(N)
        h_block[:split] = 1.0
        h_block = _project_feasible(h_block)
        starts.append(h_block)
        h_block2 = np.zeros(N)
        h_block2[split:] = 1.0
        h_block2 = _project_feasible(h_block2)
        starts.append(h_block2)

    # Linear ramps
    starts.append(_project_feasible(np.linspace(0.0, 1.0, N)))
    starts.append(_project_feasible(np.linspace(1.0, 0.0, N)))

    # Triangular
    t = np.arange(N)
    tri = 1.0 - np.abs(t - N / 2.0) / (N / 2.0)
    starts.append(_project_feasible(tri))
    tri2 = np.abs(t - N / 2.0) / (N / 2.0)
    starts.append(_project_feasible(tri2))

    # Parabolic
    par = 1.0 - ((t - N/2.0) / (N/2.0)) ** 2
    starts.append(_project_feasible(par))
    par2 = ((t - N/2.0) / (N/2.0)) ** 2
    starts.append(_project_feasible(par2))

    # Random smooth seeds with various kernel widths
    rng = np.random.RandomState(42)
    for kernel_width in [5, 7, 9, 11, 13]:
        for _ in range(8):
            noise = rng.randn(N)
            kernel = np.ones(kernel_width) / kernel_width
            smooth = np.convolve(noise, kernel, mode='same')
            h_pert = 0.5 + 0.45 * smooth
            starts.append(_project_feasible(h_pert))

    # Random blocky seeds with various block sizes
    for block_size in [3, 5, 7, 10, 15, 20]:
        for _ in range(5):
            h_rand = rng.rand(N)
            for b in range(0, N, block_size):
                h_rand[b:b+block_size] = h_rand[b]
            starts.append(_project_feasible(h_rand))

    # Poisson-based starts
    for _ in range(10):
        n_sources = rng.randint(2, 6)
        sources = np.zeros(N)
        for _ in range(n_sources):
            pos = rng.randint(0, N)
            strength = rng.uniform(-0.5, 0.5)
            sources[pos] += strength
        kernel = np.array([0.2, 0.6, 0.2])
        starts.append(_poisson_h(sources, kernel))

    # Phase 2: Refine with smooth epigraph SLSQP
    bounds = [(0.0, 1.0) for _ in range(N)] + [(-1.0, 2.0)]
    beta = 20.0

    for h0 in starts:
        if time.time() - start_time > DEADLINE - 30:
            break
        try:
            x0 = np.concatenate([h0, [_correlation_max(h0) * N / 2.0 + 0.1]])
            res = minimize(
                _smooth_objective,
                x0,
                args=(beta,),
                jac=lambda x: _smooth_objective_grad(x, beta),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10, 'disp': False}
            )
            h_res = _project_feasible(res.x[:N])
            val = _correlation_max(h_res)
            if val < best_val - 1e-10:
                best_val = val
                best_h = h_res.copy()
        except Exception:
            pass

    # Phase 3: Parametric optimization (12-dimensional sinusoid sum)
    if time.time() - start_time < DEADLINE - 25:
        # Try several parametric initializations with 4 sinusoidal components
        param_inits = [
            [0.3, 0, 1, 0.2, np.pi/2, 2, 0.1, np.pi, 3, 0.05, 0, 5],
            [0.4, 0, 1, 0.3, 0, 2, 0.2, 0, 3, 0.1, np.pi/4, 4],
            [0.3, np.pi/4, 1, 0.2, 0, 3, 0.1, 0, 5, 0.05, 0, 7],
            [0.2, 0, 2, 0.3, 0, 3, 0.1, 0, 4, 0.08, np.pi/2, 6],
            [0.3, 0, 1, 0.2, 0, 4, 0.1, 0, 7, 0.05, 0, 10],
            [0.25, 0, 1, 0.25, 0, 2, 0.15, 0, 3, 0.1, 0, 5],
            [0.35, 0, 1, 0.25, np.pi/3, 2, 0.15, np.pi/6, 3, 0.08, 0, 4],
            [0.28, np.pi/8, 1, 0.22, 0, 2, 0.12, np.pi/4, 3, 0.06, 0, 5],
        ]
        
        param_bounds = [
            (0.0, 0.49), (0, 2*np.pi), (1, 20),
            (0.0, 0.49), (0, 2*np.pi), (1, 20),
            (0.0, 0.49), (0, 2*np.pi), (1, 20),
            (0.0, 0.49), (0, 2*np.pi), (1, 20),
        ]
        
        for p0 in param_inits:
            if time.time() - start_time > DEADLINE - 20:
                break
            try:
                # Use Nelder-Mead for the 12-dimensional parametric problem
                res = minimize(
                    _parametric_objective,
                    p0,
                    method='Nelder-Mead',
                    options={'maxiter': 800, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True}
                )
                h_res = _parametric_h(res.x)
                val = _correlation_max(h_res)
                if val < best_val - 1e-10:
                    best_val = val
                    best_h = h_res.copy()
            except Exception:
                pass
        
        # Also try L-BFGS-B on the parametric form with numerical gradient
        for p0 in param_inits[:4]:
            if time.time() - start_time > DEADLINE - 15:
                break
            try:
                res = minimize(
                    _parametric_objective,
                    p0,
                    jac=_parametric_objective_grad,
                    method='L-BFGS-B',
                    bounds=param_bounds,
                    options={'maxiter': 1500, 'ftol': 1e-13, 'gtol': 1e-10, 'disp': False}
                )
                h_res = _parametric_h(res.x)
                val = _correlation_max(h_res)
                if val < best_val - 1e-10:
                    best_val = val
                    best_h = h_res.copy()
            except Exception:
                pass

    # Phase 3b: Poisson parametric optimization
    if time.time() - start_time < DEADLINE - 18:
        poisson_inits = [
            [0.3, 10, -0.3, 50, 0.2, 25, -0.2, 75, 0.1, 40],
            [0.4, 5, -0.4, 45, 0.3, 20, -0.3, 80, 0.15, 35],
            [0.25, 15, -0.25, 55, 0.2, 30, -0.2, 70, 0.1, 45],
            [0.35, 8, -0.35, 48, 0.25, 28, -0.25, 72, 0.12, 38],
            [0.2, 12, -0.2, 52, 0.15, 32, -0.15, 68, 0.08, 42],
        ]
        
        poisson_bounds = [
            (-0.5, 0.5), (0, N-1),
            (-0.5, 0.5), (0, N-1),
            (-0.5, 0.5), (0, N-1),
            (-0.5, 0.5), (0, N-1),
            (-0.5, 0.5), (0, N-1),
        ]
        
        for p0 in poisson_inits:
            if time.time() - start_time > DEADLINE - 15:
                break
            try:
                res = minimize(
                    _poisson_objective,
                    p0,
                    method='Nelder-Mead',
                    options={'maxiter': 600, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                )
                h_res = _poisson_parametric_h(res.x)
                val = _correlation_max(h_res)
                if val < best_val - 1e-10:
                    best_val = val
                    best_h = h_res.copy()
            except Exception:
                pass
        
        for p0 in poisson_inits[:3]:
            if time.time() - start_time > DEADLINE - 12:
                break
            try:
                res = minimize(
                    _poisson_objective,
                    p0,
                    jac=_poisson_objective_grad,
                    method='L-BFGS-B',
                    bounds=poisson_bounds,
                    options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-9, 'disp': False}
                )
                h_res = _poisson_parametric_h(res.x)
                val = _correlation_max(h_res)
                if val < best_val - 1e-10:
                    best_val = val
                    best_h = h_res.copy()
            except Exception:
                pass

    # Phase 4: Coordinate polish on best candidate
    if time.time() - start_time < DEADLINE - 10:
        try:
            h_polished, val_polished = _coordinate_polish(best_h, max_iter=250)
            if val_polished < best_val - 1e-12:
                best_val = val_polished
                best_h = h_polished
        except Exception:
            pass

    # Phase 5: Final high-precision optimization
    if time.time() - start_time < DEADLINE - 5:
        try:
            # First, try a higher beta smooth optimization
            beta2 = 80.0
            x0 = np.concatenate([best_h, [_correlation_max(best_h) * N / 2.0 + 0.03]])
            res = minimize(
                _smooth_objective,
                x0,
                args=(beta2,),
                jac=lambda x: _smooth_objective_grad(x, beta2),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-11, 'disp': False}
            )
            h_final = _project_feasible(res.x[:N])
            val_final = _correlation_max(h_final)
            if val_final < best_val - 1e-13:
                best_val = val_final
                best_h = h_final
        except Exception:
            pass
        
        # Final coordinate polish
        if time.time() - start_time < DEADLINE - 3:
            try:
                h_final2, val_final2 = _coordinate_polish(best_h, max_iter=100)
                if val_final2 < best_val - 1e-12:
                    best_val = val_final2
                    best_h = h_final2
            except Exception:
                pass

    # Ensure final validity
    best_h = _project_feasible(best_h)

    return best_h
