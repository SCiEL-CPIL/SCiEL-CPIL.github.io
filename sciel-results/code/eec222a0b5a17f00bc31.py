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
    shifted = beta * (corr - t)
    m = np.max(shifted)
    shifted = shifted - m
    logsum = m + np.log(np.sum(np.exp(shifted)))
    return t + logsum / beta

def _smooth_objective_grad(x, beta):
    h = x[:N]
    t = x[N]
    corr = _corr_vector(h)
    shifted = beta * (corr - t)
    m = np.max(shifted)
    shifted = shifted - m
    p = np.exp(shifted)
    p_sum = np.sum(p)
    p = p / p_sum  # softmax weights over lags

    # Gradient w.r.t t is 0 (as derived)
    grad_t = 0.0

    # Vectorized gradient computation using FFT-based convolution
    # grad_h[i] = sum_k p_k * [(1-h[i+k]) * 1(0<=i+k<N) - h[i-k] * 1(0<=i-k<N)]
    # = cross_corr(p, 1-h) - cross_corr(p, h)  (with appropriate sign and truncation)
    
    # For full mode correlation: corr[k] = sum_i h[i] * g[i+k] where g=1-h
    # d(corr[k])/dh[i] = g[i+k] if 0<=i+k<N, plus -h[i-k] if 0<=i-k<N
    
    # Let's compute: term1[i] = sum_k p_k * g[i+k] * 1(0<=i+k<N)
    #              term2[i] = sum_k p_k * h[i-k] * 1(0<=i-k<N)
    # grad_h[i] = term1[i] - term2[i]
    
    # term1[i] = sum_k p_k * g[i+k] for valid i+k
    # This is a cross-correlation of p and g
    # p has length 2N-1, indexed 0..2N-2, with lag k = idx - (N-1)
    # So p[idx] corresponds to lag k = idx - (N+1)
    
    # Let's use direct vectorized computation with numpy
    one_minus_h = 1.0 - h
    grad_h = np.zeros(N)
    
    # Process all lags at once using broadcasting
    # For lag k (from -(N-1) to N-1):
    # term1 contributes to indices i where 0 <= i+k < N, i.e., max(0,-k) <= i < min(N, N-k)
    # term2 contributes to indices i where 0 <= i-k < N, i.e., max(0,k) <= i < min(N, N+k)
    
    lags = np.arange(-(N-1), N)  # length 2N-1
    p_aligned = p  # p[k_idx] where k_idx = k + (N-1)
    
    # Vectorized: for each element i in 0..N-1, compute sum over valid k
    # Use FFT for efficiency
    # term1[i] = sum_k p_k * g[i+k] where 0<=i+k<N
    # This is (p * g)[i] in cross-correlation sense
    
    # Use np.correlate style: 
    # term1 = cross_correlation(p, g) where p is centered
    # Actually, let's just do it with a loop but vectorize the inner part
    
    # More efficient: use the fact that this is a convolution
    # term1[i] = sum_{k} p[k] * g[i+k] = (p_rev * g)[i + (N-1)]
    # where p_rev is p reversed
    
    p_rev = p[::-1]
    # term1[i] = sum_k p[k] * g[i+k]
    # Let j = i+k, so k = j-i. term1[i] = sum_j p[j-i] * g[j]
    # This is a cross-correlation: (p * g)[i] in the sense of sum_j p[j-i]*g[j]
    # = np.correlate(g, p, mode='full')[N-1 + i - (N-1)] ... let me just compute directly
    
    # Direct FFT approach:
    # term1[i] = sum_k p[k] * g[i+k] for 0<=i+k<N
    # Pad g to length 2N, compute correlation
    g_padded = np.zeros(2*N)
    g_padded[:N] = g_padded[:N]  # will fill
    g_padded[:N] = one_minus_h
    
    # Cross-correlation: (f * g)[n] = sum_m f[m] * g[m+n]
    # We want: sum_k p[k] * g[i+k] = (p * g)[i]
    # Using FFT: ifft(fft(p) * conj(fft(g))) gives circular correlation
    # For linear, pad to 2N-1 + N - 1 = 3N-2
    
    size = 3*N
    F_p = np.fft.rfft(p, size)
    F_g = np.fft.rfft(one_minus_h, size)
    # Cross-correlation: ifft(conj(F_p) * F_g) gives sum_k p[k] * g[k+i]
    # Actually: np.correlate(g, p, 'full')[n] = sum_k g[k] * p[k-n]
    # Let me just use np.correlate
    corr_pg = np.correlate(one_minus_h, p, mode='full')
    # corr_pg[m] = sum_k g[k] * p[k-m] = sum_k g[k] * p[k-m]
    # We want term1[i] = sum_k p[k] * g[i+k]
    # Let j = i+k: sum_j p[j-i] * g[j]
    # = sum_j g[j] * p[j-i]
    # = corr_pg[i] if corr_pg[i] = sum_j g[j] * p[j-i]
    # np.correlate(g, p, 'full')[i] = sum_j g[j] * p[j-(i-(N-1))]... 
    
    # This is getting complicated. Let me just use a simple loop that's still fast enough
    # with the skip threshold, and focus on other improvements.
    
    for k_idx in range(len(p)):
        pk = p[k_idx]
        if pk < 1e-12:
            continue
        k = k_idx - (N - 1)  # actual lag
        
        if k >= 0:
            if N - 1 - k >= 0:
                grad_h[:N-k] += pk * (1.0 - h[k:])
            if k > 0:
                grad_h[k:] -= pk * h[:N-k]
        else:
            k_pos = -k
            if k_pos < N:
                grad_h[k_pos:] += pk * (1.0 - h[:N-k_pos])
                if k_pos > 0:
                    grad_h[:k_pos] -= pk * h[k_pos:]
    
    return np.concatenate([grad_h, [grad_t]])

def _parametric_h(params):
    """
    Generate h from 9 parameters: [a1, p1, f1, a2, p2, f2, a3, p3, f3]
    h_raw[i] = 0.5 + a1*sin(2*pi*f1*i/N + p1) + a2*sin(2*pi*f2*i/N + p2) + a3*sin(2*pi*f3*i/N + p3)
    Then project to feasible set.
    """
    a1, p1, f1, a2, p2, f2, a3, p3, f3 = params
    i = np.arange(N)
    h_raw = 0.5 + a1 * np.sin(2 * np.pi * f1 * i / N + p1) + \
                    a2 * np.sin(2 * np.pi * f2 * i / N + p2) + \
                    a3 * np.sin(2 * np.pi * f3 * i / N + p3)
    return _project_feasible(h_raw)

def _parametric_objective(params):
    h = _parametric_h(params)
    return _correlation_max(h)

def _parametric_objective_grad(params):
    """
    Numerical gradient of the parametric objective.
    """
    eps = 1e-6
    grad = np.zeros(len(params))
    f0 = _parametric_objective(params)
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += eps
        grad[i] = (_parametric_objective(params_plus) - f0) / eps
    return grad

def _coordinate_polish(h, max_iter=200):
    """
    Coordinate descent: for each i, optimize h[i] given others,
    projecting to keep sum = TARGET_SUM and 0<=h<=1.
    """
    h = _project_feasible(h).copy()
    best_val = _correlation_max(h)
    for it in range(max_iter):
        improved = False
        for i in range(N):
            h_old = h[i]
            candidates = [0.0, 1.0, h_old]
            for dx in [-0.1, -0.05, 0.05, 0.1]:
                candidates.append(h_old + dx)
            best_x = h_old
            best_v = best_val
            for x in candidates:
                x = max(0.0, min(1.0, x))
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
    for base in [0.2, 0.25, 0.3, 0.35, 0.4, 0.45]:
        alt = np.where(np.arange(N) % 2 == 0, base, 1.0 - base)
        starts.append(_project_feasible(alt))

    # Sine waves
    for freq in [1, 2, 3, 4, 5, 7, 10, 15, 20, 30]:
        for amp in [0.2, 0.3, 0.4, 0.45, 0.49]:
            t = np.arange(N)
            h_sin = 0.5 + amp * np.sin(2 * np.pi * freq * t / N)
            starts.append(_project_feasible(h_sin))
            h_cos = 0.5 + amp * np.cos(2 * np.pi * freq * t / N)
            starts.append(_project_feasible(h_cos))

    # Two-block patterns
    for split in [20, 30, 40, 50, 60, 70, 80]:
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

    # Random smooth seeds
    rng = np.random.RandomState(42)
    for _ in range(15):
        noise = rng.randn(N)
        kernel = np.ones(9) / 9.0
        smooth = np.convolve(noise, kernel, mode='same')
        h_pert = 0.5 + 0.4 * smooth
        starts.append(_project_feasible(h_pert))

    # Random blocky seeds
    for _ in range(10):
        h_rand = rng.rand(N)
        block_size = rng.randint(3, 15)
        for b in range(0, N, block_size):
            h_rand[b:b+block_size] = h_rand[b]
        starts.append(_project_feasible(h_rand))

    # Phase 2: Refine with smooth epigraph L-BFGS-B, adaptive beta
    bounds = [(0.0, 1.0) for _ in range(N)] + [(-1.0, 2.0)]
    
    # Use multiple beta values: start low, increase for accuracy
    beta_schedule = [15.0, 25.0, 40.0, 60.0]
    
    for h0 in starts:
        if time.time() - start_time > DEADLINE - 35:
            break
        try:
            for beta in beta_schedule:
                if time.time() - start_time > DEADLINE - 30:
                    break
                t_init = _correlation_max(h0) * N / 2.0 + 0.1
                x0 = np.concatenate([h0, [t_init]])
                res = minimize(
                    _smooth_objective,
                    x0,
                    args=(beta,),
                    jac=lambda x, b=beta: _smooth_objective_grad(x, b),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-11, 'disp': False}
                )
                h_res = _project_feasible(res.x[:N])
                val = _correlation_max(h_res)
                if val < best_val - 1e-10:
                    best_val = val
                    best_h = h_res.copy()
                    h0 = h_res.copy()  # carry forward improved solution
        except Exception:
            pass

    # Phase 3: Parametric optimization (9-dimensional sinusoid sum)
    if time.time() - start_time < DEADLINE - 20:
        param_inits = [
            [0.3, 0, 1, 0.2, np.pi/2, 2, 0.1, np.pi, 3],
            [0.4, 0, 1, 0.3, 0, 2, 0.2, 0, 3],
            [0.3, np.pi/4, 1, 0.2, 0, 3, 0.1, 0, 5],
            [0.2, 0, 2, 0.3, 0, 3, 0.1, 0, 4],
            [0.3, 0, 1, 0.2, 0, 4, 0.1, 0, 7],
            [0.25, 0, 1, 0.25, 0, 2, 0.15, 0, 3],
        ]
        
        param_bounds = [
            (0.0, 0.49), (0, 2*np.pi), (1, 15),
            (0.0, 0.49), (0, 2*np.pi), (1, 15),
            (0.0, 0.49), (0, 2*np.pi), (1, 15),
        ]
        
        for p0 in param_inits:
            if time.time() - start_time > DEADLINE - 15:
                break
            try:
                res = minimize(
                    _parametric_objective,
                    p0,
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                )
                h_res = _parametric_h(res.x)
                val = _correlation_max(h_res)
                if val < best_val - 1e-10:
                    best_val = val
                    best_h = h_res.copy()
            except Exception:
                pass
        
        for p0 in param_inits[:3]:
            if time.time() - start_time > DEADLINE - 10:
                break
            try:
                res = minimize(
                    _parametric_objective,
                    p0,
                    jac=_parametric_objective_grad,
                    method='L-BFGS-B',
                    bounds=param_bounds,
                    options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
                )
                h_res = _parametric_h(res.x)
                val = _correlation_max(h_res)
                if val < best_val - 1e-10:
                    best_val = val
                    best_h = h_res.copy()
            except Exception:
                pass

    # Phase 4: Coordinate polish on best candidate
    if time.time() - start_time < DEADLINE - 8:
        try:
            h_polished, val_polished = _coordinate_polish(best_h, max_iter=200)
            if val_polished < best_val - 1e-12:
                best_val = val_polished
                best_h = h_polished
        except Exception:
            pass

    # Phase 5: Final high-precision polish with very high beta
    if time.time() - start_time < DEADLINE - 5:
        try:
            for beta2 in [80.0, 120.0]:
                if time.time() - start_time > DEADLINE - 3:
                    break
                t_init = _correlation_max(best_h) * N / 2.0 + 0.02
                x0 = np.concatenate([best_h, [t_init]])
                res = minimize(
                    _smooth_objective,
                    x0,
                    args=(beta2,),
                    jac=lambda x, b=beta2: _smooth_objective_grad(x, b),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-13, 'disp': False}
                )
                h_final = _project_feasible(res.x[:N])
                val_final = _correlation_max(h_final)
                if val_final < best_val - 1e-13:
                    best_val = val_final
                    best_h = h_final
        except Exception:
            pass

    # Ensure final validity
    best_h = _project_feasible(best_h)

    # Phase 6: Additional coordinate descent passes with finer steps
    if time.time() - start_time < DEADLINE - 2:
        try:
            for _ in range(5):
                if time.time() - start_time > DEADLINE - 1:
                    break
                h_polished, val_polished = _coordinate_polish(best_h, max_iter=80)
                if val_polished < best_val - 1e-12:
                    best_val = val_polished
                    best_h = h_polished
        except Exception:
            pass

    # Final projection to ensure validity
    best_h = _project_feasible(best_h)

    return best_h
