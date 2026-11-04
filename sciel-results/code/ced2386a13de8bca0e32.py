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

def _multi_sine_sequence(n, params):
    """
    Generate a sequence as sum of K sinusoids + DC offset.
    params = [dc, a1, f1, p1, a2, f2, p2, ..., aK, fK, pK]
    """
    x = np.zeros(n)
    x[:] = params[0]
    t = np.arange(n, dtype=float)
    for k in range(1, len(params) // 3 + 1):
        a = params[3*k - 2]
        f = params[3*k - 1]
        p = params[3*k]
        x += a * np.sin(2.0 * np.pi * f * t / n + p)
    return x

def _objective_multi_sine(params, n=N, beta=20.0):
    """Objective for multi-sine parameter optimization."""
    x = _multi_sine_sequence(n, params)
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

def _optimize_multi_sine(n=N, K=5, time_budget=60.0):
    """
    Optimize a multi-sine representation using differential evolution.
    """
    start = time.time()
    
    # Parameter vector: [dc, a1, f1, p1, ..., aK, fK, pK]
    n_params = 1 + 3 * K
    
    def bounds():
        b = [(-2.0, 2.0)]  # dc
        for _ in range(K):
            b.append((-1.0, 1.0))  # amplitude
            b.append((0.1, n / 2))  # frequency (cycles over N samples)
            b.append((-np.pi, np.pi))  # phase
        return b
    
    def objective(params):
        if time.time() - start > time_budget:
            return 1e10
        return _objective_multi_sine(params, n=n, beta=20.0)
    
    # Use differential evolution for global search
    try:
        result = scipy.optimize.differential_evolution(
            objective,
            bounds=bounds(),
            seed=42,
            maxiter=150,
            tol=1e-10,
            polish=True,
            update_interval=1,
            workers=-1,
            popsize=12
        )
        if time.time() - start < time_budget:
            best_params = result.x
            best_x = _multi_sine_sequence(n, best_params)
            best_score = _metric_offdiag(best_x)
            return best_x, best_score
    except Exception:
        pass
    
    return None, float('inf')

def _gaussian_window_sequence(n, center=0.5, width=0.1):
    """Gaussian window centered at 'center' fraction of n."""
    t = np.arange(n, dtype=float) / n
    return np.exp(-0.5 * ((t - center) / width) ** 2)

def _optimize_gaussian(n=N, time_budget=20.0):
    """Optimize Gaussian window parameters."""
    start = time.time()
    
    def gauss_obj(params):
        if time.time() - start > time_budget:
            return 1e10
        center, width, dc = params
        x = _gaussian_window_sequence(n, center, width) + dc
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
        beta = 20.0
        soft_max = m + np.log(np.sum(np.exp(beta * (abs_c - m)))) / beta
        return soft_max / (s * s)
    
    try:
        result = scipy.optimize.differential_evolution(
            gauss_obj,
            bounds=[(0.2, 0.8), (0.02, 0.5), (-0.5, 0.5)],
            seed=7,
            maxiter=80,
            tol=1e-10,
            polish=True,
            popsize=12
        )
        if time.time() - start < time_budget:
            center, width, dc = result.x
            x = _gaussian_window_sequence(n, center, width) + dc
            score = _metric_offdiag(x)
            return x, score
    except Exception:
        pass
    
    return None, float('inf')

def _spectral_design(n=N):
    """
    Design a sequence by directly specifying its DFT magnitude to be flat
    (maximally spread), then taking the real part. This is a structural
    approach: we design in the frequency domain.
    """
    size = 1
    while size < 2 * n - 1:
        size *= 2
    
    # Start with a random phase spectrum with flat magnitude
    rng = np.random.default_rng(123)
    for trial in range(20):
        # Random phases
        phases = rng.uniform(-np.pi, np.pi, size // 2 + 1)
        # Flat magnitude (all ones) - this gives a sequence with flat spectrum
        mag = np.ones(size // 2 + 1)
        # Ensure Hermitian symmetry for real signal
        spectrum = mag * np.exp(1j * phases)
        spectrum[0] = mag[0]  # DC term real
        if size % 2 == 0:
            spectrum[-1] = mag[-1]  # Nyquist term real
        
        x_full = np.fft.irfft(spectrum, size)
        x = x_full[:n]
        
        # Try different DC offsets
        for dc in [0.0, 0.1, -0.1, 0.5, -0.5]:
            x_test = x + dc
            s = np.sum(x_test)
            if abs(s) < 1e-10:
                continue
            score = _metric_offdiag(x_test)
            if score < 0.5:  # Good enough threshold
                return x_test, score
    
    return None, float('inf')

def _phase_optimization(n=N, time_budget=30.0):
    """
    Optimize the phase of the FFT directly to minimize the peak autoconvolution.
    This works in the frequency domain where the problem may be smoother.
    """
    start = time.time()
    
    size = 1
    while size < 2 * n - 1:
        size *= 2
    
    n_freq = size // 2 + 1
    
    def phase_objective(phases):
        if time.time() - start > time_budget:
            return 1e10
        
        # Construct spectrum from phases with unit magnitude
        spectrum = np.exp(1j * phases)
        spectrum[0] = 1.0  # DC real
        if size % 2 == 0:
            spectrum[-1] = 1.0  # Nyquist real
        
        x_full = np.fft.irfft(spectrum, size)
        x = x_full[:n]
        
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
        
        beta = 30.0
        soft_max = m + np.log(np.sum(np.exp(beta * (abs_c - m)))) / beta
        return soft_max / (s * s)
    
    try:
        # Start with random phases
        rng = np.random.default_rng(42)
        best_x = None
        best_score = float('inf')
        
        for trial in range(3):
            phases_init = rng.uniform(-np.pi, np.pi, n_freq)
            
            result = scipy.optimize.minimize(
                phase_objective,
                phases_init,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if time.time() - start > time_budget:
                break
            
            # Reconstruct x from optimized phases
            spectrum = np.exp(1j * result.x)
            spectrum[0] = 1.0
            if size % 2 == 0:
                spectrum[-1] = 1.0
            
            x_full = np.fft.irfft(spectrum, size)
            x = x_full[:n]
            
            score = _metric_offdiag(x)
            if score < best_score:
                best_score = score
                best_x = x.copy()
        
        if best_x is not None:
            return best_x, best_score
    except Exception:
        pass
    
    return None, float('inf')

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
            options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50},
            tol=1e-14
        )
        if time.time() - start < time_budget:
            x_opt = result.x
            score = _metric_offdiag(x_opt)
            return x_opt, score
    except Exception:
        pass
    
    return None, float('inf')

def _multi_start_refine(x_init, time_budget=60.0, n_starts=5):
    """
    Try multiple perturbed starts and keep the best result.
    """
    start = time.time()
    best_x = x_init.copy()
    best_score = _metric_offdiag(x_init)
    
    rng = np.random.default_rng(12345)
    
    for i in range(n_starts):
        if time.time() - start > time_budget - 5.0:
            break
        
        # Add small random perturbation
        noise_scale = 0.01 * (1 - i / n_starts)
        x_perturbed = x_init + noise_scale * rng.standard_normal(len(x_init))
        
        # Normalize to keep sum similar
        s_orig = np.sum(x_init)
        s_new = np.sum(x_perturbed)
        if abs(s_new) > 1e-10:
            x_perturbed = x_perturbed * (s_orig / s_new)
        
        x_opt, score = _refine_with_lbfgs(x_perturbed, time_budget=min(15.0, time_budget - (time.time() - start) - 5.0), beta=50.0)
        
        if x_opt is not None and score < best_score:
            best_score = score
            best_x = x_opt.copy()
    
    return best_x, best_score

def _gradient_polish_fast(x, max_iters=50, time_budget=15.0):
    """
    Fast gradient descent using FFT-based gradient computation.
    Much faster than the O(n^2) version.
    """
    start = time.time()
    x = x.copy()
    best_x = x.copy()
    best_score = _metric_offdiag(x)
    n = len(x)
    
    size = 1
    while size < 2 * n - 1:
        size *= 2
    
    for it in range(max_iters):
        if time.time() - start > time_budget:
            break
        
        s = np.sum(x)
        if abs(s) < 1e-12:
            break
        
        c = _autoconv_lin(x)
        abs_c = np.abs(c[1:])
        n_lags = len(abs_c)
        
        # Use top 10% of lags for smoother gradient
        threshold = np.percentile(abs_c, 90)
        active = abs_c > threshold
        if not np.any(active):
            active = abs_c == np.max(abs_c)
        
        # Compute gradient using FFT: grad = sum_k sign(c_k) * (x shifted by k + x shifted by -k)
        # This can be done efficiently with circular shifts in the padded domain
        grad = np.zeros(n)
        
        active_lags = np.where(active)[0] + 1  # +1 because we excluded lag 0
        
        for k in active_lags:
            sign_c = np.sign(c[k])
            if sign_c == 0:
                sign_c = 1.0
            
            # d(c_k)/dx_j = x_{j+k} + x_{j-k}
            # Vectorized: for each j, this is x[j+k] (if j+k<n) + x[j-k] (if j-k>=0)
            idx1 = np.arange(n) + k
            valid1 = idx1 < n
            idx2 = np.arange(n) - k
            valid2 = idx2 >= 0
            
            contrib = np.zeros(n)
            contrib[valid1] += sign_c * x[idx1[valid1]]
            contrib[valid2] += sign_c * x[idx2[valid2]]
            grad += contrib
        
        gnorm = np.linalg.norm(grad)
        if gnorm < 1e-15:
            break
        
        # Adaptive step size with backtracking
        step = 0.02
        for backtrack in range(10):
            x_new = x - step * grad / gnorm
            
            # Keep sum positive and reasonable
            s_new = np.sum(x_new)
            if abs(s_new) < 0.1:
                x_new += (1.0 - s_new) / n
            
            sc_new = _metric_offdiag(x_new)
            if sc_new < best_score:
                best_score = sc_new
                best_x = x_new.copy()
                x = x_new
                break
            else:
                step *= 0.5
        else:
            break
    
    return best_x, best_score

def construct():
    """
    Construct a signed step function minimizing max|autoconv| / (sum(x))^2.
    Structural approach: multi-frequency design + spectral design + refinement.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    best_x = None
    best_score = float('inf')
    
    def update_best(x, score):
        nonlocal best_x, best_score
        if x is not None and score < best_score:
            best_score = score
            best_x = x.copy()

    # Phase 1: Try structural designs (low-dimensional parameter spaces)
    
    # 1a: Multi-sine with 3 harmonics
    if time.time() < deadline - 70.0:
        x_ms, sc_ms = _optimize_multi_sine(N, K=3, time_budget=25.0)
        update_best(x_ms, sc_ms)
    
    # 1b: Multi-sine with 5 harmonics
    if time.time() < deadline - 45.0:
        x_ms2, sc_ms2 = _optimize_multi_sine(N, K=5, time_budget=20.0)
        update_best(x_ms2, sc_ms2)
    
    # 1c: Gaussian window
    if time.time() < deadline - 35.0:
        x_g, sc_g = _optimize_gaussian(N, time_budget=12.0)
        update_best(x_g, sc_g)
    
    # 1d: Spectral design (flat spectrum with random phases)
    if time.time() < deadline - 25.0:
        x_sp, sc_sp = _spectral_design(N)
        update_best(x_sp, sc_sp)
    
    # 1e: Phase optimization in frequency domain
    if time.time() < deadline - 20.0:
        x_ph, sc_ph = _phase_optimization(N, time_budget=15.0)
        update_best(x_ph, sc_ph)
    
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
    
    # B-spline-like (cubic)
    bspline = np.zeros(N)
    for i in range(N):
        u = i / N - 0.5
        bspline[i] = max(0, 1 - abs(u) * 2) ** 3
    candidates.append(bspline)
    
    # Hamming window
    candidates.append(np.hamming(N))
    
    # Blackman window
    candidates.append(np.blackman(N))
    
    for x in candidates:
        s = np.sum(x)
        if abs(s) < 1e-10:
            continue
        sc = _metric_offdiag(x)
        update_best(x, sc)
    
    if best_x is None:
        return np.ones(N)
    
    # Phase 3: L-BFGS-B refinement of the best candidate
    if time.time() < deadline - 25.0:
        x_ref, sc_ref = _refine_with_lbfgs(best_x, time_budget=20.0, beta=50.0)
        update_best(x_ref, sc_ref)
    
    # Phase 4: Multi-start refinement
    if time.time() < deadline - 20.0:
        x_ms_ref, sc_ms_ref = _multi_start_refine(best_x, time_budget=15.0, n_starts=3)
        update_best(x_ms_ref, sc_ms_ref)
    
    # Phase 5: Fast gradient polish
    if time.time() < deadline - 10.0:
        x_pol, sc_pol = _gradient_polish_fast(best_x, max_iters=50, time_budget=8.0)
        update_best(x_pol, sc_pol)
    
    # Phase 6: Final high-precision L-BFGS-B
    if time.time() < deadline - 5.0:
        x_ref2, sc_ref2 = _refine_with_lbfgs(best_x, time_budget=4.0, beta=100.0)
        update_best(x_ref2, sc_ref2)
    
    # Final check: ensure we have a valid result
    if best_x is None or best_score >= float('inf'):
        return np.ones(N)
    
    return best_x
