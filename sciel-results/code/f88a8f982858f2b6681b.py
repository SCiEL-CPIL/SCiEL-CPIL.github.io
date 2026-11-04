import time
import numpy as np
from scipy.optimize import minimize

N = 50
w = 0.5 / N  # step width = 0.01
_START = time.time()
DEADLINE = 145.0  # safety margin before 150s kill


def _elapsed():
    return time.time() - _START


def _time_left():
    return DEADLINE - _elapsed()


def _autoconv_lags(h):
    """Compute all lag values of autoconvolution (without the w factor).
    c[m] = sum_j h[j]*h[j+m] for m=0..N-1
    """
    c = np.zeros(N)
    c[0] = np.dot(h, h)
    for m in range(1, N):
        c[m] = np.dot(h[:N - m], h[m:])
    return c


def _soft_max(c, beta=20.0):
    """Soft max via log-sum-exp for smoothness. Returns (value, weights)."""
    m = c.max()
    exp_c = np.exp(beta * (c - m))
    s = exp_c.sum()
    if s <= 0:
        return m, np.ones_like(c) / len(c)
    weights = exp_c / s
    # Soft max value: (1/beta) * log(sum(exp(beta*c)))
    val = m + np.log(s) / beta
    return val, weights


def _autoconv_max(h, beta=20.0):
    """Compute soft max of autoconvolution = w * soft_max_m c_m."""
    c = _autoconv_lags(h)
    val, weights = _soft_max(c, beta)
    return w * val, c, weights


def _ratio(h, beta=20.0):
    """Compute ||h||_2^2 / (||h||_1 * ||h*h||_inf) using soft-max."""
    h = np.maximum(h, 0.0)
    s2 = np.dot(h, h)
    s1 = np.sum(h)
    if s1 <= 0 or s2 <= 0:
        return 0.0
    acmax, c, weights = _autoconv_max(h, beta)
    if acmax <= 0:
        return 0.0
    # ||h||_2^2 = w * s2, ||h||_1 = w * s1, ||h*h||_inf ≈ acmax
    # ratio = (w*s2) / (w*s1 * acmax) = s2 / (s1 * acmax)
    return s2 / (s1 * acmax)


def _ratio_and_grad(h, beta=20.0):
    """Compute ratio and analytic gradient using soft-max smoothing."""
    h = np.maximum(h, 0.0)
    s2 = np.dot(h, h)
    s1 = np.sum(h)
    if s1 <= 0 or s2 <= 0:
        return 0.0, np.zeros_like(h)

    c = _autoconv_lags(h)
    cmax_soft, c, weights = _soft_max(c, beta)
    if cmax_soft <= 0:
        return 0.0, np.zeros_like(h)

    acmax = w * cmax_soft
    ratio = s2 / (s1 * acmax)

    # d(s2)/dh_i = 2*h_i
    dF = 2.0 * h

    # d(cmax_soft)/dh_i = sum_m weights[m] * dc[m]/dh_i
    # dc[m]/dh_i: for m=0, 2*h_i; for m>0, h[i+m] (if i+m<N) + h[i-m] (if i>=m)
    dcmax = np.zeros_like(h)
    for m in range(N):
        wm = weights[m]
        if wm < 1e-15:
            continue
        if m == 0:
            dcmax += wm * 2.0 * h
        else:
            dcmax[:N - m] += wm * h[m:]
            dcmax[m:] += wm * h[:N - m]

    # G = s1 * cmax_soft, dG = dcmax * s1 + cmax_soft * 1 (since d(s1)/dh_i = 1)
    G = s1 * cmax_soft
    dG = s1 * dcmax + cmax_soft  # d(s1*cmax)/dh_i = s1*dcmax/dh_i + cmax*1

    # ratio = s2 / (s1 * acmax) = s2 / (s1 * w * cmax_soft) = s2 / (w * G)
    # d(ratio)/dh_i = (dF_i * G - s2 * dG_i) / (w * G^2)
    grad = (dF * G - s2 * dG) / (w * G * G)

    return ratio, grad


def _objective_neg(x, beta=20.0):
    """Negative ratio for minimization with soft-max."""
    x = np.maximum(x, 0.0)
    mx = np.max(x)
    if mx <= 0:
        return 0.0, np.zeros_like(x)
    h = x / mx
    r, g = _ratio_and_grad(h, beta)
    # Chain rule for normalization: h = x/mx
    # dh_i/dx_j = (1/mx) * delta_ij - x_i/(mx^2) * (1/mx) * (1 if x_j=mx else 0)
    # Simplified: since we normalize by max, gradient in x-space:
    # For the active max coordinate, the gradient is scaled differently.
    # For simplicity, use g/mx as approximation (exact for non-max coords)
    return -r, g / mx


def _objective_neg_scalar(x, beta=20.0):
    """Scalar negative ratio with soft-max."""
    x = np.maximum(x, 0.0)
    mx = np.max(x)
    if mx <= 0:
        return 0.0
    h = x / mx
    r, _ = _ratio_and_grad(h, beta)
    return -r


def _dft_parameterize(z):
    """
    Given a complex vector z of length N, compute h = |IDFT(z)|^2.
    This ensures h >= 0 automatically.
    """
    h = np.abs(np.fft.ifft(z)) ** 2
    return h


def _dft_from_h(h):
    """
    Given h >= 0, find a complex z such that |IDFT(z)|^2 = h (approximately).
    """
    sqrt_h = np.sqrt(np.maximum(h, 0.0))
    z = np.fft.fft(sqrt_h)
    return z


def _z_objective(xz, beta=20.0):
    """Objective in z-space: -ratio(|IDFT(z)|^2)"""
    z = xz[:N] + 1j * xz[N:]
    h = _dft_parameterize(z)
    mx = np.max(h)
    if mx <= 0:
        return 0.0
    h_norm = h / mx
    r, _ = _ratio_and_grad(h_norm, beta)
    return -r


def _z_objective_grad(xz, beta=20.0):
    """Gradient of -ratio with respect to xz using analytic chain rule."""
    z = xz[:N] + 1j * xz[N:]
    h = _dft_parameterize(z)
    mx = np.max(h)
    if mx <= 0:
        return np.zeros(2 * N)
    h_norm = h / mx
    r, gh = _ratio_and_grad(h_norm, beta)

    # h_k = |y_k|^2 where y_k = (1/N) * sum_j z_j * exp(2*pi*i*j*k/N)
    # Let W_kj = (1/N) * exp(2*pi*i*j*k/N)
    # y_k = sum_j W_kj * z_j
    # h_k = |y_k|^2 = y_k * conj(y_k)
    # dh_k/d(Re(z_j)) = Re(conj(y_k) * W_kj)
    # dh_k/d(Im(z_j)) = Im(conj(y_k) * W_kj)
    
    # For the normalized h_norm = h/mx, we need d(h_norm)/d(Re(z_j))
    # = (1/mx) * dh/d(Re(z_j)) - (h/mx^2) * d(mx)/d(Re(z_j))
    # This is complex; for now, use the unnormalized gradient scaled by 1/mx
    # as an approximation, which works when mx is roughly constant.
    
    y = np.fft.ifft(z)  # y_k = (1/N) * sum_j z_j * exp(2*pi*i*j*k/N)
    # W_kj = (1/N) * exp(2*pi*i*j*k/N)
    # dh_k/d(Re(z_j)) = Re(conj(y_k) * W_kj)
    # dh_k/d(Im(z_j)) = Im(conj(y_k) * W_kj)
    
    # Build gradient efficiently:
    # grad_Re[j] = sum_k gh[k] * Re(conj(y_k) * W_kj)
    # grad_Im[j] = sum_k gh[k] * Im(conj(y_k) * W_kj)
    # = Re(sum_k gh[k] * conj(y_k) * W_kj) and Im(...)
    # Note: sum_k gh[k] * conj(y_k) * W_kj = sum_k gh[k] * conj(y_k) * (1/N) * exp(2*pi*i*j*k/N)
    # This is (1/N) * IFFT(gh * conj(y))_j evaluated at index j
    # Actually: IFFT(f)_j = (1/N) * sum_k f_k * exp(2*pi*i*j*k/N)
    # So we need IFFT(gh * conj(y))
    
    f = gh * np.conj(y)
    grad_complex = np.fft.ifft(f)  # grad_complex[j] = (1/N) * sum_k f_k * exp(2*pi*i*j*k/N)
    
    # grad_Re[j] = Re(grad_complex[j]) / mx  (approximation ignoring d(mx)/dz)
    # grad_Im[j] = Im(grad_complex[j]) / mx
    
    grad = np.zeros(2 * N)
    grad[:N] = np.real(grad_complex) / mx
    grad[N:] = np.imag(grad_complex) / mx
    
    return grad


def _optimize_z(z0, t0, max_iter=1000, beta=20.0):
    """
    Optimize in z-space: minimize -ratio(|IDFT(z)|^2) with respect to z.
    """
    xz0 = np.concatenate([z0.real, z0.imag])
    
    try:
        res = minimize(_z_objective, xz0, jac=_z_objective_grad,
                       args=(beta,),
                       method='L-BFGS-B',
                       options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-12})
        xz = res.x
        z = xz[:N] + 1j * xz[N:]
        h = _dft_parameterize(z)
        mx = np.max(h)
        if mx > 0:
            h = h / mx
            return h, _ratio(h, beta)
    except Exception:
        pass
    
    h = _dft_parameterize(z0)
    mx = np.max(h)
    if mx > 0:
        h = h / mx
    return h, _ratio(h, beta)


def _initializations():
    """Generate diverse initializations."""
    step_centers = -0.25 + (np.arange(N) + 0.5) * w
    shapes = []

    # Gaussian
    for sig in [0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3]:
        s = np.exp(-step_centers**2 / (2 * sig * sig))
        shapes.append(s)

    # Lorentzian
    for sig in [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2]:
        s = 1.0 / (1.0 + (step_centers / sig)**2)
        shapes.append(s)

    # Triangle
    for width in [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25]:
        s = np.clip(1.0 - np.abs(step_centers) / width, 0, None)
        if np.max(s) > 0:
            shapes.append(s)

    # Box
    for width in [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15]:
        s = (np.abs(step_centers) < width).astype(float)
        shapes.append(s)

    # Single step
    s = np.zeros(N); s[N // 2] = 1.0
    shapes.append(s)

    # Two symmetric steps
    for d in [1, 2, 3, 4, 5, 7, 10]:
        s = np.zeros(N)
        c = N // 2
        if c - d >= 0 and c + d < N:
            s[c - d] = 1
            s[c + d] = 1
        shapes.append(s)

    # Three symmetric steps
    for d in [2, 3, 5]:
        s = np.zeros(N)
        c = N // 2
        if c - d >= 0 and c + d < N:
            s[c - d] = 1
            s[c] = 1
            s[c + d] = 1
        shapes.append(s)

    # Power-law decay
    for p in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
        s = np.exp(-p * np.abs(step_centers))
        shapes.append(s)

    # Sine bump
    for freq in [1, 2, 3, 4, 5, 6, 8]:
        s = np.sin(freq * np.pi * (step_centers + 0.25) / 0.5)
        s = np.maximum(s, 0)
        if np.max(s) > 0:
            shapes.append(s)

    # Cosine bump
    for freq in [1, 2, 3, 4]:
        s = np.cos(freq * np.pi * step_centers / 0.25)
        s = np.maximum(s, 0)
        if np.max(s) > 0:
            shapes.append(s)

    # Smoothstep
    for width in [0.05, 0.1, 0.15, 0.2, 0.25]:
        t = np.clip((width - np.abs(step_centers)) / (2 * width), 0, 1)
        s = t * t * (3 - 2 * t)
        if np.max(s) > 0:
            shapes.append(s)

    # Bump function
    for width in [0.05, 0.08, 0.1, 0.15, 0.2]:
        t = np.abs(step_centers) / width
        s = np.where(t < 1, np.exp(-1.0 / (1.0 - t**2)), 0.0)
        if np.max(s) > 0:
            shapes.append(s)

    # Chebyshev-based
    t = 2 * step_centers
    for deg in [2, 3, 4, 5, 6, 8, 10, 12]:
        c = np.polynomial.chebyshev.chebval(t, [0] * deg + [1])
        c = np.maximum(c, 0)
        if np.max(c) > 0:
            shapes.append(c / np.max(c))

    # Random smooth
    rng = np.random.default_rng(42)
    for _ in range(15):
        freqs = rng.integers(1, 10, size=10)
        amps = rng.random(10)
        phases = rng.random(10) * 2 * np.pi
        s = np.zeros(N)
        for f, a, ph in zip(freqs, amps, phases):
            s += a * np.cos(f * 2 * np.pi * step_centers + ph)
        s = np.maximum(s, 0)
        if np.max(s) > 0:
            shapes.append(s / np.max(s))

    # Random with smoothing
    for _ in range(20):
        base = rng.random(N)
        kernel = np.exp(-np.arange(-8, 9)**2 / 8.0)
        kernel /= kernel.sum()
        base = np.convolve(base, kernel, mode='same')
        base = np.maximum(base, 0)
        if np.max(base) > 0:
            shapes.append(base / np.max(base))

    # Pure random
    for _ in range(30):
        shapes.append(rng.random(N))

    # Bimodal distributions
    for d in [0.05, 0.1, 0.15, 0.2]:
        for sig in [0.01, 0.02, 0.03, 0.05]:
            s = np.exp(-(step_centers - d)**2 / (2*sig*sig)) + np.exp(-(step_centers + d)**2 / (2*sig*sig))
            if np.max(s) > 0:
                shapes.append(s / np.max(s))

    # Sparse patterns
    for mask_size in [3, 5, 7, 10]:
        for _ in range(5):
            s = np.zeros(N)
            indices = rng.choice(N, size=mask_size, replace=False)
            s[indices] = 1.0
            # Smooth it
            kernel = np.exp(-np.arange(-3, 4)**2 / 4.0)
            kernel /= kernel.sum()
            s = np.convolve(s, kernel, mode='same')
            s = np.maximum(s, 0)
            if np.max(s) > 0:
                shapes.append(s / np.max(s))

    return shapes


def _local_optimize(h0, t0, max_iter=2000, tol=1e-12, beta=20.0):
    """Run optimization from h0."""
    if _elapsed() > DEADLINE - 5:
        return h0, _ratio(h0, beta)

    h0 = np.array(h0, dtype=float)
    mx = np.max(h0)
    if mx <= 0:
        return h0, 0.0
    h0 = h0 / mx

    bounds = [(0.0, None)] * N

    # Try L-BFGS-B
    try:
        res = minimize(_objective_neg_scalar, h0, args=(beta,),
                       method='L-BFGS-B',
                       bounds=bounds,
                       options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-10, 'maxls': 500})
        x = np.maximum(res.x, 0.0)
        mx2 = np.max(x)
        if mx2 > 0:
            h2 = x / mx2
            f2 = _ratio(h2, beta)
            f1 = _ratio(h0, beta)
            if f2 > f1:
                return h2, f2
    except Exception:
        pass

    # Try trust-constr
    try:
        res = minimize(_objective_neg, h0, args=(beta,), method='trust-constr',
                       bounds=bounds,
                       options={'maxiter': max_iter, 'gtol': tol, 'xtol': 1e-14})
        x = np.maximum(res.x, 0.0)
        mx2 = np.max(x)
        if mx2 > 0:
            h2 = x / mx2
            f2 = _ratio(h2, beta)
            f1 = _ratio(h0, beta)
            if f2 > f1:
                return h2, f2
    except Exception:
        pass

    return h0, _ratio(h0, beta)


def _dft_optimize(h0, t0, max_iter=1000, beta=20.0):
    """Optimize using DFT parameterization."""
    if _elapsed() > DEADLINE - 5:
        return h0, _ratio(h0, beta)

    h0 = np.array(h0, dtype=float)
    mx = np.max(h0)
    if mx <= 0:
        return h0, 0.0
    h0 = h0 / mx

    # Convert to z-space
    z0 = _dft_from_h(h0)
    
    h, f = _optimize_z(z0, t0, max_iter=max_iter, beta=beta)
    f1 = _ratio(h0, beta)
    if f > f1:
        return h, f
    return h0, f1


def _polish(h, t0, beta=20.0):
    """High-precision final polish."""
    if _elapsed() > DEADLINE - 3:
        return h, _ratio(h, beta)

    h = np.array(h, dtype=float)
    mx = np.max(h)
    if mx <= 0:
        return h, 0.0
    h = h / mx

    bounds = [(0.0, None)] * N

    # SLSQP
    try:
        res = minimize(_objective_neg_scalar, h, args=(beta,),
                       method='SLSQP',
                       bounds=bounds,
                       options={'maxiter': 1000, 'ftol': 1e-15, 'eps': 1e-10})
        x = np.maximum(res.x, 0.0)
        mx2 = np.max(x)
        if mx2 > 0:
            h2 = x / mx2
            f2 = _ratio(h2, beta)
            f1 = _ratio(h, beta)
            if f2 > f1:
                h = h2
    except Exception:
        pass

    # L-BFGS-B tight
    try:
        res = minimize(_objective_neg_scalar, h, args=(beta,),
                       method='L-BFGS-B',
                       bounds=bounds,
                       options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 500})
        x = np.maximum(res.x, 0.0)
        mx2 = np.max(x)
        if mx2 > 0:
            h2 = x / mx2
            f2 = _ratio(h2, beta)
            f1 = _ratio(h, beta)
            if f2 > f1:
                h = h2
    except Exception:
        pass

    # DFT polish
    if _elapsed() < DEADLINE - 3:
        h, f = _dft_optimize(h, t0, max_iter=500, beta=beta)

    return h, _ratio(h, beta)


def _coordinate_refine(h, t0, max_rounds=10, beta=20.0):
    """Coordinate descent refinement."""
    h = np.array(h, dtype=float)
    best_f = _ratio(h, beta)

    for rnd in range(max_rounds):
        if _elapsed() > DEADLINE - 8:
            break
        improved = False
        for i in range(N):
            if _elapsed() > DEADLINE - 8:
                break
            orig = h[i]
            for delta in [0.001, -0.001, 0.005, -0.005, 0.01, -0.01,
                          0.02, -0.02, 0.05, -0.05, 0.1, -0.1]:
                h[i] = max(0.0, orig + delta)
                f = _ratio(h, beta)
                if f > best_f:
                    best_f = f
                    improved = True
                    orig = h[i]
                else:
                    h[i] = orig
        if not improved:
            break

    return h, best_f


def construct():
    t0 = time.time()
    best_h = np.ones(N)
    best_f = _ratio(best_h, 20.0)

    starts = _initializations()

    # Normalize all starts
    norm_starts = []
    for s in starts:
        s = np.array(s, dtype=float)
        mx = np.max(s)
        if mx > 0:
            norm_starts.append(s / mx)

    # Phase 1: Quick screening with moderate beta
    for idx, s in enumerate(norm_starts):
        if _elapsed() > DEADLINE - 50:
            break
        h, f = _local_optimize(s, t0, max_iter=300, tol=1e-10, beta=20.0)
        if f > best_f:
            best_f = f
            best_h = h.copy()

    # Phase 2: DFT-based optimization on best
    if _elapsed() < DEADLINE - 30:
        h, f = _dft_optimize(best_h, t0, max_iter=1000, beta=20.0)
        if f > best_f:
            best_f = f
            best_h = h.copy()

    # Phase 3: Refine best with more iterations, higher beta for sharper peak
    if _elapsed() < DEADLINE - 25:
        best_h, best_f = _local_optimize(best_h, t0, max_iter=2000, tol=1e-13, beta=50.0)

    # Phase 4: DFT refinement on refined best with high beta
    if _elapsed() < DEADLINE - 20:
        h, f = _dft_optimize(best_h, t0, max_iter=1000, beta=50.0)
        if f > best_f:
            best_f = f
            best_h = h.copy()

    # Phase 5: Coordinate descent with high beta
    if _elapsed() < DEADLINE - 10:
        best_h, best_f = _coordinate_refine(best_h, t0, max_rounds=10, beta=100.0)

    # Phase 6: High-precision polish with very high beta (near-hard max)
    if _elapsed() < DEADLINE - 5:
        best_h, best_f = _polish(best_h, t0, beta=100.0)

    # Phase 7: Random restarts from perturbed best
    if _elapsed() < DEADLINE - 15:
        rng = np.random.default_rng(999)
        for _ in range(10):
            if _elapsed() > DEADLINE - 10:
                break
            h_pert = best_h + rng.normal(0, 0.05, N)
            h_pert = np.maximum(h_pert, 0)
            mx = np.max(h_pert)
            if mx <= 0:
                continue
            h_pert = h_pert / mx
            h, f = _local_optimize(h_pert, t0, max_iter=500, tol=1e-12, beta=50.0)
            if f > best_f:
                best_f = f
                best_h = h.copy()

    # Final polish with very high beta
    if _elapsed() < DEADLINE - 3:
        best_h, best_f = _polish(best_h, t0, beta=100.0)

    best_h = np.maximum(best_h, 0.0)
    return [float(v) for v in best_h]
