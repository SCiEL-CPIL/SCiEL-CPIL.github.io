import time
import numpy as np
from scipy.optimize import minimize

N = 50
w = 0.5 / N  # step width = 0.01
_START = time.time()
DEADLINE = 145.0  # safety margin before 150s kill


def _elapsed():
    return time.time() - _START


def _autoconv_lags(h):
    """Compute all lag values of autoconvolution (without the w factor).
    c[m] = sum_j h[j]*h[j+m] for m=0..N-1
    """
    c = np.zeros(N)
    c[0] = np.dot(h, h)
    for m in range(1, N):
        c[m] = np.dot(h[:N - m], h[m:])
    return c


def _ratio(h):
    """Compute ||h||_2^2 / (||h||_1 * ||h*h||_inf)."""
    h = np.maximum(h, 0.0)
    s2 = np.dot(h, h)
    s1 = np.sum(h)
    if s1 <= 0 or s2 <= 0:
        return 0.0
    c = _autoconv_lags(h)
    acmax = w * c.max()
    if acmax <= 0:
        return 0.0
    return s2 / (s1 * acmax)


def _soft_max(c, beta):
    """Soft-max approximation of max(c)."""
    c = c - c.max()
    return c.max() + np.log(np.sum(np.exp(beta * c))) / beta


def _ratio_soft(h, beta=50.0):
    """Compute ratio with soft-max smoothing of the peak."""
    h = np.maximum(h, 0.0)
    s2 = np.dot(h, h)
    s1 = np.sum(h)
    if s1 <= 0 or s2 <= 0:
        return 0.0
    c = _autoconv_lags(h)
    cmax_soft = _soft_max(c, beta)
    acmax = w * cmax_soft
    if acmax <= 0:
        return 0.0
    return s2 / (s1 * acmax)


def _ratio_and_grad_soft(h, beta=50.0):
    """Compute ratio and analytic gradient with soft-max smoothing."""
    h = np.maximum(h, 0.0)
    s2 = np.dot(h, h)
    s1 = np.sum(h)
    if s1 <= 0 or s2 <= 0:
        return 0.0, np.zeros_like(h)

    c = _autoconv_lags(h)
    cmax_soft = _soft_max(c, beta)
    if cmax_soft <= 0:
        return 0.0, np.zeros_like(h)

    acmax = w * cmax_soft
    ratio = s2 / (s1 * acmax)

    # Gradient of soft-max: softmax weights
    c_shifted = c - c.max()
    weights = np.exp(beta * c_shifted)
    weights /= weights.sum()

    dF = 2.0 * h  # d(s2)/dh_i

    # d(c_max_soft)/dh_i = sum_m weights[m] * dc[m]/dh_i
    dcmax = np.zeros_like(h)
    for m in range(N):
        if weights[m] > 1e-15:
            if m == 0:
                dcmax += weights[m] * 2.0 * h
            else:
                dcmax[:N - m] += weights[m] * h[m:]
                dcmax[m:] += weights[m] * h[:N - m]

    dG = cmax_soft + s1 * dcmax
    G = s1 * cmax_soft
    grad = (dF * G - s2 * dG) / (w * G * G)

    return ratio, grad


def _objective_neg_soft(x, beta=50.0):
    """Scalar negative ratio with soft-max."""
    x = np.maximum(x, 0.0)
    mx = np.max(x)
    if mx <= 0:
        return 0.0
    h = x / mx
    r = _ratio_soft(h, beta)
    return -r


def _dft_parameterize(z):
    """Compute h = |FFT(z)|^2 / max(|FFT(z)|^2) from complex vector z."""
    H = np.fft.fft(z)
    h = np.abs(H) ** 2
    mx = np.max(h)
    if mx <= 0:
        return np.zeros(N)
    return h / mx


def _ratio_from_z(z, beta=50.0):
    """Compute ratio from complex parameter z via DFT."""
    h = _dft_parameterize(z)
    if np.max(h) <= 0:
        return 0.0
    return _ratio_soft(h, beta)


def _obj_z_neg(x, beta=50.0):
    """Negative ratio as function of real-flattened z."""
    z = x[:N] + 1j * x[N:]
    return -_ratio_from_z(z, beta)


def _initializations():
    """Generate diverse parametric initializations."""
    step_centers = -0.25 + (np.arange(N) + 0.5) * w
    shapes = []

    # Gaussian with various widths
    for sig in [0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3]:
        s = np.exp(-step_centers**2 / (2 * sig * sig))
        shapes.append(s)

    # Lorentzian / Cauchy
    for sig in [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2]:
        s = 1.0 / (1.0 + (step_centers / sig)**2)
        shapes.append(s)

    # Triangle
    for width in [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25]:
        s = np.clip(1.0 - np.abs(step_centers) / width, 0, None)
        if np.max(s) > 0:
            shapes.append(s)

    # Box / plateau
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

    # Cosine bump (centered)
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

    # Bump function (compact support, smooth)
    for width in [0.05, 0.08, 0.1, 0.15, 0.2]:
        t = np.abs(step_centers) / width
        s = np.where(t < 1, np.exp(-1.0 / (1.0 - t**2)), 0.0)
        if np.max(s) > 0:
            shapes.append(s)

    # Polynomial (Chebyshev-based)
    t = 2 * step_centers  # map to [-0.5, 0.5]
    for deg in [2, 3, 4, 5, 6, 8, 10, 12]:
        c = np.polynomial.chebyshev.chebval(t, [0] * deg + [1])
        c = np.maximum(c, 0)
        if np.max(c) > 0:
            shapes.append(c / np.max(c))

    # Random smooth shapes
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

    # One-sided concentrated shapes
    for alpha in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 8.0, 10.0]:
        s = np.exp(-alpha * (step_centers + 0.25))
        shapes.append(s)
        s2 = np.exp(-alpha * (0.25 - step_centers))
        shapes.append(s2)

    # Power-law one-sided
    for p in [1.0, 2.0, 3.0, 5.0, 8.0]:
        s = np.maximum(0.25 - step_centers, 0.01) ** (-p)
        s = s / np.max(s)
        shapes.append(s)
        s2 = np.maximum(step_centers + 0.25, 0.01) ** (-p)
        s2 = s2 / np.max(s2)
        shapes.append(s2)

    # Tapered plateau (raised cosine window)
    for width in [0.05, 0.1, 0.15, 0.2, 0.25]:
        s = np.where(np.abs(step_centers) < width,
                     0.5 * (1 + np.cos(np.pi * step_centers / width)), 0.0)
        if np.max(s) > 0:
            shapes.append(s / np.max(s))

    return shapes


def _local_optimize_soft(h0, max_iter=2000, beta=50.0):
    """Run optimization with soft-max objective from h0."""
    if _elapsed() > DEADLINE - 5:
        return h0, _ratio(h0)

    h0 = np.array(h0, dtype=float)
    mx = np.max(h0)
    if mx <= 0:
        return h0, 0.0
    h0 = h0 / mx

    bounds = [(0.0, None)] * N

    # Try L-BFGS-B with soft-max objective
    try:
        res = minimize(lambda x: _objective_neg_soft(x, beta), h0, method='L-BFGS-B',
                       bounds=bounds,
                       options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 200})
        x = np.maximum(res.x, 0.0)
        mx2 = np.max(x)
        if mx2 > 0:
            h2 = x / mx2
            f2 = _ratio(h2)
            f1 = _ratio(h0)
            if f2 > f1:
                return h2, f2
    except Exception:
        pass

    # Try SLSQP
    try:
        res = minimize(lambda x: _objective_neg_soft(x, beta), h0, method='SLSQP',
                       bounds=bounds,
                       options={'maxiter': 1000, 'ftol': 1e-14, 'eps': 1e-8})
        x = np.maximum(res.x, 0.0)
        mx2 = np.max(x)
        if mx2 > 0:
            h2 = x / mx2
            f2 = _ratio(h2)
            f1 = _ratio(h0)
            if f2 > f1:
                return h2, f2
    except Exception:
        pass

    return h0, _ratio(h0)


def _spectral_optimize_soft(h0, max_iter=800, beta=50.0):
    """
    Optimize over DFT parameterization z with soft-max objective.
    h = |FFT(z)|^2 / max, ensuring non-negativity.
    """
    if _elapsed() > DEADLINE - 5:
        return h0, _ratio(h0)

    h0 = np.array(h0, dtype=float)
    mx = np.max(h0)
    if mx <= 0:
        return h0, 0.0
    h0 = h0 / mx

    # Initialize z: z = IFFT(sqrt(h0))
    sqrt_h = np.sqrt(h0)
    z_init = np.fft.ifft(sqrt_h)

    z0_real = np.concatenate([z_init.real, z_init.imag])

    try:
        res = minimize(lambda x: _obj_z_neg(x, beta), z0_real, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 100})
        x = res.x
        z = x[:N] + 1j * x[N:]
        h2 = _dft_parameterize(z)
        f2 = _ratio(h2)
        f1 = _ratio(h0)
        if f2 > f1:
            return h2, f2
    except Exception:
        pass

    return h0, _ratio(h0)


def _coordinate_refine(h, max_rounds=20):
    """Coordinate descent refinement."""
    h = np.array(h, dtype=float)
    best_f = _ratio(h)

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
                f = _ratio(h)
                if f > best_f:
                    best_f = f
                    improved = True
                    orig = h[i]
                else:
                    h[i] = orig
        if not improved:
            break

    return h, best_f


def _polish(h):
    """High-precision final polish."""
    if _elapsed() > DEADLINE - 3:
        return h, _ratio(h)

    h = np.array(h, dtype=float)
    mx = np.max(h)
    if mx <= 0:
        return h, 0.0
    h = h / mx

    bounds = [(0.0, None)] * N

    # Try with very high beta (nearly exact max)
    for beta in [100.0, 200.0]:
        try:
            res = minimize(lambda x: _objective_neg_soft(x, beta), h, method='L-BFGS-B',
                           bounds=bounds,
                           options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 500})
            x = np.maximum(res.x, 0.0)
            mx2 = np.max(x)
            if mx2 > 0:
                h2 = x / mx2
                f2 = _ratio(h2)
                f1 = _ratio(h)
                if f2 > f1:
                    h = h2
        except Exception:
            pass

    return h, _ratio(h)


def construct():
    t0 = time.time()
    best_h = np.ones(N)
    best_f = _ratio(best_h)

    starts = _initializations()

    # Normalize all starts
    norm_starts = []
    for s in starts:
        s = np.array(s, dtype=float)
        mx = np.max(s)
        if mx > 0:
            norm_starts.append(s / mx)

    # Phase 1: Quick screening with soft-max optimization
    for idx, s in enumerate(norm_starts):
        if _elapsed() > DEADLINE - 45:
            break
        h, f = _local_optimize_soft(s, max_iter=300, beta=30.0)
        if f > best_f:
            best_f = f
            best_h = h.copy()

    # Phase 2: Refine best with higher beta
    if _elapsed() < DEADLINE - 25:
        best_h, best_f = _local_optimize_soft(best_h, max_iter=1500, beta=50.0)

    # Phase 3: Spectral optimization (DFT parameterization)
    if _elapsed() < DEADLINE - 20:
        h_spec, f_spec = _spectral_optimize_soft(best_h, max_iter=800, beta=50.0)
        if f_spec > best_f:
            best_f = f_spec
            best_h = h_spec.copy()

    # Phase 4: Coordinate descent refinement
    if _elapsed() < DEADLINE - 10:
        best_h, best_f = _coordinate_refine(best_h, max_rounds=15)

    # Phase 5: High-precision polish
    if _elapsed() < DEADLINE - 5:
        best_h, best_f = _polish(best_h)

    # Phase 6: Random restarts from perturbed best
    if _elapsed() < DEADLINE - 12:
        rng = np.random.default_rng(999)
        for _ in range(6):
            if _elapsed() > DEADLINE - 6:
                break
            h_pert = best_h + rng.normal(0, 0.05, N)
            h_pert = np.maximum(h_pert, 0)
            mx = np.max(h_pert)
            if mx <= 0:
                continue
            h_pert = h_pert / mx
            h, f = _local_optimize_soft(h_pert, max_iter=600, beta=50.0)
            if f > best_f:
                best_f = f
                best_h = h.copy()

    # Final polish
    if _elapsed() < DEADLINE - 3:
        best_h, best_f = _polish(best_h)

    best_h = np.maximum(best_h, 0.0)
    return [float(v) for v in best_h]
