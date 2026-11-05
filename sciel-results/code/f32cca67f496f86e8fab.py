"""Rewritten program for third autocorrelation inequality optimization."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.fft import dct, idct, fft, ifft
from scipy.optimize import minimize

def _eval_value(h):
    """Compute the objective value."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return np.inf
    conv = _convolve_fft(h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _convolve_fft(h):
    """Compute full autoconvolution using FFT."""
    n_conv = 2 * len(h) - 1
    n_fft = 1
    while n_fft < n_conv:
        n_fft *= 2
    H = fft(h, n=n_fft)
    conv = ifft(H * H, n=n_fft).real[:n_conv]
    return conv

def _soft_max(x, beta):
    """Numerically stable soft maximum."""
    m = np.max(x)
    if not np.isfinite(m):
        return m
    exp_vals = np.exp(beta * (x - m))
    sm = m + np.log(np.sum(exp_vals)) / beta
    return sm

def _objective_soft(h, beta):
    """Soft-max surrogate objective. h is normalized internally so sum(h) = N."""
    if not np.all(np.isfinite(h)):
        return 1e10
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    h_norm = h * (N / s)
    conv = _convolve_fft(h_norm)
    sm = _soft_max(conv, beta)
    return abs(2.0 * N * sm / (N * N))

def _objective_exact(h):
    """Exact objective for comparison."""
    if not np.all(np.isfinite(h)):
        return 1e10
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    h_norm = h * (N / s)
    conv = _convolve_fft(h_norm)
    return abs(2.0 * N * np.max(conv) / (N * N))

def construct():
    """
    Returns a sequence of 400 signed step heights.
    Uses FFT-based convolution, soft-max surrogate optimization,
    and coordinate descent polish.
    """
    deadline = time.time() + 130.0  # safety margin before 150s kill

    # Start with all-ones (value = 2.0)
    best_h = np.ones(N)
    best_val = 2.0

    # Phase 1: Quick parametric families to get a good starting point
    idx = np.arange(N)

    # Family 1: h[i] = 1 + a*cos(2*pi*m*i/N + phase)
    for m in range(1, 31):
        if time.time() > deadline - 90:
            break
        for a in np.linspace(-1.0, 1.0, 21):
            for phase in [0.0, np.pi / 4, np.pi / 2]:
                h = 1.0 + a * np.cos(2.0 * np.pi * m * idx / N + phase)
                s = np.sum(h)
                if abs(s) < 1e-6:
                    continue
                v = _eval_value(h)
                if v < best_val:
                    best_val = v
                    best_h = h.copy()

    # Family 2: Square wave / sign patterns
    for m in range(1, 21):
        if time.time() > deadline - 80:
            break
        for a in np.linspace(0.5, 2.0, 11):
            h = 1.0 + a * np.sign(np.sin(2.0 * np.pi * m * idx / N))
            s = np.sum(h)
            if abs(s) < 1e-6:
                continue
            v = _eval_value(h)
            if v < best_val:
                best_val = v
                best_h = h.copy()

    # Family 3: Smoothed step / tanh
    for beta in [0.5, 1.0, 2.0, 3.0, 5.0, 8.0]:
        if time.time() > deadline - 75:
            break
        for shift in np.linspace(-N/4, N/4, 11):
            h = np.tanh(beta * (idx - N / 2.0 + shift))
            s = np.sum(h)
            if abs(s) < 1e-6:
                continue
            v = _eval_value(h)
            if v < best_val:
                best_val = v
                best_h = h.copy()

    # Family 4: Two-block pattern
    for frac in np.linspace(0.1, 0.9, 17):
        if time.time() > deadline - 70:
            break
        for a in np.linspace(0.5, 3.0, 11):
            h = np.ones(N)
            h[:int(frac * N)] = a
            s = np.sum(h)
            if abs(s) < 1e-6:
                continue
            v = _eval_value(h)
            if v < best_val:
                best_val = v
                best_h = h.copy()

    # Phase 2: Soft-max surrogate optimization with L-BFGS-B
    # Multiple beta values, multiple restarts
    rng = np.random.RandomState(123)

    beta_schedule = [2.0, 5.0, 10.0, 20.0, 50.0]
    n_restarts = 5

    for beta in beta_schedule:
        if time.time() > deadline - 25:
            break

        def obj_time(h):
            return _objective_soft(h, beta)

        for trial in range(n_restarts):
            if time.time() > deadline - 20:
                break

            # Start from best_h or perturbed version
            if trial == 0:
                h_init = best_h.copy()
            elif trial == 1:
                h_init = best_h + 0.05 * rng.randn(N)
                h_init = h_init * (N / np.sum(h_init))
            elif trial == 2:
                h_init = best_h + 0.1 * rng.randn(N)
                h_init = h_init * (N / np.sum(h_init))
            elif trial == 3:
                # Random restart with structure
                h_init = 1.0 + 0.5 * rng.randn(N)
                h_init = h_init * (N / np.sum(h_init))
            else:
                # DCT-based random restart
                c = dct(best_h, type=2, norm='ortho')
                c[1:] += 0.1 * rng.randn(N - 1)
                h_init = idct(c, type=2, norm='ortho')
                h_init = h_init * (N / np.sum(h_init))

            try:
                result = minimize(
                    obj_time,
                    h_init,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12}
                )

                if result.fun < best_val:
                    h_opt = result.x
                    v_opt = _eval_value(h_opt)
                    if v_opt < best_val and np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-10:
                        best_val = v_opt
                        best_h = h_opt.copy()
            except Exception:
                continue

    # Phase 3: Coordinate descent polish in time domain
    if time.time() < deadline - 10:
        h = best_h.copy()
        step_sizes = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001]

        for step in step_sizes:
            if time.time() > deadline - 5:
                break
            improved = True
            while improved:
                improved = False
                for i in range(N):
                    if time.time() > deadline - 5:
                        break
                    old = h[i]
                    for delta in [step, -step]:
                        h[i] = old + delta
                        s = np.sum(h)
                        if abs(s) < 1e-10:
                            h[i] = old
                            continue
                        v = _eval_value(h)
                        if v < best_val:
                            best_val = v
                            best_h = h.copy()
                            improved = True
                        else:
                            h[i] = old
                    h[i] = old

    # Phase 4: Final soft-max polish with very high beta
    if time.time() < deadline - 5:
        beta_final = 100.0

        def obj_final(h):
            return _objective_soft(h, beta_final)

        try:
            result = minimize(
                obj_final,
                best_h,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-13}
            )
            if result.fun < best_val:
                h_opt = result.x
                v_opt = _eval_value(h_opt)
                if v_opt < best_val and np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-10:
                    best_val = v_opt
                    best_h = h_opt.copy()
        except Exception:
            pass

    # Ensure finite and sum != 0
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        return np.ones(N)

    return best_h
# EVOLVE-BLOCK-END
