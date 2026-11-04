"""
Signed autocorrelation max-abs variant — combined sparse/spectral + gradient polish.

Strategy:
  * Low-dimensional parameterization (DC + sparse spikes + sinusoidal tones)
    for global search via differential evolution.
  * Proper FFT-based gradient computation for the smooth soft-max surrogate.
  * Frequency-domain spectral flatness penalty to encourage uniform |X(k)|².
  * Annealed L-BFGS-B + gradient descent polish.
  * Explicit deadline guard.
"""

import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 400
DEADLINE = 138.0
_start = time.time()


def _elapsed():
    return time.time() - _start


def _autoconv(x):
    """Linear (full) autoconvolution of x, lags 0..2N-2 (circular padded)."""
    n = len(x)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    X = np.fft.rfft(x, fsize)
    c = np.fft.irfft(X * X, fsize)[:size]
    return c


def _score(x):
    """max|c_k| / (sum x)^2, including k=0."""
    s = float(np.sum(x))
    if abs(s) < 1e-12:
        return 1e12
    ac = _autoconv(x)
    return float(np.max(np.abs(ac))) / (s * s)


def _score_offdiag(x):
    """max|c_k| for k!=0 / (sum x)^2."""
    s = float(np.sum(x))
    if abs(s) < 1e-12:
        return 1e12
    ac = _autoconv(x)
    if len(ac) > 1:
        return float(np.max(np.abs(ac[1:]))) / (s * s)
    return 0.0


def _norm(x):
    s = float(np.sum(x))
    if abs(s) < 1e-12:
        return x
    return x / s


def _sparse_to_signal(p, n_spikes):
    x = np.full(N, p[0])
    for j in range(n_spikes):
        pos = int(round(p[1 + 2 * j])) % N
        mag = p[2 + 2 * j]
        x[pos] += mag
    return x


def _sine_to_signal(p, K):
    i = np.arange(N)
    x = np.full(N, p[0])
    for j in range(K):
        base = 1 + 3 * j
        f = p[base]
        ph = p[base + 1]
        a = p[base + 2]
        x += a * np.cos(2.0 * np.pi * f * i / N + ph)
    return x


def _mixed_to_signal(p, n_spikes, K):
    x = np.full(N, p[0])
    idx = 1
    for j in range(n_spikes):
        pos = int(round(p[idx])) % N
        mag = p[idx + 1]
        x[pos] += mag
        idx += 2
    i = np.arange(N)
    for j in range(K):
        f = p[idx]
        ph = p[idx + 1]
        a = p[idx + 2]
        x += a * np.cos(2.0 * np.pi * f * i / N + ph)
        idx += 3
    return x


def _de_objective_sparse(p, n_spikes):
    x = _sparse_to_signal(p, n_spikes)
    x = _norm(x)
    return _score(x)


def _de_objective_sine(p, K):
    x = _sine_to_signal(p, K)
    x = _norm(x)
    return _score(x)


def _de_objective_mixed(p, n_spikes, K):
    x = _mixed_to_signal(p, n_spikes, K)
    x = _norm(x)
    return _score(x)


def _spectral_flatness_penalty(x, weight=0.1):
    """Penalty for non-uniform |X(k)|² across non-DC bins."""
    fsize = 1
    while fsize < 2 * N:
        fsize *= 2
    X = np.fft.rfft(x, fsize)
    power = np.abs(X) ** 2
    # Exclude DC (index 0) and Nyquist if present
    non_dc = power[1:]
    if len(non_dc) < 2:
        return 0.0
    mean_p = np.mean(non_dc)
    if mean_p < 1e-15:
        return 0.0
    # Coefficient of variation squared as flatness penalty
    var_p = np.var(non_dc)
    cv2 = var_p / (mean_p * mean_p)
    return weight * cv2


def _softmax_abs_ac(x, tau):
    """Smooth soft-max of |c_k| for k!=0, divided by (sum x)^2."""
    ac = _autoconv(x)
    if len(ac) <= 1:
        return 0.0
    a = np.abs(ac[1:])
    amax = float(np.max(a))
    if amax < 1e-14:
        return 0.0
    sm = amax + tau * np.log(np.sum(np.exp((a - amax) / tau)))
    s = float(np.sum(x))
    if abs(s) < 1e-12:
        return 1e12
    return sm / (s * s)


def _smooth_objective(x, tau=0.5, flat_weight=0.05):
    """Combined smooth objective: soft-max autocorr + spectral flatness."""
    base = _softmax_abs_ac(x, tau)
    flat = _spectral_flatness_penalty(x, weight=flat_weight)
    return base + flat


def _gradient_smooth(x, tau=0.5, flat_weight=0.05):
    """
    Compute gradient of the smooth objective w.r.t. x.
    Uses FFT-based cross-correlation for the autoconvolution gradient.
    """
    s = float(np.sum(x))
    if abs(s) < 1e-12:
        return np.zeros(N), 1e12

    ac = _autoconv(x)
    if len(ac) <= 1:
        return np.zeros(N), 0.0

    abs_c = np.abs(ac[1:])
    amax = float(np.max(abs_c))
    if amax < 1e-14:
        grad_ac = np.zeros(N)
        soft_max = 0.0
    else:
        exp_vals = np.exp((abs_c - amax) / tau)
        sum_exp = np.sum(exp_vals)
        soft_max = amax + tau * np.log(sum_exp)
        # Gradient of soft_max w.r.t. abs_c
        grad_abs_c = exp_vals / sum_exp
        # Chain rule: d|c|/dc = sign(c)
        c_off = ac[1:]
        sign_c = np.sign(c_off)
        sign_c[np.abs(c_off) < 1e-15] = 0.0
        grad_c = grad_abs_c * sign_c

        # Build weight vector for cross-correlation
        # c_k = sum_i x_i * x_{i+k}, so dc_k/dx_j involves x_{j-k} + x_{j+k}
        # grad_x[j] = sum_k grad_c[k-1] * (x[j-k] + x[j+k])
        # Use FFT: grad_x = ifft(fft(w) * conj(fft(x))) where w is the weight sequence
        fsize = 1
        while fsize < 2 * N:
            fsize *= 2

        # The autoconvolution c_k for k=0..2N-2:
        # c_k = sum_i x_i x_{i+k} for k>=0 (with appropriate bounds)
        # For the gradient, we need the cross-correlation of x with the weight vector
        # w[m] where m indexes the lag

        # grad_c has length 2N-2 (lags 1..2N-2)
        # We need to compute sum_k grad_c[k-1] * x[j-k] and sum_k grad_c[k-1] * x[j+k]
        # This is equivalent to cross-correlation

        # Build full weight vector of length 2N-1
        w_full = np.zeros(2 * N - 1)
        w_full[1:] = grad_c  # lags 1..2N-2

        # Cross-correlation: (x * w)[j] = sum_k w[k] x[j-k]
        # Using FFT: ifft(fft(w) * conj(fft(x)))
        W = np.fft.rfft(w_full, fsize)
        X = np.fft.rfft(x, fsize)
        # For cross-correlation, we need ifft(conj(W) * X) or ifft(W * conj(X))
        # Let's use: grad_x = ifft(conj(fft(w_full)) * fft(x))
        grad_x_corr = np.fft.irfft(np.conj(W) * X, fsize)[:N]

        # Also need the "other side": sum_k w[k] x[j+k]
        # This is ifft(fft(w_full) * conj(fft(x)))
        grad_x_corr2 = np.fft.irfft(W * np.conj(X), fsize)[:N]

        grad_ac = grad_x_corr + grad_x_corr2

    # Gradient of soft_max / s^2
    # d/dx [soft_max / s^2] = grad_soft_max / s^2 - 2*soft_max*s^{-3} * ones
    grad_metric = grad_ac / (s * s) - 2.0 * soft_max * np.ones(N) / (s ** 3)

    # Add spectral flatness gradient (approximate)
    if flat_weight > 0:
        # Numerical gradient for spectral flatness (expensive but accurate)
        fsize = 1
        while fsize < 2 * N:
            fsize *= 2
        X = np.fft.rfft(x, fsize)
        power = np.abs(X) ** 2
        non_dc = power[1:]
        if len(non_dc) >= 2:
            mean_p = np.mean(non_dc)
            if mean_p > 1e-15:
                # d/dx_j of var(non_dc) = 2 * sum_k (power_k - mean_p) * d(power_k)/dx_j
                # d(power_k)/dx_j = 2 * Re(X_k * conj(dX_k/dx_j))
                # dX_k/dx_j = exp(-2*pi*i*k*j/fsize)
                dev = non_dc - mean_p
                # For each frequency bin k, d(power_k)/dx_j = 2 * Re(X_k * conj(exp(-2*pi*i*k*j/fsize)))
                # = 2 * Re(X_k * exp(2*pi*i*k*j/fsize))
                # Sum over k: sum_k dev[k] * 2 * Re(X_{k+1} * exp(2*pi*i*(k+1)*j/fsize))
                # This is a Fourier sum, computable via IFFT
                # Let Y_k = dev[k-1] * X_k for k=1..fsize/2
                Y = np.zeros(fsize // 2 + 1, dtype=complex)
                for k in range(1, min(len(dev) + 1, fsize // 2 + 1)):
                    Y[k] = dev[k - 1] * X[k]
                # grad_flat_j = 2 * Re(ifft(Y)[j]) * 2 (from chain rule)
                grad_flat_vec = 2.0 * np.fft.irfft(Y, fsize)[:N]
                grad_metric += flat_weight * grad_flat_vec

    return grad_metric, soft_max / (s * s)


def _gradient_descent_polish(x0, max_iter=500, initial_lr=0.01, tau=0.5,
                             flat_weight=0.05, deadline=None):
    """Gradient descent with annealed temperature and learning rate."""
    x = x0.copy().astype(float)
    s = np.sum(x)
    if abs(s) < 1e-12:
        return x
    x = x / s

    best_x = x.copy()
    best_val = _score(best_x)

    lr = initial_lr
    temp = tau

    for i in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break

        grad, obj_val = _gradient_smooth(x, tau=temp, flat_weight=flat_weight)

        # Check if gradient is small
        if np.max(np.abs(grad)) < 1e-10:
            break

        x_new = x - lr * grad

        # Project: normalize to sum=1
        s_new = np.sum(x_new)
        if abs(s_new) < 1e-12:
            x_new = x + 1e-6 * np.random.randn(N)
            s_new = np.sum(x_new)
        x_new = x_new / s_new

        val = _score(x_new)
        if val < best_val:
            best_val = val
            best_x = x_new.copy()

        x = x_new
        lr *= 0.998
        temp = max(0.1, temp * 0.999)

    return best_x


def _lbfgs_polish(x0, max_iter=300, tau=0.5, flat_weight=0.05, deadline=None):
    """L-BFGS-B polish with smooth objective."""
    best_x = x0.copy()
    best_sc = _score(x0)

    def obj(x):
        if deadline is not None and time.time() > deadline:
            return best_sc
        return _smooth_objective(x, tau=tau, flat_weight=flat_weight)

    try:
        res = minimize(obj, x0, method='L-BFGS-B', maxiter=max_iter,
                       options={'maxls': 30, 'ftol': 1e-12, 'gtol': 1e-9})
        if res.fun < best_sc:
            cand = res.x
            sc = _score(cand)
            if sc < best_sc:
                best_x = cand
                best_sc = sc
    except Exception:
        pass

    return best_x


def _local_polish(x0, max_iter=300, tau=0.5, deadline=None):
    """Combined L-BFGS-B + gradient descent polish."""
    best_x = x0.copy()
    best_sc = _score(x0)

    # Phase 1: L-BFGS-B with higher temperature
    if deadline is None or _elapsed() < deadline - 15:
        x1 = _lbfgs_polish(x0, max_iter=150, tau=tau,
                           flat_weight=0.05, deadline=deadline)
        sc1 = _score(x1)
        if sc1 < best_sc:
            best_sc = sc1
            best_x = x1

    # Phase 2: L-BFGS-B with lower temperature
    if deadline is None or _elapsed() < deadline - 10:
        x2 = _lbfgs_polish(best_x, max_iter=200, tau=max(0.1, tau * 0.3),
                           flat_weight=0.02, deadline=deadline)
        sc2 = _score(x2)
        if sc2 < best_sc:
            best_sc = sc2
            best_x = x2

    # Phase 3: Gradient descent with annealing
    if deadline is None or _elapsed() < deadline - 5:
        x3 = _gradient_descent_polish(best_x, max_iter=300, initial_lr=0.005,
                                      tau=max(0.1, tau * 0.3),
                                      flat_weight=0.02, deadline=deadline)
        sc3 = _score(x3)
        if sc3 < best_sc:
            best_sc = sc3
            best_x = x3

    # Phase 4: Coordinate descent sign flips
    if deadline is None or _elapsed() < deadline - 2:
        x = best_x.copy()
        for it in range(100):
            if deadline is not None and _elapsed() > deadline - 2:
                break
            improved = False
            idx = np.random.choice(N, size=min(15, N), replace=False)
            for i in idx:
                old = x[i]
                x[i] = -old
                sc = _score(x)
                if sc < best_sc:
                    best_sc = sc
                    best_x = x.copy()
                    improved = True
                else:
                    x[i] = old
            if not improved:
                break

    return best_x


def construct():
    global _start
    _start = time.time()
    deadline = _start + 135.0

    # Baseline: all-ones (score = 1.0)
    best_x = np.ones(N)
    best_score = _score(best_x)

    # --- Sparse spike optimization ---
    for n_spikes in [4, 6, 8, 10, 12]:
        if _elapsed() > deadline - 30:
            break
        np.random.seed(2000 + n_spikes)
        bounds = [(-2.0, 2.0)]  # DC
        for j in range(n_spikes):
            bounds.append((0.0, float(N - 1)))  # position
            bounds.append((-3.0, 3.0))          # magnitude

        try:
            de_res = differential_evolution(
                _de_objective_sparse, bounds, args=(n_spikes,),
                maxiter=50, popsize=15, tol=1e-5,
                seed=2000 + n_spikes, init='sobol',
                polish=False, updating='deferred'
            )
            x_de = _norm(_sparse_to_signal(de_res.x, n_spikes))
            sc_de = _score(x_de)
            if sc_de < best_score:
                best_score = sc_de
                best_x = x_de

            if _elapsed() < deadline - 20:
                x_ref = _local_polish(x_de, max_iter=200, tau=0.5,
                                      deadline=deadline)
                sc_ref = _score(x_ref)
                if sc_ref < best_score:
                    best_score = sc_ref
                    best_x = x_ref
        except Exception:
            continue

    # --- Sine tone optimization ---
    for K in [3, 4, 5, 6]:
        if _elapsed() > deadline - 25:
            break
        np.random.seed(3000 + K)
        bounds = [(-2.0, 2.0)]  # DC
        for j in range(K):
            bounds.append((1.0, float(N - 1)))   # frequency
            bounds.append((-np.pi, np.pi))       # phase
            bounds.append((-2.0, 2.0))           # amplitude

        try:
            de_res = differential_evolution(
                _de_objective_sine, bounds, args=(K,),
                maxiter=50, popsize=15, tol=1e-5,
                seed=3000 + K, init='sobol',
                polish=False, updating='deferred'
            )
            x_de = _norm(_sine_to_signal(de_res.x, K))
            sc_de = _score(x_de)
            if sc_de < best_score:
                best_score = sc_de
                best_x = x_de

            if _elapsed() < deadline - 15:
                x_ref = _local_polish(x_de, max_iter=200, tau=0.5,
                                      deadline=deadline)
                sc_ref = _score(x_ref)
                if sc_ref < best_score:
                    best_score = sc_ref
                    best_x = x_ref
        except Exception:
            continue

    # --- Mixed sparse + sine optimization ---
    for n_spikes, K in [(4, 2), (6, 3), (8, 2)]:
        if _elapsed() > deadline - 20:
            break
        np.random.seed(4000 + n_spikes + K)
        bounds = [(-2.0, 2.0)]  # DC
        for j in range(n_spikes):
            bounds.append((0.0, float(N - 1)))  # position
            bounds.append((-3.0, 3.0))          # magnitude
        for j in range(K):
            bounds.append((1.0, float(N - 1)))   # frequency
            bounds.append((-np.pi, np.pi))       # phase
            bounds.append((-2.0, 2.0))           # amplitude

        try:
            de_res = differential_evolution(
                _de_objective_mixed, bounds, args=(n_spikes, K),
                maxiter=40, popsize=12, tol=1e-5,
                seed=4000 + n_spikes + K, init='sobol',
                polish=False, updating='deferred'
            )
            x_de = _norm(_mixed_to_signal(de_res.x, n_spikes, K))
            sc_de = _score(x_de)
            if sc_de < best_score:
                best_score = sc_de
                best_x = x_de

            if _elapsed() < deadline - 12:
                x_ref = _local_polish(x_de, max_iter=200, tau=0.5,
                                      deadline=deadline)
                sc_ref = _score(x_ref)
                if sc_ref < best_score:
                    best_score = sc_ref
                    best_x = x_ref
        except Exception:
            continue

    # --- Final polish on the best found ---
    if _elapsed() < deadline - 8:
        x_final = _local_polish(best_x, max_iter=300, tau=0.3,
                                deadline=deadline)
        sc_final = _score(x_final)
        if sc_final < best_score:
            best_x = x_final
            best_score = sc_final

    return best_x
