"""Erdos minimum overlap - multi-strategy optimization with spectral, epigraph, and chirp approaches."""
import numpy as np
import time
import sys

N = 100
TARGET = 0.5 * N  # sum constraint


def _linear_overlap_values(h):
    """Return C_k = sum_i h[i]*(1-h[(i+k)%N]) for all k via FFT (real)."""
    H = np.fft.fft(h)
    # autocorr_k = ifft(|H|^2 * fft_e)[k] where fft_e = fft([1,0,...,0]) = ones
    # C_k = sum_i h[i] - h[i]*h[(i+k)%N] = sum(h) - ac_k
    # sum(h) = N/2
    # ac_k = ifft(|H|^2)[k]
    ac = np.real(np.fft.ifft(np.abs(H) ** 2))
    return TARGET - ac


def _score(h):
    """Compute the maximum overlap value."""
    return float(np.max(_linear_overlap_values(h)))


def _project(v, target):
    """Project onto box [0,1] with sum == target (bisection on shift)."""
    target = float(target)
    lo, hi = -2.0, 2.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        c = np.clip(v - mid, 0.0, 1.0)
        s = np.sum(c)
        if s > target:
            lo = mid
        else:
            hi = mid
    c = np.clip(v - 0.5 * (lo + hi), 0.0, 1.0)
    s = np.sum(c)
    diff = target - s
    if abs(diff) > 1e-10:
        # Distribute the difference
        n_adjust = min(N, max(1, int(abs(diff)) + 1))
        if diff > 0:
            idx = np.argsort(c)[::-1][:n_adjust]
            step = diff / len(idx)
            c[idx] = np.minimum(c[idx] + step, 1.0)
            # If still not exact, fix remaining
            s2 = np.sum(c)
            diff2 = target - s2
            if abs(diff2) > 1e-12:
                idx2 = np.argsort(c)[::-1][:max(1, int(abs(diff2)) + 1)]
                c[idx2] = np.minimum(c[idx2] + diff2 / len(idx2), 1.0)
        else:
            idx = np.argsort(c)[:n_adjust]
            step = diff / len(idx)
            c[idx] = np.maximum(c[idx] + step, 0.0)
            s2 = np.sum(c)
            diff2 = target - s2
            if abs(diff2) > 1e-12:
                idx2 = np.argsort(c)[:max(1, int(abs(diff2)) + 1)]
                c[idx2] = np.maximum(c[idx2] + diff2 / len(idx2), 0.0)
    return c


def _smooth_max_grad(vals, beta):
    """Smoothed max of vals with softmax weights; returns (value, gradient wrt vals)."""
    m = np.max(vals)
    x = vals - m
    w = np.exp(beta * x)
    w_sum = np.sum(w)
    w = w / (w_sum + 1e-300)
    value = m + np.log(w_sum + 1e-300) / beta
    return value, w


def _grad_overlap(h, beta):
    """Gradient of smoothed-max overlap wrt h (real, length N)."""
    vals = _linear_overlap_values(h)
    _, w = _smooth_max_grad(vals, beta)
    # d C_k / d h_i = (1 - h[(i-k)%N])
    # grad_h_i = sum_k w_k (1 - h[(i-k)%N]) = 1 - sum_k w_k h[(i-k)%N]
    # sum_k w_k h[(i-k)%N] = circular cross-correlation
    W = np.fft.fft(w)
    H = np.fft.fft(h)
    conv = np.real(np.fft.ifft(W * H))
    return 1.0 - conv


def _nesterov_smooth_max(h0, max_iter, deadline, beta=60.0):
    """Nesterov accelerated gradient on smoothed-max overlap with projection."""
    h = h0.copy()
    y = h.copy()
    t = 1.0
    best = h.copy()
    best_s = _score(h)
    for it in range(max_iter):
        if time.time() > deadline:
            break
        g = _grad_overlap(y, beta)
        step = 2.0 / (it + 3.0)
        h_new = _project(y - step * g, TARGET)
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        y = h_new + ((t - 1.0) / t_new) * (h_new - h)
        h = h_new
        t = t_new
        s = _score(h)
        if s < best_s:
            best_s = s
            best = h.copy()
    return best, best_s


def _epigraph_obj_grad(h, beta):
    """Epigraph formulation: minimize t + mu*sum(max(0, C_k - t)^2) + regularization.
    Returns (objective, grad_h, grad_t)."""
    vals = _linear_overlap_values(h)
    t = np.max(vals) + 0.01
    # Smooth hinge: max(0, C_k - t)
    excess = np.maximum(0.0, vals - t)
    obj = t + 0.5 * beta * np.sum(excess ** 2)
    # grad_t = 1 - beta * sum(excess)
    grad_t = 1.0 - beta * np.sum(excess)
    # grad_h_k = -beta * excess_k * dC_k/dh
    # dC_k/dh_i = 1 - h[(i-k)%N]
    # grad_h_i = -beta * sum_k excess_k * (1 - h[(i-k)%N])
    # = -beta * (sum(excess) - sum_k excess_k * h[(i-k)%N])
    E = np.fft.fft(excess)
    H = np.fft.fft(h)
    conv = np.real(np.fft.ifft(E * H))
    grad_h = -beta * (np.sum(excess) - conv)
    return obj, grad_h, grad_t


def _nesterov_epigraph(h0, max_iter, deadline, beta=100.0):
    """Nesterov on epigraph formulation."""
    h = h0.copy()
    y = h.copy()
    t = float(np.max(_linear_overlap_values(h)))
    t_y = t
    t_count = 1.0
    t_y_count = 1.0
    best = h.copy()
    best_s = _score(h)
    for it in range(max_iter):
        if time.time() > deadline:
            break
        obj, g_h, g_t = _epigraph_obj_grad(y, beta)
        step = 2.0 / (it + 3.0)
        h_new = _project(y - step * g_h, TARGET)
        t_new = t_y - step * g_t
        t_new = max(t_new, 0.0)
        t_count_new = (1.0 + np.sqrt(1.0 + 4.0 * t_count * t_count)) / 2.0
        t_y_count_new = (1.0 + np.sqrt(1.0 + 4.0 * t_y_count * t_y_count)) / 2.0
        y = h_new + ((t_count - 1.0) / t_count_new) * (h_new - h)
        t_y = t_new + ((t_y_count - 1.0) / t_y_count_new) * (t_new - t)
        h = h_new
        t = t_new
        t_count = t_count_new
        t_y_count = t_y_count_new
        s = _score(h)
        if s < best_s:
            best_s = s
            best = h.copy()
    return best, best_s


def _generate_chirp_candidates():
    """Generate chirp (linear frequency sweep) candidates with low autocorrelation."""
    candidates = []
    # Various chirp parameters
    for f0 in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
        for f1 in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
            if f1 <= f0:
                continue
            # Linear chirp: h[i] = 0.5 + A * cos(2*pi*(f0*i + 0.5*(f1-f0)*i^2/N))
            for A in [0.1, 0.2, 0.3, 0.4, 0.45]:
                i = np.arange(N)
                phase = 2.0 * np.pi * (f0 * i + 0.5 * (f1 - f0) * i ** 2 / N)
                h = 0.5 + A * np.cos(phase)
                h = _project(h, TARGET)
                candidates.append(h)
    # Also add some random candidates
    for _ in range(20):
        h = np.random.uniform(0.0, 1.0, N)
        h = _project(h, TARGET)
        candidates.append(h)
    return candidates


def _coordinate_descent_polish(h, max_iter, deadline):
    """Coordinate descent: adjust one coordinate at a time to reduce max overlap."""
    h = h.copy()
    best_s = _score(h)
    for it in range(max_iter):
        if time.time() > deadline:
            break
        improved = False
        for i in range(N):
            # Try small adjustments to h[i]
            for delta in [0.01, -0.01, 0.05, -0.05, 0.1, -0.1]:
                h_trial = h.copy()
                h_trial[i] = h[i] + delta
                if h_trial[i] < 0 or h_trial[i] > 1:
                    continue
                # Adjust another coordinate to maintain sum
                j = (i + 1) % N
                h_trial[j] = h[j] - delta
                if h_trial[j] < 0 or h_trial[j] > 1:
                    continue
                s = _score(h_trial)
                if s < best_s - 1e-12:
                    h = h_trial
                    best_s = s
                    improved = True
                    break
        if not improved:
            break
    return h, best_s


def _spectral_h_from_coeffs(coeffs):
    """Construct h from spectral coefficients (real, symmetric)."""
    # coeffs are the first N/2+1 Fourier coefficients
    # h = ifft(coeffs_extended)
    N_half = N // 2
    full_coeffs = np.zeros(N, dtype=complex)
    full_coeffs[0] = coeffs[0]
    for k in range(1, N_half + 1):
        full_coeffs[k] = coeffs[k]
        full_coeffs[N - k] = np.conj(coeffs[k])
    h = np.real(np.fft.ifft(full_coeffs))
    return _project(h, TARGET)


def _h_to_spectral_coeffs(h):
    """Extract spectral coefficients from h."""
    H = np.fft.fft(h)
    N_half = N // 2
    coeffs = H[:N_half + 1].copy()
    return coeffs


def _optimize_spectral(h0, max_iter, deadline):
    """Optimize in spectral domain."""
    coeffs = _h_to_spectral_coeffs(h0)
    h = h0.copy()
    best_s = _score(h)
    t = 1.0
    y_coeffs = coeffs.copy()
    for it in range(max_iter):
        if time.time() > deadline:
            break
        h_y = _spectral_h_from_coeffs(y_coeffs)
        beta = 50.0 + 50.0 * it / max_iter
        g_h = _grad_overlap(h_y, beta)
        # Gradient in spectral domain
        g_coeffs = np.fft.fft(g_h)[:N_half + 1]
        step = 2.0 / (it + 3.0)
        new_coeffs = y_coeffs - step * g_coeffs
        # Ensure real symmetry
        new_coeffs[0] = np.real(new_coeffs[0])
        for k in range(1, N_half + 1):
            new_coeffs[k] = np.complex128(new_coeffs[k])
        h_new = _spectral_h_from_coeffs(new_coeffs)
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        y_coeffs = new_coeffs + ((t - 1.0) / t_new) * (new_coeffs - coeffs)
        coeffs = new_coeffs
        t = t_new
        s = _score(h_new)
        if s < best_s:
            best_s = s
            h = h_new.copy()
    return h, best_s


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin

    # Start from constant 0.5
    h0 = np.full(N, 0.5)
    best_h = h0.copy()
    best_s = _score(h0)

    # Strategy 1: Nesterov smooth max with annealed beta
    for beta in [40.0, 80.0, 160.0, 320.0]:
        if time.time() > deadline:
            break
        h, s = _nesterov_smooth_max(best_h, 3000, deadline, beta)
        if s < best_s:
            best_s = s
            best_h = h

    # Strategy 2: Epigraph formulation
    if time.time() < deadline:
        for beta in [50.0, 100.0, 200.0]:
            if time.time() > deadline:
                break
            h, s = _nesterov_epigraph(best_h, 2000, deadline, beta)
            if s < best_s:
                best_s = s
                best_h = h

    # Strategy 3: Spectral optimization
    if time.time() < deadline:
        h, s = _optimize_spectral(best_h, 2000, deadline)
        if s < best_s:
            best_s = s
            best_h = h

    # Strategy 4: Chirp candidates + polish
    if time.time() < deadline:
        candidates = _generate_chirp_candidates()
        for h_c in candidates:
            if time.time() > deadline:
                break
            s_c = _score(h_c)
            if s_c < best_s:
                # Polish this candidate
                h_pol, s_pol = _nesterov_smooth_max(h_c, 2000, deadline, 100.0)
                if s_pol < best_s:
                    best_s = s_pol
                    best_h = h_pol

    # Strategy 5: Coordinate descent polish
    if time.time() < deadline:
        h_pol, s_pol = _coordinate_descent_polish(best_h, 200, deadline)
        if s_pol < best_s:
            best_s = s_pol
            best_h = h_pol

    # Final cleanup: ensure exact sum and bounds
    best_h = _project(best_h, TARGET)
    best_h = np.clip(best_h, 0.0, 1.0)
    s_final = _score(best_h)
    if s_final <= best_s:
        best_h = best_h
        best_s = s_final

    return best_h
