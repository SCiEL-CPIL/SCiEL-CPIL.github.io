"""Spectral (Fourier-domain) optimization for Erdos minimum overlap.

Structurally different from the seed: instead of designing time-domain
candidates (sines, random, binary) and doing local search, we work directly
in the DFT domain. The constraints (sum, bounds) are handled by projecting
the DFT coefficients, and the objective is a smooth log-sum-exp surrogate of
the max overlap minimized by projected gradient descent.
"""
import numpy as np
import time

N = 100
TARGET = N / 2.0  # 50.0
HALF_N = N // 2   # 50


def _overlap_from_autocorr(autocorr):
    """overlap_k = N/2 - autocorr_k; return max overlap."""
    return TARGET - np.min(autocorr)


def _autocorr_from_h(h):
    H = np.fft.fft(h)
    power = H * np.conj(H)
    return np.real(np.fft.ifft(power))


def _surrogate(autocorr, beta):
    """Smooth max-overlap surrogate via log-sum-exp on overlap_k."""
    overlap = TARGET - autocorr
    m = np.max(overlap)
    # log-sum-exp trick for stability
    if beta > 1e-6:
        s = np.sum(np.exp(beta * (overlap - m)))
        return m + np.log(s) / beta
    return m


def _objective(autocorr, beta):
    return _surrogate(autocorr, beta)


def _dft_to_h(c):
    """c is a complex array of length N representing DFT coeffs; return real h."""
    h = np.fft.ifft(c).real
    return h


def _project_dft(c):
    """
    Project complex DFT coefficients c to satisfy:
      1) h real  -> c[-k] = conj(c[k])
      2) sum h = TARGET -> c[0] = TARGET
    Returns projected c.
    """
    c = c.copy()
    # Enforce Hermitian symmetry (so ifft is real)
    for k in range(1, HALF_N + 1):
        avg = 0.5 * (c[k] + np.conj(c[-k]))
        c[k] = avg
        c[-k] = np.conj(avg)
    # Enforce sum constraint: c[0] = sum(h) = TARGET (real)
    c[0] = complex(TARGET, 0.0)
    return c


def _grad_surrogate_wrt_c(autocorr, c, beta):
    """
    Gradient of surrogate(autocorr(c)) w.r.t. complex c, using chain rule.
    autocorr_k = sum_j h[j] h[j-k]  (circular).
    We compute d(dautocorr_k/dh) then map to c via DFT.
    Simplify: work in h-space gradient then transform.
    """
    # overlap_k = TARGET - autocorr_k
    overlap = TARGET - autocorr
    m = np.max(overlap)
    if beta > 1e-6:
        w = np.exp(beta * (overlap - m))
        w = w / np.sum(w)          # softmax weights over k
        d_obj_d_overlap = w        # d/d overlap_k of (m + log(sum exp(beta(overlap-m)))/beta) = w
    else:
        # hard max: weight on argmax
        idx = np.argmax(overlap)
        w = np.zeros(N)
        w[idx] = 1.0
        d_obj_d_overlap = w

    # d_obj_d_autocorr_k = - d_obj_d_overlap_k
    d_obj_d_autocorr = -d_obj_d_overlap

    # autocorr_k = sum_j h[j] h[j-k]
    # d autocorr_k / d h[i] = h[i-k] + h[i+k]  (circular)
    # So gradient in h-space: g_h[i] = sum_k d_obj_d_autocorr_k * (h[i-k] + h[i+k])
    # This is a circular convolution: g_h = 2 * circ_conv(h, d_obj_d_autocorr)
    # where circ_conv(a,b)[i] = sum_k a[i-k] b[k]
    # Use FFT: g_h = ifft(fft(h) * fft(d_obj_d_autocorr)) * 2? Let's verify.
    # circ_conv(h, d)[i] = sum_k h[i-k] d[k]. FFT of circ_conv = FFT(h)*FFT(d).
    # We need sum_k d_obj_d_autocorr_k * (h[i-k] + h[i+k])
    #   = sum_k d_obj_d_autocorr_k h[i-k]  +  sum_k d_obj_d_autocorr_k h[i+k]
    # First term = circ_conv(h, d_obj_d_autocorr)[i]
    # Second term: sum_k d_obj_d_autocorr_k h[i+k] = sum_j d_obj_d_autocorr_{j-i} h[j]
    #   Let j = i+k -> k = j-i. = sum_j h[j] d_obj_d_autocorr[j-i] = circ_conv(d_obj_d_autocorr, h)[i]
    #   But circ_conv is commutative, so = circ_conv(h, d_obj_d_autocorr)[i].
    # So g_h[i] = 2 * circ_conv(h, d_obj_d_autocorr)[i].
    H = np.fft.fft(h)
    D = np.fft.fft(d_obj_d_autocorr)
    g_h = np.real(np.fft.ifft(H * D)) * 2.0

    # Now gradient w.r.t. c: h = ifft(c). dh/dc is the DFT matrix.
    # d_obj/dc = DFT^H (g_h) in real sense. For complex c with Hermitian constraint,
    # the gradient w.r.t. c (treating c and conj independently) maps via:
    # d_obj/dc = fft(g_h)  (since h = ifft(c) => dh/dc = ifft_matrix; chain: d_obj/dc = (dh/dc)^T g_h = fft(g_h))
    g_c = np.fft.fft(g_h)
    return g_c


def _project_h_to_bounds(h):
    """Clip h to [0,1] and fix sum to TARGET (time-domain repair)."""
    h = np.clip(h, 0.0, 1.0)
    s = np.sum(h)
    if abs(s - TARGET) > 1e-12:
        diff = TARGET - s
        if diff > 0:
            room = 1.0 - h
            if np.sum(room) > 1e-12:
                h += diff * room / np.sum(room)
        else:
            if s > 1e-12:
                h += diff * h / s
        h = np.clip(h, 0.0, 1.0)
    # final sum fix
    s = np.sum(h)
    if abs(s - TARGET) > 1e-10:
        diff = TARGET - s
        if diff > 0:
            room = 1.0 - h
            mask = room > 1e-12
            if np.sum(room[mask]) > 0:
                h[mask] += diff / np.sum(mask)
        else:
            mask = h > 1e-12
            if np.sum(h[mask]) > 0:
                h[mask] += diff / np.sum(mask)
        h = np.clip(h, 0.0, 1.0)
    return h


def _spectral_optimize(h0, deadline):
    """Projected gradient descent in DFT domain on the log-sum-exp surrogate."""
    c = np.fft.fft(h0)
    c = _project_dft(c)
    best_h = _project_h_to_bounds(_dft_to_h(c))
    best_score = _overlap_from_autocorr(_autocorr_from_h(best_h))

    # Adaptive beta: start moderate, increase for sharpness
    betas = [10.0, 30.0, 100.0, 300.0, 1000.0]
    # Learning rate schedule
    base_lr = 0.5

    for beta in betas:
        if time.time() > deadline:
            break
        # For each beta, run a few hundred steps with decaying lr
        iters = 400
        for it in range(iters):
            if time.time() > deadline:
                break
            lr = base_lr / (1.0 + it / 50.0)
            h = _dft_to_h(c)
            h = _project_h_to_bounds(h)
            autocorr = _autocorr_from_h(h)
            g_c = _grad_surrogate_wrt_c(autocorr, c, beta)
            # Gradient step in c-space
            c_new = c - lr * g_c
            c_new = _project_dft(c_new)
            # Evaluate true score of the resulting h
            h_new = _project_h_to_bounds(_dft_to_h(c_new))
            score = _overlap_from_autocorr(_autocorr_from_h(h_new))
            if score < best_score - 1e-12:
                c = c_new
                best_h = h_new
                best_score = score
            else:
                # reject, maybe halve lr implicitly by not moving
                pass
    return best_h, best_score


def _random_start(seed):
    rng = np.random.RandomState(seed)
    h = rng.uniform(0, 1, N)
    h = _project_h_to_bounds(h)
    return h


def _smooth_start(freq, amp, phase, seed):
    x = np.linspace(0, 2 * np.pi, N, endpoint=False)
    h = 0.5 + amp * np.sin(2 * np.pi * freq * x / N + phase)
    h = _project_h_to_bounds(h)
    return h


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    np.random.seed(42)

    # Baseline: uniform
    best_h = np.full(N, 0.5)
    best_score = _overlap_from_autocorr(_autocorr_from_h(best_h))

    # Generate diverse starting points
    starts = []
    for s in range(20):
        starts.append(_random_start(s))
    for freq in [1, 2, 3, 4, 5, 7, 10, 13, 17, 25, 33, 50]:
        for amp in [0.3, 0.45, 0.49]:
            for phase in [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]:
                starts.append(_smooth_start(freq, amp, phase, 0))

    # Sort starts by their initial true score, keep top
    scored = []
    for h in starts:
        if time.time() > deadline - 20:
            break
        sc = _overlap_from_autocorr(_autocorr_from_h(h))
        scored.append((sc, h))
    scored.sort(key=lambda x: x[0])

    top = [h for _, h in scored[:12]]
    # Always include the uniform and a couple random
    top.append(np.full(N, 0.5))
    top.append(_random_start(99))

    for h0 in top:
        if time.time() > deadline - 10:
            break
        h_opt, sc_opt = _spectral_optimize(h0, deadline - 5)
        if sc_opt < best_score:
            best_score = sc_opt
            best_h = h_opt

    # Final: ensure validity
    best_h = _project_h_to_bounds(best_h)
    return best_h
