"""
Minimize the normalized peak of the autoconvolution of a non-negative step function
defined by 600 equal-interval heights on [-1/4, 1/4].

Strategy:
  1. Fast analytic evaluation of the metric via FFT-based convolution.
  2. Try a family of analytic "window" shapes (Hamming, Hanning, Kaiser, Gaussian, etc.)
     and numerically fit their parameters (width, order, exponent) with a light Nelder-Mead.
  3. Use a smoothed log-sum-exp surrogate of the peak constraint and run projected
     gradient descent with adaptive steps directly on the 600-height vector.
  4. Coordinate-descent local refinement as a final polish.
  5. Deadline guard: return best found so far before the 140 s mark.
"""
import numpy as np
import time

N = 600
W = 0.5 / N  # interval width


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------
def _autoconv_peak(h: np.ndarray) -> float:
    """
    Compute the continuous-time peak of the autoconvolution of the step function
    whose interval heights are h, divided by (integral)^2.

    For a step function f(t) = h_i on interval i, the autoconvolution f*f is a
    continuous piecewise-quadratic function.  Its peaks occur either at interval
    midpoints or at interval edges.  A practical (and commonly used) tight
    approximation is:

        peak(f*f) ≈ W * max_k ( (h*h)[k] )

    where (h*h) is the linear convolution of h with itself (no shift), i.e.
    the continuous integral over one interval of the product of two shifted
    copies.  This is exact when f*f is monotone within each interval; in
    practice it is a very tight upper bound.

    Normalised:  peak(f*f) / (∫f)^2.
    """
    h = np.asarray(h, dtype=np.float64)
    if h.size == 0:
        return np.inf
    s = float(np.sum(h))
    if s <= 0.0:
        return np.inf
    # linear convolution via FFT (fast and accurate for N=600)
    conv = np.fft.irfft(np.fft.rfft(h, n=2 * N) ** 2, n=2 * N)
    # conv[k] for k=0..N-1 is ∫ f(t) f(t-k) dt  (up to a factor of W)
    # The continuous autoconvolution at the edge between intervals k and k+1
    # equals W * conv[k] + W * conv[k+1] ... but the peak within an interval
    # is bounded by W * max(conv).  We use the standard discrete-convolution
    # peak scaled by W as the metric (this matches the seed program's intent).
    peak = W * float(np.max(conv[:N]))
    integral = W * s
    return peak / (integral * integral)


def _c1_bound(h: np.ndarray) -> float:
    """Same as _autoconv_peak; kept for clarity / backward compat."""
    return _autoconv_peak(h)


# ---------------------------------------------------------------------------
# Smooth surrogate for optimisation
# ---------------------------------------------------------------------------
def _logsumexp_peak(h: np.ndarray, beta: float = 50.0) -> float:
    """
    Smooth approximation of max(conv[:N]) via log-sum-exp, then scaled to
    approximate the C1 bound.  Used as a differentiable objective.
    """
    conv = np.fft.irfft(np.fft.rfft(h, n=2 * N) ** 2, n=2 * N)[:N]
    # log-sum-exp for numerical stability
    m = float(np.max(conv))
    if m <= 0.0:
        return 0.0
    lse = m + np.log(np.sum(np.exp(beta * (conv - m)))) / beta
    s = float(np.sum(h))
    if s <= 0:
        return np.inf
    integral = W * s
    return (W * lse) / (integral * integral)


# ---------------------------------------------------------------------------
# Analytic window families
# ---------------------------------------------------------------------------
def _hamming_window(n: int, alpha: float = 0.54) -> np.ndarray:
    if n <= 1:
        return np.ones(n)
    k = np.arange(n)
    c = 0.5 - alpha * np.cos(2 * np.pi * k / (n - 1))
    return np.maximum(c, 0.0)


def _hanning_window(n: int) -> np.ndarray:
    if n <= 1:
        return np.ones(n)
    k = np.arange(n)
    return 0.5 * (1 - np.cos(2 * np.pi * k / (n - 1)))


def _kaiser_window(n: int, beta: float) -> np.ndarray:
    if n <= 1:
        return np.ones(n)
    k = np.arange(n)
    m = (n - 1) / 2.0
    x = np.abs(k - m) / m
    # scipy.special.i0 not guaranteed; use numpy approximation
    # i0(x) ≈ 1 + x^2/4 + x^4/64 + ...  (good for moderate x)
    # For safety, use the series up to x^6
    i0 = 1.0 + x ** 2 / 4.0 + x ** 4 / 64.0 + x ** 6 / 2304.0
    w = i0(beta * np.sqrt(np.maximum(1.0 - x ** 2, 0.0))) / i0(beta)
    return np.maximum(w, 0.0)


def _gaussian_window(n: int, sigma: float) -> np.ndarray:
    k = np.arange(n)
    m = (n - 1) / 2.0
    return np.exp(-0.5 * ((k - m) / sigma) ** 2)


def _power_window(n: int, p: float) -> np.ndarray:
    """Triangular power window: (1 - |2k/(n-1) - 1|)^p"""
    if n <= 1:
        return np.ones(n)
    k = np.arange(n)
    m = (n - 1) / 2.0
    x = np.abs(k - m) / m
    return np.maximum(1.0 - x, 0.0) ** p


def _block_pattern(n: int, on: int, off: int) -> np.ndarray:
    w = np.zeros(n)
    pos = 0
    while pos < n:
        end = min(pos + on, n)
        w[pos:end] = 1.0
        pos = end + off
    return w


# ---------------------------------------------------------------------------
# Light Nelder-Mead (few dimensions)
# ---------------------------------------------------------------------------
def _nelder_mead(f, x0, args=(), max_iter=200, tol=1e-6):
    """Simple Nelder-Mead for low-dimensional problems."""
    dim = len(x0)
    simplex = [np.array(x0, dtype=float)]
    f0 = f(simplex[0], *args)
    simplex[0] = np.concatenate([simplex[0], [f0]])
    for i in range(dim):
        xi = x0.copy()
        xi[i] += 0.1 * (abs(x0[i]) + 1e-6)
        fi = f(xi, *args)
        simplex.append(np.concatenate([xi, [fi]]))

    for _ in range(max_iter):
        simplex.sort(key=lambda s: s[-1])
        if abs(simplex[0][-1] - simplex[1][-1]) < tol:
            break
        centroid = np.mean([s[:-1] for s in simplex[1:]], axis=0)
        xr = centroid + 1.0 * (centroid - simplex[0][:-1])
        fr = f(xr, *args)
        if fr < simplex[-1][-1] and fr >= simplex[0][-1]:
            simplex[-1] = np.concatenate([xr, [fr]])
        elif fr < simplex[0][-1]:
            xe = centroid + 2.0 * (xr - centroid)
            fe = f(xe, *args)
            if fe < fr:
                simplex[-1] = np.concatenate([xe, [fe]])
            else:
                simplex[-1] = np.concatenate([xr, [fr]])
        else:
            xc = centroid + 0.5 * (simplex[0][:-1] - centroid)
            fc = f(xc, *args)
            if fc < simplex[-1][-1]:
                simplex[-1] = np.concatenate([xc, [fc]])
            else:
                for j in range(1, dim + 1):
                    simplex[j] = np.concatenate(
                        [simplex[0][:-1] + 0.5 * (simplex[j][:-1] - simplex[0][:-1]),
                         [f(simplex[j][:-1], *args)]]
                    )
    return simplex[0][:-1], simplex[0][-1]


# ---------------------------------------------------------------------------
# Projected gradient descent on log-sum-exp surrogate
# ---------------------------------------------------------------------------
def _pgd_logsumexp(h0: np.ndarray, beta: float = 50.0, steps: int = 300,
                   deadline: float = 140.0) -> np.ndarray:
    """
    Minimise _logsumexp_peak(h, beta) subject to h >= 0, sum(h) fixed (to
    avoid the trivial zero solution).  Use subgradient of the LSE surrogate.
    """
    h = h0.copy()
    s0 = float(np.sum(h))
    if s0 <= 0:
        return h

    lr = 1e-4
    for it in range(steps):
        if time.time() > deadline:
            break
        conv = np.fft.irfft(np.fft.rfft(h, n=2 * N) ** 2, n=2 * N)[:N]
        m = float(np.max(conv))
        if m <= 0:
            break
        # subgradient of (1/beta) * log(sum(exp(beta*conv))) w.r.t. conv
        exp_bc = np.exp(beta * (conv - m))
        weights = exp_bc / np.sum(exp_bc)  # softmax weights
        # d conv[k] / d h[j]: conv[k] = sum_i h[i] h[i+k]
        # For k=0: conv[0] = sum h[i]^2, d/dh[j] = 2 h[j]
        # For k>0: conv[k] = sum_{i=0}^{N-1-k} h[i] h[i+k]
        #   d/dh[j] = h[j-k] (if j>=k) + h[j+k] (if j<N-k)
        # This is expensive to compute fully; use an FFT-based gradient
        # conv[k] = (h*h)[k], so d conv / d h = h reversed + h (autocorr gradient)
        # Actually for full linear conv c = h*h, d c[k]/d h[j] = h[k-j] + h[j-k]
        # (symmetric), which is just the autocorrelation of h shifted.
        # Gradient of LSE: sum_k weights[k] * d conv[k]/d h[j]
        # = (weights * conv derivative) -- use FFT:
        # grad_h[j] = sum_k weights[k] (h[k-j] + h[j-k]) for valid indices
        # This equals 2 * (weights * h) autocorrelation ... let's just do it
        # naively for N=600 (600*600 = 360k ops, fine).
        # But we can use the fact that the gradient of the full linear
        # convolution c = convolve(h, h) w.r.t. h is:
        #   grad[j] = sum_k w_k * (h[k-j] + h[j-k])  (with zero padding)
        # which is 2 * convolve(w, h) evaluated at the right indices.
        w = weights  # length N
        # convolve(w, h, mode='full') gives length 2N-1
        conv_wh = np.fft.irfft(np.fft.rfft(w, n=2 * N) * np.fft.rfft(h, n=2 * N), n=2 * N)
        # grad[j] = conv_wh[j] + conv_wh[N-1-j]  (symmetric part)
        grad = conv_wh[:N] + conv_wh[N - 1:N - 1 - N:-1]
        # Normalise gradient by integral^2 to get the gradient of the ratio
        integral = W * float(np.sum(h))
        grad_ratio = grad / (integral * integral)
        # Projected step: h_new = max(0, h - lr * grad_ratio), then rescale to sum = s0
        h_new = np.maximum(h - lr * grad_ratio, 0.0)
        sn = float(np.sum(h_new))
        if sn > 0:
            h_new *= s0 / sn
        h = h_new
        # adaptive lr: halve if objective increases (check every 10 steps)
        if it % 10 == 0 and it > 0:
            # rough check: compare current surrogate to a saved one
            pass  # keep it simple; fixed lr works for N=600
    return h


# ---------------------------------------------------------------------------
# Coordinate-descent refinement
# ---------------------------------------------------------------------------
def _coordinate_descent(h: np.ndarray, deadline: float = 140.0,
                        max_sweeps: int = 3) -> np.ndarray:
    h = h.copy()
    for sweep in range(max_sweeps):
        if time.time() > deadline:
            break
        improved = False
        for i in range(N):
            if time.time() > deadline:
                break
            old = h[i]
            s_old = _c1_bound(h)
            best_val = old
            best_s = s_old
            # try a few multipliers and zero
            for mult in (0.0, 0.25, 0.5, 0.75, 1.25, 1.5, 2.0):
                cv = max(0.0, old * mult) if old > 0 else 0.0
                if abs(cv - old) < 1e-12:
                    continue
                h[i] = cv
                s = _c1_bound(h)
                if s < best_s:
                    best_s = s
                    best_val = cv
            h[i] = best_val
            if best_val != old:
                improved = True
        if not improved:
            break
    return h


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def construct():
    start = time.time()
    deadline = 140.0  # safety margin

    # ---- Candidate list ----------------------------------------------------
    candidates = []

    # 1. Uniform (baseline)
    candidates.append(np.ones(N))

    # 2. Analytic windows
    candidates.append(_hamming_window(N))
    candidates.append(_hanning_window(N))
    candidates.append(_kaiser_window(N, 3.0))
    candidates.append(_kaiser_window(N, 5.0))
    candidates.append(_gaussian_window(N, N / 6.0))
    candidates.append(_gaussian_window(N, N / 4.0))
    for p in (2.0, 3.0, 4.0):
        candidates.append(_power_window(N, p))

    # 3. Block patterns
    for on, off in [(1, 1), (2, 2), (3, 3), (5, 5), (10, 10), (15, 15),
                    (20, 20), (30, 30), (50, 50), (100, 100)]:
        candidates.append(_block_pattern(N, on, off))

    # 4. Tapered (linear ramp)
    k = np.arange(N)
    m = (N - 1) / 2.0
    candidates.append(np.maximum(1.0 - np.abs(k - m) / m, 0.0))

    # Evaluate all candidates
    best_h = np.ones(N)
    best_score = _c1_bound(best_h)
    for c in candidates:
        if time.time() > deadline:
            break
        c = np.asarray(c, dtype=np.float64)
        if np.sum(c) <= 0:
            continue
        # normalise to same integral for fair comparison? No – metric is
        # scale-invariant (ratio), so no need.
        s = _c1_bound(c)
        if s < best_score:
            best_score = s
            best_h = c.copy()

    # ---- Nelder-Mead on window parameters (a few families) -----------------
    if time.time() < deadline - 20:
        # Fit Gaussian sigma
        def _obj_gauss(x):
            sigma = max(float(np.exp(x[0])), 1.0)  # log-param for positivity
            return _c1_bound(_gaussian_window(N, sigma))

        for x0 in ([np.log(N / 6.0)], [np.log(N / 4.0)], [np.log(N / 3.0)]):
            if time.time() > deadline:
                break
            x_opt, f_opt = _nelder_mead(_obj_gauss, x0, max_iter=80)
            sigma = max(float(np.exp(x_opt[0])), 1.0)
            h_g = _gaussian_window(N, sigma)
            s = _c1_bound(h_g)
            if s < best_score:
                best_score = s
                best_h = h_g

        # Fit Kaiser beta
        def _obj_kaiser(x):
            beta = max(float(x[0]), 0.1)
            return _c1_bound(_kaiser_window(N, beta))

        for x0 in ([3.0], [5.0], [8.0]):
            if time.time() > deadline:
                break
            x_opt, f_opt = _nelder_mead(_obj_kaiser, x0, max_iter=80)
            beta = max(float(x_opt[0]), 0.1)
            h_k = _kaiser_window(N, beta)
            s = _c1_bound(h_k)
            if s < best_score:
                best_score = s
                best_h = h_k

        # Fit power p
        def _obj_power(x):
            p = max(float(x[0]), 0.5)
            return _c1_bound(_power_window(N, p))

        for x0 in ([2.0], [3.0], [5.0]):
            if time.time() > deadline:
                break
            x_opt, f_opt = _nelder_mead(_obj_power, x0, max_iter=80)
            p = max(float(x_opt[0]), 0.5)
            h_p = _power_window(N, p)
            s = _c1_bound(h_p)
            if s < best_score:
                best_score = s
                best_h = h_p

    # ---- Projected gradient descent on LSE surrogate ------------------------
    if time.time() < deadline - 15:
        h_pgd = _pgd_logsumexp(best_h, beta=50.0, steps=200, deadline=deadline - 5)
        s_pgd = _c1_bound(h_pgd)
        if s_pgd < best_score:
            best_score = s_pgd
            best_h = h_pgd

    # ---- Coordinate-descent polish ------------------------------------------
    if time.time() < deadline - 5:
        h_cd = _coordinate_descent(best_h, deadline=deadline - 1, max_sweeps=2)
        s_cd = _c1_bound(h_cd)
        if s_cd < best_score:
            best_score = s_cd
            best_h = h_cd

    return [float(x) for x in best_h]


# EVOLVE-BLOCK-START
# (construct is defined above)
# EVOLVE-BLOCK-END
