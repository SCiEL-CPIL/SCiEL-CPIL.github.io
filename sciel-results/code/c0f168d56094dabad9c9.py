"""Spectral flat-spectrum design + time-domain soft-max polish for C3'."""
import time
import numpy as np

N = 400
T_LIMIT = 150.0


def _score(x):
    """C3' = max|autoconv| / (sum x)^2. Lower is better."""
    ac = np.convolve(x, x, mode='full')
    return float(np.max(np.abs(ac))) / (float(np.sum(x)) ** 2)


def _ac_offdiag_abs(x):
    """Absolute values of autoconvolution excluding the zero-lag peak."""
    ac = np.convolve(x, x, mode='full')
    return np.abs(ac)


def _softmax_offdiag(x, s):
    """Smooth surrogate of max over off-lag |autoconv|, in normalized units."""
    denom = float(np.sum(x)) ** 2
    if denom == 0:
        return 0.0
    a = _ac_offdiag_abs(x) / denom
    m = a.max()
    return float(m + np.log(np.mean(np.exp(a - m))) / s)


def construct():
    t0 = time.time()
    best = None
    best_val = np.inf

    def consider(x):
        nonlocal best, best_val
        v = _score(x)
        if v < best_val:
            best_val = v
            best = x.copy()
        return v

    # ---------- Phase 1: spectral flat-magnitude design ----------
    # Parameterize by per-bin log-magnitude and phase; enforce constant |X|.
    # Start from a random complex spectrum with unit-ish magnitude, then
    # "whiten" magnitudes toward a constant while keeping the DC (sum) real.
    rng = np.random.default_rng(12345)
    M = N + N - 1  # full autoconv length; use an FFT size >= M
    M = 1
    while M < (2 * N - 1):
        M *= 2

    X = np.fft.rfft(np.zeros(N))
    # random complex spectrum
    Xc = rng.normal(size=M) + 1j * rng.normal(size=M)
    # enforce hermitian by building from rfft of a random real signal
    x0 = rng.normal(size=N)
    Xc = np.fft.rfft(x0, n=M)
    # flatten magnitudes: replace |X| with its geometric mean, keep phase
    mag = np.abs(Xc)
    # keep DC (bin 0) and Nyquist (last bin) real-valued
    phase = np.angle(Xc)
    gmean = np.exp(np.mean(np.log(mag + 1e-12)))
    Xflat = gmean * np.exp(1j * phase)
    # force DC real (so sum is real)
    Xflat[0] = gmean
    Xflat[-1] = gmean * np.sign(Xflat[-1].real) if Xflat[-1].real != 0 else gmean
    x_flat = np.fft.irfft(Xflat, n=M)[:N]
    # normalize sign pattern: keep continuous for now
    consider(x_flat)

    # ---------- Phase 2: time-domain soft-max polish ----------
    # Gradient of soft-max w.r.t. x via convolution with the softmax weights.
    x = best.copy()
    denom = float(np.sum(x)) ** 2

    def grad_soft(x, s):
        denom = float(np.sum(x)) ** 2
        a = _ac_offdiag_abs(x) / denom
        m = a.max()
        w = np.exp(a - m)
        w = w / w.sum()
        # d(a_i)/dx = (1/denom) * d|ac_i|/dx ; |ac| sign handled by ac sign
        ac = np.convolve(x, x, mode='full')
        sign = np.sign(ac)
        # d ac_i / dx_j = x_{i-j} + x_{i-N+1-j} style; use correlation form:
        # ac = conv(x,x); d ac_k / dx_j = x_{k-j} (for valid overlap)
        # Build gradient via convolution of weights with x (and mirror).
        # ac_k = sum_j x_j x_{k-j}. d ac_k/d x_j = x_{k-j} + (if symmetric) ...
        # For full conv, d ac_k / dx_j = x_{k-j} (index within [0,N-1])
        # So grad_j = sum_k w_k sign_k x_{k-j}
        # = (w * x)(j) with appropriate convolution.
        g = np.convolve(w * sign, x, mode='full')
        # map full conv index: g_full[j] = sum_k (w*sign)_k x_{j-k}
        # We want sum_k (w*sign)_k x_{k-j} = g_full[N-1 - j] (mirror)
        g = g[N - 1 - np.arange(N)]
        # normalize objective derivative: objective = m + log(mean exp(a-m))/s
        # d objective/d a_i = w_i / s
        g = g / s
        # also account for denom dependence (second order) -- ignore (small)
        return g

    s = 50.0
    lr = 1e-3
    patience = 0
    best_polish = best_val
    best_x = x.copy()
    for it in range(200000):
        if time.time() - t0 > T_LIMIT - 25:
            break
        g = grad_soft(x, s)
        # scale gradient by 1/denom^2 already inside; step
        x = x - lr * g
        # keep DC (sum) fixed to preserve normalization: re-center
        target_sum = float(np.sum(best_x))
        x = x + (target_sum - float(np.sum(x))) / N
        v = _score(x)
        if v < best_polish:
            best_polish = v
            best_x = x.copy()
            patience = 0
        else:
            patience += 1
            if patience % 200 == 0:
                lr *= 0.9
        if patience > 5000:
            break
    consider(best_x)

    # ---------- Phase 3: sign/rounding polish (optional discrete step) ----------
    # Try rounding to a few candidate sign-magnitude discretizations and keep best.
    if best is not None and time.time() - t0 < T_LIMIT - 15:
        xb = best
        # candidate: round to nearest 0.5
        for scale in (1.0, 0.5, 0.25):
            xr = np.round(xb / scale) * scale
            consider(xr)
        # candidate: hard sign
        xs = np.sign(xb)
        consider(xs)

    if best is None:
        best = np.ones(N)
    return best
