"""Signed autocorrelation max-abs minimization via spectral flatness gradient descent."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = time.time() + 120.0
_MARGIN = 10.0  # seconds safety margin


def _autoconv(a):
    """Full linear autoconvolution of a (length N)."""
    F = np.fft.rfft(a, n=2 * N - 1)
    return np.fft.irfft(F * F, n=2 * N - 1)


def _score(a):
    """Upper bound on C3': max|autoconv| / (sum a)^2. Assumes a in {-1,+1}."""
    ac = _autoconv(a)
    s = a.sum()
    if s == 0:
        return np.max(np.abs(ac))  # degenerate
    return np.max(np.abs(ac)) / (s * s)


def _spectral_objective(x):
    """Objective: mean |DFT|^2 (flat magnitude) + penalty on sum(x).
    x is continuous in R^N. Returns (value, grad)."""
    F = np.fft.rfft(x)
    mag2 = np.abs(F) ** 2
    # We want mag2 flat => minimize variance of mag2, equivalently minimize sum mag2^2
    # (since mean is fixed by sum x^2 which we also control).
    val = np.sum(mag2 * mag2)
    # sum penalty: encourage x to have nonzero sum (so denominator large)
    # but we'll handle normalization at the end.
    grad = np.zeros_like(x)
    # d/dx sum |F|^4 : use chain rule
    # |F|^2 = F * conj(F); d|F|^2/dx = 2 Re(conj(F) * dF/dx)
    # We'll compute via: g = 2 * mag2 * F  (in freq), then grad = 2 * irfft(g)
    g = 2.0 * mag2 * F
    grad = np.fft.irfft(g, n=N) * 2.0
    return val, grad


def construct():
    global _DEADLINE
    t0 = time.time()

    # Start from a good random sign vector
    best_signs = None
    best_val = np.inf

    # A few random initializations
    rng = np.random.default_rng(0)
    inits = []
    # all-ones baseline
    inits.append(np.ones(N))
    # random signs
    for _ in range(3):
        inits.append(rng.choice([-1.0, 1.0], size=N))
    # alternating
    inits.append(np.tile([1.0, -1.0], N // 2))

    for x0 in inits:
        if time.time() > _DEADLINE - _MARGIN:
            break
        # Evaluate baseline
        s = x0.copy()
        v = _score(s)
        if v < best_val:
            best_val = v
            best_signs = s.copy()

    # Gradient descent on continuous relaxation to flatten spectrum
    # Objective: minimize sum |DFT(x)|^4  subject to x near +/-1
    # Use projected gradient: project onto box [-1,1], then round.

    x = best_signs.astype(float).copy()
    lr = 0.01
    for it in range(2000):
        if time.time() > _DEADLINE - _MARGIN:
            break
        val, grad = _spectral_objective(x)
        # Projected gradient step
        x = x - lr * grad
        # Project to [-1, 1]
        x = np.clip(x, -1.0, 1.0)
        # Normalize magnitude to unit (keep signs)
        mag = np.abs(x)
        if np.min(mag) < 1e-6:
            x = np.sign(x)
            x[x == 0] = 1.0
        else:
            x = np.sign(x) * np.ones_like(x)  # snap to signs if too weak
        # Actually keep continuous: rescale to have |x| in [0.5, 1]
        # Better: just clip and keep continuous
        x = np.clip(x, -1.0, 1.0)

        # Round and evaluate periodically
        if it % 20 == 0 or it == 1999:
            s = np.sign(x)
            s[s == 0] = 1.0
            v = _score(s)
            if v < best_val:
                best_val = v
                best_signs = s.copy()

        # Adaptive learning rate
        if it % 200 == 199:
            lr *= 0.5

    # Local search: flip single elements to reduce peak
    s = best_signs.copy()
    v = _score(s)
    best_local = s.copy()
    best_local_v = v
    for _round in range(6):
        if time.time() > _DEADLINE - _MARGIN:
            break
        improved = False
        ac = _autoconv(s)
        peak = np.max(np.abs(ac))
        s_sum = s.sum()
        denom = s_sum * s_sum
        for i in range(0, N, 1):
            if time.time() > _DEADLINE - _MARGIN:
                break
            # Try flipping i
            s[i] = -s[i]
            # Recompute autoconv efficiently? Just recompute (N=400, fast)
            ac2 = _autoconv(s)
            peak2 = np.max(np.abs(ac2))
            s_sum2 = s.sum()
            denom2 = s_sum2 * s_sum2
            v2 = peak2 / denom2 if denom2 != 0 else peak2
            if v2 < best_local_v - 1e-12:
                best_local_v = v2
                best_local = s.copy()
                improved = True
            else:
                s[i] = -s[i]  # revert
        if not improved:
            break
        s = best_local.copy()

    if best_local_v < best_val:
        best_val = best_local_v
        best_signs = best_local.copy()

    return best_signs
# EVOLVE-BLOCK-END
