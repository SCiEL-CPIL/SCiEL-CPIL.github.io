"""Fresh design for C3-prime: signed step heights minimizing max|autoconv|/(integral)^2."""
import time
import numpy as np

N = 400


def _autoconv(x):
    # full autoconvolution: c[k] = sum_i x[i] x[i-k]  (length 2N-1)
    return np.convolve(x, x, mode="full")


def _objective(x, temp):
    # returns (soft_max_offdiag_sq, grad)
    # we minimize soft-max of squared off-diagonal autoconv values
    c = _autoconv(x)
    # off-diagonal indices: all except the center (lag 0)
    center = N - 1
    mask = np.ones(len(c), dtype=bool)
    mask[center] = False
    off = c[mask]
    sq = off * off
    # soft-max of sq via log-sum-exp
    m = sq.max()
    # guard
    if not np.isfinite(m):
        return 1e18, np.zeros_like(x)
    expv = np.exp((sq - m) / max(temp, 1e-9))
    s = expv.sum()
    if s <= 0:
        return m, np.zeros_like(x)
    w = expv / s
    # soft-max value
    val = m + temp * np.log(s)
    # gradient wrt off = 2*off * w
    grad_off = 2.0 * off * w
    # grad_off is aligned to off (masked order); rebuild full-length grad_c
    grad_c = np.zeros_like(c)
    grad_c[mask] = grad_off
    # c = conv(x,x); d c / d x is the autoconvolution kernel.
    # For full conv: c[k] = sum_i x[i] x[i-k].
    # d c[k] / d x[j] = x[j-k] + x[k-j]  (with bounds)
    # grad_x = sum_k grad_c[k] * (x[j-k]+x[k-j])
    # This is: grad_x = conv(grad_c, x, 'full')[N-1 : 2N-1+N-1] ... let's do via correlation.
    # d c[k]/dx[j] = x[j-k]+x[k-j]. Sum over k:
    #   term1 = sum_k grad_c[k] x[j-k]  = (grad_c * x)[j]  (cross-correlation)
    #   term2 = sum_k grad_c[k] x[k-j]  = (x * grad_c)[j]
    # Both equal np.convolve(grad_c, x, 'full') and np.convolve(x, grad_c, 'full') appropriately indexed.
    a = np.convolve(grad_c, x, mode="full")   # length 2N-1+2N-1-1 = 4N-3
    b = np.convolve(x, grad_c, mode="full")
    # index j (0..N-1) in 'full' of length (N + (2N-1) -1)=3N-2
    # a[j] corresponds to cross-correlation sum_k grad_c[k] x[j-k]
    ga = a[N - 1 : N - 1 + N]
    gb = b[N - 1 : N - 1 + N]
    grad_x = ga + gb
    return val, grad_x


def construct():
    t_start = time.time()
    DEADLINE = 130.0  # safety margin before 150s kill

    rng = np.random.default_rng(12345)

    # ---- initialization: random signed, normalized to unit integral ----
    def init(n):
        x = rng.standard_normal(n)
        s = x.sum()
        if abs(s) < 1e-6:
            x = rng.standard_normal(n)
            s = x.sum()
        x = x / s
        return x

    x = init(N)
    # evaluate hard objective for tracking best
    def hard_obj(x):
        c = _autoconv(x)
        center = N - 1
        m = np.ones(len(c), dtype=bool)
        m[center] = False
        return np.max(np.abs(c[m]))
    best_x = x.copy()
    best_val = hard_obj(best_x)

    # ---- phase 1: coarse soft-max subgradient descent ----
    temp = 5.0
    lr = 0.02
    iters = 4000
    for it in range(iters):
        if time.time() - t_start > DEADLINE:
            break
        val, g = _objective(x, temp)
        # normalize gradient
        gn = np.linalg.norm(g)
        if gn > 0:
            x = x - lr * g / gn
        # renormalize to unit integral (keep on the constraint surface)
        s = x.sum()
        if abs(s) > 1e-9:
            x = x / s
        # track hard objective
        hv = hard_obj(x)
        if hv < best_val:
            best_val = hv
            best_x = x.copy()
        # anneal temperature slowly
        temp *= 0.9995

    # ---- phase 2: polish pass (low lr, low temp, patience) ----
    # measured: lr=1e-05, patience=5000 gives extra squeeze
    x = best_x.copy()
    temp = 0.05
    lr = 1e-05
    patience = 5000
    best_polish = best_val
    no_improve = 0
    for it in range(patience + 5000):
        if time.time() - t_start > DEADLINE:
            break
        val, g = _objective(x, temp)
        gn = np.linalg.norm(g)
        if gn > 0:
            x = x - lr * g / gn
        s = x.sum()
        if abs(s) > 1e-9:
            x = x / s
        hv = hard_obj(x)
        if hv < best_polish - 1e-12:
            best_polish = hv
            no_improve = 0
        else:
            no_improve += 1
        if no_improve > patience:
            break

    if best_polish < best_val:
        best_val = best_polish
        best_x = x.copy()

    # ---- final: return signed step heights ----
    # keep as signed step heights (unit integral)
    return best_x


# quick self-check (not part of evaluation)
if __name__ == "__main__":
    t0 = time.time()
    r = construct()
    print("time", time.time() - t0)
    c = _autoconv(r)
    center = N - 1
    m = np.ones(len(c), dtype=bool)
    m[center] = False
    print("max|offdiag|", np.max(np.abs(c[m])), "integral", r.sum())
