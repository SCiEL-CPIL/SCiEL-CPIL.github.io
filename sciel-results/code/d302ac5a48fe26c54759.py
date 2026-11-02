"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Structural approach: continuous subgradient descent on the max-over-linear-shifts
objective with a bisection water-filling projection onto {h in [0,1], sum h = N/2}.
This breaks the bang-bang 0.5 floor toward the ~0.4 continuous optimum.
"""
import time
import numpy as np

N = 100
SCALE = 2.0 / N  # step width factor so that integral = 1


def _compute_O(h):
    """Return vector O of length 2N: O_pos[k], O_neg[k] for linear shifts k=0..N-1."""
    # O_pos[k] = SCALE * sum_{j=k}^{N-1} h[j]*(1-h[j-k])
    S1 = np.cumsum(h[::-1])[::-1]                      # S1[k] = sum_{j=k}^{N-1} h[j]
    S2 = np.array([np.dot(h[k:], h[:N - k]) for k in range(N)])
    O_pos = SCALE * (S1 - S2)
    # O_neg[k] = SCALE * sum_{j=0}^{N-1-k} h[j]*(1-h[j+k])
    csum = np.cumsum(h)
    S1n = csum[N - 1 - np.arange(N)]                   # sum_{j=0}^{N-1-k} h[j]
    S2n = np.array([np.dot(h[:N - k], h[k:]) for k in range(N)])
    O_neg = SCALE * (S1n - S2n)
    return np.concatenate([O_pos, O_neg])


def _step(h, tau):
    """Return (Omax, grad) for the softmax-smoothed max-over-shifts objective."""
    O = _compute_O(h)
    Omax = float(O.max())
    z = tau * (O - Omax)
    ez = np.exp(z)
    w = ez / ez.sum()
    w_pos, w_neg = w[:N], w[N:]

    K = np.arange(N)[:, None]
    I = np.arange(N)[None, :]
    ge = (I >= K)
    lt = (I < (N - K))
    idx_p = np.clip(I - K, 0, N - 1)   # h[i-k]
    idx_m = np.clip(I + K, 0, N - 1)   # h[i+k]
    ge_f = ge.astype(float)
    lt_f = lt.astype(float)
    Gpos = SCALE * (ge_f - ge * h[idx_p] - lt * h[idx_m])
    Gneg = SCALE * (lt_f - lt * h[idx_m] - ge * h[idx_p])
    grad = w_pos @ Gpos + w_neg @ Gneg
    return Omax, grad


def _project(z):
    """Project z onto {h in [0,1], sum h = N/2} via bisection water-filling."""
    lo = float(z.min()) - 1.0
    hi = float(z.max()) + 1.0
    target = N / 2.0
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        s = np.clip(z - mid, 0.0, 1.0).sum()
        if s > target:
            lo = mid
        else:
            hi = mid
    t = 0.5 * (lo + hi)
    return np.clip(z - t, 0.0, 1.0)


def _optimize(init, deadline):
    h = _project(np.asarray(init, dtype=float).copy())
    giter = 0
    for tau in (5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 400.0):
        for _ in range(400):
            if time.time() > deadline - 3.0:
                return h, float(_compute_O(h).max())
            eta = 0.10 / (1.0 + 0.004 * giter)
            Omax, grad = _step(h, tau)
            if not np.all(np.isfinite(grad)):
                break
            h = _project(h - eta * grad)
            giter += 1
    return h, float(_compute_O(h).max())


def construct():
    start = time.time()
    deadline = start + 140.0  # safety margin before the 150s kill

    seed = np.full(N, 0.5)
    best_h = seed.copy()
    best_val = float(_compute_O(seed).max())  # = 0.5

    idx = np.arange(N)
    inits = [seed.copy()]
    # structured continuous inits (break the bang-bang floor early)
    for f in (1, 2, 3, 5, 8):
        inits.append(0.5 + 0.30 * np.cos(2.0 * np.pi * f * idx / N))
        inits.append(0.5 + 0.40 * np.sign(np.cos(2.0 * np.pi * f * idx / N)))
    # random continuous
    rng = np.random.default_rng(0)
    for _ in range(3):
        inits.append(rng.random(N))
    # bang-bang random (N/2 ones)
    for _ in range(2):
        bb = np.zeros(N)
        bb[rng.choice(N, N // 2, replace=False)] = 1.0
        inits.append(bb)

    for init in inits:
        if time.time() > deadline - 8.0:
            break
        try:
            h, val = _optimize(init, deadline)
        except Exception:
            continue
        if np.all(np.isfinite(h)) and val < best_val:
            best_h, best_val = h, val

    # final exact projection + sum fix
    best_h = _project(best_h)
    for _ in range(4):
        d = N / 2.0 - best_h.sum()
        if abs(d) < 1e-12:
            break
        best_h = np.clip(best_h + d / N, 0.0, 1.0)
        best_h = _project(best_h)
    return best_h
