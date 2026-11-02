import numpy as np
import time

N = 50

def _ratio(h):
    """Ratio = ||h||_2^2 / ||h||_1  (valid for nonnegative h, since sup autoconv = ||h||_2^2)."""
    h = np.asarray(h, dtype=float)
    l1 = h.sum()
    if l1 <= 0:
        return 0.0
    return float(np.dot(h, h) / l1)

def _construct():
    start = time.time()
    deadline = start + 135.0
    best = np.ones(N, dtype=float)
    best_val = _ratio(best)

    def consider(h):
        nonlocal best, best_val
        h = np.asarray(h, dtype=float)
        v = _ratio(h)
        if v > best_val:
            best_val = v
            best = h.copy()

    def local_refine(h, iters=4000):
        h = h.copy()
        v = _ratio(h)
        cur = v
        for it in range(iters):
            if time.time() > deadline:
                break
            j = it % N
            old = h[j]
            # candidate amplitudes for this coordinate
            cands = [0.0, old * 0.5, old * 0.8, old * 1.2, old * 1.5, 1.0, 0.1, 0.01]
            if old > 0:
                cands.append(old * 2.0)
            best_c = old
            best_v = cur
            for c in cands:
                if c == old:
                    continue
                h[j] = c
                nv = _ratio(h)
                if nv > best_v:
                    best_v = nv
                    best_c = c
            h[j] = best_c
            if best_c != old:
                cur = best_v
                consider(h)
        return h

    rng = np.random.default_rng(12345)

    def sa(h0, steps=20000):
        h = np.maximum(h0, 0.0).copy()
        cur = _ratio(h)
        T = 0.05
        Tmin = 1e-6
        consider(h)
        for it in range(steps):
            if time.time() > deadline:
                break
            j = rng.integers(0, N)
            old = h[j]
            # multiplicative random move
            factor = np.exp(rng.normal(0.0, 0.35))
            new = min(10.0, max(0.0, old * factor))
            if new == old:
                continue
            h[j] = new
            nv = _ratio(h)
            if nv >= cur or rng.random() < np.exp((nv - cur) / max(T, 1e-9)):
                cur = nv
                consider(h)
            else:
                h[j] = old
            T *= 0.9998
            if T < Tmin:
                T = Tmin
        return h

    # Diverse initial shapes (scale-invariant, so scale is free)
    x = np.linspace(-1, 1, N)
    inits = [
        np.ones(N),
        np.maximum(0.0, 1 - np.abs(x)),
        np.exp(-2 * x**2),
        np.maximum(0.0, 1 - np.abs(x))**3,
        np.exp(-8 * (x - 0.4)**2) + np.exp(-8 * (x + 0.4)**2),
        0.5 + 0.5 * np.cos(4 * np.pi * x),
        np.linspace(1, 0.1, N),
        np.linspace(0.1, 1, N),
        np.ones(N) * 0.5 + 0.5 * np.exp(-20 * x**2),
        np.where(np.abs(x) < 0.3, 1.0, 0.3),
        np.where(np.abs(x) < 0.1, 1.0, 0.05),
        1 - x**2,
        np.maximum(0.0, np.sinc(0.5 * x)),
        0.5 + 0.5 * np.cos(np.pi * x),
    ]
    for _ in range(6):
        inits.append(rng.random(N) + 0.05)

    for h0 in inits:
        if time.time() > deadline:
            break
        h = local_refine(h0, iters=2000)
        if time.time() > deadline:
            break
        h = sa(h, steps=12000)
        h = local_refine(h, iters=3000)

    # Final polish on the global best
    if time.time() < deadline:
        local_refine(best, iters=6000)

    # Normalize (scale-invariant; keep a clean representative)
    mx = best.max()
    if mx > 0:
        best = best / mx

    return list(best)

def construct():
    return _construct()
