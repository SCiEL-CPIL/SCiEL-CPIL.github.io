"""Erdos minimum overlap: Barker-style bang-bang + continuous bisection refinement.

We minimize max_k ov[k] where ov[k]=sum_i h[i](1-h[(i+k)%N]) = N/2 - C[k],
C being the circular autocorrelation.  Min max ov <=> max min C.

Two complementary mechanisms:
  * Bang-bang (Barker) search: O(1) per flip via incremental overlap update;
    flip *pairs* to keep sum=N/2 exactly; drive C[k] toward equalization.
  * Continuous refinement: the max-overlap constraint is monotone in a scalar
    t; we bisect on t with a water-filling projection, then a subgradient pass.
"""
import numpy as np
import time

N = 100
HALF = N / 2.0
DEADLINE = 142.0  # safety margin before 150s kill


def _time_left(t0):
    return DEADLINE - (time.time() - t0)


def _overlap_vector(h):
    """ov[k] = sum_i h[i]*(1-h[(i+k)%N]) via FFT cross-correlation."""
    h = np.asarray(h, dtype=np.float64)
    H = np.fft.fft(h)
    OmH = np.fft.fft(1.0 - h)
    ov = np.fft.ifft(np.conj(H) * OmH).real
    return ov


def _max_overlap(h):
    return float(np.max(_overlap_vector(h)))


def _project(h):
    """Project to [0,1]^N with sum exactly N/2."""
    h = np.clip(np.asarray(h, dtype=np.float64), 0.0, 1.0)
    for _ in range(40):
        diff = HALF - np.sum(h)
        if abs(diff) < 1e-14:
            break
        if diff > 0:
            mask = h < 1.0 - 1e-15
            if not np.any(mask):
                break
            h[mask] += diff / np.sum(mask)
        else:
            mask = h > 1e-15
            if not np.any(mask):
                break
            h[mask] += diff / np.sum(mask)
        h = np.clip(h, 0.0, 1.0)
    diff = HALF - np.sum(h)
    if abs(diff) > 1e-14:
        idx = np.argsort(1.0 - h) if diff > 0 else np.argsort(h)
        for i in idx:
            if diff > 0:
                add = min(diff, 1.0 - h[i])
                h[i] += add
                diff -= add
            else:
                sub = min(-diff, h[i])
                h[i] -= sub
                diff += sub
            if abs(diff) < 1e-14:
                break
    return h


def _barker_init(rng):
    """Random bang-bang (0/1) with exactly N/2 ones."""
    x = np.zeros(N)
    ones = N // 2
    idx = rng.permutation(N)[:ones]
    x[idx] = 1.0
    return x


def _barker_search(x, t0, max_iter=4000000):
    """Flip-pair local search on bang-bang x minimizing max overlap.

    ov is maintained incrementally in O(1) per flip.  A flip pair (a,b) is
    accepted if it lowers the max overlap (or, near a plateau, lowers a cheap
    O(N) variance-of-C score).  Sum is preserved exactly by flipping pairs.
    """
    x = np.asarray(x, dtype=np.float64).copy()
    ov = _overlap_vector(x)
    cur = float(np.max(ov))
    rng = np.random.default_rng(123)
    it = 0
    stalls = 0
    while it < max_iter and _time_left(t0) > 12:
        it += 1
        a = int(rng.integers(N))
        b = int(rng.integers(N))
        if a == b:
            continue
        # apply flip a
        ov += 1.0 - 2.0 * x[a]
        x[a] = 1.0 - x[a]
        # apply flip b
        ov += 1.0 - 2.0 * x[b]
        x[b] = 1.0 - x[b]
        new = float(np.max(ov))
        if new < cur - 1e-13:
            cur = new
            stalls = 0
        else:
            # cheap O(N) tie-break: lower variance of C (flatter = better)
            c = HALF - ov
            var = float(np.var(c[1:]))
            if var < 0.02 and new <= cur + 1e-12:
                cur = new
                stalls = 0
            else:
                # revert
                ov -= 1.0 - 2.0 * x[b]
                x[b] = 1.0 - x[b]
                ov -= 1.0 - 2.0 * x[a]
                x[a] = 1.0 - x[a]
                stalls += 1
        if stalls > 200000:
            break
    return x, cur


def _barker_multi(x, t0):
    """Run Barker search from a few random perturbations, keep the best."""
    best_x = x.copy()
    best_v = _max_overlap(x)
    rng = np.random.default_rng(7)
    # perturb: flip a handful of pairs
    cand = x.copy()
    for _ in range(6):
        a = int(rng.integers(N)); b = int(rng.integers(N))
        if a != b:
            cand[a] = 1.0 - cand[a]; cand[b] = 1.0 - cand[b]
    xs, v = _barker_search(cand, t0, max_iter=1500000)
    if v < best_v:
        best_x, best_v = xs, v
    return best_x, best_v


def _project_t(h, t):
    """Water-filling projection: max(0, min(1, h - t)) adjusted to sum N/2.

    For t in the active range, sum is strictly decreasing in t, so we can
    bisection on t to hit sum=N/2 exactly."""
    h = np.asarray(h, dtype=np.float64)
    def sum_at(t):
        return np.sum(np.clip(h - t, 0.0, 1.0))
    lo, hi = -1.0, 1.0
    # ensure bracket
    if sum_at(lo) < HALF:
        lo = -2.0
    if sum_at(hi) > HALF:
        hi = 2.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if sum_at(mid) > HALF:
            lo = mid
        else:
            hi = mid
    t = 0.5 * (lo + hi)
    out = np.clip(h - t, 0.0, 1.0)
    # exact fix
    diff = HALF - np.sum(out)
    if abs(diff) > 1e-14:
        if diff > 0:
            mask = out < 1.0 - 1e-15
            if np.any(mask):
                out[mask] += diff / np.sum(mask)
        else:
            mask = out > 1e-15
            if np.any(mask):
                out[mask] += diff / np.sum(mask)
        out = np.clip(out, 0.0, 1.0)
    return out


def _continuous_refine(h0, t0, max_iter=4000):
    """Bisection on t to drive max overlap to the feasible optimum, then a
    projected subgradient polish.  Returns (h, val)."""
    h = np.asarray(h0, dtype=np.float64).copy()
    best_h = h.copy()
    best_v = _max_overlap(h)
    # bisection on t: larger t => flatter => lower max overlap
    lo, hi = 0.0, 1.0
    # find hi where max overlap is already low
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        h = _project_t(h0, mid)
        v = _max_overlap(h)
        if v < best_v:
            best_v = v
            best_h = h.copy()
        if v < 0.30:
            hi = mid
        else:
            lo = mid
        if _time_left(t0) < 8:
            break
    # subgradient polish around best_h
    h = best_h.copy()
    step = 0.02
    for it in range(max_iter):
        if _time_left(t0) < 8:
            break
        ov = _overlap_vector(h)
        k = int(np.argmax(ov))
        # subgradient of max ov w.r.t. h: g[i] = ov[k] - h[i] - h[(i+k)%N]
        g = ov[k] - h - np.roll(h, k)
        g -= np.mean(g)  # keep sum
        h2 = _project(h - step * g)
        v2 = _max_overlap(h2)
        if v2 < best_v:
            best_v = v2
            best_h = h2.copy()
            h = h2
            step = min(step * 1.15, 0.15)
        else:
            step *= 0.8
            if step < 1e-4:
                break
    return best_h, best_v


def construct():
    t0 = time.time()
    rng = np.random.default_rng(2024)

    best_h = None
    best_val = float('inf')

    def consider(h):
        nonlocal best_h, best_val
        h = _project(h)
        v = _max_overlap(h)
        if v < best_val:
            best_val = v
            best_h = h.copy()

    # --- Bang-bang Barker search (fast) ---
    base = _barker_init(rng)
    base, v = _barker_search(base, t0, max_iter=2500000)
    consider(base)
    if _time_left(t0) > 30:
        base2, v2 = _barker_multi(base, t0)
        consider(base2)

    # --- Continuous refinement from the best bang-bang ---
    if _time_left(t0) > 25:
        hc, vc = _continuous_refine(best_h, t0, max_iter=3000)
        consider(hc)

    # --- A few more continuous restarts from perturbed bang-bang ---
    for _ in range(3):
        if _time_left(t0) < 20:
            break
        x = best_h.copy()
        for _ in range(10):
            a = int(rng.integers(N)); b = int(rng.integers(N))
            if a != b:
                x[a] = 1.0 - x[a]; x[b] = 1.0 - x[b]
        hc, vc = _continuous_refine(x, t0, max_iter=1500)
        consider(hc)

    # --- Final polish: light SLSQP if time permits ---
    if _time_left(t0) > 15:
        try:
            from scipy.optimize import minimize
            h0 = best_h.copy()
            res = minimize(
                _max_overlap,
                h0,
                method='SLSQP',
                bounds=[(0.0, 1.0)] * N,
                constraints=[{'type': 'eq',
                              'fun': lambda x: np.sum(x) - HALF,
                              'jac': lambda x: np.ones(N)}],
                options={'maxiter': 200, 'ftol': 1e-14, 'disp': False}
            )
            consider(_project(res.x))
        except Exception:
            pass

    final_h = _project(best_h)
    final_h = np.clip(final_h, 0.0, 1.0)
    final_h = _project(final_h)
    assert abs(np.sum(final_h) - HALF) < 1e-9, f"Sum violated: {np.sum(final_h)}"
    return final_h


if __name__ == "__main__":
    t_start = time.time()
    h = construct()
    t_end = time.time()
    print(f"Time: {t_end - t_start:.2f}s")
    print("Sum:", np.sum(h))
    print("Min:", np.min(h), "Max:", np.max(h))
    print("Max overlap:", _max_overlap(h))
    baseline = _max_overlap(np.full(100, 0.5))
    print("Baseline overlap:", baseline)
    print("Score (seed/value):", baseline / _max_overlap(h))
