"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 50
DEADLINE = 145.0  # safety margin before 150s kill
T0 = time.time()

def _remaining():
    return DEADLINE - (time.time() - T0)

def _value_from_h(h):
    h = np.asarray(h, dtype=float)
    if np.any(h < 0) or not np.all(np.isfinite(h)):
        return -1.0
    g = np.convolve(h, h)          # length 2N-1 = 99
    m = len(g)
    y = np.concatenate(([0.0], g, [0.0]))  # length m+2 = 101
    w = 1.0 / (m + 1)              # linspace(-0.5,0.5,m+2) spacing
    l2 = (w / 3.0) * np.sum(y[:-1]**2 + y[:-1]*y[1:] + y[1:]**2)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_1 <= 0 or norm_inf <= 0:
        return 0.0
    return l2 / (norm_1 * norm_inf)

def _value_from_g(g):
    g = np.asarray(g, dtype=float)
    if not np.all(np.isfinite(g)) or np.any(g < -1e-12):
        return -1.0
    m = len(g)
    y = np.concatenate(([0.0], g, [0.0]))
    w = 1.0 / (m + 1)
    l2 = (w / 3.0) * np.sum(y[:-1]**2 + y[:-1]*y[1:] + y[1:]**2)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_1 <= 0 or norm_inf <= 0:
        return 0.0
    return l2 / (np.sum(np.abs(g)) / (m + 1) * np.max(np.abs(g)))

def _h_to_g(h):
    return np.convolve(np.asarray(h, dtype=float), np.asarray(h, dtype=float))

def _g_to_h(g):
    """Recover a nonnegative h (length N) whose autocorrelation is ~g.
    Use spectral factorization: h = IFFT(sqrt(FFT(g_padded))) with phase fix,
    then project to nonnegatives."""
    N = len(g)  # g length = 2N-1 -> N = (len(g)+1)//2
    n = (len(g) + 1) // 2
    # Build full circular autocorrelation spectrum
    L = 2 * n
    G = np.zeros(L, dtype=float)
    G[:len(g)] = g
    # For a real autocorrelation, the circular version: place g appropriately
    # g[k] for k=0..2n-2. Circular autocorr length L=2n:
    #   C[0]=g[0], C[k]=g[k] for 1<=k<=n-1, C[n]=0 (or g? ), C[L-k]=g[k]
    C = np.zeros(L)
    C[0] = g[0]
    C[1:n] = g[1:n]
    C[n+1:] = g[1:n][::-1]
    C[n] = 0.0
    spec = np.fft.fft(C)
    spec = np.maximum(spec.real, 0.0)
    mag = np.sqrt(spec)
    h = np.fft.ifft(mag).real
    # take first n samples
    h = h[:n]
    h = np.maximum(h, 0.0)
    return h

# --- Plateau (bimodal box) generator: h is high on two intervals, low elsewhere ---
def _plateau_h(a, b, c, d, base, peak):
    """a,b,c,d in [0,1] with a<b<c<d. h high (peak) on [a,b] and [c,d], base elsewhere."""
    x = np.linspace(0.0, 1.0, N + 1)[:-1] + 0.5 / N  # cell centers
    h = np.full(N, base)
    mask1 = (x >= a) & (x < b)
    mask2 = (x >= c) & (x < d)
    h[mask1] = peak
    h[mask2] = peak
    return h

def _optimize_plateau(p):
    a, b, c, d, base, peak = p
    # enforce ordering
    if not (a < b < c < d):
        return -1.0, None
    h = _plateau_h(a, b, c, d, base, peak)
    v = _value_from_h(h)
    return v, h

def _nelder_mead_plateau(p0, maxfev=400):
    from scipy.optimize import minimize
    def negf(p):
        if not (p[0] < p[1] < p[2] < p[3]):
            return 1e9
        v, _ = _optimize_plateau(p)
        return -v
    try:
        res = minimize(negf, p0, method='Nelder-Mead',
                       options={'maxfev': maxfev, 'xatol': 1e-4, 'fatol': 1e-5, 'adaptive': True})
        p = res.x
        if not (p[0] < p[1] < p[2] < p[3]):
            return p0
        return p
    except Exception:
        return p0

def _refine_h(h, maxfev=300):
    """Local refinement of h directly with Nelder-Mead on the 50-dim (projected) space."""
    from scipy.optimize import minimize
    h = np.maximum(np.asarray(h, dtype=float), 0.0)
    def negf(hh):
        hh = np.maximum(hh, 0.0)
        v = _value_from_h(hh)
        if v < 0:
            return 1e9
        return -v
    try:
        res = minimize(negf, h, method='Nelder-Mead',
                       options={'maxfev': maxfev, 'xatol': 1e-3, 'fatol': 1e-5, 'adaptive': True})
        hh = np.maximum(res.x, 0.0)
        if _value_from_h(hh) > _value_from_h(h):
            return hh
    except Exception:
        pass
    return h

def _spectral_refine(h, iters=60, lr=0.02):
    """Iteratively: g=autocorr(h), find h' = spectral factor of g, blend, project."""
    h = np.maximum(np.asarray(h, dtype=float), 0.0)
    best = h.copy()
    bestv = _value_from_h(best)
    for _ in range(iters):
        if _remaining() < 5.0:
            break
        g = _h_to_g(best)
        # normalize g to keep scale stable (value is scale-invariant? check)
        # value = l2/(norm1*norminf). l2 ~ h^2, norm1~h, norminf~h -> ratio scale invariant.
        hp = _g_to_h(g)
        # blend
        for t in (0.3, 0.5, 0.7):
            cand = (1 - t) * best + t * hp
            cand = np.maximum(cand, 0.0)
            v = _value_from_h(cand)
            if v > bestv:
                best = cand
                bestv = v
    return best

def construct():
    global T0
    T0 = time.time()
    rng = np.random.default_rng(12345)

    best_h = np.ones(N)
    best_v = _value_from_h(best_h)

    # --- Phase 1: random plateau restarts + Nelder-Mead ---
    n_plateau_starts = 40
    for i in range(n_plateau_starts):
        if _remaining() < 8.0:
            break
        # random bimodal plateau
        if i % 2 == 0:
            # symmetric-ish bimodal
            a = rng.uniform(0.0, 0.2)
            b = rng.uniform(0.25, 0.45)
            c = 1.0 - b
            d = 1.0 - a
            base = rng.uniform(0.0, 0.3)
            peak = rng.uniform(0.8, 2.0)
        else:
            # random 4 boundaries
            pts = np.sort(rng.uniform(0, 1, 4))
            a, b, c, d = pts
            base = rng.uniform(0.0, 0.4)
            peak = rng.uniform(0.8, 2.5)
        p0 = np.array([a, b, c, d, base, peak])
        p = _nelder_mead_plateau(p0, maxfev=250)
        h = _plateau_h(*p)
        v = _value_from_h(h)
        if v > best_v:
            best_v = v
            best_h = h.copy()

    # --- Phase 2: spectral refinement on best ---
    if _remaining() > 10.0:
        best_h = _spectral_refine(best_h)
        best_v = _value_from_h(best_h)

    # --- Phase 3: direct Nelder-Mead refinement on best_h ---
    if _remaining() > 10.0:
        best_h = _refine_h(best_h, maxfev=250)
        best_v = _value_from_h(best_h)

    # --- Phase 4: more plateau restarts focused around best (local search) ---
    if _remaining() > 10.0:
        # derive a plateau approximation from best_h to seed local restarts
        # find high regions
        thr = np.mean(best_h)
        high = best_h > thr
        # get contiguous high blocks
        idx = np.where(high)[0]
        if len(idx) >= 2:
            # split into two halves
            mid = len(idx) // 2
            a0 = idx[0] / N
            b0 = (idx[mid - 1] + 1) / N
            c0 = idx[mid] / N
            d0 = (idx[-1] + 1) / N
            base0 = np.min(best_h[~high]) if np.any(~high) else 0.0
            peak0 = np.max(best_h[high])
            for k in range(12):
                if _remaining() < 6.0:
                    break
                a = a0 + rng.uniform(-0.05, 0.05)
                b = b0 + rng.uniform(-0.05, 0.05)
                c = c0 + rng.uniform(-0.05, 0.05)
                d = d0 + rng.uniform(-0.05, 0.05)
                a = np.clip(a, 0.0, 1.0)
                b = np.clip(b, 0.0, 1.0)
                c = np.clip(c, 0.0, 1.0)
                d = np.clip(d, 0.0, 1.0)
                if not (a < b < c < d):
                    continue
                base = max(0.0, base0 + rng.uniform(-0.1, 0.1))
                peak = max(0.1, peak0 + rng.uniform(-0.2, 0.2))
                p0 = np.array([a, b, c, d, base, peak])
                p = _nelder_mead_plateau(p0, maxfev=200)
                h = _plateau_h(*p)
                v = _value_from_h(h)
                if v > best_v:
                    best_v = v
                    best_h = h.copy()
        # spectral + NM again
        if _remaining() > 8.0:
            best_h = _spectral_refine(best_h, iters=40)
            best_v = _value_from_h(best_h)
        if _remaining() > 8.0:
            best_h = _refine_h(best_h, maxfev=200)
            best_v = _value_from_h(best_h)

    # final safety: ensure valid
    best_h = np.maximum(best_h, 0.0)
    best_h = np.where(np.isfinite(best_h), best_h, 0.0)
    return list(best_h)

# EVOLVE-BLOCK-END
