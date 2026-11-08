"""Combined: DCT low-rank parameterization + multi-phase optimization."""
import time
import numpy as np
import scipy.optimize as opt

N = 50
M = 2 * N - 1  # 99
W = 1.0 / (M + 1)  # 1/100
DEADLINE = 145.0
T0 = time.time()

def _remaining():
    return DEADLINE - (time.time() - T0)

def compute_value(h):
    h = np.asarray(h, dtype=float).ravel()
    if h.size != N:
        h = h[:N]
        if h.size < N:
            h = np.pad(h, (0, N - h.size), 'constant')
    if not np.all(np.isfinite(h)) or np.any(h < -1e-12):
        return 0.0
    h = np.maximum(0.0, h)
    g = np.convolve(h, h)
    m = len(g)
    y = np.concatenate(([0.0], g, [0.0]))
    a = y[:m + 1]
    b = y[1:m + 2]
    l2 = (W / 3.0) * np.sum(a * a + a * b + b * b)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_1 <= 0 or norm_inf <= 0:
        return 0.0
    return l2 / (norm_1 * norm_inf)

def _smooth_max(g, beta=50.0):
    g_max = np.max(g)
    if g_max <= 0:
        return 0.0
    return g_max + (1.0 / beta) * np.log(np.sum(np.exp(beta * (g - g_max))))

def _value_neg_smooth(h, beta=50.0):
    h = np.maximum(0.0, h)
    g = np.convolve(h, h)
    m = len(g)
    y = np.concatenate(([0.0], g, [0.0]))
    a = y[:m + 1]
    b = y[1:m + 2]
    l2 = (W / 3.0) * np.sum(a * a + a * b + b * b)
    norm_1 = np.sum(g) / (m + 1)
    if norm_1 <= 0:
        return 1e6
    lse = _smooth_max(g, beta)
    if lse <= 0:
        return 1e6
    return -l2 / (norm_1 * lse)

def _value_neg_exact(h):
    h = np.maximum(0.0, h)
    v = compute_value(h)
    if v <= 0:
        return 1e6
    return -v

# DCT basis
def _make_dct_basis(N, K):
    n = np.arange(N)
    k = np.arange(K)
    basis = np.cos(np.pi * np.outer(2 * n + 1, k) / (2 * N))
    # Normalize
    for i in range(K):
        basis[i] /= np.linalg.norm(basis[i])
    return basis

DCT_K = 15
DCT_BASIS = _make_dct_basis(N, DCT_K)

def _amps_to_h(amps, offset=0.0):
    h = DCT_BASIS.T @ amps + offset
    return np.maximum(0.0, h)

def _h_to_amps(h):
    h = np.asarray(h, dtype=float).ravel()
    if h.size != N:
        h = h[:N]
        if h.size < N:
            h = np.pad(h, (0, N - h.size), 'constant')
    return DCT_BASIS @ h

def _neg_smooth_amps(amps, offset, beta=50.0):
    h = _amps_to_h(amps, offset)
    return _value_neg_smooth(h, beta)

def _neg_exact_amps(amps, offset):
    h = _amps_to_h(amps, offset)
    return _value_neg_exact(h)

def _optimize_amps(amps0, offset, maxfev=300, beta=50.0):
    try:
        res = opt.minimize(_neg_smooth_amps, amps0, args=(offset, beta),
                           method='Nelder-Mead',
                           options={'maxfev': maxfev, 'xatol': 1e-4, 'fatol': 1e-5, 'adaptive': True})
        return res.x
    except Exception:
        return amps0

def _polish_h(h, maxfev=200):
    h = np.maximum(0.0, h)
    try:
        res = opt.minimize(_value_neg_smooth, h, args=(50.0,),
                           method='L-BFGS-B',
                           bounds=[(0.0, None)] * N,
                           options={'maxiter': maxfev // 2, 'ftol': 1e-6})
        hh = np.maximum(res.x, 0.0)
        if _value_neg_exact(hh) < _value_neg_exact(h):
            h = hh
    except Exception:
        pass
    try:
        res = opt.minimize(_value_neg_exact, h,
                           method='Nelder-Mead',
                           options={'maxfev': maxfev, 'xatol': 1e-3, 'fatol': 1e-5, 'adaptive': True})
        hh = np.maximum(res.x, 0.0)
        if _value_neg_exact(hh) < _value_neg_exact(h):
            h = hh
    except Exception:
        pass
    return h

def _spectral_refine(h, iters=40, lr=0.03):
    h = np.maximum(0.0, h)
    best = h.copy()
    bestv = compute_value(best)
    for _ in range(iters):
        if _remaining() < 5.0:
            break
        g = np.convolve(best, best)
        n = N
        L = 2 * n
        C = np.zeros(L)
        C[0] = g[0]
        C[1:n] = g[1:n]
        C[n+1:] = g[1:n][::-1]
        C[n] = 0.0
        spec = np.fft.fft(C)
        spec = np.maximum(spec.real, 0.0)
        mag = np.sqrt(spec)
        hp = np.fft.ifft(mag).real[:n]
        hp = np.maximum(hp, 0.0)
        for t in (0.2, 0.4, 0.6):
            cand = (1 - t) * best + t * hp
            cand = np.maximum(cand, 0.0)
            v = compute_value(cand)
            if v > bestv:
                best = cand
                bestv = v
    return best

def _generate_starts():
    rng = np.random.RandomState(42)
    starts = []
    x = np.linspace(0, N - 1, N)
    
    # Uniform
    starts.append(_h_to_amps(np.ones(N)))
    
    # Gaussian bump
    gauss = np.exp(-0.5 * ((x - N / 2) / (N / 4)) ** 2)
    starts.append(_h_to_amps(gauss))
    
    # Semicircle
    semi = np.sqrt(np.maximum(0.0, 1 - ((2 * x / (N - 1) - 1)) ** 2))
    starts.append(_h_to_amps(semi))
    
    # Two peaks
    p1 = np.exp(-0.5 * ((x - N * 0.25) / (N / 8)) ** 2)
    p2 = np.exp(-0.5 * ((x - N * 0.75) / (N / 8)) ** 2)
    starts.append(_h_to_amps(p1 + p2))
    
    # Flat-top
    for frac in [0.3, 0.5, 0.7]:
        width = int(N * frac)
        flat = np.zeros(N)
        si = (N - width) // 2
        flat[si:si + width] = 1.0
        starts.append(_h_to_amps(flat))
    
    # Plateau-like
    for K_plateau in [2, 3, 4]:
        for _ in range(3):
            cuts = sorted(rng.randint(1, N, size=K_plateau - 1))
            cuts = [0] + list(cuts) + [N]
            levels = rng.uniform(0.2, 1.0, size=K_plateau)
            h_plateau = np.zeros(N)
            for i in range(K_plateau):
                h_plateau[cuts[i]:cuts[i+1]] = levels[i]
            starts.append(_h_to_amps(h_plateau))
    
    # Random smooth
    for _ in range(10):
        amps = rng.normal(0, 0.5, DCT_K)
        h_test = DCT_BASIS.T @ amps
        if np.mean(h_test) > 0.05:
            starts.append(amps)
    
    return starts

def construct():
    global T0
    T0 = time.time()
    
    best_h = np.ones(N)
    best_v = compute_value(best_h)
    
    starts = _generate_starts()
    
    # Phase 1: Optimize in DCT amplitude space
    for amps0 in starts:
        if _remaining() < 8.0:
            break
        for offset in [0.0, 0.1, 0.2]:
            if _remaining() < 5.0:
                break
            amps = _optimize_amps(amps0.copy(), offset, maxfev=200, beta=50.0)
            h = _amps_to_h(amps, offset)
            v = compute_value(h)
            if v > best_v:
                best_v = v
                best_h = h.copy()
    
    # Phase 2: Spectral refinement
    if _remaining() > 10.0:
        best_h = _spectral_refine(best_h, iters=30)
        best_v = compute_value(best_h)
    
    # Phase 3: Full-space polish
    if _remaining() > 10.0:
        best_h = _polish_h(best_h, maxfev=200)
        best_v = compute_value(best_h)
    
    # Phase 4: More DCT starts around best
    if _remaining() > 10.0:
        best_amps = _h_to_amps(best_h)
        rng = np.random.RandomState(123)
        for _ in range(8):
            if _remaining() < 5.0:
                break
            amps_pert = best_amps + rng.normal(0, 0.1, DCT_K)
            for offset in [0.0, 0.05]:
                amps = _optimize_amps(amps_pert, offset, maxfev=150, beta=50.0)
                h = _amps_to_h(amps, offset)
                v = compute_value(h)
                if v > best_v:
                    best_v = v
                    best_h = h.copy()
        if _remaining() > 8.0:
            best_h = _polish_h(best_h, maxfev=150)
            best_v = compute_value(best_h)
    
    # Final safety
    best_h = np.maximum(best_h, 0.0)
    best_h = np.where(np.isfinite(best_h), best_h, 0.0)
    return list(best_h)
