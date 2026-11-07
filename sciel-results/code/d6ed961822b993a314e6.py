"""Second autocorrelation inequality - Dinkelbach with DCT parameterization."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.fft import dct, idct

N = 50

def _compute_value(h):
    h = np.asarray(h, dtype=float)
    if not np.all(np.isfinite(h)) or np.any(h < -1e-15):
        return 0.0
    h = np.maximum(h, 0.0)
    g = np.convolve(h, h)
    m = len(g)
    y = np.concatenate(([0.0], g, [0.0]))
    w = 1.0 / (m + 1)
    y_sq = y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2
    l2 = (w / 3.0) * np.sum(y_sq)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_1 * norm_inf < 1e-300:
        return 0.0
    return l2 / (norm_1 * norm_inf)

def _aux_obj(h, lam):
    """Auxiliary: l2 - lam * norm_1 * norm_inf (for Dinkelbach)."""
    h = np.maximum(np.asarray(h, dtype=float), 0.0)
    g = np.convolve(h, h)
    m = len(g)
    y = np.concatenate(([0.0], g, [0.0]))
    w = 1.0 / (m + 1)
    y_sq = y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2
    l2 = (w / 3.0) * np.sum(y_sq)
    norm_1 = np.sum(g) / (m + 1)
    norm_inf = np.max(g)
    return l2 - lam * norm_1 * norm_inf

def _dinkelbach_dct(h0, max_iter=20, K=12, deadline=None):
    """Dinkelbach's algorithm using L-BFGS-B in DCT space."""
    h = np.maximum(np.asarray(h0, dtype=float), 0.0)
    h = h / max(h.sum(), 1e-12)
    lam = _compute_value(h)
    
    # Initial DCT coefficients
    coeffs = dct(h, type=2, norm='ortho')
    init_coeffs = np.zeros(N)
    init_coeffs[:K] = coeffs[:K]
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline - 5:
            break
            
        # Optimize auxiliary in DCT space
        def neg_aux(c):
            h_raw = np.abs(idct(c, type=2, norm='ortho'))
            s = h_raw.sum()
            if s < 1e-12:
                return 1e10
            h_norm = h_raw / s
            return -_aux_obj(h_norm, lam)
        
        try:
            res = minimize(neg_aux, init_coeffs.copy(), method='L-BFGS-B',
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
            h_new = np.abs(idct(res.x, type=2, norm='ortho'))
            s = h_new.sum()
            if s > 1e-12:
                h_new = h_new / s
                new_val = _compute_value(h_new)
                if new_val > lam:
                    h = h_new
                    lam = new_val
                    init_coeffs[:K] = dct(h, type=2, norm='ortho')[:K]
                else:
                    break
            else:
                break
        except Exception:
            break
            
    return h, _compute_value(h)

def _lbfgs_polish(h0, deadline, maxiter=500):
    """L-BFGS-B polish on the ratio directly."""
    h = np.maximum(np.asarray(h0, dtype=float), 0.0)
    h = h / max(h.sum(), 1e-12)
    
    def negv(p):
        hh = np.maximum(p, 1e-12)
        s = hh.sum()
        if s < 1e-12:
            return 1e10
        hh = hh / s
        return -_compute_value(hh)
    
    try:
        res = minimize(negv, h.copy(), method='L-BFGS-B',
                       bounds=[(1e-12, None)] * N,
                       options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-10})
        hh = np.maximum(res.x, 0.0)
        s = hh.sum()
        if s > 1e-12:
            hh = hh / s
        v = _compute_value(hh)
        if v > _compute_value(h):
            return hh, v
    except Exception:
        pass
    return h, _compute_value(h)

def _gen_starts(rng):
    """Generate diverse structured starts."""
    starts = []
    idx = np.arange(N)
    x = idx / (N - 1) - 0.5
    # Uniform
    starts.append(np.ones(N))
    # Gaussians
    for s in [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8]:
        starts.append(np.exp(-x**2 / (2 * s**2)))
    # Super/sub-Gaussian
    for p in [0.5, 1, 1.5, 2, 3, 4]:
        for s in [0.1, 0.2, 0.4, 0.7]:
            starts.append(np.exp(-np.abs(x) ** p / s))
    # Compact support
    for w in [0.1, 0.2, 0.3, 0.5]:
        starts.append(np.maximum(0, 1 - np.abs(x) / w))
    # Plateau
    for w in [0.1, 0.2, 0.3]:
        starts.append(np.where(np.abs(x) < w, 1.0, 0.0) + 1e-6)
    # Bimodal
    for c in [0.15, 0.25, 0.35]:
        for s in [0.05, 0.1, 0.15]:
            starts.append(np.exp(-(x - c)**2 / (2*s**2)) + np.exp(-(x + c)**2 / (2*s**2)))
    # Tri-modal
    for s in [0.08, 0.12, 0.2]:
        starts.append(np.exp(-(x - 0.4)**2/(2*s**2)) + np.exp(-x**2/(2*s**2)) + np.exp(-(x + 0.4)**2/(2*s**2)))
    # Lorentzian
    for a in [2, 5, 10, 20]:
        starts.append(1.0 / (1 + (a * x)**2))
    # Power-law
    for a in [0.5, 1, 2, 3]:
        starts.append(1.0 / (1 + np.abs(x) / 0.2) ** a)
    # Random smooth
    for _ in range(25):
        nfreq = int(rng.integers(2, 8))
        coeffs = rng.standard_normal(nfreq)
        phases = rng.uniform(0, 2*np.pi, nfreq)
        h_f = np.ones(N)
        for k in range(1, nfreq + 1):
            h_f += coeffs[k-1] * np.cos(2*np.pi*k*x + phases[k-1])
        starts.append(np.abs(h_f) + 0.02)
    # Random
    for _ in range(25):
        starts.append(np.abs(rng.standard_normal(N)) + 0.02)
    return starts

def construct():
    t0 = time.time()
    deadline = t0 + 140.0

    rng = np.random.default_rng(50)
    starts = _gen_starts(rng)

    best_h = np.ones(N) / N
    best_val = _compute_value(best_h)

    def try_candidate(h0):
        nonlocal best_h, best_val
        if time.time() > deadline - 10:
            return
        h0 = np.maximum(np.asarray(h0, dtype=float), 0.0)
        s = h0.sum()
        if s < 1e-12:
            return
        h0 = h0 / s
        # Dinkelbach in DCT space
        try:
            h_d, v_d = _dinkelbach_dct(h0, max_iter=15, K=12, deadline=deadline)
            if v_d > best_val:
                best_val = v_d
                best_h = h_d.copy()
        except Exception:
            pass
        # L-BFGS-B polish on current best
        try:
            h_n, v_n = _lbfgs_polish(best_h, deadline, maxiter=300)
            if v_n > best_val:
                best_val = v_n
                best_h = h_n.copy()
        except Exception:
            pass

    # Phase 1: Dinkelbach + Polish from structured starts
    for h0 in starts:
        if time.time() > deadline - 20:
            break
        try_candidate(h0)

    # Phase 2: More random restarts
    for seed in [1, 2, 3, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]:
        if time.time() > deadline - 15:
            break
        rng2 = np.random.default_rng(seed)
        for _ in range(3):
            if time.time() > deadline - 15:
                break
            h0 = np.abs(rng2.standard_normal(N)) + 0.02
            try_candidate(h0)

    # Phase 3: Final high-precision polish
    if time.time() < deadline - 5:
        try:
            h_f, v_f = _lbfgs_polish(best_h, deadline, maxiter=1000)
            if v_f > best_val:
                best_val = v_f
                best_h = h_f.copy()
        except Exception:
            pass

    # Ensure validity
    best_h = np.maximum(np.asarray(best_h, dtype=float), 0.0)
    s = best_h.sum()
    if s < 1e-12:
        best_h = np.ones(N) / N
    else:
        best_h = best_h / s
    return list(best_h)
