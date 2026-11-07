"""Second autocorrelation inequality - optimized with coordinate descent + multi-start."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _compute_value(h):
    h = np.asarray(h, dtype=float)
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

def _neg_value(params):
    h = _h_from_params(params)
    return -_compute_value(h)

def _h_from_params(params):
    e = np.exp(params - np.max(params))
    h = e / e.sum()
    return h

def _coordinate_descent(h, max_sweeps=20, tol=1e-10, deadline=None):
    """Coordinate descent: optimize each h[i] by golden section search."""
    h = np.array(h, dtype=float)
    h = np.maximum(h, 1e-10)
    h = h / h.sum()
    best = _compute_value(h)
    
    for sweep in range(max_sweeps):
        if deadline is not None and time.time() > deadline - 2:
            break
        improved = False
        for i in range(N):
            if deadline is not None and time.time() > deadline - 2:
                break
            # Save current value
            old_val = _compute_value(h)
            old_hi = h[i]
            
            # Golden section search for h[i] in [1e-8, 2.0 * old_hi + 0.1]
            lo = 1e-8
            hi = max(2.0 * old_hi + 0.1, 0.5)
            
            # Golden ratio
            gr = (np.sqrt(5) - 1) / 2
            a = lo + (1 - gr) * (hi - lo)
            b = lo + gr * (hi - lo)
            fa = _compute_value(np.where(np.arange(N) == i, a, h))
            fb = _compute_value(np.where(np.arange(N) == i, b, h))
            
            for _ in range(30):
                if deadline is not None and time.time() > deadline - 2:
                    break
                if fa < fb:
                    hi = b
                    b = a
                    fb = fa
                    a = lo + (1 - gr) * (hi - lo)
                    fa = _compute_value(np.where(np.arange(N) == i, a, h))
                else:
                    lo = a
                    a = b
                    fa = fb
                    b = lo + gr * (hi - lo)
                    fb = _compute_value(np.where(np.arange(N) == i, b, h))
            
            best_i = (a + b) / 2
            val_i = _compute_value(np.where(np.arange(N) == i, best_i, h))
            
            if val_i > old_val + tol:
                h[i] = best_i
                h = h / h.sum()
                improved = True
                best = _compute_value(h)
        
        if not improved:
            break
    
    return h, best

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 145.0

    best_h = np.ones(N)
    best_val = _compute_value(best_h)

    def try_optimize(params0, method='Nelder-Mead', maxiter=3000, **kwargs):
        nonlocal best_h, best_val
        if time.time() > deadline - 5:
            return
        try:
            if method == 'Nelder-Mead':
                opts = {'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            elif method == 'L-BFGS-B':
                opts = {'maxiter': maxiter, 'gtol': 1e-7}
            elif method == 'Powell':
                opts = {'maxiter': maxiter, 'xtol': 1e-9, 'ftol': 1e-12}
            else:
                opts = {}
            res = minimize(_neg_value, params0, method=method, options=opts)
            if res.fun < -best_val:
                best_val = -res.fun
                best_h = _h_from_params(res.x)
        except Exception:
            pass

    def try_coordinate(h0, max_sweeps=20):
        nonlocal best_h, best_val
        if time.time() > deadline - 3:
            return
        try:
            h_new, val_new = _coordinate_descent(h0, max_sweeps=max_sweeps, deadline=deadline)
            if val_new > best_val:
                best_val = val_new
                best_h = h_new
        except Exception:
            pass

    rng = np.random.default_rng(50)
    x = np.linspace(-1, 1, N)
    xh = np.linspace(-0.5, 0.5, N)

    # Generate diverse initial candidates
    candidates = []
    # Uniform
    candidates.append(np.ones(N))
    # Various Gaussians
    for a in [0.3, 0.5, 1, 2, 4, 8, 16, 32, 64]:
        candidates.append(np.exp(-a * x**2))
    # Polynomials
    for k in [1.5, 2, 2.5, 3, 4, 5, 6, 8]:
        candidates.append(np.maximum(0, 1 - np.abs(x)**k))
    # Cosine
    candidates.append(np.maximum(0, np.cos(np.pi * xh)))
    candidates.append(np.maximum(0, np.cos(2 * np.pi * xh)))
    candidates.append(np.maximum(0, np.cos(np.pi * xh)**2))
    candidates.append(np.maximum(0, np.cos(1.5 * np.pi * xh)))
    # Sine
    candidates.append(np.maximum(0, np.sin(np.pi * xh)))
    candidates.append(np.maximum(0, np.sin(np.pi * xh)**2))
    # Bimodal
    for c in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        for a in [4, 8, 12, 20, 40]:
            candidates.append(np.exp(-a*(x-c)**2) + np.exp(-a*(x+c)**2))
    # Plateau-like with smooth edges
    for start, end in [(5,45),(10,40),(15,35),(0,50),(3,47),(7,43)]:
        plat = np.zeros(N)
        plat[start:end] = 1.0
        # Smooth edges
        for j in range(start, min(start+3, N)):
            plat[j] = (j - start + 1) / 4.0
        for j in range(max(end-3, 0), end):
            plat[j] = (end - j) / 4.0
        candidates.append(np.maximum(plat, 1e-6))
    # Triangle
    candidates.append(np.maximum(0, 1 - 2*np.abs(xh)))
    # Sech-like
    candidates.append(1.0 / np.cosh(2 * x))
    candidates.append(1.0 / np.cosh(4 * x))
    # Random
    for _ in range(25):
        h_r = np.abs(rng.standard_normal(N)) + 0.01
        candidates.append(h_r)
    # Smooth random via low-frequency Fourier
    for _ in range(15):
        coeffs = rng.standard_normal(8)
        h_f = np.ones(N)
        for k in range(1, 8):
            h_f += coeffs[k-1] * np.cos(2*np.pi*k*xh)
        h_f = np.abs(h_f) + 0.01
        candidates.append(h_f)
    # Concentrated in center
    for width in [5, 8, 10, 15, 20]:
        center = N // 2
        idx = np.arange(N)
        candidates.append(np.exp(-((idx - center)**2) / (2*width**2)))

    # Phase 1: Quick NM from diverse candidates
    for h0 in candidates:
        if time.time() > deadline - 30:
            break
        h0 = np.asarray(h0, dtype=float)
        h0 = np.maximum(h0, 1e-6)
        h0 = h0 / h0.sum()
        params0 = np.log(h0)
        try_optimize(params0, 'Nelder-Mead', maxiter=1500)

    # Phase 2: Coordinate descent on best
    try_coordinate(best_h, max_sweeps=30)

    # Phase 3: Powell refinement on best
    if time.time() < deadline - 15:
        try_optimize(np.log(best_h), 'Powell', maxiter=400)

    # Phase 4: More NM restarts
    for seed in [1, 2, 3, 7, 13, 42, 99, 123, 200, 300]:
        if time.time() > deadline - 15:
            break
        rng2 = np.random.default_rng(seed)
        for _ in range(3):
            if time.time() > deadline - 12:
                break
            h0 = np.abs(rng2.standard_normal(N)) + 0.01
            h0 = h0 / h0.sum()
            params0 = np.log(h0)
            try_optimize(params0, 'Nelder-Mead', maxiter=1200)

    # Phase 5: Coordinate descent again
    try_coordinate(best_h, max_sweeps=25)

    # Phase 6: L-BFGS-B polish
    if time.time() < deadline - 8:
        try_optimize(np.log(best_h), 'L-BFGS-B', maxiter=800)

    # Phase 7: Final coordinate descent
    try_coordinate(best_h, max_sweeps=15)

    # Phase 8: Final Powell + coordinate
    if time.time() < deadline - 4:
        try_optimize(np.log(best_h), 'Powell', maxiter=200)
    try_coordinate(best_h, max_sweeps=10)

    best_h = np.maximum(best_h, 0)
    s = best_h.sum()
    if s < 1e-300:
        best_h = np.ones(N) / N
    else:
        best_h = best_h / s
    return list(best_h)
# EVOLVE-BLOCK-END
