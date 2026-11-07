"""Second autocorrelation inequality - DCT-based low-dimensional optimization."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.fftpack import dct, idct

N = 50
DCT_DIM = 8  # Number of DCT coefficients to optimize

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

def _h_from_dct_params(params):
    """Convert DCT coefficient params to normalized nonnegative heights."""
    params = np.asarray(params, dtype=float)
    # Full DCT coefficient vector (N dims), fill unused with small values
    full_coeffs = np.zeros(N)
    full_coeffs[:DCT_DIM] = params[:DCT_DIM]
    # Scale to avoid numerical issues
    full_coeffs = full_coeffs / (np.max(np.abs(full_coeffs)) + 1e-10)
    # Inverse DCT to get heights
    h = idct(full_coeffs, type=2, norm='ortho')
    # Enforce non-negativity
    h = np.maximum(h, 1e-10)
    h = h / h.sum()
    return h

def _dct_params_from_h(h):
    """Convert heights to DCT coefficient params (first DCT_DIM components)."""
    h = np.asarray(h, dtype=float)
    h = h / h.sum()
    coeffs = dct(h, type=2, norm='ortho')
    return coeffs[:DCT_DIM]

def _neg_value_dct(params):
    h = _h_from_dct_params(params)
    return -_compute_value(h)

def _gradient_dct(params):
    """Finite difference gradient for DCT params."""
    eps = 1e-7
    grad = np.zeros(DCT_DIM)
    f0 = _neg_value_dct(params)
    for i in range(DCT_DIM):
        params_p = params.copy()
        params_m = params.copy()
        params_p[i] += eps
        params_m[i] -= eps
        grad[i] = (_neg_value_dct(params_p) - _neg_value_dct(params_m)) / (2 * eps)
    return grad

def _coordinate_descent_dct(params, max_sweeps=20, deadline=None):
    """Coordinate descent in DCT parameter space."""
    params = np.array(params, dtype=float)
    best_val = -_neg_value_dct(params)
    
    for sweep in range(max_sweeps):
        if deadline is not None and time.time() > deadline - 3:
            break
        improved = False
        for i in range(DCT_DIM):
            if deadline is not None and time.time() > deadline - 3:
                break
            old_val = -_neg_value_dct(params)
            
            # Golden section search for optimal value of params[i]
            lo = params[i] - 5.0
            hi = params[i] + 5.0
            
            gr = (np.sqrt(5) - 1) / 2
            a = lo + (1 - gr) * (hi - lo)
            b = lo + gr * (hi - lo)
            
            params_a = params.copy()
            params_a[i] = a
            params_b = params.copy()
            params_b[i] = b
            fa = _neg_value_dct(params_a)
            fb = _neg_value_dct(params_b)
            
            for _ in range(30):
                if deadline is not None and time.time() > deadline - 3:
                    break
                if fa < fb:
                    hi = b
                    b = a
                    fb = fa
                    a = lo + (1 - gr) * (hi - lo)
                    params_a[i] = a
                    fa = _neg_value_dct(params_a)
                else:
                    lo = a
                    a = b
                    fa = fb
                    b = lo + gr * (hi - lo)
                    params_b[i] = b
                    fb = _neg_value_dct(params_b)
            
            best_i = (a + b) / 2
            params_new = params.copy()
            params_new[i] = best_i
            val_new = -_neg_value_dct(params_new)
            
            if val_new > old_val + 1e-12:
                params = params_new
                improved = True
                best_val = val_new
        
        if not improved:
            break
    
    return params, best_val

def _generate_dct_candidates(rng):
    """Generate diverse DCT parameter initializations."""
    candidates = []
    
    # All zeros (uniform heights)
    candidates.append(np.zeros(DCT_DIM))
    
    # Symmetric patterns (only even-indexed DCT coefficients)
    for a in [0.5, 1, 2, 4, 8, 16]:
        for k in [0, 1, 2, 3]:
            p = np.zeros(DCT_DIM)
            p[k] = a * np.cos(np.pi * k / 2)
            candidates.append(p.copy())
            p[k] = -a * np.cos(np.pi * k / 2)
            candidates.append(p.copy())
    
    # Random smooth DCT coefficients
    for _ in range(60):
        p = np.zeros(DCT_DIM)
        for k in range(DCT_DIM):
            p[k] = rng.standard_normal() * np.exp(-k * 0.3)
        candidates.append(p)
    
    # Concentrated low-frequency
    for _ in range(40):
        p = np.zeros(DCT_DIM)
        n_low = rng.integers(1, 4)
        for k in range(n_low):
            p[k] = rng.standard_normal() * rng.uniform(0.5, 3.0)
        candidates.append(p)
    
    # Bimodal in DCT space
    for _ in range(30):
        p = np.zeros(DCT_DIM)
        k1 = rng.integers(0, 3)
        k2 = rng.integers(3, DCT_DIM)
        p[k1] = rng.uniform(1, 4)
        p[k2] = rng.uniform(1, 4) * rng.choice([-1, 1])
        candidates.append(p)
    
    # From known good shapes
    xh = np.linspace(-0.5, 0.5, N)
    idx = np.arange(N)
    center = N // 2
    
    shapes = [
        np.exp(-xh**2 * 4),
        np.exp(-xh**2 * 16),
        np.exp(-xh**2 * 64),
        np.maximum(0, 1 - 2*np.abs(xh)),
        np.maximum(0, np.cos(np.pi * xh)),
        np.exp(-((idx - center)**2) / 10),
        np.exp(-((idx - center)**2) / 40),
        1.0 / (1 + (4*xh)**2),
        1.0 / np.cosh(4*xh),
        np.maximum(0, 1 - np.abs(xh)**4),
    ]
    
    for shape in shapes:
        shape = np.maximum(shape, 1e-8)
        shape = shape / shape.sum()
        coeffs = dct(shape, type=2, norm='ortho')
        candidates.append(coeffs[:DCT_DIM])
    
    # Asymmetric shapes
    for asym in [0.3, 0.7, 1.0, 1.5, 2.0]:
        shape = np.exp(-xh**2 * (4 + asym * xh * 4))
        shape = np.maximum(shape, 1e-8)
        shape = shape / shape.sum()
        coeffs = dct(shape, type=2, norm='ortho')
        candidates.append(coeffs[:DCT_DIM])
    
    # Multi-lobe
    for nlobes in [2, 3, 4]:
        for width in [0.1, 0.2, 0.3]:
            h_w = np.zeros(N)
            for k in range(nlobes):
                c = -1 + 2*k/(nlobes-1)
                h_w += np.exp(-(xh-c)**2/(2*width**2))
            h_w = np.maximum(h_w, 1e-8)
            h_w = h_w / h_w.sum()
            coeffs = dct(h_w, type=2, norm='ortho')
            candidates.append(coeffs[:DCT_DIM])
    
    return candidates

def construct():
    t0 = time.time()
    deadline = t0 + 145.0
    
    best_params = np.zeros(DCT_DIM)
    best_val = _compute_value(_h_from_dct_params(best_params))
    
    def try_optimize(params0, method='L-BFGS-B', maxiter=500, **kwargs):
        nonlocal best_params, best_val
        if time.time() > deadline - 5:
            return
        try:
            if method == 'L-BFGS-B':
                opts = {'maxiter': maxiter, 'gtol': 1e-8, 'maxfun': 2000}
            elif method == 'Nelder-Mead':
                opts = {'maxiter': maxiter, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True}
            elif method == 'Powell':
                opts = {'maxiter': maxiter, 'xtol': 1e-10, 'ftol': 1e-13}
            elif method == 'CG':
                opts = {'maxiter': maxiter, 'gtol': 1e-8}
            else:
                opts = {}
            res = minimize(_neg_value_dct, params0, method=method, options=opts)
            if res.fun < -best_val + 1e-15:
                best_val = -res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    def try_coordinate(params0, max_sweeps=20):
        nonlocal best_params, best_val
        if time.time() > deadline - 3:
            return
        try:
            p_new, val_new = _coordinate_descent_dct(params0, max_sweeps=max_sweeps, deadline=deadline)
            if val_new > best_val + 1e-15:
                best_val = val_new
                best_params = p_new
        except Exception:
            pass
    
    rng = np.random.default_rng(50)
    
    # Generate candidates
    candidates = _generate_dct_candidates(rng)
    
    # Phase 1: L-BFGS-B from diverse candidates
    phase1_deadline = t0 + 50
    for p0 in candidates:
        if time.time() > phase1_deadline:
            break
        try_optimize(p0, 'L-BFGS-B', maxiter=300)
    
    # Phase 2: Coordinate descent on best
    try_coordinate(best_params, max_sweeps=25)
    
    # Phase 3: Powell refinement
    if time.time() < deadline - 30:
        try_optimize(best_params, 'Powell', maxiter=300)
    
    # Phase 4: More random restarts
    phase4_deadline = t0 + 90
    for seed in [1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83]:
        if time.time() > phase4_deadline:
            break
        rng2 = np.random.default_rng(seed)
        for _ in range(3):
            if time.time() > phase4_deadline:
                break
            p0 = np.zeros(DCT_DIM)
            for k in range(DCT_DIM):
                p0[k] = rng2.standard_normal() * np.exp(-k * 0.4) * rng2.uniform(0.5, 3.0)
            try_optimize(p0, 'L-BFGS-B', maxiter=200)
    
    # Phase 5: Coordinate descent again
    try_coordinate(best_params, max_sweeps=25)
    
    # Phase 6: Nelder-Mead polish
    if time.time() < deadline - 20:
        try_optimize(best_params, 'Nelder-Mead', maxiter=400)
    
    # Phase 7: Structured restarts
    phase7_deadline = t0 + 125
    for seed in [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010, 2020, 3030, 4040, 5050]:
        if time.time() > phase7_deadline:
            break
        rng3 = np.random.default_rng(seed)
        p0 = np.zeros(DCT_DIM)
        n_active = rng3.integers(2, DCT_DIM)
        for k in range(n_active):
            p0[k] = rng3.standard_normal() * rng3.uniform(0.3, 2.5) * np.exp(-k * 0.2)
        try_optimize(p0, 'L-BFGS-B', maxiter=200)
    
    # Phase 8: Final coordinate descent + Powell
    try_coordinate(best_params, max_sweeps=25)
    if time.time() < deadline - 10:
        try_optimize(best_params, 'Powell', maxiter=200)
    
    # Phase 9: Final NM
    if time.time() < deadline - 5:
        try_optimize(best_params, 'Nelder-Mead', maxiter=300)
    
    # Final coordinate descent
    try_coordinate(best_params, max_sweeps=15)
    
    # Convert to heights
    best_h = _h_from_dct_params(best_params)
    best_h = np.maximum(best_h, 0)
    s = best_h.sum()
    if s < 1e-300:
        best_h = np.ones(N) / N
    else:
        best_h = best_h / s
    return list(best_h)
