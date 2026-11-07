"""Second autocorrelation inequality - Dinkelbach + direct coordinate search."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _compute_value(h):
    h = np.asarray(h, dtype=float)
    if not np.all(np.isfinite(h)):
        return 0.0
    h = np.maximum(h, 0.0)
    if h.sum() < 1e-300:
        return 0.0
    h = h / h.sum()
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

def _neg_value_direct(h):
    """Negative of value for direct optimization on h (nonnegative, sum=1)."""
    h = np.maximum(h, 0.0)
    s = h.sum()
    if s < 1e-300:
        return 0.0
    return -_compute_value(h)

def _coordinate_descent(h, max_sweeps=20, tol=1e-13, deadline=None):
    """Coordinate descent with golden section search per coordinate on normalized h."""
    h = np.array(h, dtype=float)
    h = np.maximum(h, 0.0)
    s = h.sum()
    if s < 1e-300:
        h = np.ones(N) / N
    else:
        h = h / s
    best = _compute_value(h)
    idx = np.arange(N)
    
    for sweep in range(max_sweeps):
        if deadline is not None and time.time() > deadline - 3:
            break
        improved = False
        for i in range(N):
            if deadline is not None and time.time() > deadline - 3:
                break
            old_val = _compute_value(h)
            old_hi = h[i]
            
            # Adaptive bounds
            lo = 0.0
            hi = max(5.0 * old_hi + 0.5, 2.0)
            
            gr = (np.sqrt(5) - 1) / 2
            a = lo + (1 - gr) * (hi - lo)
            b = lo + gr * (hi - lo)
            fa = _compute_value(np.where(idx == i, a, h))
            fb = _compute_value(np.where(idx == i, b, h))
            
            for _ in range(40):
                if deadline is not None and time.time() > deadline - 3:
                    break
                if fa < fb:
                    hi = b
                    b = a
                    fb = fa
                    a = lo + (1 - gr) * (hi - lo)
                    fa = _compute_value(np.where(idx == i, a, h))
                else:
                    lo = a
                    a = b
                    fa = fb
                    b = lo + gr * (hi - lo)
                    fb = _compute_value(np.where(idx == i, b, h))
            
            best_i = (a + b) / 2
            h_trial = np.where(idx == i, best_i, h)
            h_trial = np.maximum(h_trial, 0.0)
            h_trial = h_trial / h_trial.sum()
            val_i = _compute_value(h_trial)
            
            if val_i > old_val + tol:
                h = h_trial
                improved = True
                best = val_i
        
        if not improved:
            break
    
    return h, best

def _dinkelbach_iterate(h0, max_iters=30, tol=1e-12, deadline=None):
    """Dinkelbach iteration: fix ratio r, maximize l2 - r*norm_1*norm_inf."""
    h = np.array(h0, dtype=float)
    h = np.maximum(h, 0.0)
    s = h.sum()
    if s < 1e-300:
        h = np.ones(N) / N
    else:
        h = h / s
    
    r = _compute_value(h)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline - 3:
            break
        
        # Maximize l2(h) - r * norm_1(h) * norm_inf(h)
        # This is equivalent to maximizing l2/h_norm for fixed r
        # Use coordinate descent on the modified objective
        def _modified_obj(h):
            h = np.maximum(h, 0.0)
            hs = h.sum()
            if hs < 1e-300:
                return 0.0
            h = h / hs
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
            return l2 - r * norm_1 * norm_inf
        
        # Coordinate descent on modified objective
        idx = np.arange(N)
        for sweep in range(5):
            if deadline is not None and time.time() > deadline - 3:
                break
            for i in range(N):
                if deadline is not None and time.time() > deadline - 3:
                    break
                old_hi = h[i]
                lo = 0.0
                hi = max(5.0 * old_hi + 0.5, 2.0)
                
                gr = (np.sqrt(5) - 1) / 2
                a = lo + (1 - gr) * (hi - lo)
                b = lo + gr * (hi - lo)
                fa = _modified_obj(np.where(idx == i, a, h))
                fb = _modified_obj(np.where(idx == i, b, h))
                
                for _ in range(30):
                    if deadline is not None and time.time() > deadline - 3:
                        break
                    if fa < fb:
                        hi = b
                        b = a
                        fb = fa
                        a = lo + (1 - gr) * (hi - lo)
                        fa = _modified_obj(np.where(idx == i, a, h))
                    else:
                        lo = a
                        a = b
                        fa = fb
                        b = lo + gr * (hi - lo)
                        fb = _modified_obj(np.where(idx == i, b, h))
                
                best_i = (a + b) / 2
                h_trial = np.where(idx == i, best_i, h)
                h_trial = np.maximum(h_trial, 0.0)
                h_trial = h_trial / h_trial.sum()
        
        new_r = _compute_value(h)
        if abs(new_r - r) < tol:
            break
        r = new_r
    
    return h, _compute_value(h)

def _generate_candidates(rng):
    """Generate a diverse set of candidate initial points."""
    candidates = []
    x = np.linspace(-1, 1, N)
    xh = np.linspace(-0.5, 0.5, N)
    idx = np.arange(N)
    
    # Uniform
    candidates.append(np.ones(N))
    
    # Gaussian family (various widths)
    for a in [0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256]:
        candidates.append(np.exp(-a * x**2))
    
    # Super-Gaussian / sub-Gaussian
    for p in [0.5, 1, 1.5, 2, 3, 4, 6, 8]:
        for a in [1, 2, 4, 8, 16]:
            candidates.append(np.exp(-a * np.abs(x)**p))
    
    # Polynomial / power shapes
    for k in [0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 12]:
        candidates.append(np.maximum(0, 1 - np.abs(x)**k))
    
    # Cosine-based
    for mult in [0.5, 1, 1.5, 2, 2.5, 3]:
        candidates.append(np.maximum(0, np.cos(mult * np.pi * xh)))
        candidates.append(np.maximum(0, np.cos(mult * np.pi * xh)**2))
    
    # Sine-based
    for mult in [0.5, 1, 1.5, 2, 2.5, 3]:
        candidates.append(np.maximum(0, np.sin(mult * np.pi * xh)))
        candidates.append(np.maximum(0, np.sin(mult * np.pi * xh)**2))
    
    # Bimodal
    for c in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        for a in [4, 8, 12, 20, 40, 80, 120]:
            candidates.append(np.exp(-a*(x-c)**2) + np.exp(-a*(x+c)**2))
    
    # Plateau with smooth edges
    for start in [0, 2, 5, 8, 10, 12, 15]:
        for end in [N-start, N-2, N-5, N-8, N-10]:
            if end > start + 4:
                plat = np.zeros(N)
                plat[start:end] = 1.0
                edge = 3
                for j in range(start, min(start+edge, N)):
                    plat[j] = (j - start + 1) / (edge + 1)
                for j in range(max(end-edge, 0), end):
                    plat[j] = (end - j) / (edge + 1)
                candidates.append(np.maximum(plat, 1e-8))
    
    # Triangle
    candidates.append(np.maximum(0, 1 - 2*np.abs(xh)))
    candidates.append(np.maximum(0, 1 - 4*np.abs(xh)))
    candidates.append(np.maximum(0, 1 - 1.5*np.abs(xh)))
    
    # Sech / Lorentzian-like
    for a in [1, 2, 3, 4, 6, 8, 12]:
        candidates.append(1.0 / np.cosh(a * x))
        candidates.append(1.0 / (1 + (a*x)**2))
    
    # Asymmetric
    for asym in [0.3, 0.5, 1.0, 2.0, 3.0, 5.0]:
        candidates.append(np.exp(-x**2 * (1 + asym*x)))
        candidates.append(np.exp(-x**2 * (1 - asym*x)))
    
    # W-shaped / multi-lobe
    for nlobes in [3, 4, 5]:
        for width in [0.1, 0.15, 0.2, 0.3, 0.4]:
            h_w = np.zeros(N)
            for k in range(nlobes):
                c = -1 + 2*k/(nlobes-1)
                h_w += np.exp(-(x-c)**2/(2*width**2))
            candidates.append(h_w)
    
    # Random smooth (low-frequency Fourier)
    for _ in range(40):
        nfreq = rng.integers(2, 10)
        coeffs = rng.standard_normal(nfreq)
        phases = rng.uniform(0, 2*np.pi, nfreq)
        h_f = np.ones(N)
        for k in range(1, nfreq+1):
            h_f += coeffs[k-1] * np.cos(2*np.pi*k*xh + phases[k-1])
        candidates.append(np.abs(h_f) + 0.01)
    
    # Random (raw)
    for _ in range(30):
        h_r = np.abs(rng.standard_normal(N)) + 0.01
        candidates.append(h_r)
    
    # Concentrated center
    for width in [2, 3, 5, 7, 10, 15, 20]:
        center = N // 2
        candidates.append(np.exp(-((idx - center)**2) / (2*width**2)))
    
    # Edge-concentrated
    for width in [2, 3, 5, 7, 10]:
        candidates.append(np.exp(-((idx)**2) / (2*width**2)) + np.exp(-((idx - (N-1))**2) / (2*width**2)))
    
    # Single spike
    for pos in [0, 5, 10, 15, 20, 24, 25, 29, 35, 40, 45, 49]:
        spike = np.zeros(N)
        spike[pos] = 1.0
        candidates.append(spike + 1e-6)
    
    # Two spikes
    for p1 in [5, 10, 15, 20, 25]:
        for p2 in [30, 35, 40, 45, 49]:
            if p2 > p1:
                two = np.zeros(N)
                two[p1] = 1.0
                two[p2] = 1.0
                candidates.append(two + 1e-6)
    
    # Step functions
    for cut in [5, 10, 15, 20, 25, 30, 35, 40, 45]:
        step = np.zeros(N)
        step[cut:] = 1.0
        candidates.append(step + 1e-6)
        step2 = np.zeros(N)
        step2[:cut] = 1.0
        candidates.append(step2 + 1e-6)
    
    # Power-law decay from center
    for alpha in [0.5, 1, 1.5, 2, 3, 4]:
        dist = np.abs(idx - (N-1)/2.0)
        candidates.append(1.0 / (1 + dist**alpha))
    
    # Alternating
    for freq in [1, 2, 3, 5]:
        alt = 1 + 0.5*np.cos(2*np.pi*freq*idx/N)
        candidates.append(np.maximum(alt, 1e-6))
    
    # Additional: smooth bump with varying support
    for support in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        for center_frac in [0.25, 0.5, 0.75]:
            c = -1 + 2 * center_frac
            width = support
            h_b = np.exp(-((x - c) / width)**2)
            candidates.append(h_b)
    
    # Additional: product of Gaussians (narrow effective support)
    for a1 in [2, 4, 8, 16]:
        for a2 in [2, 4, 8, 16]:
            for c1 in [-0.3, 0, 0.3]:
                for c2 in [-0.3, 0, 0.3]:
                    if c1 != c2:
                        h_p = np.exp(-a1*(x-c1)**2) * np.exp(-a2*(x-c2)**2)
                        candidates.append(h_p)
    
    return candidates

def construct():
    t0 = time.time()
    deadline = t0 + 145.0
    
    best_h = np.ones(N) / N
    best_val = _compute_value(best_h)
    
    def try_coordinate(h0, max_sweeps=20):
        nonlocal best_h, best_val
        if time.time() > deadline - 3:
            return
        try:
            h_new, val_new = _coordinate_descent(h0, max_sweeps=max_sweeps, deadline=deadline)
            if val_new > best_val + 1e-15:
                best_val = val_new
                best_h = h_new
        except Exception:
            pass
    
    def try_dinkelbach(h0, max_iters=30):
        nonlocal best_h, best_val
        if time.time() > deadline - 3:
            return
        try:
            h_new, val_new = _dinkelbach_iterate(h0, max_iters=max_iters, deadline=deadline)
            if val_new > best_val + 1e-15:
                best_val = val_new
                best_h = h_new
        except Exception:
            pass
    
    def try_nm_direct(h0, maxiter=800):
        """Nelder-Mead directly on h (with nonnegativity handled by projection)."""
        nonlocal best_h, best_val
        if time.time() > deadline - 5:
            return
        try:
            def _obj(h):
                h = np.maximum(h, 0.0)
                s = h.sum()
                if s < 1e-300:
                    return 0.0
                return -_compute_value(h)
            
            h0_clipped = np.maximum(h0, 1e-8)
            h0_clipped = h0_clipped / h0_clipped.sum()
            
            res = minimize(_obj, h0_clipped, method='Nelder-Mead',
                          options={'maxiter': maxiter, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True})
            h_res = np.maximum(res.x, 0.0)
            s = h_res.sum()
            if s > 1e-300:
                h_res = h_res / s
                val_res = _compute_value(h_res)
                if val_res > best_val + 1e-15:
                    best_val = val_res
                    best_h = h_res
        except Exception:
            pass
    
    rng = np.random.default_rng(50)
    
    # Generate candidates
    candidates = _generate_candidates(rng)
    
    # Phase 1: Coordinate descent from diverse candidates (fast, ~50s)
    phase1_deadline = t0 + 50
    for h0 in candidates:
        if time.time() > phase1_deadline:
            break
        h0 = np.asarray(h0, dtype=float)
        h0 = np.maximum(h0, 1e-8)
        h0 = h0 / h0.sum()
        try_coordinate(h0, max_sweeps=10)
    
    # Phase 2: Dinkelbach on best
    try_dinkelbach(best_h, max_iters=25)
    
    # Phase 3: NM direct on best
    try_nm_direct(best_h, maxiter=600)
    
    # Phase 4: More random restarts with coordinate descent
    phase4_deadline = t0 + 80
    for seed in [1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67]:
        if time.time() > phase4_deadline:
            break
        rng2 = np.random.default_rng(seed)
        for _ in range(3):
            if time.time() > phase4_deadline:
                break
            h0 = np.abs(rng2.standard_normal(N)) + 0.01
            h0 = h0 / h0.sum()
            try_coordinate(h0, max_sweeps=8)
    
    # Phase 5: Dinkelbach again
    try_dinkelbach(best_h, max_iters=25)
    
    # Phase 6: Coordinate descent refinement
    try_coordinate(best_h, max_sweeps=25)
    
    # Phase 7: Structured restarts with coordinate descent
    phase7_deadline = t0 + 115
    structured_seeds = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010, 2020, 3030]
    for seed in structured_seeds:
        if time.time() > phase7_deadline:
            break
        rng3 = np.random.default_rng(seed)
        nfreq = rng3.integers(3, 8)
        coeffs = rng3.standard_normal(nfreq)
        phases = rng3.uniform(0, 2*np.pi, nfreq)
        xh = np.linspace(-0.5, 0.5, N)
        h_f = np.ones(N)
        for k in range(1, nfreq+1):
            h_f += coeffs[k-1] * np.cos(2*np.pi*k*xh + phases[k-1])
        h_f = np.abs(h_f) + 0.01
        h_f = h_f / h_f.sum()
        try_coordinate(h_f, max_sweeps=8)
    
    # Phase 8: Dinkelbach + coordinate descent polish
    try_dinkelbach(best_h, max_iters=20)
    try_coordinate(best_h, max_sweeps=25)
    
    # Phase 9: Final NM direct
    if time.time() < deadline - 15:
        try_nm_direct(best_h, maxiter=400)
    
    # Phase 10: Final coordinate descent with tight tolerance
    try_coordinate(best_h, max_sweeps=30)
    
    # Phase 11: Final Dinkelbach
    try_dinkelbach(best_h, max_iters=15)
    
    # Final coordinate descent
    try_coordinate(best_h, max_sweeps=20)
    
    # Ensure nonnegativity and normalization
    best_h = np.maximum(best_h, 0)
    s = best_h.sum()
    if s < 1e-300:
        best_h = np.ones(N) / N
    else:
        best_h = best_h / s
    return list(best_h)
# EVOLVE-BLOCK-END
