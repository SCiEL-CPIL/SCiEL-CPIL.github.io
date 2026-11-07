"""Second autocorrelation inequality - Dinkelbach with coordinate ascent and NM polish."""
import numpy as np
import time
from scipy.optimize import minimize

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

def _coord_step(h, lam, max_sweeps=30, tol=1e-10, deadline=None):
    h = np.maximum(np.asarray(h, dtype=float), 0.0)
    best_aux = _aux_obj(h, lam)
    N = len(h)
    m = 2 * N - 1
    w = 1.0 / (m + 1)
    factor = 2.0 * w / 3.0
    
    for sweep in range(max_sweeps):
        if deadline is not None and time.time() > deadline - 2:
            break
        improved = False
        for i in range(N):
            if deadline is not None and time.time() > deadline - 2:
                break
            
            old_t = h[i]
            
            # Compute g with current h
            g = np.convolve(h, h)
            
            # Decompose g_i(t) = t^2 + lin_i*t + const_i
            if i == 0:
                lin = np.sum(h[1:N])
                const = 0.0
                for j in range(1, N):
                    if 2 * j < N:
                        const += h[j] * h[2*j]
            else:
                lin = h[0]
                if 2 * i < N:
                    lin += h[2 * i]
                const = 0.0
                for j in range(N):
                    if j == 0 or j == i:
                        continue
                    k = i + j
                    if 0 <= k < N:
                        const += h[j] * h[k]
            
            # Precompute sums
            S2 = np.sum(g**2)
            S1 = np.sum(g[:-1] * g[1:])
            n1_total = np.sum(g) / (m + 1)
            
            S2_rest = S2 - g[i]**2
            s1r = S1
            if i > 0:
                s1r -= g[i-1] * g[i]
            if i < m - 1:
                s1r -= g[i] * g[i+1]
            S1_rest = s1r
            
            C_i = np.max(np.delete(g, i))
            n1_rest = (n1_total * (m + 1) - g[i]) / (m + 1)
            
            gn = 0.0
            if i > 0:
                gn += g[i-1]
            if i < m - 1:
                gn += g[i+1]
            
            # g_i(t) = t^2 + lin*t + const
            # l2(t) = factor * [(t^2+lin*t+const)^2 + S2_rest + (t^2+lin*t+const)*gn + S1_rest]
            # n1(t) = n1_rest + (t^2+lin*t+const)/(m+1)
            # ninf(t) = max(t^2+lin*t+const, C_i)
            
            candidates = [0.0, old_t]
            
            # Breakpoints where g_i(t) = C_i
            disc = lin*lin - 4*(const - C_i)
            if disc >= 0:
                sqrt_disc = np.sqrt(disc)
                t1 = (-lin + sqrt_disc) / 2.0
                t2 = (-lin - sqrt_disc) / 2.0
                if t1 >= 0:
                    candidates.append(t1)
                if t2 >= 0:
                    candidates.append(t2)
            
            # Branch 1: g_i(t) <= C_i, ninf = C_i
            # A1'(t) = 0 is cubic
            mc1 = (m+1) * 4 * factor
            mc2 = (m+1) * 6 * factor * lin
            mc3 = (m+1) * 2 * factor * (lin*lin + 2*const + gn) - lam * C_i * 2
            mc4 = (m+1) * 2 * factor * (lin*const + lin*gn) - lam * C_i * lin
            
            if abs(mc1) > 1e-15:
                try:
                    roots = np.roots([mc1, mc2, mc3, mc4])
                    for r in roots:
                        if np.isreal(r):
                            rv = float(np.real(r))
                            if rv >= 0:
                                gi_rv = rv*rv + lin*rv + const
                                if gi_rv <= C_i + 1e-10:
                                    candidates.append(rv)
                except Exception:
                    pass
            
            # Branch 2: g_i(t) > C_i, ninf = g_i(t)
            # Use grid search in the region
            if disc >= 0:
                sqrt_disc = np.sqrt(disc)
                t_bp = max(0, (-lin - sqrt_disc) / 2.0)
                t_search_max = max(t_bp + 3*old_t + 1, t_bp + 1, 2.0)
                grid = np.linspace(t_bp, t_search_max, 30)
                for tv in grid:
                    if tv >= 0:
                        candidates.append(tv)
            
            # Exploration points
            for mult in [0.1, 0.5, 1.5, 2, 5, 10]:
                candidates.append(old_t * mult)
            candidates.extend([0.01, 0.1, 1.0])
            
            # Evaluate candidates
            best_t = old_t
            best_v = best_aux
            for tv in candidates:
                if tv < 0 or not np.isfinite(tv):
                    continue
                h[i] = tv
                v = _aux_obj(h, lam)
                if v > best_v:
                    best_v = v
                    best_t = tv
            
            # Golden section refinement
            if best_t > 0.001:
                span = max(best_t * 0.3, 0.01)
                a = max(0, best_t - span)
                b = best_t + span
                gr = (np.sqrt(5) - 1) / 2
                c = a + (1 - gr) * (b - a)
                d = a + gr * (b - a)
                h[i] = c
                fc = _aux_obj(h, lam)
                h[i] = d
                fd = _aux_obj(h, lam)
                for _ in range(20):
                    if fc < fd:
                        b = d; d = c; fd = fc
                        c = a + (1 - gr) * (b - a)
                        h[i] = c
                        fc = _aux_obj(h, lam)
                    else:
                        a = c; c = d; fc = fd
                        d = a + gr * (b - a)
                        h[i] = d
                        fd = _aux_obj(h, lam)
                h[i] = (c + d) / 2
                v = _aux_obj(h, lam)
                if v > best_v:
                    best_v = v
                    best_t = (c + d) / 2
            
            h[i] = best_t
            
            if best_v > best_aux + tol:
                best_aux = best_v
                improved = True
        
        if not improved:
            break
    
    return h, best_aux

def _dinkelbach(h0, max_iter=35, tol=1e-10, deadline=None):
    h = np.maximum(np.asarray(h0, dtype=float), 0.0)
    s = h.sum()
    if s < 1e-12:
        h = np.ones(N) / N
        s = 1.0
    h = h / s
    lam = _compute_value(h)
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline - 3:
            break
        h, aux = _coord_step(h, lam, max_sweeps=25, deadline=deadline)
        new_val = _compute_value(h)
        if abs(new_val - lam) < tol * max(1.0, abs(lam)):
            break
        lam = new_val
    return h, _compute_value(h)

def _nm_polish(h0, deadline, maxiter=500):
    h = np.maximum(np.asarray(h0, dtype=float), 0.0)
    s = h.sum()
    if s < 1e-12:
        h = np.ones(N) / N
        s = 1.0
    h = h / s
    def negv(p):
        hh = np.maximum(p, 1e-12)
        hh = hh / hh.sum()
        return -_compute_value(hh)
    try:
        res = minimize(negv, h.copy(), method='Nelder-Mead',
                       options={'maxiter': maxiter, 'xatol': 1e-11, 'fatol': 1e-13, 'adaptive': True})
        hh = np.maximum(res.x, 0.0)
        s2 = hh.sum()
        if s2 > 1e-12:
            hh = hh / s2
        v = _compute_value(hh)
        if v > _compute_value(h):
            return hh, v
    except Exception:
        pass
    return h, _compute_value(h)

def _gen_starts(rng):
    starts = []
    idx = np.arange(N)
    x = idx / (N - 1) - 0.5
    # Uniform
    starts.append(np.ones(N))
    # Gaussians with various widths
    for s in [0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8]:
        starts.append(np.exp(-x**2 / (2 * s**2)))
    # Super/sub-Gaussian
    for p in [0.5, 1, 1.5, 2, 3, 4, 6]:
        for s in [0.08, 0.15, 0.25, 0.4, 0.7]:
            starts.append(np.exp(-np.abs(x) ** p / s))
    # Compact support (triangular)
    for w in [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]:
        starts.append(np.maximum(0, 1 - np.abs(x) / w))
    # Plateau
    for w in [0.05, 0.1, 0.15, 0.2, 0.3]:
        starts.append(np.where(np.abs(x) < w, 1.0, 0.0) + 1e-6)
    # Bimodal
    for c in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
        for s in [0.03, 0.05, 0.08, 0.1, 0.15]:
            starts.append(np.exp(-(x - c)**2 / (2*s**2)) + np.exp(-(x + c)**2 / (2*s**2)))
    # Tri-modal
    for s in [0.05, 0.08, 0.12, 0.2]:
        starts.append(np.exp(-(x - 0.4)**2/(2*s**2)) + np.exp(-x**2/(2*s**2)) + np.exp(-(x + 0.4)**2/(2*s**2)))
    # Lorentzian
    for a in [1, 2, 5, 10, 20, 50]:
        starts.append(1.0 / (1 + (a * x)**2))
    # Power-law
    for a in [0.5, 1, 1.5, 2, 3, 5]:
        starts.append(1.0 / (1 + np.abs(x) / 0.15) ** a)
    # Random smooth (Fourier)
    for _ in range(30):
        nfreq = int(rng.integers(2, 10))
        coeffs = rng.standard_normal(nfreq)
        phases = rng.uniform(0, 2*np.pi, nfreq)
        h_f = np.ones(N)
        for k in range(1, nfreq + 1):
            h_f += coeffs[k-1] * np.cos(2*np.pi*k*x + phases[k-1])
        starts.append(np.abs(h_f) + 0.02)
    # Random
    for _ in range(30):
        starts.append(np.abs(rng.standard_normal(N)) + 0.02)
    # Symmetric patterns
    for w in [0.1, 0.2, 0.3]:
        starts.append(np.where(np.abs(x) < w, 2.0, 1.0))
    # 4-point symmetric
    for c in [0.1, 0.2, 0.3]:
        pts = np.array([-c, c, -2*c, 2*c])
        vals = np.array([1.0, 1.0, 0.5, 0.5])
        h_sym = np.ones(N) * 0.1
        for p, v in zip(pts, vals):
            ii = int((p + 0.5) * (N - 1))
            ii = max(0, min(N-1, ii))
            h_sym[ii] = v
        starts.append(h_sym)
    return starts

def construct():
    t0 = time.time()
    deadline = t0 + 145.0

    rng = np.random.default_rng(50)
    starts = _gen_starts(rng)

    best_h = np.ones(N) / N
    best_val = _compute_value(best_h)

    def try_candidate(h0):
        nonlocal best_h, best_val
        if time.time() > deadline - 5:
            return
        h0 = np.maximum(np.asarray(h0, dtype=float), 0.0)
        s = h0.sum()
        if s < 1e-12:
            return
        h0 = h0 / s
        # Dinkelbach
        try:
            h_d, v_d = _dinkelbach(h0, max_iter=30, deadline=deadline)
            if v_d > best_val:
                best_val = v_d
                best_h = h_d.copy()
        except Exception:
            pass
        # NM polish on current best
        try:
            h_n, v_n = _nm_polish(best_h, deadline, maxiter=400)
            if v_n > best_val:
                best_val = v_n
                best_h = h_n.copy()
        except Exception:
            pass

    # Phase 1: Dinkelbach + NM from structured starts
    for h0 in starts:
        if time.time() > deadline - 20:
            break
        try_candidate(h0)

    # Phase 2: More random restarts
    for seed in [1, 2, 3, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]:
        if time.time() > deadline - 10:
            break
        rng2 = np.random.default_rng(seed)
        for _ in range(3):
            if time.time() > deadline - 10:
                break
            h0 = np.abs(rng2.standard_normal(N)) + 0.02
            try_candidate(h0)

    # Phase 3: Final Dinkelbach polish on the best
    if time.time() < deadline - 8:
        try:
            h_f, v_f = _dinkelbach(best_h, max_iter=40, deadline=deadline)
            if v_f > best_val:
                best_val = v_f
                best_h = h_f.copy()
        except Exception:
            pass

    # Phase 4: Final NM polish
    if time.time() < deadline - 5:
        try:
            h_n, v_n = _nm_polish(best_h, deadline, maxiter=500)
            if v_n > best_val:
                best_val = v_n
                best_h = h_n.copy()
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
