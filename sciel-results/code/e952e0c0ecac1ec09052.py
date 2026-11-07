"""Second autocorrelation inequality - Dinkelbach with vectorized coordinate updates."""
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

def _precompute_coeffs(h):
    """Precompute coefficients for fast coordinate updates.
    
    Returns arrays:
      - A[i]: coefficient of t^2 in l2 when varying h[i]
      - B[i]: coefficient of t in l2 when varying h[i]  
      - C[i]: constant in l2 when varying h[i]
      - D[i]: coefficient of t in norm_1 when varying h[i]
      - E[i]: constant in norm_1 when varying h[i]
      - F[i]: coefficient of t^2 in g_i when varying h[i]
      - G[i]: coefficient of t in g_i when varying h[i]
      - H[i]: constant in g_i when varying h[i]
      - rest_max[i]: max of g_j for j != i
    """
    h = np.maximum(np.asarray(h, dtype=float), 0.0)
    g = np.convolve(h, h)
    m = len(g)
    y = np.concatenate(([0.0], g, [0.0]))
    w = 1.0 / (m + 1)
    
    # l2 = (w/3) * sum_{i=0}^{m} (y_i^2 + y_i*y_{i+1} + y_{i+1}^2)
    # When varying h[i], g_j changes for j where the convolution window includes h[i]
    # g_j = sum_k h[k]*h[j+k], so h[i] appears in g_j when k=i (j+i valid) or j+k=i (j valid)
    # i.e., j in [i-(N-1), i] intersect [0, m-1]
    
    A = np.zeros(N)
    B = np.zeros(N)
    C = np.zeros(N)
    D = np.zeros(N)
    E = np.zeros(N)
    F = np.zeros(N)
    G = np.zeros(N)
    H = np.zeros(N)
    rest_max = np.zeros(N)
    
    # For each i, compute how l2, norm_1, g_i change with t = h[i]
    for i in range(N):
        # g_i(t) = t^2 + t*(h[2i] if 2i<N else 0) + sum_{j: 0<=j<N, 0<=i+j<N, j!=i, j!=0} h[j]*h[i+j]
        # Actually: g_i = sum_{j=0}^{N-1} h[j]*h[i+j] where h[k]=0 if k<0 or k>=N
        # Terms involving h[i]=t:
        #   j=0: h[0]*h[i] = h[0]*t (if i>0), or t*t=t^2 (if i=0)
        #   j=i: h[i]*h[2i] = t*h[2i] (if i>0 and 2i<N)
        #   j such that i+j=i: j=0 (already counted)
        #   j such that j=i: counted above
        # So:
        if i == 0:
            # g_0 = sum_j h[j]*h[j] = t^2 + sum_{j=1}^{N-1} h[j]^2
            F[i] = 1.0
            G[i] = 0.0
            H[i] = np.sum(h[1:]**2)
        else:
            # g_i = t*h[0] + (t*h[2i] if 2i<N else 0) + sum_{j: 1<=j<N, j!=i, 0<=i+j<N, j!=0} h[j]*h[i+j]
            F[i] = 0.0
            G[i] = h[0] + (h[2*i] if 2*i < N else 0.0)
            # H[i] = sum of h[j]*h[i+j] for j in 1..N-1, j!=i, i+j<N
            H[i] = 0.0
            for j in range(1, N):
                if j == i:
                    continue
                k = i + j
                if 0 <= k < N:
                    H[i] += h[j] * h[k]
        
        # norm_1 = sum(g)/m = (sum_j g_j)/m
        # sum_j g_j = sum_j sum_k h[k]*h[j+k] = sum_k h[k] * sum_j h[j+k]
        # = sum_k h[k] * (sum of h over valid indices j+k)
        # When varying h[i]=t:
        #   Terms with h[i] as first factor: t * sum_j h[j+i] for valid j
        #   Terms with h[i] as second factor: sum_j h[j] * t for valid j where j+i valid => j in [0, N-1-i]
        # So sum_j g_j = t * (sum_{j: 0<=j+i<N} h[j+i]) + t * (sum_{j: 0<=j<N, 0<=j+i<N} h[j]) + const
        # = t * (sum_{k=i}^{N-1} h[k]) + t * (sum_{j=0}^{N-1-i} h[j]) + const
        # The constant is sum_j g_j with h[i]=0
        
        # D[i] = coefficient of t in sum_j g_j
        D[i] = np.sum(h[i:]) + np.sum(h[:N-i])  # both sums include h[i] when i=0
        # Wait, need to be careful. Let me just compute directly.
        # sum_j g_j = sum_j sum_k h[k]*h[j+k]
        # = sum_k h[k] * sum_{j: 0<=j<N, 0<=j+k<N} h[j+k]
        # For fixed k, inner sum is sum_{j=max(0,-k)}^{min(N-1, N-1-k)} h[j+k]
        # = sum_{l=k}^{N-1} h[l] where l=j+k, j from max(0,-k) to min(N-1,N-1-k)
        # So sum_j g_j = sum_k h[k] * sum_{l=k}^{N-1} h[l]
        # = sum_k h[k] * S_k where S_k = sum_{l=k}^{N-1} h[l]
        # When varying h[i]=t:
        #   k=i term: t * S_i where S_i = t + sum_{l=i+1}^{N-1} h[l]
        #   Other k terms: h[k] * S_k where S_k may include h[i] if k <= i
        #   S_k = sum_{l=k}^{N-1} h[l] includes h[i] iff k <= i
        # So sum_j g_j = t*(t + sum_{l=i+1}^{N-1} h[l]) + sum_{k!=i} h[k] * (sum_{l=k}^{N-1} h[l] + (t if k<=i else 0))
        # This is getting complex. Let me use a simpler approach.
        
        # Actually, for the auxiliary, I'll just evaluate directly at candidate points.
        # But I can precompute rest_max[i] = max(g_j for j != i)
        mask = np.ones(m, dtype=bool)
        mask[i] = False
        rest_max[i] = np.max(g[mask]) if m > 1 else 0.0
    
    return A, B, C, D, E, F, G, H, rest_max

def _coord_step_fast(h, lam, max_sweeps=30, tol=1e-11, deadline=None):
    """Fast coordinate maximization using precomputed structure."""
    h = np.maximum(np.asarray(h, dtype=float), 0.0)
    N = len(h)
    
    # Precompute once per sweep
    g = np.convolve(h, h)
    m = len(g)
    y = np.concatenate(([0.0], g, [0.0]))
    w = 1.0 / (m + 1)
    
    best_aux = _aux_obj(h, lam)
    
    for sweep in range(max_sweeps):
        if deadline is not None and time.time() > deadline - 2:
            break
        improved = False
        
        # Recompute g and rest_max each sweep since h changes
        g = np.convolve(h, h)
        m = len(g)
        y = np.concatenate(([0.0], g, [0.0]))
        w = 1.0 / (m + 1)
        norm_1_base = np.sum(g) / m
        norm_inf_base = np.max(g)
        
        for i in range(N):
            if deadline is not None and time.time() > deadline - 2:
                break
            
            old_t = h[i]
            
            # g_i(t) = F*t^2 + G*t + H
            if i == 0:
                F_i, G_i, H_i = 1.0, 0.0, np.sum(h[1:]**2)
            else:
                F_i = 0.0
                G_i = h[0] + (h[2*i] if 2*i < N else 0.0)
                H_i = 0.0
                for j in range(1, N):
                    if j == i:
                        continue
                    k = i + j
                    if 0 <= k < N:
                        H_i += h[j] * h[k]
            
            # rest_max = max(g_j for j != i)
            mask = np.ones(m, dtype=bool)
            mask[i] = False
            C_i = np.max(g[mask]) if m > 1 else 0.0
            
            # l2(t): quadratic in t
            # l2 = (w/3) * sum_{k=0}^{m} (y_k^2 + y_k*y_{k+1} + y_{k+1}^2)
            # y_k = g_k for k=1..m-1, y_0=0, y_m=0
            # g_k changes when the convolution window includes h[i]
            # g_k = sum_j h[j]*h[k+j], h[i] appears if j=i (k+i valid) or k+j=i (k valid)
            # So g_k changes for k in [i-(N-1), i] intersect [0, m-1]
            
            # For efficiency, compute l2 contribution change
            # Base l2 with h[i]=0
            h_i_zero = h.copy()
            h_i_zero[i] = 0.0
            g_zero = np.convolve(h_i_zero, h_i_zero)
            y_zero = np.concatenate(([0.0], g_zero, [0.0]))
            l2_zero = (w / 3.0) * np.sum(y_zero[:m+1]**2 + y_zero[:m+1]*y_zero[1:m+2] + y_zero[1:m+2]**2)
            
            # Delta l2 when h[i] changes from 0 to t
            # This is quadratic in t. Let's compute coefficients by evaluating at t=1 and t=2
            h_t1 = h_i_zero.copy()
            h_t1[i] = 1.0
            g_t1 = np.convolve(h_t1, h_t1)
            y_t1 = np.concatenate(([0.0], g_t1, [0.0]))
            l2_t1 = (w / 3.0) * np.sum(y_t1[:m+1]**2 + y_t1[:m+1]*y_t1[1:m+2] + y_t1[1:m+2]**2)
            
            h_t2 = h_i_zero.copy()
            h_t2[i] = 2.0
            g_t2 = np.convolve(h_t2, h_t2)
            y_t2 = np.concatenate(([0.0], g_t2, [0.0]))
            l2_t2 = (w / 3.0) * np.sum(y_t2[:m+1]**2 + y_t2[:m+1]*y_t2[1:m+2] + y_t2[1:m+2]**2)
            
            # l2(t) = l2_zero + a*t + b*t^2
            # l2_t1 = l2_zero + a + b
            # l2_t2 = l2_zero + 2a + 4b
            # a + b = l2_t1 - l2_zero
            # 2a + 4b = l2_t2 - l2_zero
            # => a = (2*(l2_t1-l2_zero) - (l2_t2-l2_zero))/2
            # => b = ((l2_t2-l2_zero) - 2*(l2_t1-l2_zero))/2
            a_l2 = (2*(l2_t1 - l2_zero) - (l2_t2 - l2_zero)) / 2.0
            b_l2 = ((l2_t2 - l2_zero) - 2*(l2_t1 - l2_zero)) / 2.0
            
            # norm_1(t) = (sum_j g_j(t)) / m
            # sum_j g_j = sum_k h[k] * sum_{l=k}^{N-1} h[l] (derived earlier)
            # When h[i] changes from 0 to t:
            # sum_j g_j(t) = sum_j g_j(0) + t * D_i
            # where D_i = sum_{l=i}^{N-1} h[l] + sum_{k=0}^{i} h[k] (with h[i]=0 in these sums)
            # Actually, let me just compute it directly.
            sum_g_zero = np.sum(g_zero)
            sum_g_t1 = np.sum(g_t1)
            D_i = sum_g_t1 - sum_g_zero  # linear in t
            
            # norm_1(t) = (sum_g_zero + D_i * t) / m
            # norm_inf(t) = max(g_i(t), C_i) = max(F_i*t^2 + G_i*t + H_i, C_i)
            
            # Auxiliary A(t) = l2(t) - lam * norm_1(t) * norm_inf(t)
            # Piecewise:
            #   Where g_i(t) <= C_i: A(t) = l2_zero + a_l2*t + b_l2*t^2 - lam * ((sum_g_zero + D_i*t)/m) * C_i
            #     = (b_l2 - lam*C_i*D_i/m) * t^2 + (a_l2 - lam*C_i*sum_g_zero/m) * t + l2_zero - lam*C_i*sum_g_zero/m
            #   Where g_i(t) > C_i: A(t) = l2_zero + a_l2*t + b_l2*t^2 - lam * ((sum_g_zero + D_i*t)/m) * (F_i*t^2 + G_i*t + H_i)
            #     = quartic in t
            
            # Find where g_i(t) = C_i: F_i*t^2 + G_i*t + H_i = C_i
            # F_i*t^2 + G_i*t + (H_i - C_i) = 0
            if F_i > 1e-15:
                disc = G_i**2 - 4*F_i*(H_i - C_i)
                if disc >= 0:
                    sqrt_disc = np.sqrt(disc)
                    t_break1 = (-G_i - sqrt_disc) / (2*F_i)
                    t_break2 = (-G_i + sqrt_disc) / (2*F_i)
                    # Take positive roots
                    t_breaks = [tb for tb in [t_break1, t_break2] if tb > 0]
                else:
                    t_breaks = []
            elif abs(G_i) > 1e-15:
                t_break = -(H_i - C_i) / G_i
                t_breaks = [t_break] if t_break > 0 else []
            else:
                t_breaks = []
            
            # Candidate t values: 0, old_t, breakpoints, vertices
            candidates = [0.0, old_t]
            for tb in t_breaks:
                candidates.append(tb)
            
            # Vertex of quadratic branch (g_i <= C_i)
            # A_quad(t) = (b_l2 - lam*C_i*D_i/m) * t^2 + (a_l2 - lam*C_i*sum_g_zero/m) * t + const
            A_quad_coeff = b_l2 - lam*C_i*D_i/m
            B_quad_coeff = a_l2 - lam*C_i*sum_g_zero/m
            if abs(A_quad_coeff) > 1e-15:
                t_vertex_quad = -B_quad_coeff / (2*A_quad_coeff)
                if t_vertex_quad > 0:
                    candidates.append(t_vertex_quad)
            
            # For the quartic branch, do a simple grid search
            t_max_est = max(3*old_t + 1, 1.0)
            t_grid = np.linspace(0, t_max_est, 20)
            for tg in t_grid:
                # Check if g_i(tg) > C_i
                gi_tg = F_i*tg**2 + G_i*tg + H_i
                if gi_tg > C_i:
                    candidates.append(tg)
            
            # Evaluate auxiliary at all candidates
            best_t = old_t
            best_v = _aux_obj(h, lam)
            
            for tv in candidates:
                if tv < 0 or tv > 10*t_max_est:
                    continue
                h_trial = h.copy()
                h_trial[i] = tv
                v = _aux_obj(h_trial, lam)
                if v > best_v:
                    best_v = v
                    best_t = tv
            
            h[i] = best_t
            if best_v > best_aux + tol:
                best_aux = best_v
                improved = True
        
        if not improved:
            break
    
    return h, best_aux

def _dinkelbach(h0, max_iter=40, tol=1e-9, deadline=None):
    """Dinkelbach's algorithm for the fractional program l2/(n1*ninf)."""
    h = np.maximum(np.asarray(h0, dtype=float), 0.0)
    h = h / max(h.sum(), 1e-12)
    lam = _compute_value(h)
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline - 3:
            break
        h, aux = _coord_step_fast(h, lam, max_sweeps=20, deadline=deadline)
        new_val = _compute_value(h)
        if abs(new_val - lam) < tol * max(1.0, abs(lam)):
            break
        lam = new_val
    return h, _compute_value(h)

def _nm_polish(h0, deadline, maxiter=500):
    """Nelder-Mead polish on the ratio directly."""
    h = np.maximum(np.asarray(h0, dtype=float), 0.0)
    h = h / max(h.sum(), 1e-12)
    def negv(p):
        hh = np.maximum(p, 1e-12)
        hh = hh / hh.sum()
        return -_compute_value(hh)
    try:
        res = minimize(negv, h.copy(), method='Nelder-Mead',
                       options={'maxiter': maxiter, 'xatol': 1e-11, 'fatol': 1e-13, 'adaptive': True})
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
        if time.time() > deadline - 5:
            return
        h0 = np.maximum(np.asarray(h0, dtype=float), 0.0)
        s = h0.sum()
        if s < 1e-12:
            return
        h0 = h0 / s
        # Dinkelbach
        try:
            h_d, v_d = _dinkelbach(h0, max_iter=35, deadline=deadline)
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
        if time.time() > deadline - 15:
            break
        try_candidate(h0)

    # Phase 2: More random restarts
    for seed in [1, 2, 3, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]:
        if time.time() > deadline - 8:
            break
        rng2 = np.random.default_rng(seed)
        for _ in range(3):
            if time.time() > deadline - 8:
                break
            h0 = np.abs(rng2.standard_normal(N)) + 0.02
            try_candidate(h0)

    # Phase 3: Final Dinkelbach polish on the best
    if time.time() < deadline - 5:
        try:
            h_f, v_f = _dinkelbach(best_h, max_iter=50, deadline=deadline)
            if v_f > best_val:
                best_val = v_f
                best_h = h_f.copy()
        except Exception:
            pass

    # Phase 4: Final NM polish
    if time.time() < deadline - 3:
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
# EVOLVE-BLOCK-END
