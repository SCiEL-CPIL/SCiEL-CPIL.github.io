import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 50
w = 1.0 / 100.0
m = 2 * N - 1          # 99
m_plus_1 = m + 1       # 100

def compute_value(h):
    h = np.asarray(h, dtype=float)
    if len(h) != N:
        return 0.0
    if np.any(h < 0) or np.any(~np.isfinite(h)):
        return 0.0

    n_fft = 1
    while n_fft < 2 * N - 1:
        n_fft <<= 1
    H = np.fft.rfft(h, n_fft)
    G = H * H
    g = np.fft.irfft(G, n_fft)[:m]
    g = np.real(g)
    g = np.maximum(g, 0.0)

    sum_g_sq = np.sum(g * g)
    sum_g_g_next = np.sum(g[:-1] * g[1:])
    sum_g = np.sum(g)
    max_g = np.max(g)
    if max_g == 0 or sum_g == 0:
        return 0.0

    numerator = w * m_plus_1 * (sum_g_sq + 2.0 * sum_g_g_next)
    denominator = 3.0 * sum_g * max_g
    return numerator / denominator


def compute_value_smooth(h, smoothing=1e-4):
    h = np.asarray(h, dtype=float)
    if len(h) != N:
        return 0.0
    if np.any(h < 0) or np.any(~np.isfinite(h)):
        return 0.0

    n_fft = 1
    while n_fft < 2 * N - 1:
        n_fft <<= 1
    H = np.fft.rfft(h, n_fft)
    G = H * H
    g = np.fft.irfft(G, n_fft)[:m]
    g = np.real(g)
    g = np.maximum(g, 0.0)

    sum_g_sq = np.sum(g * g)
    sum_g_g_next = np.sum(g[:-1] * g[1:])
    sum_g = np.sum(g)

    max_g = np.max(g)
    if max_g > 0:
        shifted = (g - max_g) / smoothing
        shifted = np.clip(shifted, -50.0, 50.0)
        smooth_max = max_g + smoothing * np.log(np.sum(np.exp(shifted)))
    else:
        smooth_max = 0.0

    if smooth_max == 0 or sum_g == 0:
        return 0.0

    numerator = w * m_plus_1 * (sum_g_sq + 2.0 * sum_g_g_next)
    denominator = 3.0 * sum_g * smooth_max
    return numerator / denominator


def neg_value(log_h):
    centered = log_h - np.mean(log_h)
    h = np.exp(centered)
    return -compute_value(h)


def neg_value_smooth(log_h, smoothing=1e-4):
    centered = log_h - np.mean(log_h)
    h = np.exp(centered)
    return -compute_value_smooth(h, smoothing)


def coordinate_descent(h, max_iter=200, tol=1e-10, deadline=None):
    """Optimize one coordinate at a time in log-space."""
    h = np.maximum(h, 1e-8)
    log_h = np.log(h)
    log_h = log_h - np.mean(log_h)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(N):
            # Try small perturbations in log-space for coordinate i
            best_delta = 0.0
            best_val = compute_value(np.exp(log_h - np.mean(log_h)))
            
            # Golden section search in log-space
            for step in [0.05, 0.02, 0.01, 0.005, 0.002, 0.001]:
                for sign in [1.0, -1.0]:
                    trial = log_h.copy()
                    trial[i] += sign * step
                    trial = trial - np.mean(trial)
                    h_trial = np.exp(trial)
                    v_trial = compute_value(h_trial)
                    if v_trial > best_val + tol:
                        best_val = v_trial
                        best_delta = sign * step
            if best_delta != 0.0:
                log_h[i] += best_delta
                log_h = log_h - np.mean(log_h)
                improved = True
        if not improved:
            break
    
    h_final = np.exp(log_h - np.mean(log_h))
    h_final = np.maximum(h_final, 0.0)
    return h_final, compute_value(h_final)


def local_refine(h_start, deadline):
    """Multi-phase refinement: smooth then exact, with coordinate descent."""
    h_start = np.maximum(h_start, 1e-8)
    log_h_start = np.log(h_start)
    
    best_h = h_start.copy()
    best_v = compute_value(h_start)
    
    # Phase 1: smooth objective with decreasing smoothing
    for sm in [1e-2, 1e-3, 1e-4]:
        if time.time() > deadline - 20:
            break
        try:
            res = minimize(
                neg_value_smooth, log_h_start, args=(sm,),
                method='L-BFGS-B',
                options={'maxiter': 500, 'maxfun': 3000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            h1 = np.exp(res.x - np.mean(res.x))
            h1 = np.maximum(h1, 0.0)
            v1 = compute_value(h1)
            if v1 > best_v:
                best_v = v1
                best_h = h1.copy()
                log_h_start = res.x - np.mean(res.x)
        except Exception:
            pass
    
    # Phase 2: exact objective
    log_h1 = np.log(np.maximum(best_h, 1e-8))
    try:
        res2 = minimize(
            neg_value, log_h1,
            method='L-BFGS-B',
            options={'maxiter': 3000, 'maxfun': 20000, 'ftol': 1e-16, 'gtol': 1e-14}
        )
        h2 = np.exp(res2.x - np.mean(res2.x))
        h2 = np.maximum(h2, 0.0)
        v2 = compute_value(h2)
        if v2 > best_v:
            best_v = v2
            best_h = h2.copy()
    except Exception:
        pass
    
    # Phase 3: coordinate descent polish
    if time.time() < deadline - 10:
        h3, v3 = coordinate_descent(best_h, max_iter=100, deadline=deadline - 10)
        if v3 > best_v:
            best_v = v3
            best_h = h3.copy()
    
    return best_h, best_v


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    best_val = 0.0
    best_h = np.ones(N)
    seed_val = compute_value(best_h)
    if seed_val > best_val:
        best_val = seed_val
        best_h = best_h.copy()

    candidates = [np.ones(N)]

    # --- Generate diverse initial candidates ---
    x = np.arange(N, dtype=float)
    c = (N - 1) / 2.0
    
    # Various shapes
    h_tri = np.maximum(1.0 - np.abs(x - c) / c, 0.0)
    h_rc = 0.5 * (1.0 + np.cos(np.pi * (x - c) / c))
    h_sine = np.sin(np.pi * (x / (N - 1)))
    
    # Gaussian family with different widths
    for sigma in [3, 5, 7, 10, 15, 20]:
        h_g = np.exp(-((x - c)**2) / (2.0 * sigma**2))
        candidates.append(h_g)
    
    # Super-Gaussian
    for sigma in [5, 8, 12]:
        for beta in [2, 3, 4, 6]:
            dist = np.abs(x - c)
            h_sg = np.exp(-(dist**beta) / (2.0 * (sigma**beta)))
            candidates.append(h_sg)
    
    # Polynomial
    for R in [15, 20, 25, 30]:
        for p in [1, 2, 3, 4]:
            t = (x - c) / R
            h_poly = np.maximum(1.0 - t**2, 0.0) ** p
            candidates.append(h_poly)
    
    # Flat top
    for half_w in [5, 8, 10, 12, 15]:
        d = np.abs(x - c)
        h_ft = np.zeros(N)
        mask_inner = d < half_w
        mask_edge = (d >= half_w) & (d <= half_w * 1.5)
        h_ft[mask_inner] = 1.0
        d_edge = d[mask_edge]
        t_edge = (d_edge - half_w) / (half_w * 0.5)
        h_ft[mask_edge] = 0.5 * (1.0 + np.cos(np.pi * t_edge))
        candidates.append(h_ft)
    
    # Two bumps
    for sep in [5, 8, 10, 12]:
        for sigma in [3, 5, 7]:
            h_2b = np.exp(-((x - (c - sep))**2) / (2.0*sigma**2)) + \
                   np.exp(-((x - (c + sep))**2) / (2.0*sigma**2))
            candidates.append(h_2b)
    
    # Plateau
    for plat_w in [5, 8, 10, 12, 15]:
        h_plat = np.where(np.abs(x - c) <= plat_w, 1.0, 0.0)
        candidates.append(h_plat)
        h_plat_smooth = np.exp(-((np.abs(x - c) - plat_w).clip(0, None)**2) / 4.0)
        candidates.append(h_plat_smooth)
    
    # Random smooth candidates
    rng = np.random.default_rng(42)
    for _ in range(20):
        # Random smooth function via Fourier
        freqs = rng.uniform(0, 3, size=N)
        phases = rng.uniform(0, 2*np.pi, size=N)
        amps = rng.exponential(1.0, size=N)
        h_rand = np.zeros(N)
        for f, ph, a in zip(freqs, phases, amps):
            h_rand += a * np.cos(2*np.pi*f*x/N + ph)
        h_rand = np.maximum(h_rand, 0.0)
        if np.max(h_rand) > 0:
            h_rand = h_rand / np.max(h_rand)
            candidates.append(h_rand)
    
    # --- Local refinement of all candidates ---
    for h_cand in candidates:
        if time.time() > deadline - 15.0:
            break
        h_ref, v_ref = local_refine(h_cand, deadline)
        if v_ref > best_val:
            best_val = v_ref
            best_h = h_ref.copy()

    # --- Multi-start random perturbations around best ---
    rng2 = np.random.default_rng(12345)
    for _ in range(20):
        if time.time() > deadline - 10.0:
            break
        # perturb in log-space with varying scale
        scale = rng2.uniform(0.1, 0.5)
        log_h = np.log(np.maximum(best_h, 1e-8))
        log_h_pert = log_h + rng2.normal(0, scale, size=N)
        h_pert = np.exp(log_h_pert - np.mean(log_h_pert))
        h_pert = np.maximum(h_pert, 0.0)
        h_ref, v_ref = local_refine(h_pert, deadline)
        if v_ref > best_val:
            best_val = v_ref
            best_h = h_ref.copy()

    # --- Final high-precision polish ---
    if time.time() < deadline - 5.0:
        log_h_best = np.log(np.maximum(best_h, 1e-8))
        try:
            res = minimize(
                neg_value, log_h_best,
                method='L-BFGS-B',
                options={'maxiter': 5000, 'maxfun': 30000, 'ftol': 1e-18, 'gtol': 1e-16}
            )
            h_final = np.exp(res.x - np.mean(res.x))
            h_final = np.maximum(h_final, 0.0)
            val_final = compute_value(h_final)
            if val_final > best_val:
                best_val = val_final
                best_h = h_final.copy()
        except Exception:
            pass
        
        # Coordinate descent final polish
        if time.time() < deadline - 3.0:
            h_cd, v_cd = coordinate_descent(best_h, max_iter=200, deadline=deadline - 3.0)
            if v_cd > best_val:
                best_val = v_cd
                best_h = h_cd.copy()

    # Ensure output validity
    best_h = np.asarray(best_h, dtype=float)
    best_h = np.maximum(best_h, 0.0)
    best_h = np.where(np.isfinite(best_h), best_h, 1.0)

    return list(best_h)


if __name__ == "__main__":
    h = construct()
    val = compute_value(h)
    print(f"Value: {val}")
    print(f"Height sample: {h[:5]}")
