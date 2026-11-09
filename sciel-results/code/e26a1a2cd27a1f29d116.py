import numpy as np
import time
from scipy.optimize import minimize

def _angles_to_points(thetas):
    thetas = np.asarray(thetas, dtype=np.float64)
    return np.column_stack((np.cos(thetas), np.sin(thetas)))

def _normalize(points):
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    dists = np.linalg.norm(centered, axis=1)
    mean_dist = np.mean(dists)
    if mean_dist <= 0.01:
        return None
    return centered / mean_dist

def _is_strictly_convex(points):
    n = len(points)
    if n < 3:
        return False
    # Check no repeated points
    dists = np.linalg.norm(points[:, np.newaxis, :] - points[np.newaxis, :, :], axis=-1)
    np.fill_diagonal(dists, np.inf)
    if np.min(dists) < 1e-10:
        return False
    
    # Cross products
    p0 = points
    p1 = np.roll(points, -1, axis=0)
    p2 = np.roll(points, -2, axis=0)
    v1 = p1 - p0
    v2 = p2 - p1
    crosses = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    if np.any(np.abs(crosses) < 1e-14):
        return False
    if len(np.unique(np.sign(crosses))) != 1:
        return False
    return True

def _vertex_scores(points):
    n = len(points)
    dists = np.linalg.norm(points[:, np.newaxis, :] - points[np.newaxis, :, :], axis=-1)
    np.fill_diagonal(dists, np.inf)
    sorted_dists = np.sort(dists, axis=1)
    
    scores = np.ones(n)
    
    # Condition 1: d_0 < 0.05
    invalid1 = sorted_dists[:, 0] < 0.05
    scores[invalid1] = 0.0
    
    # Condition 2: D < 0.1
    gaps = sorted_dists[:, 1:] - sorted_dists[:, :-1]
    k = np.argmin(gaps, axis=1)
    
    idx = np.arange(n)
    d_k = sorted_dists[idx, k]
    d_k1 = sorted_dists[idx, k + 1]
    D = (d_k + d_k1) / 2.0
    
    invalid2 = D < 0.1
    scores[invalid2] = 0.0
    
    valid_mask = ~(invalid1 | invalid2)
    
    if not np.any(valid_mask):
        return scores
        
    # For valid vertices, compute raw score
    d_sel = np.zeros((n, 4))
    
    # k == 0
    m0 = valid_mask & (k == 0)
    if np.any(m0):
        d_sel[m0] = sorted_dists[m0, 0:4]
        
    # k == 17
    m17 = valid_mask & (k == 17)
    if np.any(m17):
        d_sel[m17] = sorted_dists[m17, 15:19]
        
    # else
    me = valid_mask & ~m0 & ~m17
    if np.any(me):
        ke = k[me]
        de = sorted_dists[me]
        # indices: ke-1, ke, ke+1, ke+2
        idxs = np.stack([ke - 1, ke, ke + 1, ke + 2], axis=1)
        d_sel[me] = de[np.arange(len(ke))[:, np.newaxis], idxs]
        
    D_valid = D[valid_mask]
    d_sel_valid = d_sel[valid_mask]
    
    ratios = np.maximum(d_sel_valid / D_valid[:, np.newaxis], D_valid[:, np.newaxis] / d_sel_valid)
    raw = np.mean(ratios, axis=1)
    
    scores[valid_mask] = np.minimum(1.0, 1.0 / raw)
    
    return scores

def _polygon_score(points):
    scores = _vertex_scores(points)
    return np.min(scores)

def _soft_min(scores, beta):
    if len(scores) == 0:
        return 0.0
    shifted = -beta * scores
    max_shifted = np.max(shifted)
    if not np.isfinite(max_shifted):
        return 0.0
    result = -max_shifted / beta - np.log(np.sum(np.exp(shifted - max_shifted))) / beta
    return result

def _compute_scores_from_thetas(thetas):
    points = _angles_to_points(thetas)
    normalized = _normalize(points)
    if normalized is None:
        return None
    if not _is_strictly_convex(normalized):
        return None
    scores = _vertex_scores(normalized)
    return scores

def _neg_soft_score(thetas, beta):
    scores = _compute_scores_from_thetas(thetas)
    if scores is None:
        return 0.0
    if np.any(scores == 0):
        return 0.0
    return -_soft_min(scores, beta)

def _neg_exact_score(thetas):
    scores = _compute_scores_from_thetas(thetas)
    if scores is None:
        return 0.0
    return -np.min(scores)

def _regular_angles(n=20):
    return np.linspace(0, 2 * np.pi, n, endpoint=False)

def _generate_initialization(n, rng, mode):
    base = _regular_angles(n)
    if mode == 0:
        return base.copy()
    elif mode == 1:
        # Small random jitter
        jitter = rng.normal(0, 0.03, size=n)
        thetas = base + jitter
        return np.sort(thetas % (2 * np.pi))
    elif mode == 2:
        # Medium random jitter
        jitter = rng.normal(0, 0.08, size=n)
        thetas = base + jitter
        return np.sort(thetas % (2 * np.pi))
    elif mode == 3:
        # Fourier perturbation
        n_modes = 4
        amps = rng.uniform(0.02, 0.06, size=n_modes)
        phases = rng.uniform(0, 2 * np.pi, size=n_modes)
        t = np.arange(n)
        jitter = np.zeros(n)
        for m in range(1, n_modes + 1):
            jitter += amps[m - 1] * np.sin(m * 2 * np.pi * t / n + phases[m - 1])
        thetas = base + jitter
        return np.sort(thetas % (2 * np.pi))
    elif mode == 4:
        # Random spacings
        spacings = rng.dirichlet(np.ones(n) * 2) * 2 * np.pi
        thetas = np.cumsum(spacings)
        thetas[-1] = 0  # close the loop
        return np.sort(thetas % (2 * np.pi))
    else:
        # Harmonic perturbation
        k = rng.integers(2, 8)
        amp = rng.uniform(0.01, 0.04)
        phase = rng.uniform(0, 2 * np.pi)
        t = np.arange(n)
        jitter = amp * np.sin(k * 2 * np.pi * t / n + phase)
        thetas = base + jitter
        return np.sort(thetas % (2 * np.pi))

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 600  # 30s safety margin
    
    n = 20
    best_thetas = _regular_angles(n)
    best_score = _polygon_score(_normalize(_angles_to_points(best_thetas)))
    
    rng = np.random.default_rng(2024)
    
    def time_ok():
        return time.time() < deadline
    
    # Phase 1: Coarse search with diverse initializations
    beta1 = 10.0
    n_coarse = 20
    for i in range(n_coarse):
        if not time_ok():
            break
        mode = i % 6
        x0 = _generate_initialization(n, rng, mode)
        try:
            res = minimize(
                lambda th: _neg_soft_score(th, beta1),
                x0,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-7, 'maxfun': 30000}
            )
            if res.x is not None:
                s = -_neg_exact_score(res.x)
                if s > best_score:
                    best_score = s
                    best_thetas = res.x.copy()
        except Exception:
            pass
    
    # Phase 2: Medium refinement
    beta2 = 30.0
    n_med = 15
    for i in range(n_med):
        if not time_ok():
            break
        if i == 0:
            x0 = best_thetas.copy()
        elif i < 5:
            x0 = _generate_initialization(n, rng, i % 6)
        else:
            x0 = best_thetas + rng.normal(0, 0.015, size=n)
            x0 = np.sort(x0 % (2 * np.pi))
        try:
            res = minimize(
                lambda th: _neg_soft_score(th, beta2),
                x0,
                method='L-BFGS-B',
                options={'maxiter': 4000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 50000}
            )
            if res.x is not None:
                s = -_neg_exact_score(res.x)
                if s > best_score:
                    best_score = s
                    best_thetas = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: Fine refinement
    beta3 = 60.0
    n_fine = 10
    for i in range(n_fine):
        if not time_ok():
            break
        if i == 0:
            x0 = best_thetas.copy()
        else:
            x0 = best_thetas + rng.normal(0, 0.005, size=n)
            x0 = np.sort(x0 % (2 * np.pi))
        try:
            res = minimize(
                lambda th: _neg_soft_score(th, beta3),
                x0,
                method='L-BFGS-B',
                options={'maxiter': 6000, 'ftol': 1e-13, 'gtol': 1e-9, 'maxfun': 70000}
            )
            if res.x is not None:
                s = -_neg_exact_score(res.x)
                if s > best_score:
                    best_score = s
                    best_thetas = res.x.copy()
        except Exception:
            pass
    
    # Phase 4: Very fine with high beta
    beta4 = 120.0
    n_vfine = 6
    for i in range(n_vfine):
        if not time_ok():
            break
        if i == 0:
            x0 = best_thetas.copy()
        else:
            x0 = best_thetas + rng.normal(0, 0.002, size=n)
            x0 = np.sort(x0 % (2 * np.pi))
        try:
            res = minimize(
                lambda th: _neg_soft_score(th, beta4),
                x0,
                method='L-BFGS-B',
                options={'maxiter': 8000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 90000}
            )
            if res.x is not None:
                s = -_neg_exact_score(res.x)
                if s > best_score:
                    best_score = s
                    best_thetas = res.x.copy()
        except Exception:
            pass
    
    # Phase 5: Exact objective optimization
    n_exact = 10
    for i in range(n_exact):
        if not time_ok():
            break
        if i == 0:
            x0 = best_thetas.copy()
        else:
            x0 = best_thetas + rng.normal(0, 0.0005, size=n)
            x0 = np.sort(x0 % (2 * np.pi))
        try:
            res = minimize(
                _neg_exact_score,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 20000, 'ftol': 1e-16, 'gtol': 1e-12, 'maxfun': 200000}
            )
            if res.x is not None:
                s = -_neg_exact_score(res.x)
                if s > best_score:
                    best_score = s
                    best_thetas = res.x.copy()
        except Exception:
            pass
    
    # Phase 6: Nelder-Mead local search
    if time_ok():
        try:
            res = minimize(
                _neg_exact_score,
                best_thetas.copy(),
                method='Nelder-Mead',
                options={'maxiter': 20000, 'xatol': 1e-15, 'fatol': 1e-15}
            )
            if res.x is not None:
                s = -_neg_exact_score(res.x)
                if s > best_score:
                    best_score = s
                    best_thetas = res.x.copy()
        except Exception:
            pass
    
    # Phase 7: Coordinate descent polish
    if time_ok():
        current = best_thetas.copy()
        for rnd in range(30):
            if not time_ok():
                break
            improved = False
            for i in range(n):
                if not time_ok():
                    break
                for delta in [0.0005, -0.0005, 0.0001, -0.0001, 0.00005, -0.00005, 0.00001, -0.00001]:
                    trial = current.copy()
                    trial[i] += delta
                    trial = np.sort(trial % (2 * np.pi))
                    s = -_neg_exact_score(trial)
                    if s > best_score:
                        best_score = -s
                        best_thetas = trial.copy()
                        current = trial.copy()
                        improved = True
                        break
            if not improved:
                break
    
    # Final validation
    final_points = _angles_to_points(best_thetas)
    normalized = _normalize(final_points)
    if normalized is not None and _is_strictly_convex(normalized):
        return normalized
    else:
        regular = _angles_to_points(_regular_angles(n))
        return _normalize(regular)
