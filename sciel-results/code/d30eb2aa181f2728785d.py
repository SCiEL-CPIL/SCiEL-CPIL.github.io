import numpy as np
from scipy.optimize import minimize
import time

N = 11

# Vertices of the equilateral triangle in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
TRI_AREA = 0.5 * np.sqrt(3) / 2  # area of equilateral triangle with side 1

# Precompute indices for all triples
_IDX_TRIPLES = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])
n_triples = len(_IDX_TRIPLES)  # 165

def _all_dets_from_bary(bary):
    """
    Compute all 165 determinants (twice the area in barycentric coords).
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    P = np.column_stack((u, v, w))
    
    i_idx = _IDX_TRIPLES[:, 0]
    j_idx = _IDX_TRIPLES[:, 1]
    k_idx = _IDX_TRIPLES[:, 2]
    
    Pi = P[i_idx]
    Pj = P[j_idx]
    Pk = P[k_idx]
    
    det = (Pi[:, 0] * (Pj[:, 1] * Pk[:, 2] - Pj[:, 2] * Pk[:, 1])
           - Pi[:, 1] * (Pj[:, 0] * Pk[:, 2] - Pj[:, 2] * Pk[:, 0])
           + Pi[:, 2] * (Pj[:, 0] * Pk[:, 1] - Pj[:, 1] * Pk[:, 0]))
           
    return det

def _min_area_ratio_from_bary(bary):
    """
    Returns the minimum absolute determinant (proportional to area ratio).
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    if np.any(u < -1e-9) or np.any(v < -1e-9) or np.any(w < -1e-9):
        return -1.0
        
    dets = _all_dets_from_bary(bary)
    return np.min(np.abs(dets))

def _bary_to_cart(u, v):
    """
    Convert barycentric coordinates (u, v) to Cartesian.
    w = 1 - u - v
    Point = u*V0 + v*V1 + w*V2
    """
    w = 1.0 - u - v
    x = v * 1.0 + w * 0.5
    y = w * np.sqrt(3) / 2
    return np.column_stack((x, y))

def _project_to_bary_simplex(bary):
    """
    Project points to the valid barycentric simplex.
    """
    u = np.maximum(0, bary[:, 0])
    v = np.maximum(0, bary[:, 1])
    s = u + v
    mask = s > 1.0
    if np.any(mask):
        u[mask] /= s[mask]
        v[mask] /= s[mask]
    return np.column_stack((u, v))

def _objective_smooth(x, beta=50.0, penalty_lambda=100.0):
    """
    Smooth objective for L-BFGS-B.
    x: (2N,) array of u, v coords.
    """
    bary = x.reshape(N, 2)
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    # Penalty for invalid barycentric coords (u,v,w < 0)
    penalty = 0.0
    excess = np.maximum(0, u + v - 1.0)
    penalty += penalty_lambda * np.sum(excess**2)
    excess_u = np.maximum(0, -u)
    penalty += penalty_lambda * np.sum(excess_u**2)
    excess_v = np.maximum(0, -v)
    penalty += penalty_lambda * np.sum(excess_v**2)
    
    dets = _all_dets_from_bary(bary)
    areas = np.abs(dets)
    
    if len(areas) == 0:
        return penalty
        
    m = np.min(areas)
    shifted = areas - m
    log_sum_exp = np.log(np.sum(np.exp(-beta * shifted)))
    
    smooth_min = m - (1.0 / beta) * log_sum_exp
    
    return -smooth_min + penalty

def _objective_slsqp(x):
    """
    Objective for SLSQP: minimize -min_area.
    x: (2N,) array.
    """
    bary = x.reshape(N, 2)
    u = np.clip(bary[:, 0], 0, None)
    v = np.clip(bary[:, 1], 0, None)
    s = u + v
    mask = s > 1.0
    if np.any(mask):
        u[mask] /= s[mask]
        v[mask] /= s[mask]
    
    bary_clamped = np.column_stack((u, v))
    dets = _all_dets_from_bary(bary_clamped)
    min_area = np.min(np.abs(dets))
    return -min_area

def _local_optimize_lbfgs(x0, beta=50.0, maxiter=300, gtol=1e-10):
    """Run L-BFGS-B local optimization from x0."""
    try:
        res = minimize(
            _objective_smooth, x0, method='L-BFGS-B',
            args=(beta,),
            bounds=[(0, 1)] * (2 * N),
            options={'maxiter': maxiter, 'gtol': gtol}
        )
        return res.x
    except Exception:
        return x0

def _local_optimize_slsqp(x0, maxiter=300):
    """Run SLSQP local optimization from x0."""
    try:
        res = minimize(
            _objective_slsqp, x0, method='SLSQP',
            bounds=[(0, 1)] * (2 * N),
            options={'maxiter': maxiter, 'ftol': 1e-12}
        )
        return res.x
    except Exception:
        return x0

def _local_optimize_nm(x0, maxiter=500, xatol=1e-10, fatol=1e-10):
    """Run Nelder-Mead local optimization from x0."""
    try:
        res = minimize(
            _objective_slsqp, x0, method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': xatol, 'fatol': fatol, 'adaptive': True}
        )
        return res.x
    except Exception:
        return x0

def _structured_init_fps():
    """
    Farthest Point Sampling on a grid of barycentric coordinates.
    """
    n_grid = 25
    candidates = []
    for i in range(n_grid + 1):
        for j in range(n_grid + 1 - i):
            u = i / n_grid
            v = j / n_grid
            candidates.append([u, v])
    candidates = np.array(candidates)
    
    selected = [np.array([1/3, 1/3])]
    
    for _ in range(N - 1):
        dists = []
        for c in candidates:
            min_d = np.min(np.linalg.norm(c - np.array(selected), axis=1))
            dists.append(min_d)
        dists = np.array(dists)
        max_idx = np.argmax(dists)
        selected.append(candidates[max_idx])
        
    return np.array(selected)

def _simulated_annealing(x0, n_iter=3000, t_start=0.05, t_end=0.001, time_budget=20.0):
    """
    Simulated annealing to explore the landscape.
    """
    start_time = time.time()
    x = x0.copy()
    best_x = x.copy()
    best_score = _min_area_ratio_from_bary(x.reshape(N, 2)) / (2 * TRI_AREA)
    
    rng = np.random.RandomState(42)
    
    for i in range(n_iter):
        if time.time() - start_time > time_budget:
            break
        
        # Annealing schedule
        t = t_start * (t_end / t_start) ** (i / n_iter)
        
        # Generate a perturbation
        scale = t * 0.5
        x_new = x + rng.normal(0, scale, size=x.shape)
        x_new = _project_to_bary_simplex(x_new.reshape(N, 2)).flatten()
        
        # Evaluate
        bary_new = x_new.reshape(N, 2)
        score_new = _min_area_ratio_from_bary(bary_new) / (2 * TRI_AREA)
        
        if score_new > best_score:
            best_score = score_new
            best_x = x_new.copy()
        
        # Accept or reject
        if score_new >= best_score:
            x = x_new
        else:
            delta = score_new - best_score
            if delta > 0 or rng.random() < np.exp(delta / t):
                x = x_new
    
    return best_x, best_score

def _final_polish(x0, time_budget=10.0):
    """
    Final high-precision polish: alternate between L-BFGS-B with increasing beta,
    SLSQP, and Nelder-Mead for tight convergence.
    """
    start_time = time.time()
    x = x0.copy()
    best_x = x.copy()
    best_score = _min_area_ratio_from_bary(x.reshape(N, 2)) / (2 * TRI_AREA)
    
    # Multi-scale soft-min polishing
    for beta in [50.0, 100.0, 200.0, 500.0, 1000.0]:
        if time.time() - start_time > time_budget:
            break
        x = _local_optimize_lbfgs(x, beta=beta, maxiter=200, gtol=1e-12)
        x = _project_to_bary_simplex(x.reshape(N, 2)).flatten()
        s = _min_area_ratio_from_bary(x.reshape(N, 2)) / (2 * TRI_AREA)
        if s > best_score:
            best_score = s
            best_x = x.copy()
    
    # SLSQP refinement
    if time.time() - start_time < time_budget:
        x = _local_optimize_slsqp(x, maxiter=300)
        x = _project_to_bary_simplex(x.reshape(N, 2)).flatten()
        s = _min_area_ratio_from_bary(x.reshape(N, 2)) / (2 * TRI_AREA)
        if s > best_score:
            best_score = s
            best_x = x.copy()
    
    # Nelder-Mead final polish
    if time.time() - start_time < time_budget:
        x = _local_optimize_nm(x, maxiter=500, xatol=1e-12, fatol=1e-12)
        x = _project_to_bary_simplex(x.reshape(N, 2)).flatten()
        s = _min_area_ratio_from_bary(x.reshape(N, 2)) / (2 * TRI_AREA)
        if s > best_score:
            best_score = s
            best_x = x.copy()
    
    return best_x, best_score

def score(points):
    """Compute the raw score: min triangle area / TRI_AREA."""
    bary = np.zeros((N, 2))
    for i in range(N):
        p = points[i]
        w = 2 * p[1] / np.sqrt(3)
        v = p[0] - w * 0.5
        u = 1 - v - w
        bary[i] = [u, v]
    
    dets = _all_dets_from_bary(bary)
    min_area = np.min(np.abs(dets))
    return min_area / (2 * TRI_AREA)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    def time_left():
        return deadline - time.time()
        
    best_x = None
    best_score = -float('inf')
    
    inits = []
    
    # 1. FPS Init
    inits.append(_structured_init_fps().flatten())
    
    # 2. Random Inits (more diverse)
    rng = np.random.RandomState(42)
    for seed in range(20):
        rng_local = np.random.RandomState(seed * 1000 + 7)
        pts = np.zeros((N, 2))
        for i in range(N):
            u = rng_local.random()
            v = rng_local.random()
            while u + v > 1.0:
                u = rng_local.random()
                v = rng_local.random()
            pts[i, 0] = u
            pts[i, 1] = v
        inits.append(pts.flatten())
        
    # 3. Magic Init
    magic_pts = [
        [1.0, 0.0], [0.0, 1.0], [0.0, 0.0],
        [1/3, 1/3],
        [0.5, 0.0], [0.0, 0.5], [0.5, 0.5],
        [0.25, 0.0], [0.0, 0.25], [0.25, 0.25], [0.5, 0.25],
    ]
    inits.append(np.array(magic_pts).flatten())
    
    # 4. Structured Starts
    bary1 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.0],
        [0.5, 0.5],
        [0.0, 0.5],
        [1/3, 1/3],
        [0.25, 0.0],
        [0.75, 0.0],
        [0.25, 0.25],
        [0.5, 0.75],
    ])
    inits.append(_project_to_bary_simplex(bary1).flatten())
    
    bary2 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.333, 0.0],
        [0.667, 0.0],
        [0.0, 0.333],
        [0.0, 0.667],
        [0.333, 0.333],
        [0.667, 0.333],
        [0.333, 0.667],
        [0.5, 0.5],
    ])
    inits.append(_project_to_bary_simplex(bary2).flatten())
    
    rng3 = np.random.RandomState(123)
    bary3 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.25, 0.0],
        [0.75, 0.0],
        [0.0, 0.25],
        [0.0, 0.75],
        [0.25, 0.25],
        [0.5, 0.25],
        [0.25, 0.5],
        [0.5, 0.5],
    ])
    bary3 += rng3.normal(0, 0.02, size=bary3.shape)
    inits.append(_project_to_bary_simplex(bary3).flatten())
    
    rng4 = np.random.RandomState(456)
    bary4 = []
    bary4.append([0.0, 0.0])
    bary4.append([1.0, 0.0])
    bary4.append([0.0, 1.0])
    bary4.append([1/3, 1/3])
    for _ in range(7):
        u = rng4.uniform(0.1, 0.9)
        v = rng4.uniform(0.1, 0.9)
        if u + v < 0.85:
            bary4.append([u, v])
        else:
            bary4.append([u * 0.8, v * 0.8])
    inits.append(_project_to_bary_simplex(np.array(bary4[:N])).flatten())
    
    # 5. Additional structured inits based on known good patterns
    bary5 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.2, 0.0],
        [0.8, 0.0],
        [0.0, 0.2],
        [0.0, 0.8],
        [0.33, 0.33],
        [0.67, 0.33],
        [0.33, 0.67],
        [0.5, 0.5],
    ])
    inits.append(_project_to_bary_simplex(bary5).flatten())
    
    bary6 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.15, 0.0],
        [0.85, 0.0],
        [0.0, 0.15],
        [0.0, 0.85],
        [0.4, 0.3],
        [0.6, 0.3],
        [0.4, 0.6],
        [0.5, 0.5],
    ])
    inits.append(_project_to_bary_simplex(bary6).flatten())
    
    # Optimize each initial point
    for init in inits:
        if time_left() < 8.0:
            break
        
        x0 = _project_to_bary_simplex(init.reshape(N, 2)).flatten()
        
        # Phase 1: Simulated annealing for exploration
        if time_left() > 25.0:
            x_sa, score_sa = _simulated_annealing(x0, n_iter=2000, t_start=0.04, t_end=0.0005, time_budget=15.0)
            if score_sa > best_score:
                best_score = score_sa
                best_x = x_sa.copy()
            x0 = x_sa
        
        # Phase 2: Multi-scale L-BFGS-B polishing
        if time_left() > 15.0:
            for beta in [30.0, 100.0, 300.0]:
                if time_left() < 5.0:
                    break
                x0 = _local_optimize_lbfgs(x0, beta=beta, maxiter=200, gtol=1e-10)
                x0 = _project_to_bary_simplex(x0.reshape(N, 2)).flatten()
                s = _min_area_ratio_from_bary(x0.reshape(N, 2)) / (2 * TRI_AREA)
                if s > best_score:
                    best_score = s
                    best_x = x0.copy()
        
        # Phase 3: SLSQP refinement
        if time_left() > 10.0:
            x0 = _local_optimize_slsqp(x0, maxiter=200)
            x0 = _project_to_bary_simplex(x0.reshape(N, 2)).flatten()
            s = _min_area_ratio_from_bary(x0.reshape(N, 2)) / (2 * TRI_AREA)
            if s > best_score:
                best_score = s
                best_x = x0.copy()
        
        # Phase 4: Nelder-Mead final polish
        if time_left() > 8.0:
            x0 = _local_optimize_nm(x0, maxiter=300, xatol=1e-10, fatol=1e-10)
            x0 = _project_to_bary_simplex(x0.reshape(N, 2)).flatten()
            s = _min_area_ratio_from_bary(x0.reshape(N, 2)) / (2 * TRI_AREA)
            if s > best_score:
                best_score = s
                best_x = x0.copy()
    
    # Final polish on the best solution found
    if best_x is not None and time_left() > 5.0:
        best_x, best_score = _final_polish(best_x, time_budget=min(10.0, time_left() - 1.0))
    
    # If we found a good solution, convert to Cartesian
    if best_x is not None:
        bary_best = best_x.reshape(N, 2)
        pts_best = _bary_to_cart(bary_best[:, 0], bary_best[:, 1])
        return pts_best
    
    # Fallback: return random valid points
    pts = []
    for _ in range(N):
        u, v = np.random.random(2)
        while u + v > 1:
            u, v = np.random.random(2)
        pts.append([u + 0.5*v, np.sqrt(3)/2*v])
    return np.array(pts)
