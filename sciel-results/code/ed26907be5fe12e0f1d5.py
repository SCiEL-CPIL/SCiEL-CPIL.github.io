# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import itertools

N = 11

# Vertices of the equilateral triangle in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
TRI_AREA = 0.5 * np.sqrt(3) / 2

# Precompute indices for all triples
_IDX_TRIPLES = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])

def _bary_to_cart(u, v):
    """
    Convert barycentric coordinates (u, v) to Cartesian.
    w = 1 - u - v
    Point = u*V0 + v*V1 + w*V2
    """
    w = 1.0 - u - v
    # P = u*(0,0) + v*(1,0) + w*(0.5, sqrt(3)/2)
    x = v * 1.0 + w * 0.5
    y = w * np.sqrt(3) / 2
    return np.column_stack((x, y))

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

def _objective_smooth(x, beta, penalty_lambda=100.0):
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
    # SLSQP handles bounds, but we clamp for safety in area calc
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

def _structured_init_fps():
    """
    Farthest Point Sampling on a grid of barycentric coordinates.
    """
    n_grid = 20
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
            min_d = np.min(np.linalg.norm(c - selected, axis=1))
            dists.append(min_d)
        dists = np.array(dists)
        max_idx = np.argmax(dists)
        selected.append(candidates[max_idx])
        
    return np.array(selected)

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

def _local_optimize_nm(x0, maxiter=300, xatol=1e-8, fatol=1e-8):
    """Run Nelder-Mead local optimization from x0."""
    try:
        res = minimize(
            _objective_slsqp, x0, method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': xatol, 'fatol': fatol, 'adaptive': True}
        )
        return res.x
    except Exception:
        return x0

def _local_optimize_lbfgs(x0, maxiter=200, gtol=1e-9, beta=100.0):
    """Run L-BFGS-B local optimization from x0 with smooth objective."""
    try:
        bounds = [(0.0, 1.0)] * (2 * N)
        res = minimize(
            _objective_smooth, x0, method='L-BFGS-B',
            bounds=bounds, args=(beta,),
            options={'maxiter': maxiter, 'gtol': gtol}
        )
        return res.x
    except Exception:
        return x0

def _local_optimize_slsqp(x0, maxiter=200):
    """Run SLSQP local optimization from x0."""
    try:
        bounds = [(0.0, 1.0)] * (2 * N)
        res = minimize(
            _objective_slsqp, x0, method='SLSQP',
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-10}
        )
        return res.x
    except Exception:
        return x0

def _anneal_optimize(x0, betas, maxiter=150, gtol=1e-8):
    """
    Anneal beta for L-BFGS-B to progressively sharpen the soft-min surrogate.
    """
    current = x0
    for beta in betas:
        current = _local_optimize_lbfgs(current, maxiter=maxiter, gtol=gtol, beta=beta)
        current = _project_to_bary_simplex(current.reshape(N, 2)).flatten()
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_x = None
    best_score = -float('inf')
    
    inits = []
    
    # 1. FPS Init
    inits.append(_structured_init_fps().flatten())
    
    # 2. Random Inits
    rng = np.random.RandomState(42)
    for _ in range(10):
        pts = np.zeros((N, 2))
        for i in range(N):
            u = rng.random()
            v = rng.random()
            while u + v > 1.0:
                u = rng.random()
                v = rng.random()
            pts[i, 0] = u
            pts[i, 1] = v
        inits.append(pts.flatten())
        
    # 3. "Magic" Init: Vertices, Edge Midpoints, Centroid + 4 more
    magic_pts = [
        [1.0, 0.0], # V0
        [0.0, 1.0], # V1
        [0.0, 0.0], # V2
        [1/3, 1/3], # Centroid
        [0.5, 0.0], # Mid V0-V1
        [0.0, 0.5], # Mid V1-V2
        [0.5, 0.5], # Mid V0-V2? No, 0.5+0.5=1, w=0. Mid V0-V2 is u=0.5, v=0, w=0.5?
                    # P = 0.5 V0 + 0 V1 + 0.5 V2. Yes.
        # Need 4 more. Let's add quarter points.
        [0.25, 0.0], # 1/4 V0 - 3/4 V2? No. u=0.25, v=0, w=0.75.
        [0.0, 0.25], # u=0, v=0.25, w=0.75.
        [0.25, 0.25], # u=0.25, v=0.25, w=0.5.
        [0.5, 0.25], # u=0.5, v=0.25, w=0.25.
    ]
    inits.append(np.array(magic_pts).flatten())
    
    # 4. Structured Starts from Inspiration
    # Start 1: Hexagonal lattice pattern
    bary1 = np.array([
        [0.0, 0.0],    # V0
        [1.0, 0.0],    # V1
        [0.0, 1.0],    # V2
        [0.5, 0.0],    # mid V0-V1
        [0.5, 0.5],    # mid V1-V2
        [0.0, 0.5],    # mid V2-V0
        [1/3, 1/3],    # centroid
        [0.25, 0.0],   # 1/4 V0-V1
        [0.75, 0.0],   # 3/4 V0-V1
        [0.25, 0.25],  # interior
        [0.5, 0.75],   # interior
    ])
    inits.append(_project_to_bary_simplex(bary1).flatten())
    
    # Start 2: Concentric hexagons
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
    
    # Start 3: Symmetric grid with jitter
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
    
    # Start 4: Random with structured bias
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

    # Run optimization on each init
    for init in inits:
        if time.time() > deadline:
            break
            
        try:
            # Project init to valid simplex
            init_bary = init.reshape(N, 2)
            init_bary = _project_to_bary_simplex(init_bary)
            
            # 1. Annealed L-BFGS-B
            if time.time() < deadline - 10:
                x_annealed = _anneal_optimize(init_bary.flatten(), betas=[20, 50, 100, 200])
                x_annealed = _project_to_bary_simplex(x_annealed.reshape(N, 2))
                score = _min_area_ratio_from_bary(x_annealed)
                if score > best_score:
                    best_score = score
                    best_x = x_annealed
            
            # 2. Nelder-Mead from best so far or original init
            if time.time() < deadline - 5:
                x_nm = _local_optimize_nm(init_bary.flatten(), maxiter=500)
                x_nm = _project_to_bary_simplex(x_nm.reshape(N, 2))
                score = _min_area_ratio_from_bary(x_nm)
                if score > best_score:
                    best_score = score
                    best_x = x_nm
                    
        except Exception:
            pass
            
    # If we have time, do a few more random restarts or DE?
    # Let's just do a few more NM restarts if time permits
    if time.time() < deadline - 20:
        for _ in range(5):
            if time.time() > deadline:
                break
            pts = np.zeros((N, 2))
            for i in range(N):
                u = np.random.random()
                v = np.random.random()
                while u + v > 1.0:
                    u = np.random.random()
                    v = np.random.random()
                pts[i, 0] = u
                pts[i, 1] = v
            pts = _project_to_bary_simplex(pts)
            try:
                x_nm = _local_optimize_nm(pts.flatten(), maxiter=300)
                x_nm = _project_to_bary_simplex(x_nm.reshape(N, 2))
                score = _min_area_ratio_from_bary(x_nm)
                if score > best_score:
                    best_score = score
                    best_x = x_nm
            except Exception:
                pass

    if best_x is None:
        # Fallback: FPS
        best_x = _structured_init_fps()
        
    # Convert back to Cartesian for output
    cart_points = _bary_to_cart(best_x[:, 0], best_x[:, 1])
    
    return cart_points
# EVOLVE-BLOCK-END
