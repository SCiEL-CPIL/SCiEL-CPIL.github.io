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

# Precompute indices for all triples (i < j < k)
_IDX_TRIPLES = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])
N_TRIPLES = len(_IDX_TRIPLES)

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

def _all_dets_from_bary(bary):
    """
    Compute all determinants (twice the area in barycentric coords).
    bary: (N, 2) array of (u, v)
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
    
    # Determinant of [Pi, Pj, Pk]
    det = (Pi[:, 0] * (Pj[:, 1] * Pk[:, 2] - Pj[:, 2] * Pk[:, 1])
           - Pi[:, 1] * (Pj[:, 0] * Pk[:, 2] - Pj[:, 2] * Pk[:, 0])
           + Pi[:, 2] * (Pj[:, 0] * Pk[:, 1] - Pj[:, 1] * Pk[:, 0]))
           
    return det

def _min_area_ratio_from_bary(bary):
    """
    Returns the minimum absolute determinant.
    If invalid, returns -1.0
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
    
    # Penalty for invalid barycentric coords
    penalty = 0.0
    excess = np.maximum(0, u + v - 1.0)
    penalty += penalty_lambda * np.sum(excess**2)
    
    dets = _all_dets_from_bary(bary)
    areas = np.abs(dets)
    
    if len(areas) == 0:
        return penalty
        
    m = np.min(areas)
    shifted = areas - m
    # Softmin approximation: -log(sum(exp(-beta * x))) / beta + min
    # Actually: min(x) ~ -1/beta * log(sum(exp(-beta * x)))
    # We want to maximize min, so minimize -min.
    # -min(x) ~ 1/beta * log(sum(exp(-beta * x)))
    # Let's use: - (m - (1/beta) * log(sum(exp(-beta * (areas - m)))))
    # This is a smooth approximation of -min(areas).
    
    # To avoid overflow/underflow, we shift by m
    # log_sum_exp(-beta * (areas - m)) = log(sum(exp(-beta * (areas - m))))
    # Since areas - m >= 0, exp(-beta * (areas - m)) <= 1.
    # sum(exp(-beta * (areas - m))) >= 1 (since the min term is exp(0)=1).
    # So log(sum(...)) >= 0.
    # -min(areas) = -m.
    # Smooth approx: -m + (1/beta) * log(sum(exp(-beta * (areas - m))))
    
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

def _local_optimize_nm(x0, bounds, maxiter=300, xatol=1e-8, fatol=1e-8):
    """Run Nelder-Mead local optimization from x0."""
    try:
        res = minimize(
            _objective_slsqp, x0, method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': xatol, 'fatol': fatol, 'adaptive': True}
        )
        return res.x
    except Exception:
        return x0

def _local_optimize_lbfgs(x0, bounds, maxiter=200, gtol=1e-9):
    """Run L-BFGS-B local optimization from x0."""
    try:
        res = minimize(
            _objective_smooth, x0, method='L-BFGS-B',
            bounds=bounds,
            args=(50.0, 100.0),
            options={'maxiter': maxiter, 'gtol': gtol}
        )
        return res.x
    except Exception:
        return x0

def _local_optimize_slsqp(x0, bounds, maxiter=200):
    """Run SLSQP local optimization from x0."""
    try:
        res = minimize(
            _objective_slsqp, x0, method='SLSQP',
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-10}
        )
        return res.x
    except Exception:
        return x0

def _combinatorial_refinement(x0, time_budget=5.0):
    """
    Topological sweep-line / combinatorial exchange heuristic.
    Identifies the active set of triangles defining the minimum area.
    Attempts to swap points to increase the 12th smallest area (or similar).
    This operates on the discrete ordering of areas.
    """
    start_time = time.time()
    current_bary = x0.copy()
    current_min_area = _min_area_ratio_from_bary(current_bary)
    
    if current_min_area < 0:
        return current_bary
        
    # Precompute triangle indices for faster lookup
    # We want to find the triangle with the minimum area.
    # Then try to move one of its vertices to increase the area.
    # This is a form of local search on the point positions.
    
    # A more robust approach:
    # 1. Compute all areas.
    # 2. Find the smallest area triangle T_min = (i, j, k).
    # 3. Try to move point i, j, or k slightly to increase the area of T_min.
    # 4. This is essentially what gradient-based methods do, but we can do it
    #    more aggressively by looking at the 2nd, 3rd, etc. smallest areas.
    
    # Let's implement a simple "shake" heuristic:
    # Identify the critical triangles (those with area close to min).
    # Randomly perturb the points involved in these triangles.
    
    rng = np.random.RandomState(42)
    
    for step in range(500):
        if time.time() - start_time > time_budget:
            break
            
        dets = _all_dets_from_bary(current_bary)
        areas = np.abs(dets)
        min_area = np.min(areas)
        
        if min_area <= current_min_area:
            # No improvement, maybe revert or try a different perturbation
            # For now, just continue
            pass
        else:
            current_min_area = min_area
            # Update best
            # (We will return the best at the end)
            
        # Find the index of the minimum area triangle
        min_idx = np.argmin(areas)
        i, j, k = _IDX_TRIPLES[min_idx]
        
        # Try to move one of the points i, j, k
        # We can try moving it in a random direction or towards the centroid of the other points
        # Or, more simply, try a small random perturbation to one of the points.
        
        # Let's try a "swap" like heuristic:
        # If we move point i, how does it affect the areas?
        # This is complex. Let's stick to a simple random perturbation of the critical points.
        
        point_to_move = rng.choice([i, j, k])
        
        # Generate a small random perturbation
        delta_u = rng.uniform(-0.05, 0.05)
        delta_v = rng.uniform(-0.05, 0.05)
        
        new_bary = current_bary.copy()
        new_bary[point_to_move, 0] += delta_u
        new_bary[point_to_move, 1] += delta_v
        
        # Project to simplex
        new_bary = _project_to_bary_simplex(new_bary)
        
        new_min_area = _min_area_ratio_from_bary(new_bary)
        
        if new_min_area > current_min_area:
            current_bary = new_bary
            current_min_area = new_min_area
        else:
            # Revert
            pass
            
    return current_bary

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    def time_left():
        return deadline - time.time()
        
    best_x = None
    best_score = -float('inf')
    
    inits = []
    
    # 1. FPS Init
    inits.append(_structured_init_fps().flatten())
    
    # 2. Random Inits
    rng = np.random.RandomState(42)
    for _ in range(15):
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
        
    # 3. Magic Init
    magic_pts = [
        [1.0, 0.0], [0.0, 1.0], [0.0, 0.0],
        [1/3, 1/3],
        [0.5, 0.0], [0.0, 0.5], [0.5, 0.5],
        [0.25, 0.0], [0.0, 0.25], [0.25, 0.25], [0.5, 0.25],
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
    
    bounds = [(0, 1)] * (2 * N)
    
    # Optimization Loop
    for init in inits:
        if time.time() > deadline:
            break
            
        x_init = _project_to_bary_simplex(init.reshape(N, 2)).flatten()
        
        # 1. L-BFGS-B with smooth objective
        x_lbfgs = _local_optimize_lbfgs(x_init, bounds, maxiter=100)
        x_lbfgs = _project_to_bary_simplex(x_lbfgs.reshape(N, 2)).flatten()
        
        # 2. Nelder-Mead refinement
        x_nm = _local_optimize_nm(x_lbfgs, bounds, maxiter=200)
        x_nm = _project_to_bary_simplex(x_nm.reshape(N, 2)).flatten()
        
        # 3. SLSQP refinement
        x_slsqp = _local_optimize_slsqp(x_nm, bounds, maxiter=100)
        x_slsqp = _project_to_bary_simplex(x_slsqp.reshape(N, 2)).flatten()
        
        # 4. Combinatorial Refinement
        if time.time() < deadline - 10:
            x_comb = _combinatorial_refinement(x_slsqp, time_budget=2.0)
            x_comb = _project_to_bary_simplex(x_comb.reshape(N, 2)).flatten()
        else:
            x_comb = x_slsqp
            
        # Evaluate all candidates
        candidates = [x_lbfgs, x_nm, x_slsqp, x_comb]
        
        for x_cand in candidates:
            bary_cand = x_cand.reshape(N, 2)
            score = _min_area_ratio_from_bary(bary_cand)
            
            if score > best_score:
                best_score = score
                best_x = bary_cand.copy()
                
    if best_x is None:
        # Fallback
        best_x = np.zeros((N, 2))
        best_x[:] = [1/3, 1/3]
        # Add some spread
        for i in range(N):
            best_x[i, 0] += rng.uniform(-0.1, 0.1)
            best_x[i, 1] += rng.uniform(-0.1, 0.1)
        best_x = _project_to_bary_simplex(best_x)
        
    # Convert to Cartesian
    points = _bary_to_cart(best_x[:, 0], best_x[:, 1])
    
    return points
# EVOLVE-BLOCK-END
