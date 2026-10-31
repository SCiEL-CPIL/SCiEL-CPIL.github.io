"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

# The equilateral triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2

# Precompute triangle indices for vectorized area computation
_TRI_IDX = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])
_N_TRI = len(_TRI_IDX)

def _all_areas(pts):
    """Fully vectorized computation of all triangle areas using broadcasting."""
    pi = pts[_TRI_IDX[:, 0]]
    pj = pts[_TRI_IDX[:, 1]]
    pk = pts[_TRI_IDX[:, 2]]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    areas = 0.5 * np.abs(dx1 * dy2 - dy1 * dx2)
    return areas

def _min_area(pts):
    """Compute the minimum area of any triangle formed by three of the points."""
    return float(np.min(_all_areas(pts)))

def _clamp_to_triangle(p):
    """Clamp a point to be inside the equilateral triangle."""
    x, y = p
    # Barycentric coordinates
    # p = a*V0 + b*V1 + c*V2, a+b+c=1
    # x = b + 0.5c
    # y = c * H
    c = 2 * y / np.sqrt(3)
    b = x - 0.5 * c
    a = 1 - b - c
    
    eps = 1e-12
    a = max(eps, min(1 - 2*eps, a))
    b = max(eps, min(1 - 2*eps, b))
    c = max(eps, min(1 - 2*eps, c))
    
    s = a + b + c
    a /= s
    b /= s
    c /= s
    
    return a * V0 + b * V1 + c * V2

def _clamp_all(pts):
    """Clamp all points to triangle."""
    return np.array([_clamp_to_triangle(p) for p in pts])

def _soft_min_area(pts, beta=50.0):
    """Smooth approximation of min area."""
    areas = _all_areas(pts)
    m = np.min(areas)
    if m <= 1e-15:
        return m
    shifted = -beta * (areas - m)
    # Use log-sum-exp trick for stability
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min_val = m - log_sum / beta
    return soft_min_val

def _objective_de(flat_pts):
    """
    Objective for Differential Evolution.
    Minimizes negative soft-min area + penalty for points outside triangle.
    """
    pts = flat_pts.reshape(-1, 2)
    
    # Penalty for points outside the triangle
    c = 2 * pts[:, 1] / np.sqrt(3)
    b = pts[:, 0] - 0.5 * c
    a = 1 - b - c
    
    pen_a = np.maximum(0, -a)
    pen_b = np.maximum(0, -b)
    pen_c = np.maximum(0, -c)
    penalty = np.sum(pen_a**2 + pen_b**2 + pen_c**2)
    
    pts_clamped = _clamp_all(pts)
    
    # Use a softer min for DE to allow exploration
    area_val = _soft_min_area(pts_clamped, beta=15.0)
    
    # Large penalty weight to keep points inside
    return -area_val + 1000.0 * penalty

def _polish_candidate(pts, time_budget=10.0):
    """
    Apply a sequence of local optimizers to a candidate point set.
    Returns the best point set found.
    """
    start_time = time.time()
    def time_left():
        return time_budget - (time.time() - start_time)
    
    bounds = [(0.0, 1.0)] * N + [(0.0, H)] * N
    current_best = _min_area(pts)
    best_pts = pts.copy()
    
    # Phase 1: L-BFGS-B on soft objective (high beta for precision)
    if time_left() > 2.0:
        try:
            flat_pts = best_pts.flatten()
            res_lbfgs = minimize(
                lambda x: -_soft_min_area(_clamp_all(x.reshape(-1, 2)), beta=100.0),
                flat_pts,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            polished_pts = _clamp_all(res_lbfgs.x.reshape(-1, 2))
            val = _min_area(polished_pts)
            if val > current_best:
                best_pts = polished_pts
                current_best = val
        except Exception:
            pass

    # Phase 2: Nelder-Mead on hard objective
    if time_left() > 2.0:
        try:
            res_nm = minimize(
                lambda x: -_min_area(_clamp_all(x.reshape(-1, 2))),
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            final_pts = _clamp_all(res_nm.x.reshape(-1, 2))
            val = _min_area(final_pts)
            if val > current_best:
                best_pts = final_pts
                current_best = val
        except Exception:
            pass

    # Phase 3: Powell on hard objective
    if time_left() > 2.0:
        try:
            res_powell = minimize(
                lambda x: -_min_area(_clamp_all(x.reshape(-1, 2))),
                best_pts.flatten(),
                method='Powell',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-14, 'xtol': 1e-12}
            )
            final_pts = _clamp_all(res_powell.x.reshape(-1, 2))
            val = _min_area(final_pts)
            if val > current_best:
                best_pts = final_pts
                current_best = val
        except Exception:
            pass
            
    # Phase 4: Another NM pass with tighter tolerance
    if time_left() > 1.0:
        try:
            res_nm2 = minimize(
                lambda x: -_min_area(_clamp_all(x.reshape(-1, 2))),
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
            )
            final_pts = _clamp_all(res_nm2.x.reshape(-1, 2))
            val = _min_area(final_pts)
            if val > current_best:
                best_pts = final_pts
                current_best = val
        except Exception:
            pass

    return best_pts

def _initial_placement_symmetric():
    """Symmetric initial placement."""
    pts = []
    # Center
    pts.append([1/3, H/3])
    # 3 points near vertices
    m = 0.15
    pts.append(m * V0 + (1-m)/2 * V1 + (1-m)/2 * V2)
    pts.append((1-m)/2 * V0 + m * V1 + (1-m)/2 * V2)
    pts.append((1-m)/2 * V0 + (1-m)/2 * V1 + m * V2)
    # 3 points near edge midpoints
    m2 = 0.2
    pts.append(0.5*(1-m2)*V0 + 0.5*m2*V1 + 0.5*(1-m2)*V2)
    pts.append(m2*V0 + 0.5*(1-m2)*V1 + 0.5*(1-m2)*V2)
    pts.append(0.5*(1-m2)*V0 + 0.5*(1-m2)*V1 + m2*V2)
    # 3 more points in a ring
    cx, cy = 1/3, H/3
    r = 0.25
    for angle in [np.pi/6, 5*np.pi/6, 7*np.pi/6]:
        px = cx + r * np.cos(angle)
        py = cy + r * np.sin(angle)
        pts.append([px, py])
    return np.array(pts)

def _generate_structured_candidates():
    """
    Generate a few diverse, structured initial candidates.
    This leverages the 'cluster' idea by using known symmetric patterns.
    """
    candidates = []
    
    # 1. The standard symmetric one
    candidates.append(_initial_placement_symmetric())
    
    # 2. A triangular grid pattern (approximate)
    # 4 points in a diamond, 3 on edges, 3 near vertices, 1 center
    # This is just a variation to test different local basins
    pts = []
    pts.append([1/3, H/3]) # Center
    # Ring 1 (close to center)
    r1 = 0.2
    for k in range(3):
        angle = 2 * np.pi * k / 3 + np.pi/6
        px = 1/3 + r1 * np.cos(angle)
        py = H/3 + r1 * np.sin(angle)
        pts.append([px, py])
    # Ring 2 (outer)
    r2 = 0.4
    for k in range(6):
        angle = 2 * np.pi * k / 6
        px = 1/3 + r2 * np.cos(angle)
        py = H/3 + r2 * np.sin(angle)
        # Clamp to triangle roughly
        p = _clamp_to_triangle([px, py])
        pts.append(list(p))
    # Ensure we have 11 points, pad if necessary or truncate
    if len(pts) > 11:
        pts = pts[:11]
    while len(pts) < 11:
        pts.append([1/3, H/3]) # Pad with center (will be optimized)
    candidates.append(np.array(pts))
    
    return candidates

def construct():
    start_time = time.time()
    deadline = 118.0  # Safety margin for 120s limit
    
    def time_left():
        return deadline - (time.time() - start_time)
    
    # Bounds for differential evolution
    bounds = [(0.0, 1.0)] * N + [(0.0, H)] * N
    
    # 1. Generate structured candidates
    structured_candidates = _generate_structured_candidates()
    best_pts = structured_candidates[0]
    current_best_score = _min_area(best_pts)
    
    # 2. Run Differential Evolution for global exploration
    # Allocate ~60% of the time to DE
    de_time_budget = 60.0
    if time_left() > de_time_budget:
        try:
            res_de = differential_evolution(
                _objective_de,
                bounds,
                seed=42,
                maxiter=200,
                tol=1e-8,
                popsize=20,
                polish=False,
                updating='immediate'
            )
            if res_de.success or res_de.fun < 1e6:
                de_pts = _clamp_all(res_de.x.reshape(-1, 2))
                de_score = _min_area(de_pts)
                if de_score > current_best_score:
                    best_pts = de_pts
                    current_best_score = de_score
                # Add DE result to candidates
                structured_candidates.append(de_pts)
        except Exception:
            pass

    # 3. Polish all candidates with the remaining time
    # We have ~50 seconds left.
    # We will polish the best candidate first, then others if time permits.
    
    # Sort candidates by current score
    scored_candidates = []
    for c in structured_candidates:
        s = _min_area(c)
        scored_candidates.append((s, c))
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Polish the top 3 candidates
    for score, cand in scored_candidates[:3]:
        if time_left() < 10.0:
            break
            
        # Allocate time for polish based on remaining time
        # If this is the first candidate, give it more time
        polish_budget = min(40.0, time_left() - 5.0)
        
        polished_pts = _polish_candidate(cand, time_budget=polish_budget)
        polished_score = _min_area(polished_pts)
        
        if polished_score > current_best_score:
            best_pts = polished_pts
            current_best_score = polished_score

    return _clamp_all(best_pts)

# EVOLVE-BLOCK-END
