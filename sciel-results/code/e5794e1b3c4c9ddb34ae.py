"""Heilbronn triangles: maximize minimum triangle area for 11 points in unit equilateral triangle."""
import numpy as np
import time
from itertools import combinations
from scipy.optimize import minimize, differential_evolution

N = 11

# The equilateral triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2

# Precompute triangle indices for vectorized area computation
_TRI_IDX = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])
_N_TRI = len(_TRI_IDX)

def _all_areas(pts):
    """Vectorized computation of all triangle areas."""
    # pts shape (N, 2)
    p1 = pts[_TRI_IDX[:, 0]]
    p2 = pts[_TRI_IDX[:, 1]]
    p3 = pts[_TRI_IDX[:, 2]]
    
    # Area = 0.5 * | (x2-x1)(y3-y1) - (x3-x1)(y2-y1) |
    x1, y1 = p1[:, 0], p1[:, 1]
    x2, y2 = p2[:, 0], p2[:, 1]
    x3, y3 = p3[:, 0], p3[:, 1]
    
    areas = 0.5 * np.abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
    return areas

def _min_area(pts):
    """Compute the minimum area of any triangle formed by three of the points."""
    areas = _all_areas(pts)
    return float(np.min(areas))

def _soft_min_area(pts, beta=50.0):
    """Smooth approximation of min area using log-sum-exp."""
    areas = _all_areas(pts)
    m = np.min(areas)
    shifted = -beta * (areas - m)
    # Clip for numerical stability
    shifted = np.clip(shifted, -500, 0)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min_val = m - log_sum / beta
    return soft_min_val

def _clamp_to_triangle(p):
    """Clamp a point to be inside the equilateral triangle using barycentric coordinates."""
    x, y = p
    # Barycentric coordinates: point = a*V0 + b*V1 + c*V2, a+b+c=1
    # x = b + 0.5*c, y = (sqrt(3)/2)*c
    c = 2 * y / np.sqrt(3)
    b = x - 0.5 * c
    a = 1 - b - c
    
    eps = 1e-12
    a = max(eps, a)
    b = max(eps, b)
    c = max(eps, c)
    
    s = a + b + c
    a /= s
    b /= s
    c /= s
    
    return a * V0 + b * V1 + c * V2

def _clamp_all(pts):
    """Clamp all points to triangle."""
    result = np.empty_like(pts)
    for i in range(len(pts)):
        result[i] = _clamp_to_triangle(pts[i])
    return result

def _objective_smooth(flat_pts, beta=50.0):
    """Objective to minimize: negative soft-min area (smooth)."""
    pts = flat_pts.reshape(-1, 2)
    pts = _clamp_all(pts)
    return -_soft_min_area(pts, beta)

def _objective_hard(flat_pts):
    """Objective to minimize: negative min area (non-smooth, for final polish)."""
    pts = flat_pts.reshape(-1, 2)
    pts = _clamp_all(pts)
    return -_min_area(pts)

def _initial_placement_structured():
    """Structured initial placement: triangular lattice pattern."""
    pts = []
    # Row 0 (bottom): 4 points
    y0 = 0.05
    for i in range(4):
        x = 0.05 + i * (0.9 / 3)
        pts.append([x, y0])
    # Row 1 (middle): 4 points
    y1 = 0.35
    for i in range(4):
        x = 0.15 + i * (0.7 / 3)
        pts.append([x, y1])
    # Row 2 (top): 3 points
    y2 = 0.65
    for i in range(3):
        x = 0.25 + i * (0.5 / 2)
        pts.append([x, y2])
    return np.array(pts)

def _initial_placement_symmetric():
    """Symmetric initial placement using barycentric coordinates."""
    pts = []
    # Near vertices (pulled inward)
    margin = 0.08
    pts.append(margin * V0 + (1-margin)/2 * V1 + (1-margin)/2 * V2)
    pts.append((1-margin)/2 * V0 + margin * V1 + (1-margin)/2 * V2)
    pts.append((1-margin)/2 * V0 + (1-margin)/2 * V1 + margin * V2)
    # Edge midpoints (pulled inward)
    m2 = 0.05
    pts.append(0.5*(1-m2)*V0 + 0.5*m2*V1 + 0.5*(1-m2)*V2)
    pts.append(m2*V0 + 0.5*(1-m2)*V1 + 0.5*(1-m2)*V2)
    pts.append(0.5*(1-m2)*V0 + 0.5*(1-m2)*V1 + m2*V2)
    # Center
    pts.append((V0 + V1 + V2) / 3)
    # 4 more interior points (symmetric around center)
    cx, cy = 1/3, H/3
    r = 0.12
    for angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
        px = cx + r * np.cos(angle)
        py = cy + r * np.sin(angle)
        pts.append([px, py])
    return np.array(pts)

def _initial_placement_random(seed):
    """Random initial placement inside triangle."""
    rng = np.random.default_rng(seed)
    pts = []
    for _ in range(N):
        r1 = rng.random()
        r2 = rng.random()
        if r1 + r2 > 1:
            r1 = 1 - r1
            r2 = 1 - r2
        p = r1 * V1 + r2 * V2
        pts.append(p)
    return np.array(pts)

def _initial_placement_tammes():
    """Tammes-like initial placement based on known good configurations."""
    # Based on the idea that near-optimal configurations have points
    # arranged in a pattern that avoids small triangles
    pts = []
    # Use a hexagonal-like pattern adapted to the triangle
    # 3 points near vertices
    d = 0.15
    pts.append([d, d * np.sqrt(3)])
    pts.append([1-d, d * np.sqrt(3)])
    pts.append([0.5, H - d * np.sqrt(3)])
    # 3 points near edge midpoints
    d2 = 0.1
    pts.append([0.5 - d2, d2 * np.sqrt(3) + 0.02])
    pts.append([0.5 + d2, d2 * np.sqrt(3) + 0.02])
    pts.append([0.5, H - d2 * np.sqrt(3) - 0.02])
    # 5 interior points in a pentagon-like arrangement
    cx, cy = 0.5, H/3
    r = 0.18
    for i in range(5):
        angle = 2 * np.pi * i / 5 - np.pi/2
        px = cx + r * np.cos(angle)
        py = cy + r * np.sin(angle)
        pts.append([px, py])
    return np.array(pts)

def _optimize_from_start(start_pts, deadline, beta=50.0):
    """Run L-BFGS-B optimization from a starting point."""
    start_time = time.time()
    
    def time_left():
        return deadline - (time.time() - start_time)
    
    flat_init = start_pts.flatten()
    
    try:
        res = minimize(
            lambda x, b=beta: _objective_smooth(x, b),
            flat_init,
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        if time_left() > 1.0:
            opt_pts = _clamp_all(res.x.reshape(-1, 2))
            opt_score = _min_area(opt_pts)
            return opt_pts, opt_score
    except Exception:
        pass
    
    return start_pts, _min_area(start_pts)

def _nelder_mead_polish(pts, deadline):
    """Polish with Nelder-Mead on the hard objective."""
    start_time = time.time()
    
    def time_left():
        return deadline - (time.time() - start_time)
    
    if time_left() < 2.0:
        return pts, _min_area(pts)
    
    try:
        res = minimize(
            _objective_hard,
            pts.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        nm_pts = _clamp_all(res.x.reshape(-1, 2))
        nm_score = _min_area(nm_pts)
        return nm_pts, nm_score
    except Exception:
        return pts, _min_area(pts)

def _differential_evolution_search(deadline):
    """Use differential evolution for global search."""
    start_time = time.time()
    
    def time_left():
        return deadline - (time.time() - start_time)
    
    if time_left() < 10.0:
        return None, -1.0
    
    # Bounds for each coordinate
    # x: [0, 1], y: [0, H] but constrained by triangle
    # We'll use the full bounding box and rely on clamping
    bounds = [(0, 1), (0, H)] * N
    
    def de_objective(x):
        pts = x.reshape(-1, 2)
        pts = _clamp_all(pts)
        return -_min_area(pts)
    
    try:
        result = differential_evolution(
            de_objective,
            bounds,
            maxiter=100,
            popsize=15,
            tol=1e-8,
            mutation=(0.3, 1.0),
            recombination=0.7,
            seed=42,
            polish=False,
            callback=None
        )
        de_pts = _clamp_all(result.x.reshape(-1, 2))
        de_score = _min_area(de_pts)
        return de_pts, de_score
    except Exception:
        return None, -1.0

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin for 150s limit
    
    def time_left():
        return deadline - (time.time() - start_time)
    
    # Generate multiple diverse initializations
    init_candidates = []
    init_candidates.append(_initial_placement_structured())
    init_candidates.append(_initial_placement_symmetric())
    init_candidates.append(_initial_placement_tammes())
    for seed in [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]:
        init_candidates.append(_initial_placement_random(seed))
    
    best_pts = None
    best_score = -1.0
    
    # Phase 1: Smooth optimization from each initialization
    for init in init_candidates:
        if time_left() < 5.0:
            break
        opt_pts, opt_score = _optimize_from_start(init, deadline, beta=50.0)
        if opt_score > best_score:
            best_pts = opt_pts.copy()
            best_score = opt_score
    
    if best_pts is None:
        best_pts = _initial_placement_structured()
        best_score = _min_area(best_pts)
    
    # Phase 2: Differential evolution for global search
    if time_left() > 15.0:
        de_pts, de_score = _differential_evolution_search(deadline)
        if de_pts is not None and de_score > best_score:
            best_pts = de_pts.copy()
            best_score = de_score
    
    # Phase 3: Multi-start perturbation + smooth optimization
    rng = np.random.default_rng(42)
    n_perturb = 10
    for trial in range(n_perturb):
        if time_left() < 5.0:
            break
        scale = 0.03 + 0.015 * trial
        trial_pts = best_pts + rng.normal(0, scale, size=(N, 2))
        trial_pts = _clamp_all(trial_pts)
        
        for beta in [40.0, 70.0]:
            if time_left() < 4.0:
                break
            opt_pts, opt_score = _optimize_from_start(trial_pts, deadline, beta=beta)
            if opt_score > best_score:
                best_pts = opt_pts.copy()
                best_score = opt_score
    
    # Phase 4: Simulated annealing for exploration
    if time_left() > 12.0:
        current_pts = best_pts.copy()
        current_score = best_score
        T = 0.012
        beta_sa = 40.0
        n_iter = 2000
        
        for iteration in range(n_iter):
            if time_left() < 4.0:
                break
            
            idx = rng.integers(0, N)
            delta = rng.normal(0, T, size=2)
            new_pts = current_pts.copy()
            new_pts[idx] = _clamp_to_triangle(current_pts[idx] + delta)
            
            new_soft = _soft_min_area(new_pts, beta_sa)
            cur_soft = _soft_min_area(current_pts, beta_sa)
            
            if new_soft > cur_soft:
                accept = True
            else:
                diff = new_soft - cur_soft
                accept = rng.random() < np.exp(diff / max(T * 0.1, 1e-10))
            
            if accept:
                current_pts = new_pts
                hard_score = _min_area(current_pts)
                if hard_score > best_score:
                    best_pts = current_pts.copy()
                    best_score = hard_score
            
            T *= 0.999
            if T < 0.0005:
                T = 0.0005
    
    # Phase 5: Nelder-Mead local search
    if time_left() > 6.0:
        nm_pts, nm_score = _nelder_mead_polish(best_pts, deadline)
        if nm_score > best_score:
            best_pts = nm_pts.copy()
            best_score = nm_score
    
    # Phase 6: Final high-precision polish
    if time_left() > 3.0:
        try:
            res_final = minimize(
                _objective_hard,
                best_pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-14}
            )
            final_pts = _clamp_all(res_final.x.reshape(-1, 2))
            final_score = _min_area(final_pts)
            if final_score > best_score:
                best_pts = final_pts.copy()
                best_score = final_score
        except Exception:
            pass
    
    # Phase 7: One more Nelder-Mead pass
    if time_left() > 2.0:
        nm2_pts, nm2_score = _nelder_mead_polish(best_pts, deadline)
        if nm2_score > best_score:
            best_pts = nm2_pts.copy()
            best_score = nm2_score
    
    # Final clamp and return
    best_pts = _clamp_all(best_pts)
    return best_pts
