"""Heilbronn convex regions: maximize min triangle area / convex hull area for 13 points."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

# Precompute triangle indices
TRI_INDICES = list(combinations(range(N), 3))
NUM_TRIS = len(TRI_INDICES)  # 286

def _all_tri_areas(pts):
    """Compute all 286 triangle areas for a point set."""
    pts = np.asarray(pts, dtype=np.float64)
    areas = np.empty(NUM_TRIS)
    for idx, (i, j, k) in enumerate(TRI_INDICES):
        v1 = pts[j] - pts[i]
        v2 = pts[k] - pts[i]
        areas[idx] = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
    return areas

def _convex_hull_area(pts):
    """Compute area of convex hull."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score(pts):
    """Score = min triangle area / convex hull area."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape != (N, 2):
        return 0.0
    if np.any(np.isnan(pts)) or np.any(np.isinf(pts)):
        return 0.0
    areas = _all_tri_areas(pts)
    min_area = np.min(areas)
    hull_area = _convex_hull_area(pts)
    if hull_area < 1e-12:
        return 0.0
    return min_area / hull_area

def _soft_min_ratio(pts, temperature):
    """
    Smooth surrogate: soft-min of triangle areas / hull area.
    Uses log-sum-exp for numerical stability.
    """
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape != (N, 2):
        return -1e9
    
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
        if hull_area < 1e-12:
            return -1e9
    except Exception:
        return -1e9
    
    areas = _all_tri_areas(pts)
    areas = np.maximum(areas, 1e-15)
    
    # soft_min = -T * log(sum(exp(-a/T)))
    scaled = -areas / temperature
    scaled_max = np.max(scaled)
    log_sum = np.log(np.sum(np.exp(scaled - scaled_max))) + scaled_max
    soft_min_area = -log_sum * temperature
    
    return soft_min_area / hull_area

def _objective(x, temperature):
    """Objective for L-BFGS-B: minimize negative soft-min ratio."""
    pts = x.reshape(N, 2)
    ratio = _soft_min_ratio(pts, temperature)
    if ratio < -1e8:
        return 1e8
    return -ratio

def _halton_sequence(n, base):
    """Generate n points from the Halton sequence in [0,1)."""
    result = np.zeros(n)
    for i in range(n):
        f = 1.0
        idx = i + 1
        while idx > 0:
            f /= base
            idx = (idx - 1) // base
        result[i] = f
    return result

def _generate_halton_points(n, bases=(2, 3)):
    """Generate n points in [0,1)^2 using Halton sequences."""
    return np.column_stack([_halton_sequence(n, b) for b in bases])

def _regular_polygon(n, radius=1.0, center=(0,0)):
    """Generate regular polygon vertices."""
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    return np.column_stack([center[0] + radius*np.cos(angles), 
                            center[1] + radius*np.sin(angles)])

def _concentric_rings(n, n_rings=3):
    """Generate points on concentric rings for diverse initialization."""
    pts = []
    # Center point
    pts.append([0.0, 0.0])
    remaining = n - 1
    for ring in range(n_rings):
        if remaining <= 0:
            break
        n_on_ring = min(remaining, max(3, (n-1)//n_rings))
        if ring == n_rings - 1:
            n_on_ring = remaining
        radius = (ring + 1) / n_rings
        angles = np.linspace(0, 2*np.pi, n_on_ring, endpoint=False) + ring * np.pi/6
        ring_pts = np.column_stack([radius*np.cos(angles), radius*np.sin(angles)])
        pts.append(ring_pts)
        remaining -= n_on_ring
    
    if remaining > 0:
        angles = np.linspace(0, 2*np.pi, remaining, endpoint=False)
        ring_pts = np.column_stack([1.0*np.cos(angles), 1.0*np.sin(angles)])
        pts.append(ring_pts)
    
    result = np.vstack(pts)
    return result[:n]

def _structured_grid(n):
    """Generate n points on a structured grid (sqrt-ish layout)."""
    pts = []
    # Center
    pts.append([0.5, 0.5])
    # Ring 1: 4 points
    for angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
        pts.append([0.5 + 0.35*np.cos(angle), 0.5 + 0.35*np.sin(angle)])
    # Ring 2: 4 points
    for angle in [np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4]:
        pts.append([0.5 + 0.45*np.cos(angle), 0.5 + 0.45*np.sin(angle)])
    # Ring 3: 4 points
    for angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
        pts.append([0.5 + 0.48*np.cos(angle + np.pi/8), 0.5 + 0.48*np.sin(angle + np.pi/8)])
    return np.array(pts)

def _optimize_from_start(initial_pts, time_budget, rng):
    """
    Multi-phase optimization from a single starting point:
    1. Coarse soft-min optimization with decreasing temperature
    2. Fine polish with very low temperature
    3. Nelder-Mead on exact objective
    4. Local perturbation search on the best found
    """
    start_time = time.time()
    x0 = initial_pts.flatten().copy()
    best_x = x0.copy()
    best_score = _score(initial_pts)
    
    # Phase 1: Soft-min optimization with annealing
    temperatures = [0.05, 0.02, 0.01, 0.005, 0.002]
    for temp in temperatures:
        if time.time() - start_time > time_budget - 3:
            break
        
        def obj(x):
            return _objective(x, temp)
        
        try:
            result = minimize(
                obj, x0, method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-9}
            )
            if result.fun < 1e7:
                cand = result.x
                cand_score = _score(cand.reshape(N, 2))
                if cand_score > best_score:
                    best_score = cand_score
                    best_x = cand.copy()
                x0 = cand
            else:
                x0 = result.x
        except Exception:
            pass
    
    # Phase 2: Nelder-Mead on exact objective
    if time.time() - start_time < time_budget - 2:
        def exact_obj(x):
            pts = x.reshape(N, 2)
            r = _score(pts)
            return -r
        
        try:
            result = minimize(
                exact_obj, best_x, method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10,
                         'initial_simplex': 1e-3}
            )
            if result.fun < 1e7:
                cand = result.x
                cand_score = _score(cand.reshape(N, 2))
                if cand_score > best_score:
                    best_score = cand_score
                    best_x = cand.copy()
        except Exception:
            pass
    
    # Phase 3: Local perturbation search on the best found
    if time.time() - start_time < time_budget - 1:
        best_pts = best_x.reshape(N, 2)
        for _ in range(50):
            if time.time() - start_time > time_budget - 0.5:
                break
            # Perturb the two points of the minimum triangle
            areas = _all_tri_areas(best_pts)
            min_idx = np.argmin(areas)
            i, j, k = TRI_INDICES[min_idx]
            
            # Try moving one of the three points slightly
            for pidx in [i, j, k]:
                for scale in [0.01, 0.005, 0.002]:
                    if time.time() - start_time > time_budget - 0.3:
                        break
                    trial = best_pts.copy()
                    trial[pidx] += rng.normal(0, scale, size=2)
                    trial_score = _score(trial)
                    if trial_score > best_score:
                        best_score = trial_score
                        best_x = trial.flatten().copy()
                        best_pts = trial
                        break
    
    return best_x.reshape(N, 2), best_score

def construct():
    """Main entry point: find optimal 13-point configuration."""
    start_time = time.time()
    total_budget = 145.0  # Safety margin for 150s limit
    
    rng = np.random.RandomState(12345)
    
    best_points = None
    best_ratio = -1e9
    
    # Generate diverse starting points
    starts = []
    
    # 1. Halton (2,3)
    starts.append(_generate_halton_points(N, (2, 3)))
    # 2. Halton (3,5)
    starts.append(_generate_halton_points(N, (3, 5)))
    # 3. Regular 13-gon
    starts.append(_regular_polygon(N, radius=1.0))
    # 4. Concentric rings (3 rings)
    starts.append(_concentric_rings(N, n_rings=3))
    # 5. Concentric rings (4 rings)
    starts.append(_concentric_rings(N, n_rings=4))
    # 6. Structured grid
    starts.append(_structured_grid(N))
    
    # 7-10: Random points in unit disk
    for _ in range(4):
        angles = rng.uniform(0, 2*np.pi, N)
        radii = rng.uniform(0, 1, N)
        pts = np.column_stack([radii*np.cos(angles), radii*np.sin(angles)])
        starts.append(pts)
    
    # Evaluate initial candidates
    for cand in starts:
        if time.time() - start_time > total_budget:
            break
        try:
            s = _score(cand)
            if s > best_ratio:
                best_ratio = s
                best_points = cand.copy()
        except Exception:
            pass
    
    # Optimize from each starting point, allocating time budget
    n_starts = len(starts)
    time_per_start = (total_budget - 5.0) / n_starts  # 5s safety margin
    
    for i, start in enumerate(starts):
        if time.time() - start_time > total_budget - 2:
            break
        
        remaining_time = total_budget - (time.time() - start_time)
        alloc_time = min(time_per_start, remaining_time - 1.0)
        if alloc_time < 2.0:
            continue
        
        try:
            opt_pts, opt_score = _optimize_from_start(start, alloc_time, rng)
            if opt_score > best_ratio:
                best_ratio = opt_score
                best_points = opt_pts.copy()
        except Exception:
            pass
    
    # Final refinement on the best found
    if best_points is not None and time.time() - start_time < total_budget - 3:
        final_time = total_budget - (time.time() - start_time) - 1.0
        if final_time > 5.0:
            try:
                final_pts, final_score = _optimize_from_start(best_points, final_time, rng)
                if final_score > best_ratio:
                    best_ratio = final_score
                    best_points = final_pts.copy()
            except Exception:
                pass
    
    if best_points is None:
        # Fallback: regular polygon
        best_points = _regular_polygon(N, radius=1.0)
    
    return best_points
# EVOLVE-BLOCK-END
