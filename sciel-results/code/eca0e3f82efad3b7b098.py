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

def _regular_polygon(n, radius=1.0, center=(0,0)):
    """Generate regular polygon vertices."""
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    return np.column_stack([center[0] + radius*np.cos(angles), 
                            center[1] + radius*np.sin(angles)])

def _concentric_rings(n, n_rings=3):
    """Generate points on concentric rings for diverse initialization."""
    pts = []
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

def _halton_sequence(n, base):
    """Generate n Halton sequence points in [0,1)."""
    result = []
    f = 1.0
    for i in range(1, n + 1):
        x = 0.0
        i_val = i
        while i_val > 0:
            i_val, digit = divmod(i_val, base)
            x += digit * f
            f /= base
        result.append(x)
    return np.array(result)

def _generate_halton_points(n, base1=2, base2=3):
    """Generate n points using 2D Halton sequence in unit square, mapped to disk."""
    x = _halton_sequence(n, base1)
    y = _halton_sequence(n, base2)
    r = np.sqrt(x)
    theta = 2 * np.pi * y
    pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    return pts

def _angular_to_cartesian(angles, radii):
    """Convert angles and radii to Cartesian coordinates."""
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

def _cartesian_to_angular(pts):
    """Convert Cartesian to angles and radii."""
    radii = np.linalg.norm(pts, axis=1)
    angles = np.arctan2(pts[:, 1], pts[:, 0])
    angles = angles % (2 * np.pi)
    return angles, radii

def _local_search_angular(angles, radii, time_budget, rng):
    """
    Coordinate-wise local search in angular parameterization.
    Perturb one angle or radius at a time with adaptive step sizes.
    """
    start_time = time.time()
    best_angles = angles.copy()
    best_radii = radii.copy()
    best_score = _score(_angular_to_cartesian(best_angles, best_radii))
    
    angle_step = 0.05
    radius_step = 0.05
    no_improve_count = 0
    
    while time.time() - start_time < time_budget - 1:
        improved = False
        for i in range(N):
            # Try perturbing angle
            for delta in [-angle_step, angle_step]:
                trial_angles = best_angles.copy()
                trial_angles[i] += delta
                trial_score = _score(_angular_to_cartesian(trial_angles, best_radii))
                if trial_score > best_score:
                    best_angles = trial_angles
                    best_score = trial_score
                    improved = True
                    no_improve_count = 0
            
            # Try perturbing radius
            for delta in [-radius_step, radius_step]:
                trial_radii = best_radii.copy()
                trial_radii[i] = np.maximum(0.01, trial_radii[i] + delta)
                trial_score = _score(_angular_to_cartesian(best_angles, trial_radii))
                if trial_score > best_score:
                    best_radii = trial_radii
                    best_score = trial_score
                    improved = True
                    no_improve_count = 0
        
        if not improved:
            no_improve_count += 1
            if no_improve_count >= 3:
                angle_step *= 0.5
                radius_step *= 0.5
                no_improve_count = 0
            if angle_step < 1e-6:
                break
    
    return best_angles, best_radii, best_score

def _optimize_from_start(initial_pts, time_budget, rng):
    """
    Optimize from a starting point using multi-stage approach:
    1. Convert to angular parameterization
    2. L-BFGS with smooth surrogate in angular space (temperature annealing)
    3. Local search in angular space
    4. Nelder-Mead on exact objective in Cartesian space
    """
    start_time = time.time()
    best_pts = initial_pts.copy()
    best_score = _score(initial_pts)
    
    angles, radii = _cartesian_to_angular(best_pts)
    x0 = np.concatenate([angles, radii])
    
    # Stage 1: L-BFGS with decreasing temperatures in angular space
    temperatures = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    
    for temp in temperatures:
        if time.time() - start_time > time_budget - 3:
            break
        
        def obj(x):
            a = x[:N]
            r = np.maximum(x[N:], 1e-6)
            pts = _angular_to_cartesian(a, r)
            ratio = _soft_min_ratio(pts, temp)
            if ratio < -1e8:
                return 1e8
            return -ratio
        
        try:
            result = minimize(
                obj, x0, method='L-BFGS-B',
                bounds=[(None, None)] * N + [(1e-6, None)] * N,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 40}
            )
            if result.fun < 1e7:
                cand = result.x
                cand_pts = _angular_to_cartesian(cand[:N], np.maximum(cand[N:], 1e-6))
                cand_score = _score(cand_pts)
                if cand_score > best_score:
                    best_score = cand_score
                    best_pts = cand_pts.copy()
                x0 = cand
        except Exception:
            pass
    
    # Stage 2: Local search in angular space
    if time.time() - start_time < time_budget - 5:
        remaining = time_budget - (time.time() - start_time) - 5
        if remaining > 2:
            try:
                angles, radii = _cartesian_to_angular(best_pts)
                new_angles, new_radii, new_score = _local_search_angular(
                    angles, radii, remaining, rng
                )
                if new_score > best_score:
                    best_score = new_score
                    best_pts = _angular_to_cartesian(new_angles, new_radii).copy()
            except Exception:
                pass
    
    # Stage 3: Nelder-Mead on exact objective in Cartesian space
    if time.time() - start_time < time_budget - 1:
        def exact_obj(x):
            pts = x.reshape(N, 2)
            r = _score(pts)
            return -r
        
        try:
            result = minimize(
                exact_obj, best_pts.flatten(), method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12,
                         'initial_simplex': 1e-3}
            )
            if result.fun < 1e7:
                cand = result.x
                cand_score = _score(cand.reshape(N, 2))
                if cand_score > best_score:
                    best_score = cand_score
                    best_pts = cand.reshape(N, 2).copy()
        except Exception:
            pass
    
    return best_pts, best_score

def construct():
    """Main entry point: find optimal 13-point configuration."""
    start_time = time.time()
    total_budget = 145.0  # Safety margin for 150s limit
    
    rng = np.random.RandomState(12345)
    
    best_points = None
    best_ratio = -1e9
    
    # Generate diverse starting points
    starts = []
    
    # 1. Regular 13-gon
    starts.append(_regular_polygon(N, radius=1.0))
    # 2. Concentric rings (3 rings)
    starts.append(_concentric_rings(N, n_rings=3))
    # 3. Concentric rings (4 rings)
    starts.append(_concentric_rings(N, n_rings=4))
    
    # 4. Halton sequence points
    starts.append(_generate_halton_points(N, base1=2, base2=3))
    starts.append(_generate_halton_points(N, base1=3, base2=5))
    
    # 5-7: Random points in unit disk with good spread
    for seed in [12345, 67890, 11111]:
        r_rng = np.random.RandomState(seed)
        angles = r_rng.uniform(0, 2*np.pi, N)
        angles.sort()
        radii = r_rng.uniform(0.3, 1.0, N)
        pts = np.column_stack([radii*np.cos(angles), radii*np.sin(angles)])
        starts.append(pts)
    
    # 8: Jittered regular polygon
    base_poly = _regular_polygon(N, radius=1.0)
    jitter = rng.normal(0, 0.05, (N, 2))
    starts.append(base_poly + jitter)
    
    # 9: Two-ring structure (7 inner, 6 outer)
    inner_angles = np.linspace(0, 2*np.pi, 7, endpoint=False)
    outer_angles = np.linspace(np.pi/12, 2*np.pi, 6, endpoint=False)
    inner_pts = np.column_stack([0.5*np.cos(inner_angles), 0.5*np.sin(inner_angles)])
    outer_pts = np.column_stack([1.0*np.cos(outer_angles), 1.0*np.sin(outer_angles)])
    starts.append(np.vstack([inner_pts, outer_pts]))
    
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
    
    # Optimize from each starting point
    n_starts = len(starts)
    time_per_start = (total_budget - 5.0) / n_starts
    
    for i, start in enumerate(starts):
        if time.time() - start_time > total_budget - 2:
            break
        
        remaining_time = total_budget - (time.time() - start_time)
        alloc_time = min(time_per_start, remaining_time - 1.0)
        if alloc_time < 3.0:
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
        if final_time > 10.0:
            try:
                final_pts, final_score = _optimize_from_start(best_points, final_time, rng)
                if final_score > best_ratio:
                    best_ratio = final_score
                    best_points = final_pts.copy()
            except Exception:
                pass
    
    if best_points is None:
        best_points = _regular_polygon(N, radius=1.0)
    
    return best_points
# EVOLVE-BLOCK-END
