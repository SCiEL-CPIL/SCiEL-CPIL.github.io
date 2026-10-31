import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

N = 13

# Pre-compute triangle indices for N=13
_TRI_INDICES = np.array(list(combinations(range(N), 3)))
_I_IDX = _TRI_INDICES[:, 0]
_J_IDX = _TRI_INDICES[:, 1]
_K_IDX = _TRI_INDICES[:, 2]
_NUM_TRI = len(_TRI_INDICES)

def _get_triangle_areas(points):
    """
    Vectorized calculation of areas for all triples.
    points: (N, 2) array
    returns: array of shape (num_triangles,)
    """
    pi = points[_I_IDX]
    pj = points[_J_IDX]
    pk = points[_K_IDX]
    
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return areas

def _hull_area(points):
    """Calculate convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Calculate the raw score: min_triangle_area / hull_area."""
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    areas = _get_triangle_areas(points)
    if len(areas) == 0:
        return 0.0
    min_area = np.min(areas)
    return min_area / hull_area

def _softmin_score(points, alpha):
    """
    Smooth approximation of the score using softmin of triangle areas.
    """
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    
    areas = _get_triangle_areas(points)
    if len(areas) == 0:
        return 0.0
    
    # Softmin: -1/alpha * log(sum(exp(-alpha * areas)))
    a_min = np.min(areas)
    exp_terms = np.exp(-alpha * (areas - a_min))
    log_sum = np.log(np.sum(exp_terms))
    softmin_val = a_min - np.log(np.sum(exp_terms)) / alpha
    
    return softmin_val / hull_area

def _neg_softmin_score(params, alpha):
    """Objective for softmin-based optimization."""
    points = params.reshape(N, 2)
    s = _softmin_score(points, alpha)
    return -s

def _neg_score(params):
    """Objective function for minimization (maximizing score)."""
    points = params.reshape(N, 2)
    s = _score(points)
    return -s

def _coordinate_local_search(points, deadline, rng, initial_step=0.05, min_step=1e-8, max_iter=300):
    """
    Coordinate-wise pattern search on the hard score.
    Tries moving each coordinate in +/- direction, accepting improvements.
    Adapts step size based on success/failure.
    """
    best_pts = points.copy()
    best_score = _score(best_pts)
    step = initial_step
    n_iter = 0
    stall_count = 0
    
    while time.time() < deadline - 5 and step > min_step and n_iter < max_iter:
        improved = False
        for i in range(N):
            if time.time() > deadline - 5:
                break
            for dim in range(2):
                if time.time() > deadline - 5:
                    break
                for direction in [1, -1]:
                    trial = best_pts.copy()
                    trial[i, dim] += direction * step
                    trial_score = _score(trial)
                    if trial_score > best_score:
                        best_pts = trial
                        best_score = trial_score
                        improved = True
                        stall_count = 0
                        break
                if improved:
                    break
            if improved:
                break
        
        n_iter += 1
        if not improved:
            stall_count += 1
            step *= 0.5
            if stall_count > 10:
                break
        else:
            stall_count = 0
            step = min(step * 1.3, initial_step)
    
    return best_pts, best_score

def _make_ring_config(n_points, radius, phase_offset=0.0, n_rings=1, ring_radii=None):
    """Generate a ring-based initial configuration."""
    if ring_radii is None:
        if n_rings == 1:
            ring_radii = [radius]
        else:
            ring_radii = np.linspace(0.3, radius, n_rings).tolist()
    
    points = []
    per_ring = n_points // n_rings
    remainder = n_points % n_rings
    
    for ring_idx, r in enumerate(ring_radii):
        count = per_ring + (1 if ring_idx < remainder else 0)
        for i in range(count):
            angle = 2 * np.pi * i / count + phase_offset + ring_idx * 0.3
            points.append([r * np.cos(angle), r * np.sin(angle)])
    
    # Fill remaining points if needed
    while len(points) < n_points:
        idx = len(points)
        angle = 2 * np.pi * idx / n_points + phase_offset
        points.append([radius * 0.5 * np.cos(angle), radius * 0.5 * np.sin(angle)])
    
    return np.array(points[:n_points])

def _optimize_from_start(start_pts, deadline, rng):
    """
    Multi-stage optimization from a starting point:
    1. Softmin-based Nelder-Mead with decreasing alpha (annealing)
    2. L-BFGS-B polish with softmin
    3. Coordinate-wise local search on hard score
    4. Ultra-fine Nelder-Mead polish on hard score
    Returns best points and score found.
    """
    best_pts = start_pts.copy()
    best_score = _score(start_pts)
    
    # Stage 1: Softmin optimization with annealing alpha
    alphas = [200, 100, 50, 25, 10, 5, 2, 1]
    for alpha in alphas:
        if time.time() > deadline - 20:
            break
        try:
            res = minimize(
                _neg_softmin_score,
                best_pts.flatten(),
                args=(alpha,),
                method='Nelder-Mead',
                options={'maxiter': 800, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
            )
            cand_pts = res.x.reshape(N, 2)
            score = _score(cand_pts)
            if score > best_score:
                best_score = score
                best_pts = cand_pts
        except Exception:
            pass
    
    # Stage 2: L-BFGS-B polish with softmin
    for alpha_polish in [3.0, 1.5, 0.5]:
        if time.time() < deadline - 15:
            try:
                res = minimize(
                    _neg_softmin_score,
                    best_pts.flatten(),
                    args=(alpha_polish,),
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                cand_pts = res.x.reshape(N, 2)
                score = _score(cand_pts)
                if score > best_score:
                    best_score = score
                    best_pts = cand_pts
            except Exception:
                pass
    
    # Stage 3: Coordinate-wise local search on hard score
    if time.time() < deadline - 12:
        try:
            cand_pts, score = _coordinate_local_search(
                best_pts, deadline, rng, initial_step=0.05, min_step=1e-8, max_iter=200
            )
            if score > best_score:
                best_score = score
                best_pts = cand_pts
        except Exception:
            pass
    
    # Stage 4: Ultra-fine Nelder-Mead polish on hard score
    if time.time() < deadline - 5:
        try:
            res = minimize(
                _neg_score,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            cand_pts = res.x.reshape(N, 2)
            score = _score(cand_pts)
            if score > best_score:
                best_score = score
                best_pts = cand_pts
        except Exception:
            pass
    
    return best_pts, best_score

def _generate_diverse_configs(rng):
    """Generate a diverse set of initial configurations."""
    configs = []
    
    # Regular polygon
    angles = 2 * np.pi * np.arange(N) / N
    configs.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # Slightly perturbed regular polygons
    for perturb in [0.01, 0.03, 0.05, 0.08, 0.12]:
        angles_p = 2 * np.pi * np.arange(N) / N + rng.uniform(-perturb, perturb, N)
        radii_p = 1.0 + rng.uniform(-perturb, perturb, N)
        configs.append(np.column_stack([radii_p * np.cos(angles_p), radii_p * np.sin(angles_p)]))
    
    # Ring configurations with various parameters
    for radius in [0.9, 1.0, 1.1, 1.2]:
        for phase in [0.0, np.pi / N, np.pi / (2 * N), np.pi / (3 * N)]:
            configs.append(_make_ring_config(N, radius, phase_offset=phase))
    
    # Two-ring structures
    for r1, r2 in [(0.4, 1.0), (0.5, 1.0), (0.5, 1.1), (0.6, 0.9), (0.3, 1.2)]:
        for phase in [0.0, np.pi / N, np.pi / (2 * N)]:
            configs.append(_make_ring_config(N, 1.0, phase_offset=phase, n_rings=2, ring_radii=[r1, r2]))
    
    # Three-ring structures
    for r1, r2, r3 in [(0.3, 0.6, 1.0), (0.4, 0.7, 1.0), (0.2, 0.5, 0.9)]:
        configs.append(_make_ring_config(N, 1.0, phase_offset=0.0, n_rings=3, ring_radii=[r1, r2, r3]))
    
    # Radial perturbation of regular polygon
    for amp in [0.05, 0.1, 0.15, 0.2, 0.3]:
        for freq in [2, 3, 4, 5, 6]:
            angles = 2 * np.pi * np.arange(N) / N
            radii = 1.0 + amp * np.sin(freq * angles)
            configs.append(np.column_stack([radii * np.cos(angles), radii * np.sin(angles)]))
    
    # Random configurations
    for _ in range(12):
        configs.append(rng.uniform(-1, 1, (N, 2)))
    
    # Concentric circles with varying radii
    for n_inner in [1, 2, 3]:
        n_outer = N - n_inner
        inner_angles = 2 * np.pi * np.arange(n_inner) / n_inner
        outer_angles = 2 * np.pi * np.arange(n_outer) / n_outer + np.pi / n_outer
        inner_pts = np.column_stack([0.4 * np.cos(inner_angles), 0.4 * np.sin(inner_angles)])
        outer_pts = np.column_stack([1.0 * np.cos(outer_angles), 1.0 * np.sin(outer_angles)])
        configs.append(np.vstack([inner_pts, outer_pts]))
    
    return configs

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s kill
    
    best_score = 0.0
    best_points = None
    rng = np.random.RandomState(42)
    
    # Generate diverse initial configurations
    initial_configs = _generate_diverse_configs(rng)
    
    # Score all initial configs and sort
    scored_configs = []
    for config in initial_configs:
        try:
            s = _score(config)
            scored_configs.append((s, config))
        except Exception:
            pass
    scored_configs.sort(key=lambda x: -x[0])
    
    # Try top configs with multi-stage optimization
    # Prioritize top-8 to balance quality and quantity
    top_configs = scored_configs[:8]
    
    for s_init, config in top_configs:
        if time.time() > deadline - 35:
            break
        
        try:
            cand_pts, score = _optimize_from_start(config, deadline, rng)
            if score > best_score:
                best_score = score
                best_points = cand_pts
        except Exception:
            continue
    
    # Perturbation-based restarts with coordinate search
    if best_points is not None and time.time() < deadline - 25:
        n_restarts = 0
        while time.time() < deadline - 8:
            n_restarts += 1
            # Adaptive noise: start moderate, decrease over time
            frac = min(n_restarts / 20.0, 1.0)
            noise_scale = 0.15 * (1.0 - frac) + 0.002
            perturbed = best_points + noise_scale * rng.normal(0, 1, (N, 2))
            
            try:
                cand_pts, score = _optimize_from_start(perturbed, deadline, rng)
                if score > best_score:
                    best_score = score
                    best_points = cand_pts
            except Exception:
                continue
    
    # Final ultra-fine polish on the best result
    if best_points is not None and time.time() < deadline - 3:
        try:
            # Very tight Nelder-Mead
            res = minimize(
                _neg_score,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True}
            )
            cand_pts = res.x.reshape(N, 2)
            score = _score(cand_pts)
            if score > best_score:
                best_score = score
                best_points = cand_pts
        except Exception:
            pass
        
        # Final coordinate search with very small steps
        if time.time() < deadline - 1:
            try:
                cand_pts, score = _coordinate_local_search(
                    best_points, deadline, rng, initial_step=0.01, min_step=1e-10, max_iter=500
                )
                if score > best_score:
                    best_score = score
                    best_points = cand_pts
            except Exception:
                pass
    
    # Fallback to regular polygon if everything failed
    if best_points is None:
        angles = 2 * np.pi * np.arange(N) / N
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return best_points
