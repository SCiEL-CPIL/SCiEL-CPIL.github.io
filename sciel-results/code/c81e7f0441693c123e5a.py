import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
import time
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

N = 13

# Precompute triangle indices
tri_indices = np.array(list(combinations(range(N), 3)))
_I_IDX = tri_indices[:, 0]
_J_IDX = tri_indices[:, 1]
_K_IDX = tri_indices[:, 2]

def _get_triangle_areas(points):
    """Vectorized calculation of areas for all triples."""
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

def compute_score(points):
    """
    Compute the ratio of minimum triangle area to convex hull area.
    """
    points = np.asarray(points, dtype=float)
    
    # Calculate areas of all triangles
    areas = _get_triangle_areas(points)
    min_area = np.min(areas)
    
    # Calculate convex hull area
    hull_area = _hull_area(points)
    if hull_area == 0:
        return 0.0
        
    return min_area / hull_area

def _softmin_score(points, alpha):
    """
    Smooth approximation of the score using softmin of triangle areas.
    This provides a differentiable-like landscape for optimization.
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
    softmin_val = a_min - log_sum / alpha
    
    return softmin_val / hull_area

def _neg_score(params):
    """
    Negative score for minimization.
    """
    points = params.reshape(N, 2)
    return -compute_score(points)

def _neg_softmin_score(params, alpha):
    """
    Negative softmin score for minimization.
    """
    points = params.reshape(N, 2)
    return -_softmin_score(points, alpha)

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
    
    while len(points) < n_points:
        idx = len(points)
        angle = 2 * np.pi * idx / n_points + phase_offset
        points.append([radius * 0.5 * np.cos(angle), radius * 0.5 * np.sin(angle)])
    
    return np.array(points[:n_points])

def _hierarchical_points(affine_params, angular_devs):
    """
    Generate points from hierarchical parameterization:
    - affine_params: [a11, a12, a21, a22, tx, ty] (2x2 matrix + translation)
    - angular_devs: N angular deviations from regular polygon angles
    
    Points start as regular N-gon, then angular perturbations are applied,
    then affine transformation is applied.
    """
    # Base regular polygon angles
    base_angles = 2 * np.pi * np.arange(N) / N
    
    # Apply angular deviations
    angles = base_angles + angular_devs
    
    # Points on unit circle with perturbed angles
    base_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Apply affine transformation: M @ p + t
    a11, a12, a21, a22, tx, ty = affine_params
    M = np.array([[a11, a12], [a21, a22]])
    t = np.array([tx, ty])
    
    transformed = base_pts @ M.T + t
    return transformed

def _neg_hierarchical_score(affine_angular_params):
    """
    Negative score for hierarchical parameterization.
    First 6 params are affine, next N params are angular deviations.
    """
    affine_params = affine_angular_params[:6]
    angular_devs = affine_angular_params[6:6+N]
    
    points = _hierarchical_points(affine_params, angular_devs)
    return -compute_score(points)

def _neg_hierarchical_softmin(affine_angular_params, alpha):
    """
    Negative softmin score for hierarchical parameterization.
    """
    affine_params = affine_angular_params[:6]
    angular_devs = affine_angular_params[6:6+N]
    
    points = _hierarchical_points(affine_params, angular_devs)
    return -_softmin_score(points, alpha)

def _optimize_hierarchical(deadline, rng):
    """
    Two-stage hierarchical optimization:
    Stage 1: DE on affine params (6) with fixed regular angles
    Stage 2: Polish angular deviations with Nelder-Mead
    Stage 3: Joint polish of all params
    """
    best_score = 0.0
    best_params = None
    
    # Stage 1: Optimize affine frame with DE
    # Bounds: affine matrix entries in [-2, 2], translation in [-1, 1]
    bounds = [(-2, 2)] * 4 + [(-1, 1)] * 2
    
    def _neg_affine_only(affine_params):
        # Regular angles (zero deviations)
        zero_devs = np.zeros(N)
        points = _hierarchical_points(affine_params, zero_devs)
        return -compute_score(points)
    
    if time.time() < deadline - 60:
        try:
            de_result = differential_evolution(
                _neg_affine_only,
                bounds=bounds,
                maxiter=100,
                popsize=20,
                tol=1e-8,
                seed=int(rng.integers(0, 2**31)),
                polish=False,
                init='latinhypercube'
            )
            best_affine = de_result.x
            score = compute_score(_hierarchical_points(best_affine, np.zeros(N)))
            if score > best_score:
                best_score = score
                best_params = np.concatenate([best_affine, np.zeros(N)])
        except Exception:
            # Fallback: identity affine
            best_affine = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
            best_params = np.concatenate([best_affine, np.zeros(N)])
            best_score = compute_score(_hierarchical_points(best_affine, np.zeros(N)))
    
    if best_params is None:
        best_params = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0] + [0.0] * N)
        best_score = compute_score(_hierarchical_points(best_params[:6], np.zeros(N)))
    
    # Stage 2: Polish angular deviations with Nelder-Mead
    if time.time() < deadline - 30:
        try:
            res = minimize(
                _neg_hierarchical_softmin,
                best_params,
                args=(10.0,),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
            )
            cand_params = res.x
            cand_score = compute_score(_hierarchical_points(cand_params[:6], cand_params[6:]))
            if cand_score > best_score:
                best_score = cand_score
                best_params = cand_params
        except Exception:
            pass
    
    # Stage 3: Joint polish of all 6+N params with decreasing alpha
    alphas = [5.0, 2.0, 1.0]
    for alpha in alphas:
        if time.time() > deadline - 15:
            break
        try:
            res = minimize(
                _neg_hierarchical_softmin,
                best_params,
                args=(alpha,),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True}
            )
            cand_params = res.x
            cand_score = compute_score(_hierarchical_points(cand_params[:6], cand_params[6:]))
            if cand_score > best_score:
                best_score = cand_score
                best_params = cand_params
        except Exception:
            pass
    
    # Convert back to point coordinates
    final_points = _hierarchical_points(best_params[:6], best_params[6:])
    final_score = compute_score(final_points)
    
    return final_points, max(best_score, final_score)

def _optimize_from_start(start_pts, deadline, rng):
    """
    Multi-stage optimization from a starting point:
    1. Softmin-based Nelder-Mead with decreasing alpha
    2. L-BFGS-B polish
    3. Nelder-Mead on hard score
    Returns best points and score found.
    """
    best_pts = start_pts.copy()
    best_score = compute_score(start_pts)
    
    # Stage 1: Softmin optimization with annealing alpha
    alphas = [100, 50, 20, 10, 5, 2]
    for alpha in alphas:
        if time.time() > deadline - 10:
            break
        try:
            res = minimize(
                _neg_softmin_score,
                best_pts.flatten(),
                args=(alpha,),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True}
            )
            cand_pts = res.x.reshape(N, 2)
            score = compute_score(cand_pts)
            if score > best_score:
                best_score = score
                best_pts = cand_pts
        except Exception:
            pass
    
    # Stage 2: L-BFGS-B polish with softmin
    if time.time() < deadline - 15:
        try:
            res = minimize(
                _neg_softmin_score,
                best_pts.flatten(),
                args=(2.0,),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            cand_pts = res.x.reshape(N, 2)
            score = compute_score(cand_pts)
            if score > best_score:
                best_score = score
                best_pts = cand_pts
        except Exception:
            pass
    
    # Stage 3: Nelder-Mead on hard score
    if time.time() < deadline - 5:
        try:
            res = minimize(
                _neg_score,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
            )
            cand_pts = res.x.reshape(N, 2)
            score = compute_score(cand_pts)
            if score > best_score:
                best_score = score
                best_pts = cand_pts
        except Exception:
            pass
    
    return best_pts, best_score

def _polish_with_local_search(points, deadline):
    """
    Final high-precision polish: coordinate descent on the hard objective.
    Iteratively move each point to maximize the minimum triangle area.
    """
    best_pts = points.copy()
    best_score = compute_score(best_pts)
    
    # Coordinate descent: for each point, try small perturbations
    for outer_iter in range(50):
        if time.time() > deadline - 5:
            break
        
        improved = False
        for i in range(N):
            if time.time() > deadline - 5:
                break
            
            orig = best_pts[i].copy()
            best_local = orig
            best_local_score = best_score
            
            # Try perturbations in x and y directions
            for dx in [-0.05, -0.02, -0.01, 0.0, 0.01, 0.02, 0.05]:
                for dy in [-0.05, -0.02, -0.01, 0.0, 0.01, 0.02, 0.05]:
                    cand = best_pts.copy()
                    cand[i] = orig + np.array([dx, dy])
                    s = compute_score(cand)
                    if s > best_local_score + 1e-12:
                        best_local_score = s
                        best_local = orig + np.array([dx, dy])
            
            if np.any(best_local != orig):
                best_pts[i] = best_local
                improved = True
        
        if not improved:
            break
        
        current_score = compute_score(best_pts)
        if current_score > best_score:
            best_score = current_score
    
    return best_pts, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 150s kill
    
    best_score = 0.0
    best_points = None
    rng = np.random.default_rng(42)
    
    # First: try hierarchical parameterization (promising approach)
    try:
        hier_pts, hier_score = _optimize_hierarchical(deadline, rng)
        if hier_score > best_score:
            best_score = hier_score
            best_points = hier_pts
    except Exception:
        pass
    
    # Generate diverse initial configurations for standard optimization
    initial_configs = []
    
    # Regular polygon
    angles = 2 * np.pi * np.arange(N) / N
    initial_configs.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # Ring configurations with various parameters
    for radius in [1.0, 1.2, 0.8]:
        for phase in [0.0, np.pi / N, np.pi / (2 * N)]:
            initial_configs.append(_make_ring_config(N, radius, phase_offset=phase))
    
    # Two-ring structures
    for r1, r2 in [(0.5, 1.0), (0.4, 1.1), (0.6, 0.9)]:
        for phase in [0.0, np.pi / N]:
            initial_configs.append(_make_ring_config(N, 1.0, phase_offset=phase, n_rings=2, ring_radii=[r1, r2]))
    
    # Radial perturbation of regular polygon
    for amp in [0.1, 0.2, 0.3]:
        pts = []
        for i in range(N):
            angle = 2 * np.pi * i / N
            r = 1.0 + amp * np.sin(3 * angle)
            pts.append([r * np.cos(angle), r * np.sin(angle)])
        initial_configs.append(np.array(pts))
    
    # Grid-like
    init_grid = np.array([
        [0, 0], [1, 0], [2, 0],
        [0, 1], [1, 1], [2, 1],
        [0, 2], [1, 2], [2, 2],
        [0.5, 0.5], [1.5, 0.5], [0.5, 1.5], [1.5, 1.5]
    ])
    initial_configs.append(init_grid)
    
    # Random
    init_rand = rng.random((N, 2))
    initial_configs.append(init_rand)
    
    # Optimize from each initial configuration if time permits
    for init in initial_configs:
        if time.time() > deadline - 20:
            break
            
        try:
            cand_pts, cand_score = _optimize_from_start(init, deadline, rng)
            if cand_score > best_score:
                best_score = cand_score
                best_points = cand_pts
        except Exception:
            pass
    
    # Final polish with local search if we have time
    if best_points is not None and time.time() < deadline - 10:
        try:
            best_points, best_score = _polish_with_local_search(best_points, deadline)
        except Exception:
            pass
    
    if best_points is None:
        # Fallback to regular polygon if optimization failed
        angles = 2 * np.pi * np.arange(N) / N
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_points
