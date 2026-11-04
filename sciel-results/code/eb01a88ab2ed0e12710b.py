"""Heilbronn convex regions: Maximize minimum triangle area / hull area."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 13

# EVOLVE-BLOCK-START

# Pre-compute triangle indices for efficiency
# i < j < k
_idx_i, _idx_j, _idx_k = np.indices((N, N, N)).reshape(3, -1)
_mask = (_idx_i < _idx_j) & (_idx_j < _idx_k)
_TRI_I = _idx_i[_mask]
_TRI_J = _idx_j[_mask]
_TRI_K = _idx_k[_mask]
_NUM_TRI = len(_TRI_I)

def _compute_score(pts):
    """
    Compute the minimum triangle area normalized by the convex hull area.
    Returns 0 if invalid.
    """
    pts = np.asarray(pts)
    if pts.shape != (N, 2):
        return 0.0
        
    # Compute Convex Hull Area
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
        if hull_area < 1e-9:
            return 0.0
    except Exception:
        return 0.0

    # Compute all triangle areas vectorized
    p_i = pts[_TRI_I]
    p_j = pts[_TRI_J]
    p_k = pts[_TRI_K]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    
    return min_area / hull_area

def _angles_to_points(angles):
    """Convert angles (and optional radii) to 2D points on/near unit circle."""
    # angles shape: (N,) or (N, 2) where second column is log-radius
    if angles.ndim == 1:
        a = angles
        r = np.ones(N)
    else:
        a = angles[:, 0]
        r = np.exp(angles[:, 1])
    return np.column_stack((r * np.cos(a), r * np.sin(a)))

def _smooth_min_objective(params, temperature):
    """
    Smooth log-sum-exp approximation of -min(triangle areas) / hull_area.
    Uses angular + log-radius representation.
    """
    pts = _angles_to_points(params)
    
    # Compute triangle areas
    p_i = pts[_TRI_I]
    p_j = pts[_TRI_J]
    p_k = pts[_TRI_K]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Smooth minimum via log-sum-exp
    # min(x) ≈ -1/β * log(sum(exp(-β*x)))
    beta = 1.0 / temperature
    neg_beta_areas = -beta * areas
    max_val = np.max(neg_beta_areas)
    log_sum = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
    smooth_min = -log_sum / beta
    
    # Compute convex hull area (use polygon area for sorted-by-angle points)
    # Since points are on/near circle in angular order, hull is the polygon
    # Sort by angle to get convex hull vertices
    a = params[:, 0] if params.ndim > 1 else params
    sort_idx = np.argsort(a)
    hull_pts = pts[sort_idx]
    
    # Shoelace formula for polygon area
    x = hull_pts[:, 0]
    y = hull_pts[:, 1]
    n_hull = len(x)
    hull_area = 0.5 * np.abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    
    if hull_area < 1e-9:
        return 1e9
    
    ratio = smooth_min / hull_area
    return -ratio

def _generate_initial_params(rng):
    """Generate diverse initial parameter sets (angles + log-radii)."""
    inits = []
    
    # 1. Regular polygon angles, unit radius
    angles_reg = np.linspace(0, 2 * np.pi, N, endpoint=False)
    params_reg = np.column_stack((angles_reg, np.zeros(N)))
    inits.append(params_reg)
    
    # 2. Halton-based angles
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    halton_angles = np.array([halton(i, 2) * 2 * np.pi for i in range(1, N + 1)])
    halton_angles = np.sort(halton_angles)
    params_halton = np.column_stack((halton_angles, np.zeros(N)))
    inits.append(params_halton)
    
    # 3. Jittered regular polygons
    for trial in range(8):
        noise = rng.normal(0, 0.15, N)
        angles_jit = angles_reg + noise
        angles_jit = np.sort(angles_jit)
        # Wrap to [0, 2π)
        angles_jit = angles_jit % (2 * np.pi)
        angles_jit = np.sort(angles_jit)
        radii = rng.uniform(-0.2, 0.2, N)
        params_jit = np.column_stack((angles_jit, radii))
        inits.append(params_jit)
    
    # 4. Random angular distributions
    for trial in range(8):
        angles_rand = np.sort(rng.uniform(0, 2 * np.pi, N))
        radii = rng.uniform(-0.3, 0.3, N)
        params_rand = np.column_stack((angles_rand, radii))
        inits.append(params_rand)
    
    # 5. Grid-inspired: cluster angles
    for trial in range(4):
        base_angles = np.linspace(0, 2 * np.pi, 4, endpoint=False)
        angles_grid = np.repeat(base_angles, 4)[:N]
        noise = rng.normal(0, 0.2, N)
        angles_grid = np.sort(angles_grid + noise)
        angles_grid = angles_grid % (2 * np.pi)
        angles_grid = np.sort(angles_grid)
        radii = rng.uniform(-0.1, 0.1, N)
        params_grid = np.column_stack((angles_grid, radii))
        inits.append(params_grid)
    
    return inits

def _optimize_from_start(params0, start_time, time_limit):
    """Run multi-temperature L-BFGS-B optimization from a single start."""
    best_params = params0.copy()
    best_score = _compute_score(_angles_to_points(params0))
    
    # Multi-scale temperature annealing
    temperatures = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    
    for temp in temperatures:
        if (time.time() - start_time) > time_limit - 5.0:
            break
        
        try:
            def obj(p):
                return _smooth_min_objective(p, temp)
            
            res = minimize(
                obj,
                best_params,
                method='L-BFGS-B',
                options={'maxiter': 200, 'gtol': 1e-6, 'maxfun': 1000}
            )
            
            if res.fun < 1e8:
                new_params = res.x
                new_score = _compute_score(_angles_to_points(new_params))
                if new_score > best_score:
                    best_score = new_score
                    best_params = new_params.copy()
                    
        except Exception:
            continue
    
    return best_params, best_score

def construct():
    start_time = time.time()
    time_limit = 140.0  # Safety margin for 150s limit
    
    rng = np.random.default_rng(42)
    
    best_params = None
    best_score = 0.0
    
    inits = _generate_initial_params(rng)
    
    # Optimize each start
    for p0 in inits:
        if (time.time() - start_time) > time_limit - 10.0:
            break
            
        try:
            params, score = _optimize_from_start(p0, start_time, time_limit)
            if score > best_score:
                best_score = score
                best_params = params.copy()
                
        except Exception:
            continue
    
    # Final polish with very low temperature
    if best_params is not None and (time.time() - start_time) < time_limit - 5.0:
        try:
            def obj_final(p):
                return _smooth_min_objective(p, 0.0005)
            
            res_final = minimize(
                obj_final,
                best_params,
                method='L-BFGS-B',
                options={'maxiter': 500, 'gtol': 1e-8, 'maxfun': 2000}
            )
            
            if res_final.fun < 1e8:
                final_params = res_final.x
                final_score = _compute_score(_angles_to_points(final_params))
                if final_score > best_score:
                    best_score = final_score
                    best_params = final_params.copy()
        except Exception:
            pass
    
    if best_params is None:
        # Fallback to regular polygon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        return np.column_stack((np.cos(angles), np.sin(angles)))
        
    return _angles_to_points(best_params)

# EVOLVE-BLOCK-END
