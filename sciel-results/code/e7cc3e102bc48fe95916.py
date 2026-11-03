import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

N = 14
N_HALF = 7

# Precompute indices for all combinations of 3 points
_indices = np.arange(N)
IDX_I, IDX_J, IDX_K = np.meshgrid(_indices, _indices, _indices, indexing='ij')
_mask = (IDX_I < IDX_J) & (IDX_J < IDX_K)
IDX_I = IDX_I[_mask].flatten()
IDX_J = IDX_J[_mask].flatten()
IDX_K = IDX_K[_mask].flatten()

def _all_areas(points):
    pi = points[IDX_I]
    pj = points[IDX_J]
    pk = points[IDX_K]
    v1x = pj[:, 0] - pi[:, 0]
    v1y = pj[:, 1] - pi[:, 1]
    v2x = pk[:, 0] - pi[:, 0]
    v2y = pk[:, 1] - pi[:, 1]
    areas = 0.5 * np.abs(v1x * v2y - v1y * v2x)
    return areas

def _compute_min_triangle_area(points):
    return np.min(_all_areas(points))

def _compute_hull_area(points):
    try:
        # Check for degenerate cases (all points collinear or identical)
        if np.allclose(points[:, 0], points[0, 0]) or np.allclose(points[:, 1], points[0, 1]):
            return 0.0
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    hull_area = _compute_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _compute_min_triangle_area(points)
    return min_area / hull_area

def _params_to_points(params):
    """
    Convert 14 parameters (7 angles + 7 radii) to 14 centrally symmetric points.
    params[0:7]: angles in (0, pi) for the 7 half-points
    params[7:14]: radii for the 7 half-points
    
    The 14 points are: p_i = r_i * (cos(a_i), sin(a_i)) and -p_i for i=0..6
    """
    angles = params[:N_HALF]
    radii = params[N_HALF:]
    
    # Ensure angles are in valid range
    angles = np.clip(angles, 0.02, np.pi - 0.02)
    radii = np.clip(radii, 0.05, 3.0)
    
    half_points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    all_points = np.vstack([half_points, -half_points])
    return all_points

def _objective(params):
    """Negative score for minimization."""
    points = _params_to_points(params)
    score = _score(points)
    return -score

def _objective_soft(params, beta=200.0):
    """Soft-min objective for smoother optimization landscape."""
    points = _params_to_points(params)
    if points.shape != (N, 2) or np.any(~np.isfinite(points)):
        return 1e10
    
    hull_area = _compute_hull_area(points)
    if hull_area < 1e-10:
        return 1e10
    
    areas = _all_areas(points)
    if np.min(areas) < 1e-12:
        return 1e10
    
    # Soft min via log-sum-exp: approximates min(areas) smoothly
    z = -beta * areas
    max_z = np.max(z)
    if max_z < -700:
        log_sum = max_z
    else:
        log_sum = max_z + np.log(np.sum(np.exp(z - max_z)))
    soft_min = -log_sum / beta
    
    # Penalize if hull area is too small or points are degenerate
    penalty = 0.0
    if hull_area < 0.1:
        penalty += (0.1 - hull_area) * 100
    if np.min(areas) < 0.01:
        penalty += (0.01 - np.min(areas)) * 50
        
    return -(soft_min / hull_area) + penalty

def _bounds_sym():
    """
    Bounds for centrally symmetric parameters.
    """
    bounds = []
    # Angles: (0.02, pi-0.02)
    for _ in range(N_HALF):
        bounds.append((0.02, np.pi - 0.02))
    # Radii: (0.05, 3.0)
    for _ in range(N_HALF):
        bounds.append((0.05, 3.0))
    return bounds

def construct():
    t0 = time.time()
    deadline = t0 + 145.0  # 5s safety margin
    
    best_score = 0.0
    best_params = None
    best_points = None
    
    bounds = _bounds_sym()
    
    # Generate initial parameter sets for local refinement
    inits = []
    rng = np.random.default_rng(42)
    
    # 1. Regular symmetric configuration (approximation of regular 14-gon)
    base_angles = np.linspace(0.1, np.pi - 0.1, N_HALF)
    base_radii = np.ones(N_HALF)
    inits.append(np.concatenate([base_angles, base_radii]))
    
    # 2. Varying radii scales
    for r_scale in [0.5, 0.7, 1.0, 1.3]:
        radii = np.full(N_HALF, r_scale)
        inits.append(np.concatenate([base_angles, radii]))
        
    # 3. Random perturbations around regular
    for _ in range(15):
        angles = np.sort(rng.uniform(0.1, np.pi - 0.1, N_HALF))
        radii = rng.uniform(0.5, 1.5, N_HALF)
        inits.append(np.concatenate([angles, radii]))
        
    # 4. More diverse random starts
    for _ in range(15):
        angles = np.sort(rng.uniform(0.05, np.pi - 0.05, N_HALF))
        radii = rng.uniform(0.2, 2.5, N_HALF)
        inits.append(np.concatenate([angles, radii]))

    # Phase 1: Global Exploration with Differential Evolution using soft objective
    # Use soft objective for smoother landscape, then verify with exact score
    if time.time() < deadline - 70:
        try:
            de_result = differential_evolution(
                _objective_soft,
                bounds,
                seed=42,
                maxiter=50,
                popsize=15,
                tol=1e-6,
                polish=False,
                init='sobol',
                mutation=(0.4, 1.0),
                recombination=0.8
            )
            if np.isfinite(de_result.fun):
                # Evaluate with exact score
                exact_score = -_objective(de_result.x)
                if exact_score > best_score:
                    best_score = exact_score
                    best_params = de_result.x.copy()
                    best_points = _params_to_points(de_result.x)
        except Exception:
            pass
    
    # Phase 2: Local Refinement from multiple starts using soft objective
    # Score initial candidates with exact objective
    scored_inits = []
    for init_params in inits:
        if time.time() > deadline - 50:
            break
        try:
            s = -_objective(init_params)
            scored_inits.append((s, init_params))
        except Exception:
            pass
            
    # Sort by score descending
    scored_inits.sort(key=lambda x: x[0], reverse=True)
    
    # Refine the best DE result and top random starts
    candidates = []
    if best_params is not None:
        candidates.append(best_params)
        
    for s, p in scored_inits[:8]:
        candidates.append(p)
        
    # Regular polygon
    regular_params = np.concatenate([np.linspace(0.1, np.pi - 0.1, N_HALF), np.ones(N_HALF)])
    candidates.append(regular_params)
    
    # Add some diverse symmetric configurations
    for r_base in [0.6, 0.8, 1.2]:
        angles = np.linspace(0.15, np.pi - 0.15, N_HALF)
        radii = np.full(N_HALF, r_base)
        # Perturb angles slightly
        angles = angles + rng.uniform(-0.05, 0.05, N_HALF)
        angles = np.sort(angles)
        candidates.append(np.concatenate([angles, radii]))
    
    for init_params in candidates:
        if time.time() > deadline - 15:
            break
            
        # First pass: Nelder-Mead with soft objective for smoother convergence
        try:
            res_soft = minimize(
                lambda p: _objective_soft(p, beta=100.0), 
                init_params, 
                method='Nelder-Mead',
                options={
                    'maxiter': 3000,
                    'xatol': 1e-12,
                    'fatol': 1e-14,
                    'adaptive': True
                }
            )
            
            if np.isfinite(res_soft.fun):
                # Verify with exact objective
                exact_score = -_objective(res_soft.x)
                if exact_score > best_score:
                    best_score = exact_score
                    best_params = res_soft.x.copy()
                    best_points = _params_to_points(res_soft.x)
                    
        except Exception:
            pass
        
        # Second pass: Powell on exact objective for precision
        if best_score > 0 and time.time() < deadline - 8:
            try:
                res_powell = minimize(
                    _objective,
                    best_params,
                    method='Powell',
                    options={
                        'maxiter': 1000,
                        'xtol': 1e-14,
                        'ftol': 1e-16
                    }
                )
                score_powell = -res_powell.fun
                if np.isfinite(score_powell) and score_powell > best_score:
                    best_score = score_powell
                    best_params = res_powell.x.copy()
                    best_points = _params_to_points(res_powell.x)
            except Exception:
                pass

    # Phase 3: Final high-precision polish with multiple methods
    if best_params is not None:
        # Try different beta values for soft-min to escape local optima
        for beta in [50.0, 100.0, 200.0, 500.0]:
            if time.time() > deadline - 3:
                break
            try:
                res_soft_final = minimize(
                    lambda p, b=beta: _objective_soft(p, b),
                    best_params,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 2000,
                        'xatol': 1e-13,
                        'fatol': 1e-15,
                        'adaptive': True
                    }
                )
                if np.isfinite(res_soft_final.fun):
                    exact_score = -_objective(res_soft_final.x)
                    if exact_score > best_score:
                        best_score = exact_score
                        best_params = res_soft_final.x.copy()
                        best_points = _params_to_points(res_soft_final.x)
            except Exception:
                pass
        
        # Final Powell polish
        if time.time() < deadline - 1:
            try:
                res_final_powell = minimize(
                    _objective,
                    best_params,
                    method='Powell',
                    options={
                        'maxiter': 2000,
                        'xtol': 1e-15,
                        'ftol': 1e-17
                    }
                )
                score_final = -res_final_powell.fun
                if np.isfinite(score_final) and score_final > best_score:
                    best_score = score_final
                    best_params = res_final_powell.x.copy()
                    best_points = _params_to_points(res_final_powell.x)
            except Exception:
                pass

    # Final fallback if nothing found
    if best_points is None:
        # Return regular 14-gon
        angles = 2 * np.pi * np.arange(N) / N
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        
    # Normalize to unit area
    hull_area = _compute_hull_area(best_points)
    if hull_area > 1e-12:
        scale = np.sqrt(1.0 / hull_area)
        best_points = best_points * scale
    else:
        best_points = best_points * 1.0
        
    return best_points
