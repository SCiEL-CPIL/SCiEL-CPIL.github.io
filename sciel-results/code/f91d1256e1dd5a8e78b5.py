import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time

N = 14

# Pre-compute indices for all triples
_IDX_I = []
_IDX_J = []
_IDX_K = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _IDX_I.append(i)
            _IDX_J.append(j)
            _IDX_K.append(k)
_IDX_I = np.array(_IDX_I)
_IDX_J = np.array(_IDX_J)
_IDX_K = np.array(_IDX_K)

def _soft_min(arr, temperature=0.01):
    """
    Smooth approximation to the minimum using log-sum-exp.
    As temperature -> 0, this approaches the true minimum.
    """
    if len(arr) == 0:
        return 0.0
    # Use scaled values for numerical stability
    shifted = arr / temperature
    max_val = np.max(shifted)
    log_sum_exp = max_val + np.log(np.sum(np.exp(shifted - max_val)))
    return -temperature * log_sum_exp

def _score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    """
    try:
        if np.any(np.isnan(points)) or np.any(np.isinf(points)):
            return 0.0
        
        # Compute convex hull area
        try:
            hull = ConvexHull(points)
            hull_area = hull.volume
        except Exception:
            return 0.0
            
        if hull_area < 1e-10:
            return 0.0
        
        # Compute triangle areas
        pi = points[_IDX_I]
        pj = points[_IDX_J]
        pk = points[_IDX_K]
        
        v1 = pj - pi
        v2 = pk - pi
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        
        min_area = np.min(areas)
        
        return min_area / hull_area
    except Exception:
        return 0.0

def _angles_to_points(angles):
    """Convert sorted angles on unit circle to points."""
    angles = np.sort(angles) % (2 * np.pi)
    return np.column_stack([np.cos(angles), np.sin(angles)])

def _circular_objective(angles):
    """
    Objective for circular parameterization: minimize -score
    """
    try:
        if np.any(np.isnan(angles)) or np.any(np.isinf(angles)):
            return 1e6
        points = _angles_to_points(angles)
        s = _score(points)
        if s <= 1e-15:
            return 1e6
        return -s
    except Exception:
        return 1e6

def _smooth_objective(angles, temperature=0.01):
    """
    Smooth objective using soft-min of triangle areas.
    This enables gradient-based optimization.
    """
    try:
        if np.any(np.isnan(angles)) or np.any(np.isinf(angles)):
            return 1e6
        
        points = _angles_to_points(angles)
        
        # Compute convex hull area
        try:
            hull = ConvexHull(points)
            hull_area = hull.volume
        except Exception:
            return 1e6
            
        if hull_area < 1e-10:
            return 1e6
        
        # Compute triangle areas
        pi = points[_IDX_I]
        pj = points[_IDX_J]
        pk = points[_IDX_K]
        
        v1 = pj - pi
        v2 = pk - pi
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        
        # Use soft-min for smoothness
        min_area_smooth = _soft_min(areas, temperature)
        
        score = min_area_smooth / hull_area
        if score <= 1e-15:
            return 1e6
        return -score
    except Exception:
        return 1e6

def construct():
    start_time = time.time()
    # Evaluator kills after 150s. We use 140s as a hard stop to ensure we return.
    deadline = start_time + 140.0
    
    best_points = None
    best_score = -np.inf
    best_angles = None
    
    # Initial guesses
    guesses = []
    
    # 1. Regular polygon
    angles_reg = np.linspace(0, 2 * np.pi, N, endpoint=False)
    guesses.append(angles_reg.copy())
    
    # 2. Perturbed regular polygons with varying noise levels
    rng = np.random.default_rng(42)
    noise_levels = [0.05, 0.1, 0.15, 0.2, 0.3]
    for noise_level in noise_levels:
        for _ in range(4):
            noise = rng.normal(0, noise_level, N)
            guesses.append(angles_reg + noise)
        
    # 3. Random angles
    for _ in range(15):
        guesses.append(np.sort(rng.uniform(0, 2 * np.pi, N)))
    
    # 4. Slightly rotated regular polygons
    for rot in np.linspace(0, 2*np.pi/N, 5, endpoint=False):
        guesses.append(angles_reg + rot)
    
    # Evaluate initial guesses
    scored_guesses = []
    for g in guesses:
        points = _angles_to_points(g)
        s = _score(points)
        scored_guesses.append((s, g))
        if s > best_score:
            best_score = s
            best_points = points.copy()
            best_angles = g.copy()
            
    # Sort by score
    scored_guesses.sort(key=lambda x: x[0], reverse=True)
    
    # Phase 1: Use smooth objective with L-BFGS-B for efficient gradient-based optimization
    # Start with higher temperature for smoother landscape, then anneal
    temperatures = [0.05, 0.02, 0.01, 0.005]
    
    for temp in temperatures:
        if time.time() > deadline - 20:
            break
            
        # Refine top candidates with smooth objective
        num_refines_smooth = min(8, len(scored_guesses))
        for idx in range(num_refines_smooth):
            if time.time() > deadline - 10:
                break
                
            s_init, angles = scored_guesses[idx]
            
            try:
                res = minimize(
                    lambda a: _smooth_objective(a, temperature=temp),
                    angles,
                    method='L-BFGS-B',
                    options={
                        'maxiter': 500,
                        'ftol': 1e-12,
                        'gtol': 1e-8,
                    }
                )
                # Evaluate actual score (not smooth)
                points = _angles_to_points(res.x)
                s = _score(points)
                if s > best_score:
                    best_score = s
                    best_points = points.copy()
                    best_angles = res.x.copy()
                    # Update scored_guesses with improved solution
                    scored_guesses[0] = (s, res.x.copy())
            except Exception:
                pass
        
        # Re-sort after each temperature phase
        scored_guesses.sort(key=lambda x: x[0], reverse=True)
    
    # Phase 2: Nelder-Mead refinement on best candidates with exact objective
    if time.time() < deadline - 15:
        num_refines_nm = min(5, len(scored_guesses))
        for idx in range(num_refines_nm):
            if time.time() > deadline - 5:
                break
                
            s_init, angles = scored_guesses[idx]
            
            try:
                res = minimize(
                    _circular_objective,
                    angles,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 3000, 
                        'xatol': 1e-10, 
                        'fatol': 1e-12, 
                        'adaptive': True
                    }
                )
                s = -res.fun
                if s > best_score:
                    best_score = s
                    best_points = _angles_to_points(res.x)
                    best_angles = res.x
            except Exception:
                pass
    
    # Phase 3: Final high-precision polish
    if best_angles is not None and time.time() < deadline - 2:
        try:
            # Use very tight tolerances for final polish
            res = minimize(
                _circular_objective,
                best_angles,
                method='Nelder-Mead',
                options={
                    'maxiter': 2000, 
                    'xatol': 1e-12, 
                    'fatol': 1e-14, 
                    'adaptive': True
                }
            )
            s = -res.fun
            if s > best_score:
                best_score = s
                best_points = _angles_to_points(res.x)
                best_angles = res.x
        except Exception:
            pass

    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = _angles_to_points(angles)
        
    return best_points
