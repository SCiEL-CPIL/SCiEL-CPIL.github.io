import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
from itertools import combinations

N = 14

# Precompute indices for all triples
def _get_triple_indices(n):
    indices = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                indices.append((i, j, k))
    return np.array(indices, dtype=int)

_TRIPLE_INDICES = _get_triple_indices(N)

def _score(points):
    """
    Compute the score: min triangle area / convex hull area.
    """
    if len(points) < 3:
        return 0.0
    
    # Calculate Convex Hull Area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume # In 2D, volume is area
    except Exception:
        return 0.0
        
    if hull_area < 1e-12:
        return 0.0
        
    # Calculate Minimum Triangle Area
    # Vectorized calculation
    i_idx = _TRIPLE_INDICES[:, 0]
    j_idx = _TRIPLE_INDICES[:, 1]
    k_idx = _TRIPLE_INDICES[:, 2]
    
    p_i = points[i_idx]
    p_j = points[j_idx]
    p_k = points[k_idx]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    
    # Cross product in 2D
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    
    return min_area / hull_area

def _soft_min_score(points, beta=50.0):
    """
    Smooth approximation of min triangle area / hull area.
    Uses log-sum-exp trick for numerical stability.
    """
    if len(points) < 3:
        return 0.0
    
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0
        
    if hull_area < 1e-12:
        return 0.0
        
    i_idx = _TRIPLE_INDICES[:, 0]
    j_idx = _TRIPLE_INDICES[:, 1]
    k_idx = _TRIPLE_INDICES[:, 2]
    
    p_i = points[i_idx]
    p_j = points[j_idx]
    p_k = points[k_idx]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Soft minimum using log-sum-exp
    log_areas = np.log(areas + 1e-15)
    m = np.min(log_areas)
    soft_min = m - np.log(np.sum(np.exp(log_areas - m))) / beta
    
    return np.exp(soft_min) / hull_area

def _neg_soft_score(points, beta=50.0):
    return -_soft_min_score(points, beta)

def _jittered_14gon(radius_jitter, angular_jitter, seed=0):
    """
    Generate a jittered regular 14-gon.
    radius_jitter: amplitude of radial displacement
    angular_jitter: amplitude of angular displacement
    """
    rng = np.random.RandomState(seed)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    
    # Radial jitter: small perturbations
    r_jit = 1.0 + radius_jitter * rng.normal(0, 1, N)
    # Angular jitter
    a_jit = angles + angular_jitter * rng.normal(0, 1, N)
    
    pts = np.column_stack([r_jit * np.cos(a_jit), r_jit * np.sin(a_jit)])
    return pts

def _coordinate_polish(points, beta=50.0, max_iter=200, deadline=None):
    """
    Coordinate descent polish: optimize each coordinate one at a time.
    """
    pts = points.copy()
    for iteration in range(max_iter):
        if deadline and time.time() > deadline:
            break
        improved = False
        for i in range(N):
            for coord in range(2):
                # Try small steps in each direction
                best_val = _neg_soft_score(pts, beta)
                best_step = 0.0
                
                for step in [0.01, 0.005, 0.001, -0.01, -0.005, -0.001]:
                    pts_old = pts[i, coord]
                    pts[i, coord] = pts_old + step
                    val = _neg_soft_score(pts, beta)
                    if val < best_val:
                        best_val = val
                        best_step = step
                    pts[i, coord] = pts_old
                
                if best_step != 0.0:
                    pts[i, coord] += best_step
                    improved = True
        
        if not improved:
            break
    
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    best_score = 0
    best_points = None
    current_points = None
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Regular 14-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    init_reg = np.column_stack([np.cos(angles), np.sin(angles)])
    candidates.append(init_reg)
    
    # 2. Two concentric heptagons with various radii and offsets
    angles_inner = np.linspace(0, 2*np.pi, 7, endpoint=False)
    angles_outer = np.linspace(np.pi/7, 2*np.pi + np.pi/7, 7, endpoint=False)
    
    for r_in in [0.3, 0.5, 0.7, 0.8]:
        for offset in [0, np.pi/14, np.pi/7]:
            inner_pts = r_in * np.column_stack([np.cos(angles_inner + offset), np.sin(angles_inner + offset)])
            outer_pts = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
            candidates.append(np.vstack([inner_pts, outer_pts]))
    
    # 3. Jittered regular 14-gons with different jitter levels
    for rj in [0.05, 0.1, 0.15, 0.2]:
        for aj in [0.05, 0.1, 0.15]:
            for seed in [0, 1, 2]:
                candidates.append(_jittered_14gon(rj, aj, seed))
    
    # 4. Random points in unit disk
    rng = np.random.RandomState(42)
    for _ in range(5):
        r = np.sqrt(rng.rand(N))
        theta = 2 * np.pi * rng.rand(N)
        candidates.append(np.column_stack([r * np.cos(theta), r * np.sin(theta)]))
    
    # Evaluate all candidates
    for c in candidates:
        s = _score(c)
        if s > best_score:
            best_score = s
            best_points = c.copy()
            current_points = c.copy()
    
    if best_points is None:
        return init_reg
    
    # Phase 1: Nelder-Mead with soft-min objective
    if time.time() < deadline - 30:
        def objective_soft(x):
            if time.time() > deadline - 2:
                raise TimeoutError()
            p = x.reshape(N, 2)
            return _neg_soft_score(p, beta=50.0)
        
        try:
            res = minimize(objective_soft, current_points.flatten(), method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-7, 'fatol': 1e-9, 'maxfev': 5000})
            if res.fun < 0:
                new_pts = res.x.reshape(N, 2)
                s = _score(new_pts)
                if s > best_score:
                    best_score = s
                    best_points = new_pts.copy()
                    current_points = new_pts.copy()
        except TimeoutError:
            pass
        except Exception:
            pass
    
    # Phase 2: Coordinate polish on the best point
    if time.time() < deadline - 15:
        polished = _coordinate_polish(current_points, beta=50.0, max_iter=100, deadline=deadline - 5)
        s = _score(polished)
        if s > best_score:
            best_score = s
            best_points = polished.copy()
            current_points = polished.copy()
    
    # Phase 3: Final Nelder-Mead with hard score for fine-tuning
    if time.time() < deadline - 10:
        def objective_hard(x):
            if time.time() > deadline - 2:
                raise TimeoutError()
            p = x.reshape(N, 2)
            return -_score(p)
        
        try:
            res = minimize(objective_hard, current_points.flatten(), method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10, 'maxfev': 3000})
            if res.fun < 0:
                new_pts = res.x.reshape(N, 2)
                s = _score(new_pts)
                if s > best_score:
                    best_score = s
                    best_points = new_pts.copy()
        except TimeoutError:
            pass
        except Exception:
            pass
    
    # Phase 4: Try jittered 14-gon optimization with different seeds
    if time.time() < deadline - 20:
        for seed in [10, 20, 30]:
            if time.time() > deadline - 10:
                break
            jit_pts = _jittered_14gon(0.15, 0.15, seed)
            
            def objective_jit(x):
                if time.time() > deadline - 2:
                    raise TimeoutError()
                p = x.reshape(N, 2)
                return _neg_soft_score(p, beta=80.0)
            
            try:
                res = minimize(objective_jit, jit_pts.flatten(), method='Nelder-Mead',
                               options={'maxiter': 2000, 'xatol': 1e-7, 'fatol': 1e-9, 'maxfev': 2000})
                if res.fun < 0:
                    new_pts = res.x.reshape(N, 2)
                    s = _score(new_pts)
                    if s > best_score:
                        best_score = s
                        best_points = new_pts.copy()
            except TimeoutError:
                pass
            except Exception:
                pass
    
    # Final coordinate polish
    if time.time() < deadline - 3:
        polished = _coordinate_polish(best_points, beta=100.0, max_iter=50, deadline=deadline - 1)
        s = _score(polished)
        if s > best_score:
            best_score = s
            best_points = polished.copy()
    
    return best_points
