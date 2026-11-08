import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
import sys

N = 14
DEADLINE = 140.0  # Safety margin before 150s kill

def _hull_area(pts):
    """Compute convex hull area."""
    try:
        if len(pts) < 3:
            return 0.0
        # Check for collinearity or degeneracy
        if np.allclose(pts[:, 0], pts[0, 0]) or np.allclose(pts[:, 1], pts[0, 1]):
            return 0.0
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _all_triangle_areas(points):
    """
    Return array of all triangle areas (N choose 3).
    Vectorized calculation for speed.
    """
    n = len(points)
    x = points[:, 0]
    y = points[:, 1]
    
    num_tris = n * (n - 1) * (n - 2) // 6
    areas = np.empty(num_tris)
    idx = 0
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            # Vectorize over k > j
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            cross = (x[j] - x[i]) * dy_ki - dx_ki * (y[j] - y[i])
            tri_areas = 0.5 * np.abs(cross)
            areas[idx:idx+len(tri_areas)] = tri_areas
            idx += len(tri_areas)
            
    return areas

def _min_triangle_area_vec(points):
    """Compute the minimum triangle area."""
    areas = _all_triangle_areas(points)
    if len(areas) == 0:
        return 0.0
    return np.min(areas)

def _score(pts):
    """Compute min triangle area / hull area."""
    hull_a = _hull_area(pts)
    if hull_a <= 1e-10:
        return 0.0
    min_area = _min_triangle_area_vec(pts)
    return min_area / hull_a

def _normalize_unit_area(pts):
    """Scale points so convex hull area is 1."""
    hull_a = _hull_area(pts)
    if hull_a > 0:
        return pts / np.sqrt(hull_a)
    return pts

def _objective_de(pts_flat):
    """
    Objective for Differential Evolution.
    We want to maximize min_area / hull_area.
    Since DE minimizes, we return -score.
    To handle scale invariance, we normalize inside.
    """
    points = pts_flat.reshape(-1, 2)
    if np.any(~np.isfinite(points)):
        return 1e10
    
    # Normalize to unit area hull
    norm_pts = _normalize_unit_area(points)
    
    # If normalization failed (e.g., collinear), return high penalty
    hull_a = _hull_area(norm_pts)
    if hull_a < 1e-10:
        return 1e10
        
    min_area = _min_triangle_area_vec(norm_pts)
    
    # We want to maximize min_area. 
    # Score is min_area / 1.0 = min_area.
    # So we minimize -min_area.
    return -min_area

def _objective_local(pts_flat):
    """
    Objective for local polishing (Powell/Nelder-Mead).
    Minimizes -score.
    """
    points = pts_flat.reshape(-1, 2)
    if np.any(~np.isfinite(points)):
        return 1e10
    
    # Normalize to unit area hull
    norm_pts = _normalize_unit_area(points)
    
    hull_a = _hull_area(norm_pts)
    if hull_a < 1e-10:
        return 1e10
        
    min_area = _min_triangle_area_vec(norm_pts)
    
    return -min_area

def _generate_initial_configs():
    """Generate diverse initial configurations."""
    configs = []
    rng = np.random.RandomState(42)
    
    # 1. Regular 14-gon
    angles = 2 * np.pi * np.arange(N) / N
    p_reg = np.column_stack([np.cos(angles), np.sin(angles)])
    configs.append(p_reg)
    
    # 2. Regular 14-gon with noise
    for seed in range(10):
        noise = rng.normal(0, 0.1, size=(N, 2))
        p_noisy = p_reg + noise
        configs.append(p_noisy)
        
    # 3. Two concentric rings (7 and 7)
    ang1 = 2 * np.pi * np.arange(7) / 7
    r1 = 0.6
    p1 = np.column_stack([r1 * np.cos(ang1), r1 * np.sin(ang1)])
    
    ang2 = 2 * np.pi * np.arange(7) / 7 + np.pi/7
    r2 = 1.0
    p2 = np.column_stack([r2 * np.cos(ang2), r2 * np.sin(ang2)])
    
    p_rings = np.vstack([p1, p2])
    configs.append(p_rings)
    
    # 4. Three concentric rings (5, 5, 4)
    ang_a = 2 * np.pi * np.arange(5) / 5
    p_a = np.column_stack([0.4 * np.cos(ang_a), 0.4 * np.sin(ang_a)])
    ang_b = 2 * np.pi * np.arange(5) / 5 + np.pi/5
    p_b = np.column_stack([0.7 * np.cos(ang_b), 0.7 * np.sin(ang_b)])
    ang_c = 2 * np.pi * np.arange(4) / 4 + np.pi/4
    p_c = np.column_stack([1.0 * np.cos(ang_c), 1.0 * np.sin(ang_c)])
    p_3rings = np.vstack([p_a, p_b, p_c])
    configs.append(p_3rings)
    
    # 5. Random points in disk
    for seed in [1, 2, 3, 4, 5]:
        rng2 = np.random.RandomState(seed)
        r = rng2.uniform(0.2, 1, N) ** 0.5
        th = rng2.uniform(0, 2*np.pi, N)
        p_rand = np.column_stack([r * np.cos(th), r * np.sin(th)])
        configs.append(p_rand)
        
    # 6. Square-ish
    corners = np.array([[1,1],[-1,1],[-1,-1],[1,-1]], dtype=float)
    edge_pts = []
    for i in range(3):
        s = (i+1)/4.0
        edge_pts.append([1, -1+2*s])
        edge_pts.append([-1+2*s, 1])
        edge_pts.append([-1, 1-2*s])
        edge_pts.append([1-2*s, -1])
    edge_pts = np.array(edge_pts[:10])
    p_sq = np.vstack([corners, edge_pts])
    configs.append(p_sq)
    
    # Normalize all to unit area for consistency
    normalized_configs = []
    for c in configs:
        nc = _normalize_unit_area(c)
        if _hull_area(nc) > 1e-6:
            normalized_configs.append(nc)
            
    return normalized_configs

def _polish_candidate(pts, start_time, deadline, method='Powell'):
    """
    Polish a candidate configuration using local optimization.
    """
    if time.time() - start_time > deadline - 5:
        return pts
        
    x0 = pts.flatten()
    
    def obj(x):
        return _objective_local(x)
        
    try:
        if method == 'Powell':
            # Powell is robust for non-smooth
            res = minimize(obj, x0, method='Powell', 
                           options={'maxiter': 1000, 'ftol': 1e-12, 'xtol': 1e-10})
        elif method == 'Nelder-Mead':
            res = minimize(obj, x0, method='Nelder-Mead',
                           options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12})
        else:
            return pts
            
        if res.success or res.fun < obj(x0):
            new_pts = res.x.reshape(N, 2)
            # Ensure finite
            if np.all(np.isfinite(new_pts)):
                return new_pts
    except Exception:
        pass
        
    return pts

def construct():
    start_time = time.time()
    
    # Get initial configs
    inits = _generate_initial_configs()
    
    # Evaluate initial scores
    best_score = -np.inf
    best_pts = inits[0]
    
    for init in inits:
        s = _score(init)
        if s > best_score:
            best_score = s
            best_pts = init
            
    # Sort inits by score (best first)
    init_scores = [_score(c) for c in inits]
    sorted_indices = np.argsort(init_scores)[::-1]
    top_inits = [inits[i] for i in sorted_indices[:5]]
    
    # Phase 1: Differential Evolution
    # We search in Cartesian space, normalized inside objective.
    # Bounds: Since we normalize, the actual scale doesn't matter for the score,
    # but DE needs bounds. We use a box that contains typical unit-area configs.
    bounds = [(-2, 2)] * (2 * N)
    
    de_deadline = start_time + 60.0  # Allocate 60s to DE
    
    if time.time() < de_deadline:
        try:
            # Use a smaller population to fit in time
            # popsize=10, maxiter=50
            result = differential_evolution(
                _objective_de,
                bounds,
                maxiter=30,
                popsize=15,
                tol=1e-5,
                seed=42,
                polish=False,
                init='sobol'
            )
            
            if result.x is not None:
                points_de = result.x.reshape(N, 2)
                score_de = _score(points_de)
                if score_de > best_score:
                    best_score = score_de
                    best_pts = points_de
                # Add DE result to top candidates
                top_inits.append(points_de)
        except Exception:
            pass
    
    # Phase 2: Local Polishing
    # Polish the best DE result and the top initial configs
    candidates = list(set([id(c) for c in [best_pts] + top_inits]))
    # We can't use id for numpy arrays easily in a set if they are different objects.
    # Just take the unique best ones.
    unique_candidates = []
    seen = []
    for c in [best_pts] + top_inits:
        # Simple uniqueness check based on first point and shape
        key = tuple(c[0])
        if key not in seen:
            seen.append(key)
            unique_candidates.append(c)
            
    # Sort candidates by current score
    unique_candidates.sort(key=lambda x: _score(x), reverse=True)
    
    # Polish top 5 candidates with Powell
    polish_deadline = start_time + 120.0
    
    for i, cand in enumerate(unique_candidates[:5]):
        if time.time() > polish_deadline:
            break
            
        # Use Powell for robustness
        polished = _polish_candidate(cand, start_time, polish_deadline, method='Powell')
        
        # Normalize and score
        cand_pts = _normalize_unit_area(polished)
        s = _score(cand_pts)
        
        if s > best_score:
            best_score = s
            best_pts = cand_pts
    
    # Phase 3: Final Nelder-Mead polish on the absolute best
    if time.time() < start_time + 140:
        final_polished = _polish_candidate(best_pts, start_time, start_time + 140, method='Nelder-Mead')
        final_pts = _normalize_unit_area(final_polished)
        s_final = _score(final_pts)
        
        if s_final > best_score:
            best_score = s_final
            best_pts = final_pts
    
    return best_pts

# The entry point must be construct()
if __name__ == "__main__":
    pts = construct()
    print(_score(pts))
