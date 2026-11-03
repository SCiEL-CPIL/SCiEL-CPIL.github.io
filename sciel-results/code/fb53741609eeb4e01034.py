import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

# Precompute indices for all 286 triples
# This is done once at module load to speed up evaluation
_triples = list(combinations(range(N), 3))
_i_idx = np.array([t[0] for t in _triples])
_j_idx = np.array([t[1] for t in _triples])
_k_idx = np.array([t[2] for t in _triples])

def _min_triangle_area_vectorized(points):
    """
    Calculate the minimum area of all triangles formed by the points.
    Vectorized for speed.
    """
    x = points[:, 0]
    y = points[:, 1]
    
    v1x = x[_j_idx] - x[_i_idx]
    v1y = y[_j_idx] - y[_i_idx]
    v2x = x[_k_idx] - x[_i_idx]
    v2y = y[_k_idx] - y[_i_idx]
    
    # Area = 0.5 * |cross product|
    cross = v1x * v2y - v1y * v2x
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _convex_hull_area(points):
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """
    Calculate the raw score: min_triangle_area / convex_hull_area.
    """
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    min_tri = _min_triangle_area_vectorized(points)
    return min_tri / hull_area

def _soft_min_score(points, beta=50.0):
    """
    Smooth surrogate for the score using log-sum-exp on the negative areas.
    Minimizing this is equivalent to maximizing the soft-min of the areas.
    """
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-9:
        return 1e6

    x = points[:, 0]
    y = points[:, 1]
    
    v1x = x[_j_idx] - x[_i_idx]
    v1y = y[_j_idx] - y[_i_idx]
    v2x = x[_k_idx] - x[_i_idx]
    v2y = y[_k_idx] - y[_i_idx]
    
    cross = v1x * v2y - v1y * v2x
    areas = 0.5 * np.abs(cross)
    
    # Soft min: -1/beta * log(sum(exp(-beta * areas)))
    # To avoid overflow, we subtract the max
    neg_beta_areas = -beta * areas
    max_val = np.max(neg_beta_areas)
    
    # If all areas are 0, max_val is 0, sum is N, log is log(N), result is -log(N)/beta
    # If max_val is very negative, exp will underflow to 0, which is fine.
    if max_val < -700: # exp(-700) is ~1e-304, close to underflow
        return 1e6
        
    log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
    soft_min_area = -log_sum_exp / beta
    
    score = soft_min_area / hull_area
    return -score

def _regular_polygon(n, center=(0.0, 0.0), radius=1.0, angle_offset=0.0):
    angles = np.linspace(0, 2*np.pi, n, endpoint=False) + angle_offset
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)
    return np.column_stack([x, y])

def _generate_initializations():
    """
    Generate diverse initial configurations, including the specific
    'Barycentric' style structures mentioned in the strategy:
    Regular polygons with radial displacements or concentric rings.
    """
    rng = np.random.RandomState(123)
    inits = []
    
    # 1. Regular 13-gon (Baseline)
    inits.append(_regular_polygon(N))
    
    # 2. Two-ring configurations (Concentric)
    # The strategy suggests looking at radial displacements.
    # These are structured searches over (r_inner, offset)
    for n1, n2 in [(7, 6), (6, 7), (8, 5), (5, 8), (9, 4), (4, 9), (12, 1), (1, 12), (10, 3), (3, 10)]:
        # Search a few radii for the inner ring
        for r2 in [0.2, 0.35, 0.5, 0.65, 0.8]:
            # Offset to maximize separation
            offset = np.pi / max(n2, 1) if n2 > 1 else 0
            pts = np.vstack([_regular_polygon(n1, radius=1.0), _regular_polygon(n2, radius=r2, angle_offset=offset)])
            inits.append(pts)
            
    # 3. Three-ring configurations
    for n1, n2, n3 in [(6, 4, 3), (5, 4, 4), (7, 3, 3), (8, 3, 2), (5, 5, 3), (4, 4, 5)]:
        for r2 in [0.4, 0.6]:
            for r3 in [0.2, 0.35]:
                # Interleaving angles
                o1 = 0
                o2 = np.pi / n2
                o3 = np.pi / n3
                ring1 = _regular_polygon(n1, radius=1.0, angle_offset=o1)
                ring2 = _regular_polygon(n2, radius=r2, angle_offset=o2)
                ring3 = _regular_polygon(n3, radius=r3, angle_offset=o3)
                inits.append(np.vstack([ring1, ring2, ring3]))
                
    # 4. "Barycentric" specific: 12 on hull, 1 inside
    # This is a specific case of the 12-1 split above, but let's add random internal points
    for _ in range(5):
        r_in = rng.uniform(0.1, 0.6)
        theta_in = rng.uniform(0, 2*np.pi)
        inner_pt = np.array([[r_in * np.cos(theta_in), r_in * np.sin(theta_in)]])
        outer_pts = _regular_polygon(12)
        inits.append(np.vstack([outer_pts, inner_pt]))
        
    # 5. Random points in disk (for diversity)
    for _ in range(10):
        r = rng.rand(N) ** 0.5
        theta = rng.rand(N) * 2 * np.pi
        inits.append(np.column_stack([r * np.cos(theta), r * np.sin(theta)]))
        
    return inits

def _refine_soft_min(pts, deadline):
    """
    Refine points using L-BFGS with annealing beta.
    """
    current_pts = pts.copy()
    current_score = _score(current_pts)
    
    # Start with smoother approximations and sharpen
    betas = [10.0, 50.0, 100.0, 200.0, 500.0]
    
    for beta in betas:
        if time.time() > deadline - 10.0:
            break
            
        def soft_obj(x, b=beta):
            if time.time() > deadline:
                return 1e6
            points = x.reshape(-1, 2)
            return _soft_min_score(points, b)
            
        try:
            res = minimize(
                soft_obj, current_pts.flatten(), 
                method='L-BFGS-B',
                options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            cand = res.x.reshape(-1, 2)
            cand_score = _score(cand)
            if cand_score > current_score:
                current_pts = cand
                current_score = cand_score
        except Exception:
            pass
            
    return current_pts

def _refine_nelder_mead(pts, deadline, maxiter=300):
    """
    Final polish on the hard (nonsmooth) objective.
    """
    current_pts = pts.copy()
    current_score = _score(current_pts)
    
    if time.time() > deadline - 5.0:
        return current_pts
        
    def hard_obj(x):
        if time.time() > deadline:
            return 0.0
        points = x.reshape(-1, 2)
        return -_score(points)
        
    try:
        res = minimize(
            hard_obj, current_pts.flatten(), 
            method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True}
        )
        cand = res.x.reshape(-1, 2)
        cand_score = _score(cand)
        if cand_score > current_score:
            current_pts = cand
            current_score = cand_score
    except Exception:
        pass
        
    return current_pts

def construct():
    start_time = time.time()
    deadline = 140.0 # Safety margin for 150s limit

    # 1. Generate diverse initializations
    inits = _generate_initializations()
    
    best_points = inits[0]
    best_score = _score(best_points)
    
    # 2. Quick evaluation of all inits to find a good starting point for refinement
    for init in inits:
        s = _score(init)
        if s > best_score:
            best_score = s
            best_points = init
            
    # 3. Differential Evolution (Global Search)
    # We run DE on the soft objective to help it navigate the landscape, 
    # or just on hard score if fast enough. 
    # Given 13 points (26 dims), DE is expensive. We'll use a small pop.
    if time.time() - start_time < deadline - 40:
        bounds = [(-2.0, 2.0)] * (2 * N)
        
        def de_objective(x):
            if time.time() - start_time > deadline:
                return 0.0
            points = x.reshape(-1, 2)
            return -_score(points)
            
        try:
            # popsize=8, maxiter=15 is very small, but DE is stochastic.
            # We rely on the subsequent refinement.
            res_de = differential_evolution(
                de_objective, bounds, 
                maxiter=15, popsize=8, 
                tol=1e-3, seed=42, 
                polish=False, init='latinhypercube'
            )
            if res_de.fun < -best_score:
                candidate = res_de.x.reshape(-1, 2)
                cand_score = _score(candidate)
                if cand_score > best_score:
                    best_score = cand_score
                    best_points = candidate
        except Exception:
            pass

    # 4. Refine the best candidate found so far (and potentially a few others if time permits)
    # We prioritize the global best.
    refined_best = _refine_soft_min(best_points, deadline)
    s_ref = _score(refined_best)
    if s_ref > best_score:
        best_score = s_ref
        best_points = refined_best
        
    # 5. If time allows, refine a few other diverse high-scoring inits
    # Sort inits by score (excluding the one we just refined)
    # Actually, let's just pick a couple of the "structured" ones (rings) that might be local optima
    candidates_to_refine = []
    for init in inits:
        if np.allclose(init, best_points):
            continue
        s = _score(init)
        candidates_to_refine.append((s, init))
        
    candidates_to_refine.sort(reverse=True, key=lambda x: x[0])
    
    for s, init in candidates_to_refine[:3]: # Try top 3 others
        if time.time() - start_time > deadline - 20:
            break
        # Quick refinement
        refined = _refine_soft_min(init, deadline)
        s_ref = _score(refined)
        if s_ref > best_score:
            best_score = s_ref
            best_points = refined
            
    # 6. Final Polish on the absolute best
    if time.time() - start_time < deadline - 5:
        final_pts = _refine_nelder_mead(best_points, deadline, maxiter=500)
        s_final = _score(final_pts)
        if s_final > best_score:
            best_score = s_final
            best_points = final_pts

    return best_points
