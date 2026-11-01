import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
import time
import sys

N = 14

# Precompute triangle indices for vectorized area calculation
# C(14, 3) = 364 triangles
_IDX = np.array([(i, j, k) for i in range(N) for j in range(i + 1, N) for k in range(j + 1, N)])
_I, _J, _K = _IDX[:, 0], _IDX[:, 1], _IDX[:, 2]

def compute_score(points):
    """
    Compute min_triangle_area / convex_hull_area.
    points: (N, 2) array
    """
    if len(points) < 3:
        return 0.0
    
    # Check for degenerate hull
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-10:
            return 0.0
    except Exception:
        return 0.0
        
    # Vectorized triangle area calculation
    p_i = points[_I]
    p_j = points[_J]
    p_k = points[_K]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    if min_area < 1e-10:
        return 0.0
        
    return min_area / hull_area

def neg_score(x):
    """Objective function for minimizers: -score"""
    points = x.reshape((N, 2))
    return -compute_score(points)

def ring_to_points(params):
    """
    Convert ring parameters to point coordinates.
    params: [r1, r2, r3, r4, th1, th2, th3, th4, n1, n2, n3, n4]
    n_i are counts, r_i radii, th_i angular offsets.
    """
    r1, r2, r3, r4, th1, th2, th3, th4, n1, n2, n3, n4 = params
    
    # Round counts to integers and adjust to sum to N
    n1 = int(round(n1))
    n2 = int(round(n2))
    n3 = int(round(n3))
    n4 = int(round(n4))
    
    # Adjust to ensure sum is N
    total = n1 + n2 + n3 + n4
    while total > N:
        # Reduce the largest count
        counts = [n1, n2, n3, n4]
        max_idx = counts.index(max(counts))
        if max_idx == 0: n1 -= 1
        elif max_idx == 1: n2 -= 1
        elif max_idx == 2: n3 -= 1
        else: n4 -= 1
        total -= 1
    while total < N:
        # Add to the smallest count (or just n4)
        counts = [n1, n2, n3, n4]
        min_idx = counts.index(min(counts))
        if min_idx == 0: n1 += 1
        elif min_idx == 1: n2 += 1
        elif min_idx == 2: n3 += 1
        else: n4 += 1
        total += 1
        
    points = []
    
    if n1 > 0:
        angles = th1 + np.linspace(0, 2*np.pi, n1, endpoint=False)
        points.append(np.column_stack((r1*np.cos(angles), r1*np.sin(angles))))
    if n2 > 0:
        angles = th2 + np.linspace(0, 2*np.pi, n2, endpoint=False)
        points.append(np.column_stack((r2*np.cos(angles), r2*np.sin(angles))))
    if n3 > 0:
        angles = th3 + np.linspace(0, 2*np.pi, n3, endpoint=False)
        points.append(np.column_stack((r3*np.cos(angles), r3*np.sin(angles))))
    if n4 > 0:
        angles = th4 + np.linspace(0, 2*np.pi, n4, endpoint=False)
        points.append(np.column_stack((r4*np.cos(angles), r4*np.sin(angles))))
        
    if not points:
        return np.zeros((N, 2))
        
    result = np.vstack(points)
    
    # Ensure exactly N points
    if len(result) < N:
        # Fill with center points
        result = np.vstack([result, np.zeros((N - len(result), 2))])
    elif len(result) > N:
        result = result[:N]
        
    return result

def neg_ring_score(params):
    """Objective for differential evolution on ring parameters."""
    try:
        points = ring_to_points(params)
        return -compute_score(points)
    except Exception:
        return 0.0

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin for 150s limit
    
    best_points = None
    best_score = -np.inf
    rng = np.random.default_rng(42)
    
    def update_best(points):
        nonlocal best_points, best_score
        s = compute_score(points)
        if s > best_score:
            best_score = s
            best_points = points.copy()
        return s

    # --- Phase 1: Global Search with Differential Evolution on Ring Parameters ---
    # Parameters: r1, r2, r3, r4 (radii), th1, th2, th3, th4 (angles), n1, n2, n3, n4 (counts)
    # Constraints: n1+n2+n3+n4 = 14. DE doesn't handle equality constraints well directly, 
    # but our ring_to_points handles it by rounding/adjusting. 
    # We bound n_i to [1, 10] to ensure diversity.
    
    bounds = [
        (0.5, 1.5),  # r1 (outer)
        (0.1, 1.0),  # r2
        (0.0, 0.6),  # r3
        (0.0, 0.3),  # r4 (inner)
        (0, 2*np.pi),  # th1
        (0, 2*np.pi),  # th2
        (0, 2*np.pi),  # th3
        (0, 2*np.pi),  # th4
        (1, 10),  # n1
        (1, 10),  # n2
        (1, 10),  # n3
        (1, 10),  # n4
    ]
    
    # Run DE for a few iterations
    # popsize=15, maxiter=50 is roughly 750 evaluations. Each eval is fast (vectorized).
    # Let's try to run 2-3 DE instances if time permits.
    
    de_restarts = 3
    for i in range(de_restarts):
        if time.time() - start_time > deadline - 40:
            break
            
        try:
            # Use different seeds for diversity
            seed = 42 + i * 1000
            res = differential_evolution(
                neg_ring_score,
                bounds,
                seed=seed,
                maxiter=60,
                popsize=15,
                mutation=(0.3, 1.2),
                recombination=0.8,
                tol=1e-6,
                polish=False, # We will do local polish separately on the resulting points
                init='latinhypercube'
            )
            
            if res.fun < 0: # Valid score found
                best_params = res.x
                points = ring_to_points(best_params)
                update_best(points)
                
        except Exception as e:
            pass

    # --- Phase 2: Local Refinement on Coordinates ---
    # Take the best points found so far (from DE or initial regular polygon)
    # and refine them using a local optimizer.
    
    if best_points is None:
        # Fallback: Regular polygon
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
        update_best(best_points)
        
    # We will try to refine the best points found.
    # We might also try refining a few other good candidates if we tracked them, 
    # but for simplicity, we refine the current best and a few perturbations.
    
    candidates_to_refine = [best_points]
    
    # Add some structured perturbations to the best points
    for _ in range(3):
        if time.time() - start_time > deadline - 20:
            break
        perturbed = best_points + rng.normal(0, 0.05, (N, 2))
        candidates_to_refine.append(perturbed)
        
    # Also add a standard 2-ring configuration as a candidate
    a_out = np.linspace(0, 2*np.pi, 7, endpoint=False)
    a_in = a_out + np.pi/7
    two_ring = np.vstack((
        np.column_stack((1.0*np.cos(a_out), 1.0*np.sin(a_out))),
        np.column_stack((0.5*np.cos(a_in), 0.5*np.sin(a_in)))
    ))
    candidates_to_refine.append(two_ring)
    
    for cand in candidates_to_refine:
        if time.time() - start_time > deadline - 10:
            break
            
        x0 = cand.flatten()
        
        try:
            # Powell is robust for non-smooth objectives (min of areas)
            res = minimize(
                neg_score, 
                x0, 
                method='Powell', 
                options={'maxiter': 200, 'xtol': 1e-6, 'ftol': 1e-6}
            )
            
            if res.fun < -best_score:
                best_points = res.x.reshape((N, 2))
                best_score = -res.fun
                
        except Exception:
            pass

    return best_points

# EVOLVE-BLOCK-END
