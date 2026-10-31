import numpy as np
import time
import scipy.spatial as spat
import scipy.optimize
import itertools

N = 13

def calculate_score(points):
    """
    Calculate the minimum triangle area divided by the convex hull area.
    Returns 0 if points are collinear or hull area is 0.
    """
    if len(points) < 3:
        return 0.0
    
    # Compute convex hull area
    try:
        hull = spat.ConvexHull(points)
        hull_area = hull.volume # In 2D, volume is area
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0

    pts = points
    n = len(pts)
    
    # Efficient vectorized triangle area calculation
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_flat = i_idx[mask]
    j_flat = j_idx[mask]
    k_flat = k_idx[mask]
    
    v1 = pts[j_flat] - pts[i_flat] 
    v2 = pts[k_flat] - pts[i_flat]
    
    # Determinant in 2D
    det = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(det)
    
    min_area = np.min(areas)
    
    return min_area / hull_area

def _negative_score_flat(params):
    """
    Objective function for differential evolution.
    params: flat array of shape (2*N,)
    """
    pts = params.reshape(N, 2)
    
    # Add a small penalty if points are too close to prevent numerical issues in hull
    # though DE usually handles this.
    # We want to maximize score, so minimize -score.
    score = calculate_score(pts)
    
    # If hull fails, score is 0, which is bad (minimizing -0 = 0).
    # We can add a penalty for non-convex or degenerate configurations if needed,
    # but calculate_score handles exceptions by returning 0.
    
    return -score

def _local_polish(points, max_iter=500, step_size=0.01, deadline=None):
    """
    Coordinate descent polish on the hard objective.
    """
    current_pts = points.copy()
    current_score = calculate_score(current_pts)
    
    if current_score <= 0:
        return current_pts, current_score
    
    # Precompute indices for speed
    i_idx, j_idx, k_idx = np.indices((N, N, N))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_flat = i_idx[mask]
    j_flat = j_idx[mask]
    k_flat = k_idx[mask]
    
    # For each point, we need to know which triangles it participates in.
    # But recalculating all triangles for each point move is O(N^3) per point, 
    # which is 2197 * 13 * 500 ~ 14M ops. This might be slow in Python loops.
    # Let's stick to a simpler local search: random perturbations or small grid search around current best.
    
    # Actually, let's use a simple local search: try moving each point in 4 directions
    # by a small amount, and keep if it improves.
    
    directions = np.array([
        [1, 0], [-1, 0], [0, 1], [0, -1],
        [0.707, 0.707], [-0.707, 0.707], [0.707, -0.707], [-0.707, -0.707]
    ])
    
    step = step_size
    for _ in range(max_iter):
        if deadline and time.time() > deadline - 2.0:
            break
            
        improved = False
        for i in range(N):
            if deadline and time.time() > deadline - 2.0:
                break
                
            original = current_pts[i].copy()
            
            for d in directions:
                candidate = current_pts.copy()
                candidate[i] = original + d * step
                
                # Check if candidate is valid (non-zero hull)
                # We can skip hull check for speed if we trust the score function, 
                # but score function is expensive.
                # Let's just call calculate_score.
                new_score = calculate_score(candidate)
                
                if new_score > current_score:
                    current_pts[i] = candidate[i]
                    current_score = new_score
                    improved = True
                    break # Restart directions for this point? Or continue to next point?
                    # Usually, if we move, we should re-evaluate other directions for this point
                    # or just move to next point. Moving to next point is standard coordinate descent.
        
        if not improved:
            # Shrink step size
            step *= 0.8
            if step < 1e-5:
                break
                
    return current_pts, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 5s safety margin
    
    best_score = -1.0
    best_points = None
    
    # Initial guess: Regular 13-gon or Grid
    # Let's start with a grid-like structure as a good initial population seed
    # But DE will explore the space.
    
    # Bounds for points: Let's constrain to [-2, 2] to keep numbers manageable.
    # The score is scale-invariant, so the absolute scale doesn't matter, 
    # but the relative positions do.
    bounds = [(-2, 2)] * (2 * N)
    
    # Use Differential Evolution
    # popsize=15 is default. 
    # maxiter=100 might be too many if each eval is slow.
    # calculate_score is O(N^3) ~ 2000 ops + hull.
    # 15 * 100 * 2000 = 3M ops per iteration? No, 15 pop * 100 iter = 1500 evals.
    # 1500 * 2000 = 3M flops. This is very fast.
    # Even with overhead, it should finish in < 1s.
    
    # Let's try a more aggressive setting.
    # seed=42 for reproducibility
    
    try:
        result = scipy.optimize.differential_evolution(
            _negative_score_flat, 
            bounds,
            seed=42,
            maxiter=200,
            popsize=20,
            tol=1e-6,
            polish=False # We will do our own polish
        )
        
        best_params = result.x
        best_pts = best_params.reshape(N, 2)
        best_score = calculate_score(best_pts)
        
        if best_score > -1e-9:
            best_points = best_pts.copy()
            
    except Exception as e:
        # Fallback to simple grid if DE fails
        base_coords = []
        for i in range(4):
            for j in range(4):
                if not (i == 3 and j == 3):
                    base_coords.append([i, j])
        best_points = np.array(base_coords, dtype=float)
        best_score = calculate_score(best_points)
    
    # Local Polish
    if best_points is not None:
        polished_pts, polished_score = _local_polish(
            best_points, 
            max_iter=1000, 
            step_size=0.05, 
            deadline=deadline
        )
        
        if polished_score > best_score:
            best_score = polished_score
            best_points = polished_pts
            
    # Fallback if still bad
    if best_points is None or best_score < 1e-6:
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
        best_score = calculate_score(best_points)
        
    return best_points
