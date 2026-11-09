import numpy as np
from scipy.optimize import minimize

def calculate_score(angles):
    # angles: array of 20 angles
    # Ensure they are sorted and span 2*pi
    # The optimizer will handle the constraints, but we should be robust
    
    n = len(angles)
    x = np.cos(angles)
    y = np.sin(angles)
    pts = np.stack([x, y], axis=1)
    
    # Check convexity (turning angles)
    # Cross product of consecutive edges
    # For CCW order, cross product should be positive
    # If any cross product is <= 0, it's not strictly convex
    # We assume the optimizer keeps it convex, but we add a penalty
    convex_penalty = 0.0
    for i in range(n):
        p0 = pts[i]
        p1 = pts[(i+1)%n]
        p2 = pts[(i+2)%n]
        v1 = p1 - p0
        v2 = p2 - p1
        cross = v1[0]*v2[1] - v1[1]*v2[0]
        if cross <= 0:
            convex_penalty += 100.0 # Large penalty
    
    if convex_penalty > 0:
        return -convex_penalty

    # Normalize
    centroid = np.mean(pts, axis=0)
    pts = pts - centroid
    dists = np.linalg.norm(pts, axis=1)
    mean_dist = np.mean(dists)
    if mean_dist <= 0.01:
        return 0.0
    pts = pts / mean_dist
    
    # Calculate score for each vertex
    min_score = 1.0
    for i in range(n):
        vi = pts[i]
        # Distances to other vertices
        dists_i = np.linalg.norm(pts - vi, axis=1)
        # Remove self distance (index i)
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        dists_i = dists_i[mask]
        
        # Check min distance
        if np.min(dists_i) < 0.05:
            return 0.0 # Or a very low score for this vertex
        
        # Sort
        sorted_dists = np.sort(dists_i)
        
        # Find min gap
        gaps = np.diff(sorted_dists)
        k = np.argmin(gaps)
        D = (sorted_dists[k] + sorted_dists[k+1]) / 2.0
        
        if D < 0.1:
            return 0.0
            
        # Select 4 distances
        if k == 0:
            idxs = [0, 1, 2, 3]
        elif k == 17: # n-1-1 = 18-1 = 17? 
            # n=20, 19 dists. indices 0..18.
            # if k=17, gap is between 17 and 18.
            # take 15..18
            idxs = [15, 16, 17, 18]
        else:
            idxs = [k-1, k, k+1, k+2]
            
        selected_dists = sorted_dists[idxs]
        
        # Raw score
        ratios = np.maximum(selected_dists / D, D / selected_dists)
        raw = np.mean(ratios)
        score = min(1.0, 1.0 / raw)
        
        if score < min_score:
            min_score = score
            
    return min_score

def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    # Initial guess: regular polygon gaps
    delta0 = np.full(n, 2.0 * np.pi / n)
    
    # Objective: maximize min score -> minimize -min score
    def objective(deltas):
        # Ensure deltas are positive and sum to 2*pi
        # SLSQP will handle constraints
        angles = np.cumsum(deltas)
        angles = np.roll(angles, -1) # theta_0 = 0? 
        # Let's define theta_0 = 0
        # theta_i = sum(delta_0..delta_{i-1})
        # If we pass deltas, we compute angles
        # But the objective function in minimize takes the variables
        # Let's make the variables the angles directly, with constraints
        pass

    # Better: Optimize angles directly with constraints
    # x: 20 angles
    # Constraints: 
    # 1. sum of (x_{i+1}-x_i) = 2*pi (cyclic)
    # 2. x_{i+1} > x_i (sorted)
    
    # This is tricky with SLSQP on cyclic variables.
    # Let's optimize the gaps delta_i.
    # x: delta_0..delta_19
    # Constraints:
    # sum(x) = 2*pi
    # x_i > 0
    
    def obj(deltas):
        # deltas: array of 20
        # angles: theta_0=0, theta_i = sum(deltas[:i])
        angles = np.zeros(n)
        for i in range(1, n):
            angles[i] = np.sum(deltas[:i])
        return -calculate_score(angles)
        
    def constraint_sum(deltas):
        return np.sum(deltas) - 2.0 * np.pi

    constraints = [{'type': 'eq', 'fun': constraint_sum}]
    bounds = [(1e-5, 2*np.pi)] * n # Small positive lower bound
    
    x0 = np.full(n, 2.0 * np.pi / n)
    
    # Run optimization
    # Nelder-Mead doesn't support constraints well.
    # SLSQP does.
    # But SLSQP might be slow or get stuck.
    # Let's try a few iterations or a different method.
    
    # Given the time limit, we might need a faster heuristic.
    # Let's try a simple local search.
    
    best_deltas = x0.copy()
    best_score = -obj(best_deltas)
    
    # Simple hill climbing
    # We can use scipy.optimize.minimize with method='Powell' or 'Nelder-Mead' on the projected space?
    # No, let's stick to SLSQP but with a limited number of iterations.
    
    try:
        res = minimize(obj, x0, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 500, 'ftol': 1e-6})
        if res.success and -res.fun > best_score:
            best_deltas = res.x
            best_score = -res.fun
    except Exception as e:
        pass
        
    # If SLSQP failed or didn't improve, try a few random perturbations?
    # Or just return the regular polygon if no improvement.
    
    angles = np.zeros(n)
    for i in range(1, n):
        angles[i] = np.sum(best_deltas[:i])
        
    x = np.cos(angles)
    y = np.sin(angles)
    polygon = np.stack([x, y], axis=1)
    
    return polygon
