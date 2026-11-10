# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull, Delaunay
from itertools import combinations

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the ratio of the minimum
    triangle area to the convex hull area.
    """
    n = 13
    start_time = time.time()
    time_limit = 590  # Safety margin from 630s

    # Pre-compute triangle indices (i < j < k)
    idx = np.arange(n)
    i, j, k = np.meshgrid(idx, idx, idx, indexing='ij')
    i = i.flatten()
    j = j.flatten()
    k = k.flatten()
    mask = (i < j) & (j < k)
    tri_i, tri_j, tri_k = i[mask], j[mask], k[mask]
    n_tris = len(tri_i)

    def calculate_score(points):
        """
        Returns the score: min_triangle_area / hull_area.
        Returns -inf if invalid.
        """
        if not np.all(np.isfinite(points)):
            return -np.inf
        
        p1 = points[tri_i]
        p2 = points[tri_j]
        p3 = points[tri_k]
        
        v1 = p2 - p1
        v2 = p3 - p1
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        
        min_area = np.min(areas)
        
        try:
            hull = ConvexHull(points)
            hull_area = hull.volume
        except:
            return -np.inf
            
        if hull_area < 1e-10:
            return -np.inf
            
        return min_area / hull_area

    def objective(flat_pts, beta=1000):
        """
        Objective function for minimization.
        Minimizes -min_triangle_area / hull_area.
        """
        pts = flat_pts.reshape(n, 2)
        if not np.all(np.isfinite(pts)):
            return 1e10
            
        p1 = pts[tri_i]
        p2 = pts[tri_j]
        p3 = pts[tri_k]
        
        v1 = p2 - p1
        v2 = p3 - p1
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        
        # Soft min for smoothness
        if beta < 50:
            a_max = np.max(areas)
            exp_terms = np.exp(-beta * (areas - a_max))
            log_sum = np.log(np.sum(exp_terms))
            soft_min = a_max - log_sum / beta
        else:
            soft_min = np.min(areas)
            
        try:
            hull = ConvexHull(pts)
            hull_area = hull.volume
        except:
            return 1e10
            
        if hull_area < 1e-10:
            return 1e10
            
        return -soft_min / hull_area

    # Initializations
    inits = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    init_reg = np.column_stack([np.cos(angles), np.sin(angles)])
    inits.append(init_reg)
    
    # 2. Hexagonal packing
    pts = []
    pts.append([0, 0])
    for i in range(6):
        angle = i * np.pi / 3
        pts.append([np.cos(angle), np.sin(angle)])
    for i in range(6):
        angle = (i + 0.5) * np.pi / 3
        pts.append([2 * np.cos(angle), 2 * np.sin(angle)])
    init_hex = np.array(pts)
    inits.append(init_hex)
    
    # 3. Square-ish layout
    # Try a grid-like structure
    grid_pts = []
    # 4x4 grid minus one corner? 13 points
    # Let's try a 3x3 grid + 4 corners?
    # Or just perturb a square
    square = np.array([
        [0, 0], [1, 0], [2, 0], [3, 0],
        [0, 1], [1, 1], [2, 1], [3, 1],
        [0, 2], [1, 2], [2, 2], [3, 2],
        [0, 3]
    ])
    inits.append(square)
    
    # 4. Random perturbations
    rng = np.random.default_rng(42)
    for _ in range(5):
        init_rand = rng.uniform(-1, 1, (n, 2))
        inits.append(init_rand)

    best_points = None
    best_score = -np.inf
    
    # Evaluate inits
    for init in inits:
        score = calculate_score(init)
        if score > best_score:
            best_score = score
            best_points = init.copy()

    if time.time() - start_time > time_limit:
        return best_points

    bounds = [(-10, 10)] * (2 * n)
    
    # Optimization loop
    # We use a multi-stage approach:
    # 1. Coarse optimization with soft min
    # 2. Fine optimization with hard min
    # 3. Local search around best points
    
    current_points = best_points
    current_flat = current_points.flatten()
    
    # Stage 1: Coarse optimization
    betas = [10, 50, 200]
    
    for beta in betas:
        if time.time() - start_time > time_limit - 30:
            break
            
        res = minimize(
            objective, 
            current_flat, 
            args=(beta,),
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if res.success or res.fun < 1e9:
            new_points = res.x.reshape(n, 2)
            new_score = calculate_score(new_points)
            if new_score > best_score:
                best_score = new_score
                best_points = new_points
                current_flat = new_points.flatten()
            else:
                current_flat = new_points.flatten()

    # Stage 2: Fine optimization with hard min
    if time.time() - start_time < time_limit - 10:
        res = minimize(
            objective, 
            current_flat, 
            args=(1000,),
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        
        if res.success or res.fun < 1e9:
            new_points = res.x.reshape(n, 2)
            new_score = calculate_score(new_points)
            if new_score > best_score:
                best_score = new_score
                best_points = new_points

    # Stage 3: Local search with random restarts around best point
    # Try to improve by perturbing and re-optimizing
    if time.time() - start_time < time_limit - 20:
        for _ in range(5):
            if time.time() - start_time > time_limit - 10:
                break
            # Perturb best points
            noise_scale = 0.1
            perturbed = best_points + rng.normal(0, noise_scale, (n, 2))
            perturbed_flat = perturbed.flatten()
            
            res = minimize(
                objective, 
                perturbed_flat, 
                args=(1000,),
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if res.success or res.fun < 1e9:
                new_points = res.x.reshape(n, 2)
                new_score = calculate_score(new_points)
                if new_score > best_score:
                    best_score = new_score
                    best_points = new_points

    # Ensure validity
    if not np.all(np.isfinite(best_points)):
        return inits[0] # Fallback
        
    return best_points

# EVOLVE-BLOCK-END
