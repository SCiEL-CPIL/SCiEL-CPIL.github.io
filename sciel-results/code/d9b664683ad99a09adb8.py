"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by three points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]))
                if area < min_area:
                    min_area = area
    return min_area


def _min_pairwise_distance(points):
    """Compute the minimum pairwise distance."""
    n = len(points)
    min_d = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(points[i] - points[j])
            if d < min_d:
                min_d = d
    return min_d


def _barycentric_to_cartesian(bary):
    """Convert barycentric coordinates (u, v) to Cartesian for triangle (0,0), (1,0), (0.5, sqrt(3)/2).
    The third barycentric coord is 1-u-v. Must have u>=0, v>=0, u+v<=1."""
    h = np.sqrt(3) / 2
    x = bary[0] + 0.5 * bary[1]
    y = h * bary[1]
    return np.array([x, y])


def _cartesian_to_barycentric(pt):
    """Convert Cartesian to barycentric coordinates."""
    h = np.sqrt(3) / 2
    x, y = pt
    v = y / h
    u = x - 0.5 * v
    w = 1.0 - u - v
    return np.array([u, v, w])


def _project_to_triangle(p):
    """Project point p onto the equilateral triangle (0,0), (1,0), (0.5, sqrt(3)/2)."""
    x, y = p
    h = np.sqrt(3) / 2
    
    if y < 0:
        y = 0
    if y > h:
        y = h
    
    # Left boundary: x >= y / sqrt(3)
    if x < y / np.sqrt(3):
        x = y / np.sqrt(3)
    
    # Right boundary: x <= 1 - y / sqrt(3)
    if x > 1.0 - y / np.sqrt(3):
        x = 1.0 - y / np.sqrt(3)
    
    return np.array([x, y])


def construct():
    """Construct 11 points in the unit equilateral triangle to maximize minimum triangle area."""
    h = np.sqrt(3) / 2
    start_time = time.time()
    
    # Phase 1: Lattice search in barycentric coordinates
    # We search over a 1D lattice in the space of barycentric coordinates
    # Use a fine grid and select points to maximize minimum pairwise distance
    
    # Generate candidate points on a lattice in barycentric space
    # Barycentric coords: (u, v) with u >= 0, v >= 0, u + v <= 1
    # Map to a 2D grid in (u, v) space
    
    grid_res = 50
    us = np.linspace(0, 1, grid_res)
    vs = np.linspace(0, 1, grid_res)
    
    candidates_bary = []
    candidates_cart = []
    for u in us:
        for v in vs:
            if u >= -1e-9 and v >= -1e-9 and u + v <= 1.0 + 1e-9:
                b = np.array([max(0, u), max(0, v), max(0, 1 - u - v)])
                c = _barycentric_to_cartesian(b)
                candidates_bary.append(b)
                candidates_cart.append(c)
    
    candidates_bary = np.array(candidates_bary)
    candidates_cart = np.array(candidates_cart)
    n_cand = len(candidates_cart)
    
    # Phase 2: Use furthest-point sampling to find good packings
    # We'll do multiple restarts and keep the top packings
    
    n_restarts = 200
    packings = []
    
    for r in range(n_restarts):
        if time.time() - start_time > 80:
            break
            
        np.random.seed(2000 + r)
        picked_indices = []
        remaining_indices = list(range(n_cand))
        
        # Start with a random point
        start_idx = np.random.choice(remaining_indices)
        picked_indices.append(start_idx)
        remaining_indices.remove(start_idx)
        
        while len(picked_indices) < N:
            picked_pts = candidates_cart[picked_indices]
            diff = candidates_cart[remaining_indices][:, np.newaxis, :] - picked_pts[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            min_dists = np.min(dists, axis=1)
            
            best_idx_local = np.argmax(min_dists)
            best_idx_global = remaining_indices[best_idx_local]
            picked_indices.append(best_idx_global)
            remaining_indices.remove(best_idx_global)
        
        pts = candidates_cart[picked_indices]
        min_dist = _min_pairwise_distance(pts)
        packings.append((min_dist, pts))
    
    # Sort by min_dist descending and keep top 10
    packings.sort(key=lambda x: x[0], reverse=True)
    top_packings = [p[1] for p in packings[:10]]
    
    best_final_points = top_packings[0].copy()
    best_final_area = _min_triangle_area(best_final_points)
    
    # Phase 3: Local polish to maximize minimum triangle area
    def neg_min_area(x_flat):
        pts = x_flat.reshape(N, 2)
        pts_clipped = np.array([_project_to_triangle(p) for p in pts])
        return -_min_triangle_area(pts_clipped)
    
    # Polish each top packing
    for init_pts in top_packings:
        if time.time() - start_time > 100:
            break
            
        x0 = init_pts.flatten()
        
        # Run Nelder-Mead with tighter tolerances
        result = minimize(neg_min_area, x0, method='Nelder-Mead', 
                          options={'maxiter': 5000, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True})
        
        trial_points = result.x.reshape(N, 2)
        trial_points = np.array([_project_to_triangle(p) for p in trial_points])
        trial_area = _min_triangle_area(trial_points)
        
        if trial_area > best_final_area:
            best_final_area = trial_area
            best_final_points = trial_points
    
    # Phase 4: Additional refinement - try perturbing the best solution
    if time.time() - start_time < 110:
        for _ in range(5):
            if time.time() - start_time > 110:
                break
            # Add small random perturbation
            perturbed = best_final_points + np.random.normal(0, 0.01, size=(N, 2))
            perturbed = np.array([_project_to_triangle(p) for p in perturbed])
            
            x0 = perturbed.flatten()
            result = minimize(neg_min_area, x0, method='Nelder-Mead', 
                              options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True})
            
            trial_points = result.x.reshape(N, 2)
            trial_points = np.array([_project_to_triangle(p) for p in trial_points])
            trial_area = _min_triangle_area(trial_points)
            
            if trial_area > best_final_area:
                best_final_area = trial_area
                best_final_points = trial_points
    
    return best_final_points

# EVOLVE-BLOCK-END
