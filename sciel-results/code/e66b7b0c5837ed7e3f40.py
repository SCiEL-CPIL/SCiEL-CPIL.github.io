"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import minimize, differential_evolution
from itertools import combinations

# Triangle vertices
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, np.sqrt(3) / 2])

# Pre-compute all pairs of indices for triangles to speed up area calculation
# There are C(11, 3) = 165 triangles
_indices = list(combinations(range(N), 3))

def barycentric_to_cartesian(bary_coords):
    """
    Convert barycentric coordinates (u, v, w) for each point to Cartesian (x, y).
    bary_coords shape: (N, 3) where each row sums to 1 and elements >= 0.
    """
    # x = 0.5 * w + v
    # y = (sqrt(3)/2) * w
    # Note: In standard barycentric for (0,0), (1,0), (0.5, sqrt(3)/2):
    # P = u*A + v*B + w*C
    # x = u*0 + v*1 + w*0.5 = v + 0.5w
    # y = u*0 + v*0 + w*sqrt(3)/2 = w*sqrt(3)/2
    x = bary_coords[:, 1] + 0.5 * bary_coords[:, 2]
    y = (np.sqrt(3) / 2) * bary_coords[:, 2]
    return np.column_stack((x, y))

def cartesian_to_barycentric(points):
    """
    Convert Cartesian points to barycentric coordinates.
    points shape: (N, 2)
    Returns: (N, 3)
    """
    x = points[:, 0]
    y = points[:, 1]
    
    # w = 2 * y / sqrt(3)
    w = 2 * y / np.sqrt(3)
    # v = x - 0.5 * w
    v = x - 0.5 * w
    # u = 1 - v - w
    u = 1 - v - w
    
    # Clip to ensure validity (small numerical errors might push outside)
    u = np.clip(u, 0, 1)
    v = np.clip(v, 0, 1)
    w = np.clip(w, 0, 1)
    
    # Normalize to sum to 1
    sum_uvw = u + v + w
    sum_uvw[sum_uvw == 0] = 1
    u /= sum_uvw
    v /= sum_uvw
    w /= sum_uvw
    
    return np.column_stack((u, v, w))

def compute_min_area_cartesian(points):
    """
    Calculate the minimum area of any triangle formed by 3 points.
    points shape: (N, 2)
    """
    min_area = float('inf')
    for i, j, k in _indices:
        p1 = points[i]
        p2 = points[j]
        p3 = points[k]
        # Area = 0.5 * | (x2-x1)(y3-y1) - (x3-x1)(y2-y1) |
        dx12 = p2[0] - p1[0]
        dy12 = p2[1] - p1[1]
        dx13 = p3[0] - p1[0]
        dy13 = p3[1] - p1[1]
        area = 0.5 * abs(dx12 * dy13 - dx13 * dy12)
        if area < min_area:
            min_area = area
    return min_area

def objective_min_area_flat(flat_params):
    """
    Objective function for minimizers.
    flat_params: 2N vector representing Cartesian coordinates.
    We map to barycentric to ensure constraints, then back to Cartesian for area calc.
    Actually, it's better to optimize in barycentric space directly to avoid projection issues.
    Let's switch strategy: Optimize in barycentric space (3N params).
    """
    bary = flat_params.reshape(N, 3)
    # Normalize barycentric coords to sum to 1
    bary = bary / np.sum(bary, axis=1, keepdims=True)
    # Clip to non-negative
    bary = np.clip(bary, 0, 1)
    # Re-normalize after clipping
    bary = bary / np.sum(bary, axis=1, keepdims=True)
    
    pts = barycentric_to_cartesian(bary)
    area = compute_min_area_cartesian(pts)
    return -area

def objective_smooth_flat(flat_params):
    """
    Smooth objective using log-sum-exp approximation of the min area.
    This helps gradient-free/gradient-based methods navigate the non-smooth landscape.
    """
    bary = flat_params.reshape(N, 3)
    bary = bary / np.sum(bary, axis=1, keepdims=True)
    bary = np.clip(bary, 0, 1)
    bary = bary / np.sum(bary, axis=1, keepdims=True)
    
    pts = barycentric_to_cartesian(bary)
    
    areas = []
    for i, j, k in _indices:
        p1 = pts[i]
        p2 = pts[j]
        p3 = pts[k]
        dx12 = p2[0] - p1[0]
        dy12 = p2[1] - p1[1]
        dx13 = p3[0] - p1[0]
        dy13 = p3[1] - p1[1]
        area = 0.5 * abs(dx12 * dy13 - dx13 * dy12)
        areas.append(area)
        
    areas = np.array(areas)
    # Soft min: -log(sum(exp(-areas/beta))) * beta
    # For maximization of min, we minimize -soft_min.
    # soft_min(areas) approximates min(areas) as beta -> 0
    beta = 0.01
    # Handle numerical stability
    max_a = np.max(areas)
    exp_terms = np.exp((max_a - areas) / beta)
    soft_min = max_a - beta * np.log(np.sum(exp_terms))
    
    return -soft_min

def generate_random_barycentric():
    """Generate random points in barycentric coordinates."""
    bary = np.random.dirichlet(np.ones(N))
    return bary.flatten()

def generate_grid_barycentric():
    """Generate a grid-based start in barycentric coordinates."""
    # Simple 3x3x3 grid subset or similar
    # We need 11 points.
    # Let's try a specific pattern known to be good:
    # Vertices, midpoints, center, etc.
    # A=(1,0,0), B=(0,1,0), C=(0,0,1) in barycentric
    # Midpoints: (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)
    # Center: (1/3, 1/3, 1/3)
    # This gives 7 points.
    # Add 4 more.
    
    pts_bary = [
        [1.0, 0.0, 0.0], # A
        [0.0, 1.0, 0.0], # B
        [0.0, 0.0, 1.0], # C
        [0.5, 0.5, 0.0], # Mid AB
        [0.5, 0.0, 0.5], # Mid AC
        [0.0, 0.5, 0.5], # Mid BC
        [1/3.0, 1/3.0, 1/3.0], # Center
        [0.25, 0.25, 0.5],
        [0.5, 0.25, 0.25],
        [0.25, 0.5, 0.25],
        [0.4, 0.4, 0.2]
    ]
    return np.array(pts_bary).flatten()

def local_refine(params):
    """Perform local refinement using Nelder-Mead on the smooth objective."""
    # Ensure params are valid barycentric
    bary = params.reshape(N, 3)
    bary = bary / np.sum(bary, axis=1, keepdims=True)
    bary = np.clip(bary, 0, 1)
    bary = bary / np.sum(bary, axis=1, keepdims=True)
    
    res = minimize(
        objective_smooth_flat, 
        bary.flatten(), 
        method='Nelder-Mead', 
        options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
    )
    return res.x

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_params = None
    best_score = -1.0
    
    # Initializers
    initializers = [
        generate_grid_barycentric,
        generate_random_barycentric,
        generate_random_barycentric,
        generate_random_barycentric
    ]
    
    # Try Differential Evolution first for global search
    # DE works in Cartesian or Barycentric? Barycentric is constrained.
    # DE can handle bounds.
    # Bounds for barycentric coords are [0, 1] for each coord, but they must sum to 1.
    # Standard DE doesn't handle linear constraints like sum=1 well without projection.
    # We can use DE in Cartesian space [0, 1] x [0, sqrt(3)/2] and project.
    # Or use DE on 3N barycentric coords with bounds [0,1] and normalize in objective.
    
    # Let's use DE on barycentric coords (3N dimensions).
    # Bounds: [0, 1] for all 33 params.
    # The objective function normalizes them.
    
    try:
        # DE is slow, so limit it.
        # popsize=10, maxiter=50
        de_result = differential_evolution(
            objective_min_area_flat, 
            bounds=[(0.0, 1.0)] * (3 * N), 
            seed=42, 
            maxiter=50, 
            popsize=15,
            tol=1e-8,
            polish=False, # We will polish manually
            init='sobol'
        )
        
        if time.time() < deadline:
            params = de_result.x
            # Refine
            params = local_refine(params)
            bary = params.reshape(N, 3)
            bary = bary / np.sum(bary, axis=1, keepdims=True)
            bary = np.clip(bary, 0, 1)
            bary = bary / np.sum(bary, axis=1, keepdims=True)
            pts = barycentric_to_cartesian(bary)
            score = compute_min_area_cartesian(pts)
            
            if score > best_score:
                best_score = score
                best_params = params
                
    except Exception:
        pass
    
    # Local refinement from multiple starts
    for init_func in initializers:
        if time.time() > deadline - 10.0: # Need 10s for final checks
            break
            
        try:
            params = init_func()
            # Refine
            params = local_refine(params)
            
            bary = params.reshape(N, 3)
            bary = bary / np.sum(bary, axis=1, keepdims=True)
            bary = np.clip(bary, 0, 1)
            bary = bary / np.sum(bary, axis=1, keepdims=True)
            pts = barycentric_to_cartesian(bary)
            score = compute_min_area_cartesian(pts)
            
            if score > best_score:
                best_score = score
                best_params = params
                
        except Exception:
            continue
            
    # If no valid points found, return a default
    if best_params is None:
        # Return grid start
        params = generate_grid_barycentric()
        bary = params.reshape(N, 3)
        bary = bary / np.sum(bary, axis=1, keepdims=True)
        bary = np.clip(bary, 0, 1)
        bary = bary / np.sum(bary, axis=1, keepdims=True)
        return barycentric_to_cartesian(bary)
        
    # Final cleanup: ensure points are strictly inside or on boundary
    bary = best_params.reshape(N, 3)
    bary = bary / np.sum(bary, axis=1, keepdims=True)
    bary = np.clip(bary, 0, 1)
    bary = bary / np.sum(bary, axis=1, keepdims=True)
    pts = barycentric_to_cartesian(bary)
    
    return pts

# EVOLVE-BLOCK-END
