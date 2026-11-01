"""
Caging formulation: treat the 12 inner hexagons as a single convex polytope
(convex hull of all 72 vertices) and optimize the outer hexagon to contain it.
Uses dual relationship between support functions for exact containment.
Optimizes positions and orientations via BFGS with a differentiable objective.
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 12
DEADLINE = 140.0

# Unit hexagon vertices (flat-top: vertices at 0,60,120,...)
UNIT_VERTS = np.array([
    [np.cos(np.radians(60 * i)), np.sin(np.radians(60 * i))] for i in range(6)
])

# Outward normals for outer hexagon edges (at 30,90,150,210,270,330)
OUTER_NORMALS = np.array([
    [np.cos(np.radians(60 * i + 30)), np.sin(np.radians(60 * i + 30))] for i in range(6)
])

def get_all_vertices(params):
    """
    params: (N, 3) array of [x, y, angle_deg]
    Returns: (72, 2) array of all vertices
    """
    n = len(params)
    all_verts = np.zeros((n * 6, 2))
    for i in range(n):
        x, y, a_deg = params[i]
        a_rad = np.radians(a_deg)
        c, s = np.cos(a_rad), np.sin(a_rad)
        R = np.array([[c, -s], [s, c]])
        all_verts[i*6:(i+1)*6] = UNIT_VERTS @ R.T + np.array([x, y])
    return all_verts

def hexagon_support(verts, normal):
    """Support function: max dot product of vertices with normal."""
    return np.max(verts @ normal)

def compute_outer_side_from_hull(hull_verts, outer_center, outer_angle):
    """
    Given the convex hull vertices, compute minimum outer hexagon side length.
    The outer hexagon has center, angle, and we minimize its size.
    """
    oa_rad = np.radians(outer_angle)
    c_o, s_o = np.cos(-oa_rad), np.sin(-oa_rad)
    R_o = np.array([[c_o, -s_o], [s_o, c_o]])
    
    v_rel = hull_verts - np.array(outer_center)
    v_rot = v_rel @ R_o.T
    
    # Project onto all 6 outward normals and their negatives
    projs = v_rot @ OUTER_NORMALS.T  # (M, 6)
    # For containment, we need max_k |v . n_k| <= apothem
    # But actually for a regular hexagon centered at outer_center with angle outer_angle,
    # containment means for each vertex v: max_k |v_rel_rot . n_k| <= apothem
    abs_projs = np.abs(projs)
    max_extent = np.max(abs_projs)
    
    # side = apothem / (sqrt(3)/2)
    return max_extent / (np.sqrt(3) / 2)

def objective(params, outer_center, outer_angle):
    """
    Objective: minimize outer side length needed to contain all inner hexagons.
    We use the convex hull for efficiency.
    """
    all_verts = get_all_vertices(params)
    
    # Compute convex hull
    try:
        hull = ConvexHull(all_verts)
        hull_verts = all_verts[hull.vertices]
    except Exception:
        # Fallback: use all vertices
        hull_verts = all_verts
    
    side = compute_outer_side_from_hull(hull_verts, outer_center, outer_angle)
    
    # Add small regularization to prevent degeneracy
    centers = params[:, :2]
    spread = np.mean(np.sum((centers - centers.mean(axis=0))**2, axis=1))
    
    return side + 1e-6 * spread

def sat_check(verts_i, verts_j, tol=1e-7):
    """
    Check if two convex polygons are disjoint using SAT.
    Returns True if disjoint (no overlap).
    """
    edges_i = np.roll(verts_i, -1, axis=0) - verts_i
    edges_j = np.roll(verts_j, -1, axis=0) - verts_j
    
    axes = np.vstack([edges_i, edges_j])
    norms = np.linalg.norm(axes, axis=1, keepdims=True) + 1e-15
    axes = axes / norms
    
    proj_i = verts_i @ axes.T
    proj_j = verts_j @ axes.T
    
    sep = np.minimum(
        proj_i.min(axis=0) - proj_j.max(axis=0),
        proj_j.min(axis=0) - proj_i.max(axis=0)
    )
    
    return np.max(sep) >= -tol

def check_no_overlap(params, tol=1e-7):
    """Check that no two inner hexagons overlap."""
    all_verts = get_all_vertices(params)
    n = len(params)
    for i in range(n):
        v_i = all_verts[i*6:(i+1)*6]
        for j in range(i+1, n):
            v_j = all_verts[j*6:(j+1)*6]
            if not sat_check(v_i, v_j, tol):
                return False
    return True

def generate_initial_configs():
    """Generate diverse initial configurations."""
    configs = []
    
    a1 = np.array([np.sqrt(3), 0.0])
    a2 = np.array([np.sqrt(3)/2.0, 1.5])
    
    def make_config(coords, angle=0.0):
        centers = []
        for i, j in coords:
            pos = i * a1 + j * a2
            centers.append([pos[0], pos[1], angle])
        return np.array(centers)
    
    # Various hexagonal lattice patterns
    patterns = [
        # 3-4-3-2
        [(0,0),(1,0),(2,0),(-1,1),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3)],
        # 2-4-4-2
        [(0,0),(1,0),(-1,1),(0,1),(1,1),(2,1),(-1,2),(0,2),(1,2),(2,2),(0,3),(1,3)],
        # 1-3-4-3-1
        [(0,0),(-1,1),(0,1),(1,1),(-2,2),(-1,2),(0,2),(1,2),(-1,3),(0,3),(1,3),(0,4)],
        # 4-4-4
        [(-1,0),(0,0),(1,0),(2,0),(-1,1),(0,1),(1,1),(2,1),(-1,2),(0,2),(1,2),(2,2)],
        # 3-3-3-3
        [(-1,0),(0,0),(1,0),(-1,1),(0,1),(1,1),(-1,2),(0,2),(1,2),(-1,3),(0,3),(1,3)],
        # 2-3-4-3
        [(0,0),(1,0),(-1,1),(0,1),(1,1),(-2,2),(-1,2),(0,2),(1,2),(-1,3),(0,3),(1,3)],
        # 6-6 two rows
        [(-1,0),(0,0),(1,0),(-1,1),(0,1),(1,1),(-2,2),(-1,2),(0,2),(-1,3),(0,3),(1,3)],
        # Centered cluster
        [(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,-1),(1,-1),(-1,1),(2,0),(0,2),(0,-2)],
    ]
    
    for p in patterns:
        configs.append(make_config(p))
    
    # Add rotated versions
    for p in patterns[:4]:
        configs.append(make_config(p, angle=30.0))
        configs.append(make_config(p, angle=60.0))
    
    # Random perturbations
    np.random.seed(123)
    for _ in range(6):
        base = configs[np.random.randint(len(configs))]
        perturbed = base + np.random.uniform(-0.2, 0.2, size=(N, 3))
        configs.append(perturbed)
    
    return configs

def optimize_config(initial_params, time_budget, outer_center=(0.0, 0.0), outer_angle=0.0):
    """
    Optimize a single configuration using BFGS.
    """
    start = time.time()
    
    def obj(x):
        if time.time() - start > time_budget:
            return 1e10
        try:
            return objective(x, outer_center, outer_angle)
        except Exception:
            return 1e10
    
    try:
        res = minimize(
            obj, initial_params, method='BFGS',
            options={'maxiter': 100, 'gtol': 1e-6, 'disp': False}
        )
        return res.x
    except Exception:
        return initial_params

def construct():
    start_time = time.time()
    
    best_side = float('inf')
    best_params = None
    
    configs = generate_initial_configs()
    
    # Shuffle for diversity
    np.random.shuffle(configs)
    
    for i, config in enumerate(configs):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 10:
            break
        
        # Time budget per config
        remaining = DEADLINE - elapsed - 5
        budget_per = max(2.0, remaining / max(1, len(configs) - i))
        budget = min(12.0, budget_per)
        
        if budget < 1.0:
            continue
        
        # Optimize
        try:
            optimized = optimize_config(config, budget)
        except Exception:
            optimized = config
        
        # Validate: check no overlaps
        if not check_no_overlap(optimized):
            # Try original
            if check_no_overlap(config):
                side = compute_outer_side_from_hull(get_all_vertices(config), (0.0, 0.0), 0.0)
                if side < best_side:
                    best_side = side
                    best_params = config
            continue
        
        # Compute side
        side = compute_outer_side_from_hull(get_all_vertices(optimized), (0.0, 0.0), 0.0)
        
        if side < best_side:
            best_side = side
            best_params = optimized.copy()
    
    # Final result
    if best_params is not None:
        inner = [[float(p[0]), float(p[1]), float(p[2])] for p in best_params]
        return inner, (0.0, 0.0), float(best_side), 0.0
    
    # Fallback: circular arrangement
    inner_default = []
    for i in range(N):
        theta = 2 * np.pi * i / N
        x = 3.5 * np.cos(theta)
        y = 3.5 * np.sin(theta)
        inner_default.append([x, y, 0.0])
    
    side = compute_outer_side_from_hull(get_all_vertices(np.array(inner_default)), (0.0, 0.0), 0.0)
    return inner_default, (0.0, 0.0), float(side), 0.0

if __name__ == "__main__":
    result = construct()
    print(f"Best Side: {result[2]}")
