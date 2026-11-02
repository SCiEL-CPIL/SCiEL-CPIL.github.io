"""Rewritten program using SLSQP with barycentric coordinates and auxiliary variable t."""
import time
import numpy as np
from itertools import combinations
from scipy.optimize import minimize

N = 11
SQRT3 = np.sqrt(3.0)
# Vertices of the equilateral triangle in Cartesian coordinates
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, SQRT3 / 2.0])

# Precompute all pairs of point indices
INDICES = list(range(N))
TRIS = list(combinations(INDICES, 3))
NUM_TRIS = len(TRIS)

# Barycentric to Cartesian: x = b1*V1 + b2*V2 (since b0 = 1 - b1 - b2)
# x = b1 * 1.0 + b2 * 0.5
# y = b1 * 0.0 + b2 * SQRT3/2
def bary_to_cart(b1, b2):
    x = b1 + 0.5 * b2
    y = 0.5 * SQRT3 * b2
    return x, y

def cart_to_bary(x, y):
    b2 = y / (0.5 * SQRT3)
    b1 = x - 0.5 * b2
    return b1, b2

def compute_areas_cartesian(points):
    """Compute areas of all triangles formed by 3 points. Returns array of shape (NUM_TRIS,)."""
    areas = np.empty(NUM_TRIS)
    for i, (i1, i2, i3) in enumerate(TRIS):
        p1 = points[i1]
        p2 = points[i2]
        p3 = points[i3]
        # Area = 0.5 * |cross(p2-p1, p3-p1)|
        d1 = p2 - p1
        d2 = p3 - p1
        areas[i] = 0.5 * abs(d1[0] * d2[1] - d1[1] * d2[0])
    return areas

def objective_and_constraints(z, t):
    """
    z: array of shape (2*N,) representing barycentric coords (b1, b2) for each point.
    t: auxiliary variable (minimum area to maximize).
    Returns (objective_value, constraints_list, gradients_info).
    We minimize -t subject to:
      - For each triangle: area >= t
      - For each point: b1 >= 0, b2 >= 0, b1 + b2 <= 1
    """
    # Extract barycentric coordinates
    b1 = z[0::2]
    b2 = z[1::2]
    
    # Convert to Cartesian
    x = b1 + 0.5 * b2
    y = 0.5 * SQRT3 * b2
    points = np.column_stack([x, y])
    
    # Compute all triangle areas
    areas = np.empty(NUM_TRIS)
    for i, (i1, i2, i3) in enumerate(TRIS):
        p1 = points[i1]
        p2 = points[i2]
        p3 = points[i3]
        d1 = p2 - p1
        d2 = p3 - p1
        areas[i] = 0.5 * abs(d1[0] * d2[1] - d1[1] * d2[0])
    
    # Objective: minimize -t (equivalent to maximizing t)
    obj_val = -t
    
    # Constraints: area_i - t >= 0 for all i
    # And barycentric constraints: b1 >= 0, b2 >= 0, 1 - b1 - b2 >= 0
    cons = []
    for i in range(NUM_TRIS):
        cons.append({
            'type': 'ineq',
            'fun': lambda i=i, a=areas[i]: a - t
        })
    
    # Barycentric non-negativity: b1 >= 0
    for j in range(N):
        cons.append({
            'type': 'ineq',
            'fun': lambda j=j: b1[j]
        })
        cons.append({
            'type': 'ineq',
            'fun': lambda j=j: b2[j]
        })
        cons.append({
            'type': 'ineq',
            'fun': lambda j=j: 1.0 - b1[j] - b2[j]
        })
    
    return obj_val, cons, areas

def solve_slsqp(z0, t0, max_iter=200):
    """Run SLSQP optimization starting from z0, t0."""
    def obj(z_t):
        z = z_t[:-1]
        t = z_t[-1]
        val, cons, _ = objective_and_constraints(z, t)
        return val
    
    def cons_wrapper(z_t):
        z = z_t[:-1]
        t = z_t[-1]
        _, cons, _ = objective_and_constraints(z, t)
        return [c['fun']() for c in cons]
    
    # Build constraint functions properly
    def make_constraint_funcs(z_t):
        z = z_t[:-1]
        t = z_t[-1]
        b1 = z[0::2]
        b2 = z[1::2]
        x = b1 + 0.5 * b2
        y = 0.5 * SQRT3 * b2
        points = np.column_stack([x, y])
        
        areas = np.empty(NUM_TRIS)
        for i, (i1, i2, i3) in enumerate(TRIS):
            p1 = points[i1]
            p2 = points[i2]
            p3 = points[i3]
            d1 = p2 - p1
            d2 = p3 - p1
            areas[i] = 0.5 * abs(d1[0] * d2[1] - d1[1] * d2[0])
        
        funcs = []
        for i in range(NUM_TRIS):
            funcs.append(areas[i] - t)
        for j in range(N):
            funcs.append(b1[j])
            funcs.append(b2[j])
            funcs.append(1.0 - b1[j] - b2[j])
        return funcs
    
    # SLSQP requires a flat constraint array
    def constraints(z_t):
        return make_constraint_funcs(z_t)
    
    # Bounds for z_t: b1 in [0,1], b2 in [0,1], t in [0, 1]
    bounds = []
    for j in range(N):
        bounds.append((0.0, 1.0))  # b1_j
        bounds.append((0.0, 1.0))  # b2_j
    bounds.append((0.0, 1.0))  # t
    
    z_t0 = np.concatenate([z0, [t0]])
    
    result = minimize(
        obj,
        z_t0,
        method='SLSQP',
        bounds=bounds,
        constraints={'type': 'ineq', 'fun': constraints},
        options={'maxiter': max_iter, 'ftol': 1e-10}
    )
    
    if result.success:
        z_opt = result.x[:-1]
        t_opt = result.x[-1]
        # Verify
        b1 = z_opt[0::2]
        b2 = z_opt[1::2]
        x = b1 + 0.5 * b2
        y = 0.5 * SQRT3 * b2
        points = np.column_stack([x, y])
        areas = compute_areas_cartesian(points)
        min_area = np.min(areas)
        # Check validity
        if np.all(b1 >= -1e-9) and np.all(b2 >= -1e-9) and np.all(b1 + b2 <= 1 + 1e-9):
            # Clip to valid range
            b1 = np.clip(b1, 0, 1)
            b2 = np.clip(b2, 0, 1)
            # Ensure b1+b2 <= 1
            excess = b1 + b2 - 1.0
            mask = excess > 0
            if np.any(mask):
                # Scale down
                scale = 1.0 / np.maximum(1.0, b1[mask] + b2[mask])
                b1[mask] *= scale
                b2[mask] *= scale
            x = b1 + 0.5 * b2
            y = 0.5 * SQRT3 * b2
            points = np.column_stack([x, y])
            areas = compute_areas_cartesian(points)
            min_area = np.min(areas)
            return points, min_area
    return None, None

def random_initial_points(rng):
    """Generate random initial points in barycentric coordinates."""
    z = np.zeros(2 * N)
    for j in range(N):
        u = rng.random()
        v = rng.random()
        if u + v > 1.0:
            u = 1.0 - u
            v = 1.0 - v
        z[2*j] = u
        z[2*j+1] = v
    return z

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    rng = np.random.RandomState(42)
    
    best_points = None
    best_min_area = 0.0
    
    # Try multiple random starts
    num_starts = 50
    for start_idx in range(num_starts):
        if time.time() > deadline:
            break
        
        z0 = random_initial_points(rng)
        t0 = 0.01
        
        points, min_area = solve_slsqp(z0, t0, max_iter=100)
        
        if points is not None and min_area > best_min_area:
            best_min_area = min_area
            best_points = points.copy()
        
        # Also try with a slightly higher initial t
        if time.time() > deadline:
            break
        t0_high = 0.05
        points, min_area = solve_slsqp(z0, t0_high, max_iter=100)
        if points is not None and min_area > best_min_area:
            best_min_area = min_area
            best_points = points.copy()
    
    # If no valid result, fall back to a simple construction
    if best_points is None:
        # Use the original seed-like construction
        import random
        rng2 = random.Random(7)
        pts = []
        while len(pts) < N:
            u, v = rng2.random(), rng2.random()
            if u + v < 0.98 and u > 0.01 and v > 0.01:
                pts.append([u + 0.5 * v, SQRT3 / 2 * v])
        best_points = np.array(pts)
    
    # Final verification and clipping
    b1, b2 = cart_to_bary(best_points[:, 0], best_points[:, 1])
    b1 = np.clip(b1, 0, 1)
    b2 = np.clip(b2, 0, 1)
    excess = b1 + b2 - 1.0
    mask = excess > 0
    if np.any(mask):
        scale = 1.0 / np.maximum(1.0, b1[mask] + b2[mask])
        b1[mask] *= scale
        b2[mask] *= scale
    x = b1 + 0.5 * b2
    y = 0.5 * SQRT3 * b2
    final_points = np.column_stack([x, y])
    
    return final_points
