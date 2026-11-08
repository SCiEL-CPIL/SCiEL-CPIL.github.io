"""Heilbronn triangles: maximize min triangle area for 11 points in unit-area equilateral triangle."""
import numpy as np
import random
import time
from scipy.optimize import minimize

N = 11

# Equilateral triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2.0])
AREA = np.sqrt(3) / 4.0  # area of the equilateral triangle

# Precompute all triangle indices for fast evaluation
def _get_triangle_indices(n):
    """Return list of (i,j,k) for all C(n,3) triangles."""
    indices = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                indices.append((i, j, k))
    return np.array(indices)

TRI_INDICES = _get_triangle_indices(N)


def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by 3 of the points, normalized by AREA."""
    # Vectorized: compute all triangle areas at once
    # For each triangle (i,j,k), area = 0.5 * |(pj-pi) x (pk-pi)|
    pi = points[TRI_INDICES[:, 0]]
    pj = points[TRI_INDICES[:, 1]]
    pk = points[TRI_INDICES[:, 2]]
    
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross) / AREA
    
    return np.min(areas)


def _is_inside_triangle(p):
    """Check if point p is inside or on the boundary of the equilateral triangle."""
    x, y = p
    if y < -1e-9:
        return False
    if x < -1e-9 or x > 1.0 + 1e-9:
        return False
    max_y = np.sqrt(3) * x if x <= 0.5 else np.sqrt(3) * (1.0 - x)
    return y <= max_y + 1e-9


def _project_to_triangle(points):
    """Project points onto the interior of the equilateral triangle."""
    pts = points.copy()
    # Clip to triangle boundaries
    for i in range(len(pts)):
        x, y = pts[i]
        # Bottom edge: y >= 0
        if y < 0:
            y = 0
        # Left edge: y <= sqrt(3)*x
        if y > np.sqrt(3) * x:
            # Project onto line y = sqrt(3)*x
            t = x / (1 + 3)  # x + y/sqrt(3) = x + sqrt(3)*y/sqrt(3)
            # Actually, let's use a simpler approach
            # If above left edge, project to that edge
            # Line: y = sqrt(3)*x, normal direction
            # Distance from (x,y) to line: (sqrt(3)*x - y) / 2
            # But we just need to clip
            scale = x / (x + y/np.sqrt(3)) if (x + y/np.sqrt(3)) > 1e-12 else 0.5
            x = x * scale
            y = y * scale
        # Right edge: y <= sqrt(3)*(1-x)
        if y > np.sqrt(3) * (1 - x):
            scale = (1-x) / ((1-x) + y/np.sqrt(3)) if ((1-x) + y/np.sqrt(3)) > 1e-12 else 0.5
            x = x * scale
            y = y * scale
        pts[i] = [x, y]
    return pts


def _params_to_points(params):
    """Convert 2*N parameters to N points inside the triangle using barycentric coordinates."""
    # params shape (2*N,)
    # For each point, we use 2 params u, v such that u >= 0, v >= 0, u+v <= 1
    # To map from R^2 to the triangle domain, we can use sigmoid or tanh to bound u, v
    # Let's use u = sigmoid(p1), v = sigmoid(p2) * (1 - u)
    # This ensures u in (0,1) and v in (0, 1-u)
    points = []
    for i in range(N):
        u = 1.0 / (1.0 + np.exp(-params[2*i]))
        v = 1.0 / (1.0 + np.exp(-params[2*i+1])) * (1.0 - u)
        p = V0 * (1 - u - v) + V1 * u + V2 * v
        points.append(p)
    return np.array(points)


def _soft_min_area(params, beta=50.0):
    """
    Objective function for minimization: negative of the soft-minimum triangle area.
    Uses log-sum-exp trick to approximate min(a_i) as -1/beta * log(sum(exp(-beta * a_i))).
    This creates a smooth surrogate for the non-differentiable min function.
    """
    points = _params_to_points(params)
    n = len(points)
    
    # Collect all triangle areas
    # To save memory and time, we can compute them in chunks or just loop i
    # For N=11, total triangles is 165. We can store them all.
    areas = []
    
    for i in range(n):
        pi = points[i]
        vecs = points - pi
        X = vecs[:, 0]
        Y = vecs[:, 1]
        C = np.outer(X, Y) - np.outer(Y, X)
        
        C_abs = np.abs(C)
        C_abs[i, :] = float('inf')
        C_abs[:, i] = float('inf')
        np.fill_diagonal(C_abs, float('inf'))
        
        # Get upper triangle values (j < k)
        # Since C is antisymmetric, C[j,k] = -C[k,j]. 
        # We want area = 0.5 * |C[j,k]| / AREA
        # We can just take the upper triangle of C_abs
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        # Exclude i
        mask[i, :] = False
        mask[:, i] = False
        
        vals = C_abs[mask]
        # Normalize to area
        # Area = 0.5 * val / AREA
        areas.extend((0.5 * vals) / AREA)
        
    areas = np.array(areas)
    
    # Soft min calculation
    # min(areas) approx -1/beta * log(sum(exp(-beta * areas)))
    # To avoid overflow, subtract max
    # Note: we are minimizing -soft_min, which is maximizing soft_min.
    # soft_min = -1/beta * logsumexp(-beta * areas)
    
    # If any area is 0, soft_min is 0.
    # We add a small epsilon to avoid log(0) issues if points coincide, 
    # though the mapping should prevent exact coincidence unless params are identical.
    areas_safe = np.maximum(areas, 1e-12)
    
    # LogSumExp: log(sum(exp(x)))
    # x = -beta * areas_safe
    x = -beta * areas_safe
    x_max = np.max(x)
    sum_exp = np.sum(np.exp(x - x_max))
    log_sum_exp = x_max + np.log(sum_exp)
    
    soft_min_val = -1.0 / beta * log_sum_exp
    
    # We want to maximize soft_min_val, so minimize -soft_min_val
    return -soft_min_val


def _generate_structured_initial():
    """Generate a structured initial configuration based on known good patterns."""
    # Vertices, midpoints, centroid
    pts = [
        V0 + np.array([0.05, 0.05]),
        V1 + np.array([-0.05, 0.05]),
        V2 + np.array([0.0, -0.05]),
        (V0 + V1) / 2.0,
        (V1 + V2) / 2.0,
        (V2 + V0) / 2.0,
        (V0 + V1 + V2) / 3.0,
        (V0 + 2*V1 + V2) / 4.0,
        (2*V0 + V1 + V2) / 4.0,
        (V0 + V1 + 2*V2) / 4.0,
        (2*V0 + 2*V1 + V2) / 5.0
    ]
    return np.array(pts)


def _force_directed_optimize(initial_points, max_steps=2000, dt=0.005, 
                             initial_target=0.005, target_growth=1.01,
                             damping=0.95, time_limit=None, start_time=None):
    """
    Force-directed optimization using symplectic (velocity Verlet) integration.
    
    Args:
        initial_points: Initial point configuration
        max_steps: Maximum number of simulation steps
        dt: Time step
        initial_target: Initial target minimum area
        target_growth: Multiplicative growth factor for target area
        damping: Velocity damping factor
        time_limit: If set, stop when time.time() > start_time + time_limit
        start_time: Start time for timeout check
    
    Returns:
        Best points found
    """
    points = initial_points.copy()
    velocities = np.zeros_like(points)
    best_points = points.copy()
    best_min_area = _min_triangle_area(points)
    
    target_area = initial_target
    step = 0
    
    while step < max_steps:
        # Check time limit
        if time_limit is not None and start_time is not None:
            if time.time() > start_time + time_limit:
                break
        
        # Compute forces
        forces = _compute_forces(points, target_area)
        
        # Velocity Verlet integration
        # v_half = v + 0.5 * dt * F/m  (m=1)
        v_half = velocities + 0.5 * dt * forces
        
        # x_new = x + dt * v_half
        points_new = points + dt * v_half
        
        # Project to triangle
        points_new = _project_to_triangle(points_new)
        
        # Compute new forces
        forces_new = _compute_forces(points_new, target_area)
        
        # v_new = v_half + 0.5 * dt * F_new
        velocities = v_half + 0.5 * dt * forces_new
        
        # Apply damping
        velocities *= damping
        
        # Update state
        points = points_new
        step += 1
        
        # Evaluate and track best
        current_min = _min_triangle_area(points)
        if current_min > best_min_area:
            best_min_area = current_min
            best_points = points.copy()
            
            # Grow target area if we've exceeded it
            if current_min > target_area:
                target_area *= target_growth
    
    return best_points, best_min_area


def _generate_random_points(rng, n=N):
    """Generate n random points inside the triangle using rejection sampling."""
    points = []
    while len(points) < n:
        u = rng.random()
        v = rng.random()
        if u + v <= 1.0:
            p = V0 * (1 - u - v) + V1 * u + V2 * v
            points.append(p)
    return np.array(points)


def _generate_grid_points():
    """Generate points on a triangular grid inside the triangle."""
    points = []
    # Use a few rows of a triangular lattice
    for row in range(5):
        for col in range(5 - row):
            # Barycentric coordinates
            u = (col + 0.5) / 5.0
            v = (row + 0.5) / 5.0
            if u + v < 1.0:
                p = V0 * (1 - u - v) + V1 * u + V2 * v
                points.append(p)
    
    # Take first N points
    return np.array(points[:N])


# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s
    
    best_points = None
    best_min = 0.0
    
    rng = random.Random(42)
    
    # Strategy: Force-directed optimization with multiple initializations
    # 1. Start with structured initial
    # 2. Start with random initializations
    # 3. Start with grid-based initializations
    
    initial_configs = []
    initial_configs.append(_generate_structured_initial())
    
    # Add random starts
    for _ in range(8):
        initial_configs.append(_generate_random_points(rng))
    
    # Add grid-based start
    initial_configs.append(_generate_grid_points())
    
    # Run force-directed optimization on each
    for idx, init_pts in enumerate(initial_configs):
        if time.time() > deadline:
            break
        
        # Determine time budget for this run
        remaining_time = deadline - time.time()
        time_for_this_run = min(remaining_time, 15.0)  # Cap at 15s per run
        
        # Adjust parameters based on index
        # First run gets more steps
        max_steps = 3000 if idx == 0 else 2000
        dt = 0.005
        initial_target = 0.003 if idx == 0 else 0.002
        target_growth = 1.015
        damping = 0.95
        
        try:
            result_points, result_min = _force_directed_optimize(
                init_pts,
                max_steps=max_steps,
                dt=dt,
                initial_target=initial_target,
                target_growth=target_growth,
                damping=damping,
                time_limit=time_for_this_run,
                start_time=time.time()
            )
            
            if result_min > best_min:
                best_min = result_min
                best_points = result_points.copy()
        except Exception as e:
            continue
    
    # If no valid points found (should not happen), fallback
    if best_points is None:
        best_points = _generate_random_points(rng)

    return best_points
# EVOLVE-BLOCK-END
