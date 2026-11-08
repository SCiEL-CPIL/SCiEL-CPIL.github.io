"""Packing 12 unit hexagons in a regular hexagon; minimize outer side length."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12
DEADLINE = 115.0  # safety margin before 150s kill

def hexagon_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx, cy)."""
    angle_rad = np.radians(angle_deg)
    angles = angle_rad + np.arange(6) * np.pi / 3
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def min_outer_side_for_points(points, outer_center, outer_angle_deg):
    """
    Compute minimum side length s of a regular hexagon centered at outer_center
    with rotation outer_angle_deg that contains all given points.
    
    A regular hexagon with side s has inradius r = s * sqrt(3)/2.
    The hexagon is the intersection of 6 half-planes. For a point p to be inside,
    for each of the 6 normal directions n_k, we need (p - c) dot n_k <= r.
    So s >= 2/sqrt(3) * max_k max_p (p - c) dot n_k.
    """
    cx, cy = outer_center
    angle_rad = np.radians(outer_angle_deg)
    # The 6 outward normal directions of a regular hexagon
    # For a hexagon with vertices at angles alpha + k*pi/3, the normals are at alpha + (k+0.5)*pi/3
    # Actually, let me think more carefully.
    # A regular hexagon with side s centered at origin, one vertex at angle 0:
    # Vertices at angles 0, 60, 120, 180, 240, 300 degrees, at distance s from center.
    # The edges are between consecutive vertices. The normal to the edge between
    # vertex at angle 0 and vertex at angle 60 points at angle 30 degrees.
    # Inradius = s * cos(30) = s * sqrt(3)/2.
    # So the 6 outward normals are at angles: alpha + 30 + k*60 for k=0..5
    # where alpha is the angle of the first vertex.
    
    # With rotation angle_deg, the first vertex is at angle angle_deg.
    # Normals are at angle_deg + 30 + k*60 degrees.
    normal_angles = np.radians(outer_angle_deg) + np.pi/6 + np.arange(6) * np.pi/3
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    # For each point, compute the max dot product with each normal
    rel_points = points - np.array([cx, cy])
    # rel_points shape: (M, 2), normals shape: (6, 2)
    # dot products: (M, 6)
    dots = rel_points @ normals.T
    max_dot = np.max(dots, axis=0)  # shape (6,)
    r_needed = np.max(max_dot)
    if r_needed < 0:
        r_needed = 0.0
    s = 2.0 / np.sqrt(3.0) * r_needed
    return s

def check_inside(points, outer_center, outer_side, outer_angle_deg):
    """Check if all points are inside the outer hexagon."""
    s = min_outer_side_for_points(points, outer_center, outer_angle_deg)
    return s <= outer_side + 1e-9

def compute_min_side(inner, outer_center, outer_angle_deg):
    """Compute minimum outer side length for given inner configuration."""
    all_points = []
    for x, y, ang in inner:
        v = hexagon_vertices(x, y, 1.0, ang)
        all_points.append(v)
    all_points = np.vstack(all_points)
    return min_outer_side_for_points(all_points, outer_center, outer_angle_deg)

def objective(params, inner_angles):
    """
    Objective: minimize outer side length.
    params: flattened array of [x1, y1, x2, y2, ..., x12, y12] (24 values)
    inner_angles: array of 12 angles (fixed during optimization)
    """
    pos = params.reshape(N, 2)
    inner = [[pos[i, 0], pos[i, 1], inner_angles[i]] for i in range(N)]
    
    all_points = []
    for i in range(N):
        v = hexagon_vertices(pos[i, 0], pos[i, 1], 1.0, inner_angles[i])
        all_points.append(v)
    all_points = np.vstack(all_points)
    
    # Outer hexagon centered at origin
    outer_center = (0.0, 0.0)
    s = min_outer_side_for_points(all_points, outer_center, 0.0)
    
    # Add penalty for overlapping hexagons
    # Two hexagons overlap if their distance is less than the minimum distance
    # For two unit hexagons with the same orientation, minimum non-overlapping distance is 2*sqrt(3)
    # For different orientations, it's more complex. Use a simpler penalty.
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(pos[i] - pos[j])
            # Minimum distance for non-overlap of two unit hexagons (same orientation) is 2*sqrt(3)
            min_d = 2.0 * np.sqrt(3.0)
            if d < min_d:
                penalty += (min_d - d) ** 2
    
    return s + penalty

def construct():
    start_time = time.time()
    
    best_score = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_side = 40.0
    best_angle = 0.0
    
    def try_config(inner, center, angle):
        nonlocal best_score, best_inner, best_center, best_side, best_angle
        s = compute_min_side(inner, center, angle)
        if s < best_score:
            best_score = s
            best_inner = [list(x) for x in inner]
            best_center = center
            best_side = s
            best_angle = angle
    
    # Initial configurations
    
    # Config 1: Linear
    inner1 = [[4.0 * k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
    try_config(inner1, (0.0, 0.0), 0.0)
    
    # Config 2: 3x4 hexagonal grid
    sqrt3 = np.sqrt(3)
    positions = []
    for i in range(3):
        for j in range(4):
            x = j * 2 * sqrt3 + (i % 2) * sqrt3
            y = i * 3
            positions.append((x, y))
    cx = np.mean([p[0] for p in positions])
    cy = np.mean([p[1] for p in positions])
    inner2 = [[p[0] - cx, p[1] - cy, 0.0] for p in positions]
    try_config(inner2, (0.0, 0.0), 0.0)
    
    # Config 3: 4x3 hexagonal grid (transposed)
    positions3 = []
    for i in range(4):
        for j in range(3):
            x = j * 2 * sqrt3 + (i % 2) * sqrt3
            y = i * 3
            positions3.append((x, y))
    cx3 = np.mean([p[0] for p in positions3])
    cy3 = np.mean([p[1] for p in positions3])
    inner3 = [[p[0] - cx3, p[1] - cy3, 0.0] for p in positions3]
    try_config(inner3, (0.0, 0.0), 0.0)
    
    # Config 4: Hexagonal cluster - center + ring
    # Center at (0,0), 6 around it, then 5 more
    positions4 = [(0, 0)]
    for k in range(6):
        ang = k * np.pi / 3
        positions4.append((2*sqrt3*np.cos(ang), 2*sqrt3*np.sin(ang)))
    # Add 5 more in a ring further out
    for k in range(5):
        ang = k * 2*np.pi/5 + np.pi/5
        positions4.append((4*sqrt3*np.cos(ang), 4*sqrt3*np.sin(ang)))
    cx4 = np.mean([p[0] for p in positions4])
    cy4 = np.mean([p[1] for p in positions4])
    inner4 = [[p[0] - cx4, p[1] - cy4, 0.0] for p in positions4]
    try_config(inner4, (0.0, 0.0), 0.0)
    
    # Try different outer rotations for best configs
    for cfg in [inner1, inner2, inner3, inner4]:
        if time.time() - start_time > DEADLINE:
            break
        for outer_ang in [0, 30, 60, 90, 120, 150]:
            if time.time() - start_time > DEADLINE:
                break
            try_config(cfg, (0.0, 0.0), float(outer_ang))
    
    # Optimization: use scipy minimize to improve positions
    # Start from best config found so far
    if best_inner is not None:
        if time.time() - start_time < DEADLINE - 5:
            # Use the best configuration as starting point
            init_pos = np.array([[x, y] for x, y, a in best_inner])
            init_angles = np.array([a for x, y, a in best_inner])
            
            # Try multiple random restarts
            n_restarts = 5
            for r in range(n_restarts):
                if time.time() - start_time > DEADLINE - 5:
                    break
                
                if r == 0:
                    x0 = init_pos.flatten()
                else:
                    # Perturb
                    x0 = init_pos.flatten() + 0.5 * np.random.randn(2*N)
                
                try:
                    result = minimize(
                        objective,
                        x0,
                        args=(init_angles,),
                        method='L-BFGS-B',
                        options={'maxiter': 200, 'ftol': 1e-8}
                    )
                    
                    if result.success or result.fun < best_score:
                        pos = result.fun  # This is wrong, result.fun is the objective value
                        # Actually result.fun is the value of the objective at the optimum
                        # We need to extract positions from result.x
                        pos_opt = result.x.reshape(N, 2)
                        inner_opt = [[pos_opt[i, 0], pos_opt[i, 1], init_angles[i]] for i in range(N)]
                        s_opt = compute_min_side(inner_opt, (0.0, 0.0), 0.0)
                        if s_opt < best_score:
                            best_score = s_opt
                            best_inner = inner_opt
                            best_center = (0.0, 0.0)
                            best_side = s_opt
                            best_angle = 0.0
                except Exception:
                    pass
    
    # Add safety margin to ensure validity
    best_side += 0.02
    
    return best_inner, best_center, best_side, best_angle
