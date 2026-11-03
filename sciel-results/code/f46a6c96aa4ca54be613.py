import numpy as np
from scipy.optimize import minimize
import time

N = 12
TOTAL_BUDGET = 120.0  # seconds
SAFETY_MARGIN = 5.0

# Precompute vertex offsets for unit hexagon (circumradius 1)
HEX_VERTICES = np.array([
    [1.0, 0.0],
    [0.5, np.sqrt(3)/2],
    [-0.5, np.sqrt(3)/2],
    [-1.0, 0.0],
    [-0.5, -np.sqrt(3)/2],
    [0.5, -np.sqrt(3)/2]
])

def get_hex_vertices(x, y, angle_deg):
    angle = np.radians(angle_deg)
    c, s = np.cos(angle), np.sin(angle)
    rot = np.array([[c, -s], [s, c]])
    return np.array([x, y]) + HEX_VERTICES @ rot.T

def compute_required_side(hex_centers, hex_angles, outer_angle_deg=0.0):
    """Compute minimum side length of outer hexagon (fixed orientation) to contain all inner hexagons."""
    angle = np.radians(outer_angle_deg)
    # Normals to the 6 sides of a regular hexagon with a vertex pointing at angle=0
    # If vertex is at angle 0, the sides are perpendicular to angles 30, 90, 150, 210, 270, 330
    # Actually, for a hexagon with vertices at 0,60,...,300:
    # Side 0 is between v300 and v060? No, v000 to v060.
    # The normal to the side connecting v000 and v060 is at angle 30 deg.
    # Let's just use the 6 normals at 30, 90, 150, 210, 270, 330 degrees.
    normals_angles = np.radians([30, 90, 150, 210, 270, 330])
    normals = np.stack([np.cos(normals_angles), np.sin(normals_angles)], axis=1)
    
    inradius_ratio = np.sqrt(3) / 2  # inradius = side * sqrt(3)/2
    
    max_proj = -np.inf
    for i in range(N):
        verts = get_hex_vertices(hex_centers[i][0], hex_centers[i][1], hex_angles[i])
        # Project all vertices onto all normals
        projections = verts @ normals.T  # shape (6, 6)
        max_proj = max(max_proj, np.max(projections))
    
    # max_proj = max(n_k . v)
    # We need n_k . v <= inradius = S * sqrt(3)/2
    # So S >= max_proj / (sqrt(3)/2) = 2 * max_proj / sqrt(3)
    return 2.0 * max_proj / np.sqrt(3)

def compute_overlap_penalty(hex_centers, hex_angles, eps=0.001):
    """Compute a smooth penalty for overlapping inner hexagons."""
    penalty = 0.0
    for i in range(N):
        verts_i = get_hex_vertices(hex_centers[i][0], hex_centers[i][1], hex_angles[i])
        for j in range(i+1, N):
            verts_j = get_hex_vertices(hex_centers[j][0], hex_centers[j][1], hex_angles[j])
            # For each vertex of i, check if inside j
            # A point P is inside hexagon j if for all 6 normals n_k of j:
            # n_k . (P - center_j) <= inradius_j
            # inradius_j = 1 * sqrt(3)/2
            center_j = hex_centers[j]
            angle_j = hex_angles[j]
            normals_angles = np.radians([30, 90, 150, 210, 270, 330] + np.radians(angle_j))
            normals_j = np.stack([np.cos(normals_angles), np.sin(normals_angles)], axis=1)
            inradius_j = np.sqrt(3)/2
            
            for v_i in verts_i:
                # Check if v_i is inside hexagon j
                projections = (v_i - center_j) @ normals_j.T
                # If any projection > inradius, the point is outside
                # We want to penalize when points are inside (penalize violation of non-overlap)
                # Actually, we want to penalize when hexagons overlap.
                # A simple measure: for each vertex of i, compute the minimum "distance" to the boundary of j.
                # If all 6 projections <= inradius, the point is inside.
                # The "penetration depth" is max(0, inradius - min(projections))? No.
                # If a vertex is inside, we want to push it out.
                # Let's use: for each vertex of i, compute the maximum over k of (projection_k - inradius).
                # If this max is negative, the vertex is inside. We want to penalize this.
                # Actually, if the vertex is inside, all projections are <= inradius.
                # The "slack" is inradius - max(projections)? No.
                # Let's define: for a point P, the "signed distance" to the hexagon boundary (positive if outside)
                # is min_k (inradius - n_k . (P - center))
                # If this min is negative, P is inside.
                # We want to penalize when P is inside, i.e., when min_k (inradius - n_k . (P - center)) < 0.
                # So penalty += max(0, -min_k (inradius - n_k . (P - center)))
                # = max(0, max_k (n_k . (P - center) - inradius))
                max_violation = np.max(projections - inradius_j)
                if max_violation > 0:
                    penalty += max_violation ** 2
    return penalty

def objective(params):
    """Total objective: outer side length + overlap penalty."""
    # Unpack params
    # For each hex: x, y, angle (3 params per hex)
    hex_centers = np.zeros((N, 2))
    hex_angles = np.zeros(N)
    for i in range(N):
        hex_centers[i] = params[3*i:3*i+2]
        hex_angles[i] = params[3*i+2]
    
    side = compute_required_side(hex_centers, hex_angles)
    penalty = compute_overlap_penalty(hex_centers, hex_angles)
    return side + 1000.0 * penalty  # Heavy penalty for overlap

def construct():
    start_time = time.time()
    
    # Initial guess: hexagonal grid
    # For 12 hexagons, a 4x3 grid or similar
    # Let's use a honeycomb-like arrangement
    # Centers in a triangular lattice
    dx = 1.5  # horizontal distance between centers
    dy = np.sqrt(3)/2 * 2  # vertical distance between rows? 
    # Actually, for unit hexagons in a honeycomb, the centers form a triangular lattice
    # with nearest neighbor distance = 2 * inradius = sqrt(3)
    # So dx = sqrt(3), dy = 3/2? 
    # Let's just use a simple grid and let the optimizer figure it out.
    # But a good initial guess helps.
    # 12 hexagons: 4 columns, 3 rows
    # In a honeycomb, rows are offset
    # Let's use:
    # Row 0: 4 hexagons at x = 0, sqrt(3), 2*sqrt(3), 3*sqrt(3)
    # Row 1: 3 hexagons at x = sqrt(3)/2, 3*sqrt(3)/2, 5*sqrt(3)/2, y = 1.5
    # Row 2: 4 hexagons at x = 0, sqrt(3), 2*sqrt(3), 3*sqrt(3), y = 3
    # Wait, that's 11. Let's do 4x3 grid.
    
    params_list = []
    sqrt3 = np.sqrt(3)
    # 4 columns, 3 rows
    for row in range(3):
        for col in range(4):
            x = col * sqrt3
            y = row * 1.5
            angle = 0.0
            params_list.append([x, y, angle])
    
    x0 = np.array([p for hex_params in params_list for p in hex_params])
    
    # Bounds: x, y in [-10, 10], angle in [0, 60]
    bounds = []
    for _ in range(N):
        bounds.append((-10, 10))
        bounds.append((-10, 10))
        bounds.append((0, 60))
    
    # Use Nelder-Mead with many iterations
    # But Nelder-Mead can be slow. Let's try L-BFGS-B with a finite difference approximation.
    # However, the objective is not smooth (max of linear functions).
    # Let's use a multi-start approach with Nelder-Mead.
    
    best_result = None
    best_fval = np.inf
    
    # Run several restarts
    for restart in range(5):
        if time.time() - start_time > TOTAL_BUDGET - SAFETY_MARGIN:
            break
        
        if restart == 0:
            x_current = x0.copy()
        else:
            # Perturb initial guess
            x_current = x0 + np.random.normal(0, 0.5, size=x0.shape)
            # Ensure angles are in [0, 60]
            for i in range(N):
                x_current[3*i+2] = x_current[3*i+2] % 60
        
        # Run Nelder-Mead
        result = minimize(
            objective, 
            x_current, 
            method='Nelder-Mead',
            bounds=bounds,
            options={'maxiter': 2000, 'fatol': 1e-8, 'xatol': 1e-8}
        )
        
        if result.fun < best_fval:
            best_fval = result.fun
            best_result = result.x
        
        # Check if we have a valid solution (low penalty)
        # If penalty is near zero, we can stop
        if best_fval < 12.0:  # Rough estimate: side ~11-12, penalty ~0
            # Polish with tighter tolerance
            result_polish = minimize(
                objective, 
                best_result, 
                method='Nelder-Mead',
                bounds=bounds,
                options={'maxiter': 5000, 'fatol': 1e-12, 'xatol': 1e-12}
            )
            if result_polish.fun < best_fval:
                best_fval = result_polish.fun
                best_result = result_polish.x
            break
    
    if best_result is None:
        # Fallback: return initial guess
        best_result = x0
    
    # Unpack best_result
    inner = []
    hex_centers = np.zeros((N, 2))
    hex_angles = np.zeros(N)
    for i in range(N):
        x, y, angle = best_result[3*i:3*i+3]
        hex_centers[i] = [x, y]
        hex_angles[i] = angle
        inner.append([x, y, angle])
    
    # Compute the actual side length (without penalty)
    side = compute_required_side(hex_centers, hex_angles)
    
    # The outer hexagon is centered at the centroid of the inner hexagons?
    # Actually, the outer hexagon center should be chosen to minimize the side length.
    # For a fixed orientation, the optimal center is the one that minimizes the max projection.
    # This is a linear programming problem, but we can approximate by using the centroid.
    # Or, we can shift the outer hexagon to be centered at the "middle" of the bounding box.
    # For simplicity, let's use the centroid of all inner hexagon centers.
    center = np.mean(hex_centers, axis=0)
    
    # We can also try to optimize the outer center and angle, but that's more complex.
    # For now, let's return the centroid as the outer center.
    # And use angle 0 for the outer hexagon.
    
    return inner, (float(center[0]), float(center[1])), float(side), 0.0
