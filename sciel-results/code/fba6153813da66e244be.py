"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 12

# EVOLVE-BLOCK-START

def hex_vertices(cx, cy, side, angle_deg):
    """Compute vertices of a regular hexagon centered at (cx, cy) with given side and rotation."""
    angle = np.radians(angle_deg)
    angles = angle + np.pi / 3 * np.arange(6)
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point (px, py) is inside hexagon centered at (cx, cy) with given side and rotation."""
    # Transform point to hexagon's local coordinate system
    dx = px - cx
    dy = py - cy
    angle = -np.radians(angle_deg)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    lx = cos_a * dx + sin_a * dy
    ly = -sin_a * dx + cos_a * dy
    
    # A regular hexagon with side length s centered at origin has vertices at angles 0, 60, 120, 180, 240, 300
    # The hexagon can be described as intersection of 6 half-planes
    # For a hexagon with "pointy" top (vertex at 90 degrees), the edges are at 0, 60, 120, 180, 240, 300
    # Actually, let's use the standard flat-top hexagon where vertices are at 0, 60, 120, 180, 240, 300
    # The inradius is s * sqrt(3)/2
    # The six edge normals point at 30, 90, 150, 210, 270, 330 degrees (for flat-top)
    # Wait, let me reconsider. A regular hexagon with side s has:
    # - Circumradius = s
    # - Inradius = s * sqrt(3)/2
    # For a hexagon with vertices at angles 0, 60, ..., the edges are perpendicular to directions 30, 90, 150, ...
    # The distance from center to each edge (inradius) is s * sqrt(3)/2
    
    # The hexagon is the intersection of half-planes:
    # For k = 0..5, the edge normal direction is at angle (30 + 60*k) degrees
    # The constraint is: lx * cos(theta_k) + ly * sin(theta_k) <= s * sqrt(3)/2
    
    inradius = side * np.sqrt(3) / 2
    for k in range(6):
        theta = np.radians(30 + 60 * k)
        dist = lx * np.cos(theta) + ly * np.sin(theta)
        if dist > inradius + 1e-10:
            return False
    return True

def compute_min_outer_side(inner_positions, outer_center, outer_angle_deg):
    """
    Given inner hexagon positions and outer hexagon center/angle, find minimum outer side length
    such that all inner hexagons fit inside the outer hexagon.
    
    inner_positions: list of (x, y, angle_deg) for each inner hexagon
    outer_center: (cx, cy)
    outer_angle_deg: rotation of outer hexagon
    
    Returns: minimum side length
    """
    cx, cy = outer_center
    # For a given outer side length S, check if all inner hexagons fit
    # We need to find the minimum S such that all inner hexagon vertices are inside the outer hexagon
    
    # Binary search on S
    lo = 1.0
    hi = 50.0
    
    # First, check if hi works
    def all_inside(S):
        for (ix, iy, iang) in inner_positions:
            verts = hex_vertices(ix, iy, 1.0, iang)
            for (vx, vy) in verts:
                if not point_in_hexagon(vx, vy, cx, cy, S, outer_angle_deg):
                    return False
        return True
    
    if not all_inside(hi):
        # Expand search range
        hi = 100.0
        if not all_inside(hi):
            hi = 200.0
            if not all_inside(hi):
                return hi  # Return a large value
    
    # Binary search
    for _ in range(50):
        mid = (lo + hi) / 2
        if all_inside(mid):
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-6:
            break
    
    return hi

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Optimize over: 12 inner hexagons (x, y, angle each) + outer center (x, y) + outer angle
    # Total: 12*3 + 2 + 1 = 39 parameters
    # But we can fix outer center at origin and outer angle to 0 by symmetry? 
    # Actually, the outer hexagon can be rotated, so we need to optimize that too.
    # However, we can fix outer center at (0,0) without loss of generality (translation invariance).
    # So: 12*3 + 1 = 37 parameters
    
    def objective(params):
        # Extract parameters
        inner_params = params[:36].reshape(12, 3)
        outer_angle = params[36]
        
        inner_positions = [(inner_params[i, 0], inner_params[i, 1], inner_params[i, 2]) for i in range(12)]
        outer_center = (0.0, 0.0)
        
        # Check for overlaps between inner hexagons
        # For each pair, check if any vertex of one is inside the other
        for i in range(12):
            for j in range(i + 1, 12):
                xi, yi, ai = inner_positions[i]
                xj, yj, aj = inner_positions[j]
                
                # Check vertices of hex i against hex j
                verts_i = hex_vertices(xi, yi, 1.0, ai)
                for (vx, vy) in verts_i:
                    if point_in_hexagon(vx, vy, xj, yj, 1.0, aj):
                        return 1e6  # Penalty for overlap
                
                # Check vertices of hex j against hex i
                verts_j = hex_vertices(xj, yj, 1.0, aj)
                for (vx, vy) in verts_j:
                    if point_in_hexagon(vx, vy, xi, yi, 1.0, ai):
                        return 1e6  # Penalty for overlap
        
        # Compute minimum outer side length
        side = compute_min_outer_side(inner_positions, outer_center, outer_angle)
        return side
    
    # Initial guess: place hexagons in a honeycomb-like pattern
    # A honeycomb packing of unit hexagons has center-to-center distance of sqrt(3) * 2 / sqrt(3) = 2? 
    # Actually, for flat-top hexagons, adjacent centers are at distance 2 * inradius = 2 * sqrt(3)/2 = sqrt(3)
    # Wait, inradius = sqrt(3)/2 * side = sqrt(3)/2 for side=1
    # So adjacent center distance = 2 * inradius = sqrt(3) ≈ 1.732
    
    # Let's use a hexagonal lattice arrangement
    # Row 0: 4 hexagons
    # Row 1: 4 hexagons (offset)
    # Row 2: 4 hexagons
    
    # For flat-top hexagons:
    # Horizontal spacing between centers in same row: 2 * circumradius * cos(30°) = 2 * 1 * sqrt(3)/2 = sqrt(3)
    # Vertical spacing between rows: 3/2 * circumradius = 3/2
    # Offset for alternating rows: sqrt(3)/2
    
    sqrt3 = np.sqrt(3)
    x_spacing = sqrt3
    y_spacing = 1.5
    x_offset = sqrt3 / 2
    
    initial_inner = []
    # Center the arrangement
    # 4 per row, 3 rows
    total_width = 3 * x_spacing
    total_height = 2 * y_spacing
    start_x = -total_width / 2
    start_y = -total_height / 2
    
    for row in range(3):
        for col in range(4):
            x = start_x + col * x_spacing
            y = start_y + row * y_spacing
            if row % 2 == 1:
                x += x_offset
            initial_inner.append((x, y, 0.0))
    
    # Convert to parameter vector
    init_params = np.zeros(37)
    for i, (x, y, a) in enumerate(initial_inner):
        init_params[3*i] = x
        init_params[3*i+1] = y
        init_params[3*i+2] = a
    init_params[36] = 0.0  # outer angle
    
    # Use differential evolution for global search, then polish with L-BFGS-B
    best_result = None
    best_obj = np.inf
    
    # Define bounds
    bounds = []
    for i in range(12):
        bounds.append((-10, 10))  # x
        bounds.append((-10, 10))  # y
        bounds.append((0, 60))     # angle (0 to 60 degrees, hexagon symmetry)
    bounds.append((0, 60))  # outer angle
    
    # Run differential evolution
    try:
        de_result = differential_evolution(
            objective,
            bounds,
            seed=42,
            maxiter=200,
            popsize=15,
            tol=1e-4,
            polish=True,
            init='sobol',
            updating='immediate'
        )
        
        if de_result.success or de_result.fun < best_obj:
            best_obj = de_result.fun
            best_result = de_result.x
            
            # Polish with L-BFGS-B
            if time.time() < deadline:
                polish_result = minimize(
                    objective,
                    best_result,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                if polish_result.fun < best_obj:
                    best_obj = polish_result.fun
                    best_result = polish_result.x
    except Exception as e:
        pass
    
    # If DE didn't find a good result, try L-BFGS-B from initial guess
    if best_result is None or best_obj > 10:
        if time.time() < deadline:
            try:
                lbfgs_result = minimize(
                    objective,
                    init_params,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                if lbfgs_result.fun < best_obj:
                    best_obj = lbfgs_result.fun
                    best_result = lbfgs_result.x
            except Exception as e:
                pass
    
    # If still no good result, use the initial arrangement
    if best_result is None:
        best_result = init_params
    
    # Extract final parameters
    inner_params = best_result[:36].reshape(12, 3)
    outer_angle = best_result[36]
    
    inner = []
    for i in range(12):
        inner.append([float(inner_params[i, 0]), float(inner_params[i, 1]), float(inner_params[i, 2])])
    
    outer_center = (0.0, 0.0)
    side = float(best_obj)
    
    return inner, outer_center, side, float(outer_angle)

# EVOLVE-BLOCK-END
