"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START

def hexagon_vertices(center, side, angle_deg):
    """Return vertices of a regular hexagon."""
    cx, cy = center
    theta = np.radians(angle_deg)
    angles = theta + np.arange(6) * np.pi / 3.0
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def signed_distance_point_hexagon(px, py, center, side, angle_deg):
    """
    Compute signed distance from point (px, py) to a regular hexagon.
    Positive if inside, negative if outside.
    Uses the fact that a regular hexagon is the intersection of 6 half-planes.
    The distance to the hexagon is the minimum over the 6 edges of the
    distance to the supporting line, but we need to be careful.
    
    Actually, for a convex polygon, the signed distance can be computed as:
    For each edge, compute the distance from the point to the line containing the edge.
    The point is inside if it's on the correct side of all edges.
    The signed distance to the polygon is the minimum over all edges of the
    signed distance to the edge's supporting line (where negative means outside).
    """
    cx, cy = center
    theta = np.radians(angle_deg)
    
    # Transform point to hexagon's local coordinate system (centered, unrotated)
    dx = px - cx
    dy = py - cy
    # Rotate by -theta
    cos_t = np.cos(-theta)
    sin_t = np.sin(-theta)
    lx = dx * cos_t - dy * sin_t
    ly = dx * sin_t + dy * cos_t
    
    # In local coords, hexagon with side 'side' has vertices at:
    # (side*cos(k*pi/3), side*sin(k*pi/3)) for k=0..5
    # The 6 edges are defined by 6 lines. The hexagon is the intersection of:
    # |x| <= side, |y| <= side * sqrt(3)/2, |x + y/sqrt(3)| <= side, etc.
    # More precisely, for a regular hexagon with "radius" (center to vertex) = side,
    # the apothem (center to edge midpoint) = side * sqrt(3)/2.
    
    # The 6 supporting lines have normals pointing outward at angles 0, 60, 120, 180, 240, 300 degrees
    # In local coords, the hexagon is defined by:
    # For each k in 0..5, the edge is perpendicular to direction angle k*pi/3
    # The distance from center to each edge (apothem) is a = side * sqrt(3)/2
    
    apothem = side * np.sqrt(3) / 2.0
    
    # Normals to the 6 edges in local coords
    normal_angles = np.arange(6) * np.pi / 3.0
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    # For each edge, the signed distance from the point to the supporting line
    # The line is: n · p = apothem (for the outward normal n)
    # Signed distance = apothem - n · p (positive if inside relative to that edge)
    dots = normals @ np.array([lx, ly])
    signed_dists = apothem - dots
    
    # The point is inside if all signed_dists >= 0
    # The signed distance to the hexagon is the minimum of signed_dists
    return np.min(signed_dists)

def objective(params):
    """
    Objective function to minimize.
    params: [x1, y1, a1, x2, y2, a2, ..., xN, yN, aN, S]
    where (xk, yk, ak) are center and angle of inner hexagon k,
    S is the outer hexagon side length.
    
    We want to minimize S subject to:
    1. Inner hexagons don't overlap each other
    2. Inner hexagons are inside the outer hexagon
    """
    # Parse parameters
    inner_params = params[:3*N].reshape(N, 3)
    S = params[3*N]
    
    # Outer hexagon center is at origin for simplicity (we can fix it)
    outer_center = (0.0, 0.0)
    outer_angle = 0.0
    
    # Constraint 1: Each inner hexagon must be inside the outer hexagon
    # For each inner hexagon, all its vertices must be inside the outer hexagon.
    # We check the minimum signed distance from each inner hexagon's vertices to the outer hexagon.
    penalty = 0.0
    
    for i in range(N):
        xi, yi, ai = inner_params[i]
        # Get vertices of inner hexagon i
        verts = hexagon_vertices((xi, yi), 1.0, ai)
        for j in range(6):
            vx, vy = verts[j]
            dist = signed_distance_point_hexagon(vx, vy, outer_center, S, outer_angle)
            if dist < 0:
                penalty += dist * dist
    
    # Constraint 2: Inner hexagons don't overlap each other
    # Two convex polygons don't overlap if there exists a separating axis.
    # For two regular hexagons, we can check all 12 edge normals (6 from each).
    # If for any axis, the projections are disjoint, they don't overlap.
    # If they overlap, we need a penalty.
    # A simpler approach: check if any vertex of hex i is inside hex j, or vice versa.
    # This is not sufficient for all cases, but for regular hexagons it's often good.
    # Better: use the separating axis theorem properly.
    
    for i in range(N):
        for j in range(i+1, N):
            xi, yi, ai = inner_params[i]
            xj, yj, aj = inner_params[j]
            
            verts_i = hexagon_vertices((xi, yi), 1.0, ai)
            verts_j = hexagon_vertices((xj, yj), 1.0, aj)
            
            # Get all edge normals from both hexagons
            all_norms = []
            for k in range(6):
                # Edge from vertex k to vertex k+1
                v1 = verts_i[k]
                v2 = verts_i[(k+1) % 6]
                edge = v2 - v1
                # Normal (perpendicular)
                n = np.array([-edge[1], edge[0]])
                n_norm = np.linalg.norm(n)
                if n_norm > 1e-10:
                    all_norms.append(n / n_norm)
            
            for k in range(6):
                v1 = verts_j[k]
                v2 = verts_j[(k+1) % 6]
                edge = v2 - v1
                n = np.array([-edge[1], edge[0]])
                n_norm = np.linalg.norm(n)
                if n_norm > 1e-10:
                    all_norms.append(n / n_norm)
            
            # Check if there's a separating axis
            separated = False
            for n in all_norms:
                proj_i = verts_i @ n
                proj_j = verts_j @ n
                min_i = np.min(proj_i)
                max_i = np.max(proj_i)
                min_j = np.min(proj_j)
                max_j = np.max(proj_j)
                if max_i < min_j or max_j < min_i:
                    separated = True
                    break
            
            if not separated:
                # Overlapping: compute a penalty based on how much they overlap
                # Simple penalty: sum of squared distances of vertices inside the other
                for v in verts_i:
                    d = signed_distance_point_hexagon(v[0], v[1], (xj, yj), 1.0, aj)
                    if d < 0:
                        penalty += d * d
                for v in verts_j:
                    d = signed_distance_point_hexagon(v[0], v[1], (xi, yi), 1.0, ai)
                    if d < 0:
                        penalty += d * d
    
    # Return S + penalty
    return S + 10.0 * penalty

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # 120 second limit
    
    best_S = float('inf')
    best_params = None
    
    # Try multiple random initializations
    np.random.seed(42)
    
    n_restarts = 20
    for restart in range(n_restarts):
        if time.time() > deadline - 30.0:
            break
        
        # Random initialization: place hexagons in a rough hexagonal pattern
        # Scale factor to ensure they start non-overlapping
        scale = 2.2  # Slightly more than 2 to avoid initial overlap
        
        # Hexagonal lattice positions
        positions = []
        for row in range(-2, 3):
            for col in range(-2, 3):
                x = scale * (col + 0.5 * (row & 1))
                y = scale * row * np.sqrt(3) / 2
                positions.append((x, y))
        
        # Shuffle and take N positions
        np.random.shuffle(positions)
        positions = positions[:N]
        
        # Add some random perturbation
        params = []
        for i in range(N):
            x, y = positions[i]
            x += np.random.uniform(-0.1, 0.1)
            y += np.random.uniform(-0.1, 0.1)
            angle = np.random.uniform(0, 60)
            params.extend([x, y, angle])
        
        # Initial S: large enough to contain all hexagons
        max_radius = max(np.sqrt(x**2 + y**2) for x, y in positions) + 1.0
        S_init = max_radius / (np.sqrt(3) / 2) + 0.5  # Account for apothem
        params.append(S_init)
        
        params = np.array(params)
        
        # Bounds
        bounds = [(-10, 10)] * N + [(-10, 10)] * N + [(0, 180)] * N + [(2.0, 20.0)]
        
        # Optimize
        try:
            result = minimize(objective, params, method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': 200, 'ftol': 1e-6, 'gtol': 1e-4})
            
            if result.success or result.fun < best_S:
                S_val = result.fun
                # Extract the actual S from params
                S_actual = result.x[3*N]
                if S_actual < best_S:
                    best_S = S_actual
                    best_params = result.x.copy()
        except Exception as e:
            pass
    
    # If we didn't find a good solution, use a fallback
    if best_params is None:
        # Fallback: simple hexagonal packing
        positions = []
        for row in range(-2, 3):
            for col in range(-2, 3):
                x = 2.1 * (col + 0.5 * (row & 1))
                y = 2.1 * row * np.sqrt(3) / 2
                positions.append((x, y))
        positions = positions[:N]
        
        inner = []
        for x, y in positions:
            inner.append([x, y, 0.0])
        
        # Estimate outer side
        max_r = max(np.sqrt(x**2 + y**2) for x, y in positions) + 1.0
        S = max_r / (np.sqrt(3) / 2) + 0.3
        return inner, (0.0, 0.0), S, 0.0
    
    # Extract results from best_params
    inner_params = best_params[:3*N].reshape(N, 3)
    S = best_params[3*N]
    
    inner = []
    for i in range(N):
        x, y, angle = inner_params[i]
        inner.append([float(x), float(y), float(angle)])
    
    return inner, (0.0, 0.0), float(S), 0.0

# EVOLVE-BLOCK-END
