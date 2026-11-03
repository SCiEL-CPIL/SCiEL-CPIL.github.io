import numpy as np
import time

N = 12
START_TIME = time.time()
DEADLINE = 145.0  # Safety margin before 150s

def hexagon_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon with given center, side, rotation."""
    angle = np.radians(angle_deg)
    vertices = []
    for k in range(6):
        a = angle + k * np.pi / 3
        x = cx + side * np.cos(a)
        y = cy + side * np.sin(a)
        vertices.append((x, y))
    return np.array(vertices)

def outer_hexagon_constraint(vertices, center, side, angle_deg):
    """
    Return the minimum side length of outer hexagon (with given center, rotation)
    that contains all given vertices.
    """
    cx, cy = center
    theta = np.radians(angle_deg)
    # Edge normals are at theta + 30° + k*60°
    normals = []
    for k in range(6):
        a = theta + np.pi/6 + k * np.pi/3
        normals.append(np.array([np.cos(a), np.sin(a)]))
    normals = np.array(normals)  # shape (6, 2)
    
    # For each vertex, compute max projection onto each normal
    # vertices shape: (M, 2)
    # projections: (M, 6)
    rel = vertices - np.array(center)
    proj = rel @ normals.T  # (M, 6)
    max_proj = proj.max()  # maximum over all vertices and all normals
    
    # side = max_proj / (sqrt(3)/2)
    return max_proj / (np.sqrt(3) / 2)

def objective(params):
    """
    params: [x1, y1, a1, x2, y2, a2, ..., x12, y12, a12, cx, cy, outer_angle]
    Returns the outer hexagon side length.
    """
    n_inner = N
    inner_params = params[:n_inner * 3]
    cx = params[n_inner * 3]
    cy = params[n_inner * 3 + 1]
    outer_angle = params[n_inner * 3 + 2]
    
    all_vertices = []
    for i in range(n_inner):
        x = inner_params[i * 3]
        y = inner_params[i * 3 + 1]
        a = inner_params[i * 3 + 2]
        verts = hexagon_vertices(x, y, 1.0, a)
        all_vertices.append(verts)
    
    all_vertices = np.concatenate(all_vertices, axis=0)  # (72, 2)
    
    return outer_hexagon_constraint(all_vertices, (cx, cy), 1.0, outer_angle)

def initial_config():
    """Create a reasonable initial configuration."""
    params = []
    # Place hexagons in a hexagonal lattice pattern
    # 12 hexagons: let's do a 2x3 grid offset pattern or a flower pattern
    # Flower: 1 center, 6 around, 5 more
    # Simpler: arrange in a roughly circular pattern
    
    # Let's use a hexagonal close-packed arrangement
    # Row 1: 4 hexagons, Row 2: 4 hexagons, Row 3: 4 hexagons (with offsets)
    # Or: 2 rows of 6
    
    # For hexagonal packing, adjacent hexagons center-to-center distance = 2 * inradius = 2 * sqrt(3)/2 = sqrt(3)
    # if aligned, or 2 * circumradius = 2 if touching at vertices
    
    # Let's place them in a grid pattern with spacing sqrt(3) horizontally and 1.5 vertically (for staggered)
    # Actually, for unit hexagons (side=1), the distance between centers of adjacent hexagons in a hexagonal lattice is 2 * sqrt(3)/2 = sqrt(3) ≈ 1.732
    
    dx = np.sqrt(3)
    dy = 1.5
    
    positions = []
    # 4 rows of 3, or 3 rows of 4
    # Let's do 3 rows of 4
    for row in range(3):
        for col in range(4):
            x = col * dx + (row % 2) * dx / 2
            y = row * dy
            positions.append((x, y, 0.0))
    
    # Center the configuration
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    
    for i, (x, y, a) in enumerate(positions):
        params.append(x - cx)
        params.append(y - cy)
        params.append(a)
    
    # Outer hexagon center and angle
    params.append(0.0)
    params.append(0.0)
    params.append(0.0)
    
    return np.array(params)

def construct():
    from scipy.optimize import minimize
    
    best_params = None
    best_score = float('inf')
    
    # Try multiple initial configurations
    configs = [initial_config()]
    
    # Also try a circular arrangement
    circ_params = []
    for i in range(N):
        angle = 2 * np.pi * i / N
        r = 3.0
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        circ_params.extend([x, y, 0.0])
    circ_params.extend([0.0, 0.0, 0.0])
    configs.append(np.array(circ_params))
    
    # Try a more compact arrangement
    compact_params = []
    # 2 rings: inner ring of 6 at radius sqrt(3), outer ring of 6 at radius 2*sqrt(3)
    for i in range(6):
        angle = 2 * np.pi * i / 6
        x = np.sqrt(3) * np.cos(angle)
        y = np.sqrt(3) * np.sin(angle)
        compact_params.extend([x, y, 0.0])
    for i in range(6):
        angle = 2 * np.pi * i / 6 + np.pi / 6
        x = 2 * np.sqrt(3) * np.cos(angle)
        y = 2 * np.sqrt(3) * np.sin(angle)
        compact_params.extend([x, y, 0.0])
    compact_params.extend([0.0, 0.0, 0.0])
    configs.append(np.array(compact_params))
    
    for config in configs:
        if time.time() > DEADLINE:
            break
        
        try:
            result = minimize(
                objective,
                config,
                method='L-BFGS-B',
                options={'maxiter': 200, 'maxfun': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            score = result.fun
            if score < best_score:
                best_score = score
                best_params = result.x.copy()
                
        except Exception as e:
            pass
    
    if best_params is None:
        # Fallback
        return [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)], (0.0, 0.0), 40.0, 0.0
    
    # Extract results
    inner = []
    for i in range(N):
        x = best_params[i * 3]
        y = best_params[i * 3 + 1]
        a = np.degrees(best_params[i * 3 + 2])
        inner.append([x, y, a])
    
    cx = best_params[N * 3]
    cy = best_params[N * 3 + 1]
    outer_angle = np.degrees(best_params[N * 3 + 2])
    
    side = best_score
    
    # Verify and clamp side to be reasonable
    if side < 1.0:
        side = 1.0
    
    return inner, (cx, cy), side, outer_angle
