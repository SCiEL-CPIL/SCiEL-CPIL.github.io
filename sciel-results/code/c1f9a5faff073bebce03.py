import numpy as np
from scipy.optimize import minimize

N = 11

def get_hex_vertices(cx, cy, angle_deg, side=1.0):
    angle_rad = np.deg2rad(angle_deg)
    # Pointy-top vertices
    # Angles: 90, 30, -30, -90, -150, 150
    angles = np.deg2rad([90, 30, -30, -90, -150, 150]) + angle_rad
    return np.array([
        [cx + side * np.cos(a), cy + side * np.sin(a)] for a in angles
    ])

def required_side_length(points, outer_angle_deg=0.0):
    # points: array of shape (M, 2)
    # outer_angle_deg: rotation of the outer hexagon
    # We want to find min S such that all points are inside the outer hexagon.
    # Outer hexagon is centered at (0,0).
    # We can rotate the points by -outer_angle_deg and check against a standard hexagon.
    
    if len(points) == 0:
        return 0.0
        
    theta = np.deg2rad(-outer_angle_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    
    # Rotate points
    x = points[:, 0]
    y = points[:, 1]
    x_rot = x * cos_t - y * sin_t
    y_rot = x * sin_t + y * cos_t
    
    # Standard pointy-top hexagon inequalities for side S:
    # 1. |y| <= S
    # 2. |x + sqrt(3)y| <= sqrt(3)S  => S >= |x + sqrt(3)y| / sqrt(3)
    # 3. |x - sqrt(3)y| <= sqrt(3)S  => S >= |x - sqrt(3)y| / sqrt(3)
    
    S1 = np.max(np.abs(y_rot))
    S2 = np.max(np.abs(x_rot + np.sqrt(3) * y_rot)) / np.sqrt(3)
    S3 = np.max(np.abs(x_rot - np.sqrt(3) * y_rot)) / np.sqrt(3)
    
    return max(S1, S2, S3)

def check_overlap(params, n_hex):
    # params: flat array [x1, y1, a1, x2, y2, a2, ...]
    # Check if any two hexagons overlap.
    # Approximation: Check distance between centers.
    # Min distance for disjoint unit hexagons is sqrt(3) if aligned.
    # If rotated, it can be smaller, but sqrt(3) is a safe lower bound for "touching sides".
    # Actually, if they rotate, they can interlock? No, regular hexagons don't interlock without overlapping.
    # The minimum distance between centers of two disjoint unit hexagons is sqrt(3).
    
    positions = []
    for i in range(n_hex):
        x = params[3*i]
        y = params[3*i+1]
        positions.append((x,y))
        
    for i in range(n_hex):
        for j in range(i+1, n_hex):
            dx = positions[i][0] - positions[j][0]
            dy = positions[i][1] - positions[j][1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < np.sqrt(3) - 1e-6:
                return True
    return False

def objective(params):
    # params: [x1, y1, a1, ..., xN, yN, aN, outer_angle]
    n_hex = N
    # Extract positions and angles
    hexes = []
    for i in range(n_hex):
        x = params[3*i]
        y = params[3*i+1]
        a = params[3*i+2]
        hexes.append((x, y, a))
    
    outer_angle = params[3*n_hex]
    
    # Check overlap penalty
    if check_overlap(params, n_hex):
        return 1000.0 + required_side_length(np.array([[x, y] for x, y, _ in hexes]), outer_angle)
        
    # Calculate required side length
    # We need to check all vertices of all hexagons
    all_points = []
    for x, y, a in hexes:
        verts = get_hex_vertices(x, y, a)
        all_points.append(verts)
        
    all_points = np.vstack(all_points)
    
    S = required_side_length(all_points, outer_angle)
    
    return S

def construct():
    # Initial guess: Compact cluster
    # 7 hexes in K=2 cluster
    sqrt3 = np.sqrt(3)
    init_params = []
    
    # Center
    init_params.extend([0.0, 0.0, 0.0])
    
    # 6 neighbors
    init_params.extend([sqrt3, 0.0, 0.0])
    init_params.extend([-sqrt3, 0.0, 0.0])
    init_params.extend([sqrt3/2, 1.5, 0.0])
    init_params.extend([-sqrt3/2, 1.5, 0.0])
    init_params.extend([sqrt3/2, -1.5, 0.0])
    init_params.extend([-sqrt3/2, -1.5, 0.0])
    
    # 4 more: extend the shape
    # Top
    init_params.extend([0.0, 3.0, 0.0])
    # Bottom
    init_params.extend([0.0, -3.0, 0.0])
    # Top-Right
    init_params.extend([sqrt3, 1.5, 0.0])
    # Bottom-Left
    init_params.extend([-sqrt3, -1.5, 0.0])
    
    # Outer angle
    init_params.append(0.0)
    
    # Optimize
    # Use Nelder-Mead
    res = minimize(objective, init_params, method='Nelder-Mead', 
                   options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-4})
    
    best_params = res.x
    
    # Extract results
    inner = []
    for i in range(N):
        x = best_params[3*i]
        y = best_params[3*i+1]
        a = best_params[3*i+2]
        inner.append([x, y, a])
        
    outer_center = (0.0, 0.0)
    side = objective(best_params)
    outer_angle = best_params[3*N]
    
    return inner, outer_center, side, outer_angle

# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    from scipy.optimize import minimize
    
    N = 11
    sqrt3 = np.sqrt(3)
    
    def get_hex_vertices(cx, cy, angle_deg, side=1.0):
        angle_rad = np.deg2rad(angle_deg)
        angles = np.deg2rad([90, 30, -30, -90, -150, 150]) + angle_rad
        return np.array([
            [cx + side * np.cos(a), cy + side * np.sin(a)] for a in angles
        ])
        
    def required_side_length(points, outer_angle_deg=0.0):
        if len(points) == 0:
            return 0.0
        theta = np.deg2rad(-outer_angle_deg)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        x = points[:, 0]
        y = points[:, 1]
        x_rot = x * cos_t - y * sin_t
        y_rot = x * sin_t + y * cos_t
        S1 = np.max(np.abs(y_rot))
        S2 = np.max(np.abs(x_rot + np.sqrt(3) * y_rot)) / np.sqrt(3)
        S3 = np.max(np.abs(x_rot - np.sqrt(3) * y_rot)) / np.sqrt(3)
        return max(S1, S2, S3)
        
    def check_overlap(params, n_hex):
        positions = []
        for i in range(n_hex):
            x = params[3*i]
            y = params[3*i+1]
            positions.append((x,y))
        for i in range(n_hex):
            for j in range(i+1, n_hex):
                dx = positions[i][0] - positions[j][0]
                dy = positions[i][1] - positions[j][1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < np.sqrt(3) - 1e-6:
                    return True
        return False
        
    def objective(params):
        n_hex = N
        hexes = []
        for i in range(n_hex):
            x = params[3*i]
            y = params[3*i+1]
            a = params[3*i+2]
            hexes.append((x, y, a))
        outer_angle = params[3*n_hex]
        
        if check_overlap(params, n_hex):
            return 1000.0
            
        all_points = []
        for x, y, a in hexes:
            verts = get_hex_vertices(x, y, a)
            all_points.append(verts)
        all_points = np.vstack(all_points)
        S = required_side_length(all_points, outer_angle)
        return S
        
    # Initial guess
    init_params = []
    init_params.extend([0.0, 0.0, 0.0])
    init_params.extend([sqrt3, 0.0, 0.0])
    init_params.extend([-sqrt3, 0.0, 0.0])
    init_params.extend([sqrt3/2, 1.5, 0.0])
    init_params.extend([-sqrt3/2, 1.5, 0.0])
    init_params.extend([sqrt3/2, -1.5, 0.0])
    init_params.extend([-sqrt3/2, -1.5, 0.0])
    init_params.extend([0.0, 3.0, 0.0])
    init_params.extend([0.0, -3.0, 0.0])
    init_params.extend([sqrt3, 1.5, 0.0])
    init_params.extend([-sqrt3, -1.5, 0.0])
    init_params.append(0.0)
    
    try:
        res = minimize(objective, init_params, method='Nelder-Mead', 
                       options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-4})
        best_params = res.x
    except:
        best_params = init_params
        
    inner = []
    for i in range(N):
        x = best_params[3*i]
        y = best_params[3*i+1]
        a = best_params[3*i+2]
        inner.append([x, y, a])
        
    outer_center = (0.0, 0.0)
    side = objective(best_params)
    outer_angle = best_params[3*N]
    
    return inner, outer_center, side, outer_angle
# EVOLVE-BLOCK-END
