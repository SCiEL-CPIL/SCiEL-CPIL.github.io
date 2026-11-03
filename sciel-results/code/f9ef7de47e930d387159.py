import numpy as np
import time
import scipy.optimize

N = 11

def hex_vertices(x, y, angle_deg):
    """Generate vertices of a unit hexagon centered at (x,y) with given rotation."""
    angle_rad = np.deg2rad(angle_deg)
    verts = np.array([
        [1.0, 0.0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[c, -s], [s, c]])
    verts = verts @ R.T
    verts += np.array([x, y])
    return verts

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin

    # Initial guess: Use a hexagonal grid pattern for 11 hexagons
    # A tight cluster of 7 hexagons (1 center + 6 around) plus 4 more
    d = np.sqrt(3)  # distance between adjacent hex centers
    
    # 7-hex cluster centers
    init_centers = [
        (0, 0),
        (d, 0), (d/2, 1.5), (-d/2, 1.5),
        (-d, 0), (-d/2, -1.5), (d/2, -1.5)
    ]
    
    # Add 4 more hexagons extending the pattern
    # Place them in positions that maintain hexagonal symmetry
    additional_centers = [
        (2*d, 0),
        (1.5*d, 3.0),
        (-1.5*d, 3.0),
        (0, 4.5)
    ]
    
    init_params = []
    all_centers = init_centers + additional_centers
    for i in range(N):
        x, y = all_centers[i]
        ang = 0.0
        init_params.extend([x, y, ang])
    
    init_params = np.array(init_params)

    def calc_penalty_and_side(params):
        # Reshape to N x 3
        pts = params.reshape(N, 3)
        xs = pts[:, 0]
        ys = pts[:, 1]
        thetas = pts[:, 2]
        
        # 1. Calculate required outer hexagon side
        # Normals of outer hexagon (centered at 0,0, rotation 0)
        # Faces are perpendicular to vectors at angles 0, 60, ...
        normals = np.array([
            [1.0, 0.0],
            [0.5, np.sqrt(3)/2],
            [-0.5, np.sqrt(3)/2],
            [-1.0, 0.0],
            [-0.5, -np.sqrt(3)/2],
            [0.5, -np.sqrt(3)/2]
        ])
        
        max_proj = 0.0
        for k in range(N):
            verts = hex_vertices(xs[k], ys[k], thetas[k])
            # Project all vertices onto all normals
            projs = verts @ normals.T  # 6x6
            max_proj = max(max_proj, np.max(projs))
        
        # Apothem A = max_proj
        # Side S = 2 * A / sqrt(3)
        side = 2.0 * max_proj / np.sqrt(3)
        
        # 2. Calculate overlap penalty
        penalty = 0.0
        min_dist = np.sqrt(3)
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < min_dist:
                    penalty += (min_dist - dist)**2
        
        return side, penalty

    def objective(params):
        side, penalty = calc_penalty_and_side(params)
        return side + 100.0 * penalty

    best_side = float('inf')
    best_params = init_params.copy()
    
    # Use Nelder-Mead for local search
    # We will do several random restarts with different initial configurations
    np.random.seed(42)
    
    num_restarts = 15
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        # Generate different initial configurations
        if r == 0:
            current_params = init_params.copy()
        elif r == 1:
            # Slightly perturbed version
            current_params = init_params + np.random.normal(0, 0.5, N*3)
            current_params[2::3] = 0.0  # Keep angles at 0
        elif r == 2:
            # Rotated initial configuration
            angle = np.deg2rad(30)
            c, s = np.cos(angle), np.sin(angle)
            R = np.array([[c, -s], [s, c]])
            rotated = []
            for i in range(N):
                x, y, a = init_params[i*3], init_params[i*3+1], init_params[i*3+2]
                new_x, new_y = R @ np.array([x, y])
                new_a = a + 30
                rotated.extend([new_x, new_y, new_a])
            current_params = np.array(rotated)
        elif r == 3:
            # Different layout: more compact
            current_params = init_params.copy()
            # Move some hexagons closer
            current_params[15] += 0.3  # x of 6th hex
            current_params[16] -= 0.2  # y of 6th hex
            current_params[18] -= 0.3  # x of 7th hex
            current_params[19] -= 0.2  # y of 7th hex
        elif r == 4:
            # Random scatter with constraint
            current_params = np.zeros(N*3)
            for i in range(N):
                if i == 0:
                    current_params[i*3] = 0
                    current_params[i*3+1] = 0
                else:
                    # Place in a ring around center
                    angle = 2*np.pi*i/(N-1)
                    radius = 2.5 + np.random.uniform(-0.5, 0.5)
                    current_params[i*3] = radius * np.cos(angle)
                    current_params[i*3+1] = radius * np.sin(angle)
                current_params[i*3+2] = np.random.uniform(-10, 10)
        elif r == 5:
            # Grid-like pattern
            current_params = np.zeros(N*3)
            positions = [
                (0, 0), (d, 0), (-d, 0),
                (d/2, 1.5), (-d/2, 1.5),
                (d/2, -1.5), (-d/2, -1.5),
                (2*d, 0), (-2*d, 0),
                (d, 3), (-d, 3)
            ]
            for i in range(N):
                current_params[i*3] = positions[i][0]
                current_params[i*3+1] = positions[i][1]
                current_params[i*3+2] = 0
        else:
            # Random perturbation of best found so far
            if best_side < float('inf'):
                current_params = best_params + np.random.normal(0, 1.0, N*3)
                current_params[2::3] = np.random.uniform(-10, 10, N)
            else:
                current_params = init_params + np.random.normal(0, 1.0, N*3)
                current_params[2::3] = np.random.uniform(-10, 10, N)
        
        # Run optimization
        try:
            res = scipy.optimize.minimize(
                objective, 
                current_params, 
                method='Nelder-Mead', 
                options={'maxiter': 800, 'xatol': 1e-4, 'fatol': 1e-4}
            )
            
            side, pen = calc_penalty_and_side(res.x)
            if pen < 1e-4:  # Valid packing
                if side < best_side:
                    best_side = side
                    best_params = res.x.copy()
        except Exception:
            pass
    
    # If no valid packing found, try a simpler grid or return seed
    if best_side == float('inf'):
        # Fallback: return seed-like structure
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    # Construct output
    inner = []
    for i in range(N):
        x, y, ang = best_params[i*3], best_params[i*3+1], best_params[i*3+2]
        inner.append([float(x), float(y), float(ang)])
        
    return inner, (0.0, 0.0), float(best_side), 0.0
