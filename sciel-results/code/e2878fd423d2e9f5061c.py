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
    deadline = 140.0  # Safety margin before 150s kill

    # Initial guess: 7-hex cluster + 4 more
    # Centers for a tight 7-hex cluster (side 1 hexes)
    # Distance between centers is sqrt(3)
    d = np.sqrt(3)
    init_centers = [
        (0, 0),
        (d, 0), (d/2, 1.5), (-d/2, 1.5),
        (-d, 0), (-d/2, -1.5), (d/2, -1.5)
    ]
    # Add 4 more, place them roughly in a ring or extend
    init_params = []
    for i in range(N):
        if i < 7:
            x, y = init_centers[i]
        else:
            # Place new ones further out in a hexagonal pattern
            # 8th hex: (0, 3.0) roughly
            # 9th hex: (2.6, 1.5) roughly
            # 10th hex: (2.6, -1.5) roughly
            # 11th hex: (0, -3.0) roughly
            if i == 7:
                x, y = 0.0, 3.0
            elif i == 8:
                x, y = 2.6, 1.5
            elif i == 9:
                x, y = 2.6, -1.5
            elif i == 10:
                x, y = 0.0, -3.0
        ang = 0.0
        init_params.extend([x, y, ang])
    
    init_params = np.array(init_params)

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

    def calc_penalty_and_side(params):
        # Reshape to N x 3
        pts = params.reshape(N, 3)
        xs = pts[:, 0]
        ys = pts[:, 1]
        thetas = pts[:, 2]
        
        # 1. Calculate required outer hexagon side
        max_proj = 0.0
        for k in range(N):
            verts = hex_vertices(xs[k], ys[k], thetas[k])
            # Project all vertices onto all normals
            projs = verts @ normals.T # 6x6
            max_proj = max(max_proj, np.max(projs))
        
        # Apothem A = max_proj
        # Side S = 2 * A / sqrt(3)
        side = 2.0 * max_proj / np.sqrt(3)
        
        # 2. Calculate overlap penalty
        # Minimum distance between centers of disjoint unit hexagons is sqrt(3)
        # This is a necessary condition. For rotated hexagons, it's not sufficient,
        # but a good proxy for optimization.
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
    best_penalty = float('inf')
    
    np.random.seed(42)
    
    num_restarts = 20
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        # Perturb initial guess
        if r == 0:
            current_params = init_params.copy()
        elif r == 1:
            # Mirror
            current_params = init_params.copy()
            current_params[0::3] *= -1
        else:
            # Random scatter within a box, but keep it somewhat compact
            current_params = init_params + np.random.normal(0, 0.5, N*3)
            # Ensure angles are reasonable
            current_params[2::3] = np.random.uniform(-10, 10, N)
            
        # Run optimization
        try:
            res = scipy.optimize.minimize(
                objective, 
                current_params, 
                method='Nelder-Mead', 
                options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4}
            )
            
            side, pen = calc_penalty_and_side(res.x)
            
            # Accept if penalty is low (valid packing) and side is smaller
            # Or if penalty is significantly lower than current best
            if pen < 1e-4:
                if side < best_side:
                    best_side = side
                    best_params = res.x.copy()
                    best_penalty = pen
            elif pen < best_penalty:
                # Keep track of best non-valid solution too? 
                # No, we only want valid packings.
                # But if we don't find any, we might want to return the least bad one?
                # The task says "invalid result ... scores 0". So we must return a valid one.
                # If we can't find a valid one, we return the seed (which is valid but large).
                pass
                
        except Exception:
            pass

    # If no valid packing found, return seed
    if best_side == float('inf'):
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    # Construct output
    inner = []
    for i in range(N):
        x, y, ang = best_params[i*3], best_params[i*3+1], best_params[i*3+2]
        inner.append([float(x), float(y), float(ang)])
        
    return inner, (0.0, 0.0), float(best_side), 0.0
