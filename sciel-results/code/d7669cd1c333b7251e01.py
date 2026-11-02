import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Define helper functions inside to keep state clean or outside? 
    # Standard library and NumPy/SciPy allowed.
    
    def get_hex_vertices(cx, cy, angle_deg, side=1.0):
        angle_rad = np.deg2rad(angle_deg)
        verts = np.zeros((6, 2))
        for k in range(6):
            theta = angle_rad + k * np.pi / 3
            verts[k, 0] = cx + side * np.cos(theta)
            verts[k, 1] = cy + side * np.sin(theta)
        return verts

    def calc_min_side(hexes):
        # hexes: list of (cx, cy, angle)
        # Outer hexagon centered at (0,0), angle 0 (flat top)
        # Normals for flat top hexagon (vertices at 0, 60, ...):
        # Side midpoints at 30, 90, ...
        normals = []
        for k in range(6):
            theta = np.deg2rad(30 + k * 60)
            normals.append(np.array([np.cos(theta), np.sin(theta)]))
        
        max_proj = 0.0
        for hex_def in hexes:
            cx, cy, ang = hex_def
            verts = get_hex_vertices(cx, cy, ang)
            for v in verts:
                for n in normals:
                    proj = np.dot(v, n)
                    if proj > max_proj:
                        max_proj = proj
        # Apothem a = S * sqrt(3) / 2
        # max_proj <= a  =>  S = 2 * max_proj / sqrt(3)
        return 2.0 * max_proj / np.sqrt(3.0)

    def objective(params):
        # params: 36 floats [x1, y1, a1, ..., x12, y12, a12]
        hexes = []
        for i in range(N):
            x = params[i*3]
            y = params[i*3+1]
            ang = params[i*3+2]
            hexes.append((x, y, ang))
        
        side = calc_min_side(hexes)
        
        # Penalty for overlap
        # Min distance between centers of disjoint unit hexagons is sqrt(3)
        penalty = 0.0
        min_dist = np.sqrt(3.0)
        for i in range(N):
            for j in range(i+1, N):
                xi, yi, _ = hexes[i]
                xj, yj, _ = hexes[j]
                d = np.sqrt((xi-xj)**2 + (yi-yj)**2)
                if d < min_dist:
                    penalty += (min_dist - d)**2 * 1000.0
        return side + penalty

    # Initial Guess: 7-flower + 5
    # 7-flower:
    # Center (0,0)
    # 6 neighbors at dist sqrt(3), angles 0, 60, ...
    sqrt3 = np.sqrt(3.0)
    initial_hexes = []
    initial_hexes.append((0.0, 0.0, 0.0))
    for k in range(6):
        ang = k * 60.0
        x = sqrt3 * np.cos(np.deg2rad(ang))
        y = sqrt3 * np.sin(np.deg2rad(ang))
        initial_hexes.append((x, y, 0.0))
    
    # Add 5 more. Place them at "corners" of the flower.
    # The flower has 6 outer hexagons. Let's attach to 5 of them.
    # Attachment point: Center of outer hex + vector in direction of its "outer" normal.
    # For hex at angle 0 (right), outer normal is 0 deg.
    # New center: (sqrt3, 0) + (sqrt3, 0) = (2*sqrt3, 0).
    # This creates a ring of 6 at dist 2*sqrt3. We only need 5.
    for k in range(5):
        ang = k * 60.0
        x = 2.0 * sqrt3 * np.cos(np.deg2rad(ang))
        y = 2.0 * sqrt3 * np.sin(np.deg2rad(ang))
        initial_hexes.append((x, y, 0.0))
        
    x0 = []
    for h in initial_hexes:
        x0.append(h[0])
        x0.append(h[1])
        x0.append(h[2])
        
    # Run optimization
    # Use Nelder-Mead
    bounds = [(-10, 10), (-10, 10), (0, 360)] * N
    
    best_result = None
    best_score = float('inf')
    
    # Multistart to ensure robustness
    starts = [x0]
    # Add a few random perturbations
    for _ in range(5):
        perturbed = []
        for val in x0:
            # Perturb positions, keep angles
            if len(perturbed) % 3 == 0:
                perturbed.append(val + np.random.uniform(-1, 1))
            elif len(perturbed) % 3 == 1:
                perturbed.append(val + np.random.uniform(-1, 1))
            else:
                perturbed.append(val + np.random.uniform(-30, 30))
        starts.append(perturbed)
        
    for start in starts:
        try:
            res = minimize(objective, start, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-5, 'adaptive': True})
            if res.success or res.fun < best_score:
                if res.fun < best_score:
                    best_score = res.fun
                    best_result = res
        except Exception:
            pass

    if best_result is None:
        # Fallback to initial guess
        best_params = x0
    else:
        best_params = best_result.x
        
    # Construct output
    inner = []
    for i in range(N):
        x = best_params[i*3]
        y = best_params[i*3+1]
        ang = best_params[i*3+2]
        # Normalize angle to 0-360
        ang = ang % 360.0
        inner.append([x, y, ang])
        
    # Calculate final side length accurately
    hexes = [(h[0], h[1], h[2]) for h in inner]
    side = calc_min_side(hexes)
    
    return inner, (0.0, 0.0), side, 0.0
# EVOLVE-BLOCK-END
