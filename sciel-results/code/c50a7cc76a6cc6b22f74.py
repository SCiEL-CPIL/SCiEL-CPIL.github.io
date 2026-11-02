import math
import numpy as np
import time
from scipy.optimize import minimize

START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0
EPS = 1e-9

# Constants
APOTHEM = math.sqrt(3) / 2.0
MIN_DIST = math.sqrt(3) - 1e-9 # Min distance between centers of non-overlapping unit hexes (aligned)

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        # Fallback: simple grid
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    N = 11
    
    # Initial configuration: 7-hex flower + 4 extensions
    # Axial coordinates (q, r)
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    
    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    centers = axial_to_cartesian(axial_coords)
    
    # Initial outer hexagon parameters
    # Center of mass
    cx_out = np.mean(centers[:, 0])
    cy_out = np.mean(centers[:, 1])
    # Initial apothem estimate
    # Max projection of any vertex
    max_proj = 0
    for px, py in centers:
        # Vertices of unit hexagon at (px, py)
        # Relative vertices: (1,0), (0.5, h), (-0.5, h), (-1,0), (-0.5, -h), (0.5, -h)
        for vx, vy in [(1,0), (0.5, APOTHEM), (-0.5, APOTHEM), (-1,0), (-0.5, -APOTHEM), (0.5, -APOTHEM)]:
            # Project onto x and y axes initially (rotation 0)
            proj_x = abs((px + vx) - cx_out)
            proj_y = abs((py + vy) - cy_out)
            # For a hexagon aligned with axes, the constraint is on the 6 normals.
            # Normals for rotation 0:
            # (1, 0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1, 0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
            # Actually, let's just compute the required apothem for rotation 0
            # The condition is max over normals of | (P - C) . n | <= A
            pass
    
    # Let's use a more robust initialization for A_out
    # Calculate the bounding box of all vertices
    all_vertices = []
    for px, py in centers:
        for vx, vy in [(1,0), (0.5, APOTHEM), (-0.5, APOTHEM), (-1,0), (-0.5, -APOTHEM), (0.5, -APOTHEM)]:
            all_vertices.append((px + vx, py + vy))
    all_vertices = np.array(all_vertices)
    
    # For a hexagon with rotation 0, the apothem A must satisfy:
    # A >= max( |x - cx|, |y - cy| * 2/sqrt(3) ... )? No.
    # The width in x direction is 2A * 1? No.
    # The projection of a point (x,y) onto normal (1,0) is x.
    # The projection onto normal (0.5, sqrt(3)/2) is 0.5x + sqrt(3)/2 y.
    # We need A >= max_k | (P - C) . n_k |
    
    def calc_required_apothem(centers, cx, cy, rotation):
        max_proj = 0
        # Normals for the outer hexagon
        # The normals are perpendicular to the sides.
        # If the hexagon is rotated by 'rotation', the normals are also rotated.
        # For a standard hexagon (flat top/bottom), the normals are at angles:
        # 0, 60, 120, 180, 240, 300 degrees?
        # Wait, a regular hexagon has 6 sides.
        # If side is horizontal, normal is vertical (90 deg).
        # Let's define the hexagon such that at rotation=0, one normal is (1,0).
        # This means one side is vertical? No, normal (1,0) means side is vertical (x=const).
        # Let's stick to the definition:
        # Normal k: angle = rotation + k * pi / 3
        # This covers 6 directions.
        
        normals = []
        for k in range(6):
            ang = rotation + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))
        
        for px, py in centers:
            # Check all 6 vertices of the inner hexagon (assumed rotation 0)
            for vx, vy in [(1,0), (0.5, APOTHEM), (-0.5, APOTHEM), (-1,0), (-0.5, -APOTHEM), (0.5, -APOTHEM)]:
                vx_abs = px + vx
                vy_abs = py + vy
                for nx, ny in normals:
                    proj = abs((vx_abs - cx) * nx + (vy_abs - cy) * ny)
                    if proj > max_proj:
                        max_proj = proj
        return max_proj

    A_init = calc_required_apothem(centers, cx_out, cy_out, 0.0)
    
    # Variables:
    # x0..x10, y0..y10 (22)
    # cx, cy (2)
    # A_out (1)
    # rotation (1)
    # Total 26
    
    x0 = np.zeros(26)
    for i in range(N):
        x0[2*i] = centers[i, 0]
        x0[2*i+1] = centers[i, 1]
    x0[22] = cx_out
    x0[23] = cy_out
    x0[24] = A_init
    x0[25] = 0.0 # rotation

    def objective(params):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return 1e10
            
        # Extract centers
        c_in = np.zeros((N, 2))
        for i in range(N):
            c_in[i, 0] = params[2*i]
            c_in[i, 1] = params[2*i+1]
        
        cx = params[22]
        cy = params[23]
        A = params[24]
        rot = params[25]
        
        # Penalty for overlap
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = c_in[i, 0] - c_in[j, 0]
                dy = c_in[i, 1] - c_in[j, 1]
                d_sq = dx*dx + dy*dy
                d = math.sqrt(d_sq)
                if d < MIN_DIST:
                    # Quadratic penalty
                    penalty += (MIN_DIST - d) ** 2
        
        # Penalty for containment violation
        # We want A to be large enough.
        # If A is smaller than required, penalize.
        required_A = calc_required_apothem(c_in, cx, cy, rot)
        
        # We minimize A + penalties
        # If A < required_A, the containment is violated.
        # We can just add (required_A - A)^2 if A < required_A
        if A < required_A:
            penalty += (required_A - A) ** 2
        
        # Also, we want to minimize A itself.
        # So objective is A + lambda * penalty
        # But if we minimize A, it will go to 0 unless penalized.
        # Better: Minimize required_A? No, required_A is a function of positions.
        # If we minimize A, and penalize A < required_A, the optimal A will be exactly required_A.
        # So minimizing A is equivalent to minimizing required_A.
        
        return A + 100.0 * penalty

    # Run optimization
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
        
        if res.fun < 1e9:
            # Extract results
            c_in = np.zeros((N, 2))
            for i in range(N):
                c_in[i, 0] = res.x[2*i]
                c_in[i, 1] = res.x[2*i+1]
            
            cx = res.x[22]
            cy = res.x[23]
            A = res.x[24]
            rot = res.x[25]
            
            # Refine A to be exactly the required value to ensure validity
            A = calc_required_apothem(c_in, cx, cy, rot)
            # Add a tiny safety margin
            A += 1e-9
            
            side = 2.0 * A / math.sqrt(3)
            
            inner = []
            for i in range(N):
                inner.append([float(c_in[i, 0]), float(c_in[i, 1]), 0.0])
            
            center = [float(cx), float(cy)]
            rotation = float(math.degrees(rot))
            
            return inner, center, float(side), rotation
    except Exception:
        pass

    # Fallback if optimization fails or returns bad result
    # Use the initial configuration
    A = calc_required_apothem(centers, cx_out, cy_out, 0.0)
    A += 1e-9
    side = 2.0 * A / math.sqrt(3)
    
    inner = []
    for px, py in centers:
        inner.append([float(px), float(py), 0.0])
    
    center = [float(cx_out), float(cy_out)]
    rotation = 0.0
    
    return inner, center, float(side), rotation
