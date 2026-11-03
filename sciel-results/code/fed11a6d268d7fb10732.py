import math
import numpy as np
import time
from scipy.optimize import linprog

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Axial coordinates for a compact hexagonal packing of 11 hexagons
    # Center + 6 neighbors + 4 from the next ring (corners of the hexagon)
    # This is a known dense packing for 11 hexagons.
    # Let's use a slightly more symmetric one if possible, or just the standard cluster.
    # Standard cluster: 1 center, 6 around, 4 more.
    # Total 11.
    
    # The "spiral" in the original code:
    # (0,0)
    # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # (2,0), (-2,0), (0,2), (1,-2)
    
    # Let's define a few candidate sets of axial coordinates
    axial_sets = []
    
    # Set 1: The one from the original code
    s1 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (1, -2)
    ]
    axial_sets.append(s1)
    
    # Set 2: A more symmetric "cross" shape
    # Center + 6 neighbors + 4 more placed at (1,1), (-1,2), (-2,1), (2,-1)? No, that's 10.
    # Let's try:
    # (0,0)
    # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # (1,1), (-1,2), (-2,1), (2,-1) -> This is 11 points.
    s2 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ]
    axial_sets.append(s2)
    
    # Set 3: Linear-ish but compact
    # (0,0)
    # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # (2,-1), (-2,1), (2,0), (-2,0)
    s3 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, -1), (-2, 1), (2, 0), (-2, 0)
    ]
    axial_sets.append(s3)

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    # Pre-compute cartesian coordinates for each set
    cart_sets = [axial_to_cartesian(s) for s in axial_sets]
    
    # We will test these sets with various rotations of the OUTER hexagon.
    # Note: The inner hexagons are fixed at 0 rotation (axis-aligned).
    # The outer hexagon rotates.
    
    best_S = float('inf')
    best_beta = 0.0
    best_center = None
    best_pts = None

    # Search over outer hexagon rotation (beta)
    # Symmetry period is pi/6 (30 degrees).
    # Let's scan finely.
    
    num_betas = 1200
    betas = np.linspace(0, math.pi/6, num_betas, endpoint=False)
    
    # For speed, we can parallelize or just loop. 1200 * 3 sets = 3600 LPs.
    # Each LP is small (3 vars, ~132 constraints). Should be fast.
    
    for pts in cart_sets:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
            
        for beta in betas:
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
                
            # Normals for the outer hexagon with rotation beta
            # The sides of the outer hexagon are perpendicular to these normals.
            # For a regular hexagon with side S, the apothem is A = S * sqrt(3)/2.
            # The distance from center to each side is A.
            # If the outer hexagon is rotated by beta, its normals are at angles:
            # beta + pi/6 + k * pi/3 for k=0..5
            # Wait, standard hexagon (flat top) has normals at 0, 60, 120...
            # Pointy top has normals at 30, 90, 150...
            # Let's stick to the definition used in the previous code which worked.
            # Previous code:
            # normals = np.array([
            #     [math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
            #      math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0)]
            #     for k in range(6)
            # ])
            
            normals = np.array([
                [math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
                 math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0)]
                for k in range(6)
            ])
            
            # Variables: cx, cy, D (where D is the apothem of the outer hexagon)
            # Minimize D (or S = (D+A)/A, which is monotonic with D)
            c_obj = np.array([0.0, 0.0, 1.0])
            
            A_ub = []
            b_ub = []
            
            # Constraints:
            # For each normal n_k = (nx, ny) and each point p_i = (px, py):
            # The point must be inside the hexagon.
            # The hexagon is defined by | (p - c) . n_k | <= D for all k?
            # No, the hexagon is the intersection of 6 half-planes.
            # (p - c) . n_k <= D  AND  (p - c) . (-n_k) <= D
            # i.e., -D <= (p - c) . n_k <= D
            # (px - cx) * nx + (py - cy) * ny <= D
            # (cx - px) * nx + (cy - py) * ny <= D
            
            for nx, ny in normals:
                for px, py in pts:
                    # (px - cx)nx + (py - cy)ny <= D
                    # -nx*cx - ny*cy - D <= -px*nx - py*ny
                    # A_ub row: [-nx, -ny, -1]
                    # b_ub val: -px*nx - py*ny
                    A_ub.append([-nx, -ny, -1.0])
                    b_ub.append(-(px * nx + py * ny))
                    
                    # (cx - px)nx + (cy - py)ny <= D
                    # nx*cx + ny*cy - D <= px*nx + py*ny
                    # A_ub row: [nx, ny, -1]
                    # b_ub val: px*nx + py*ny
                    A_ub.append([nx, ny, -1.0])
                    b_ub.append(px * nx + py * ny)
            
            A_ub = np.array(A_ub)
            b_ub = np.array(b_ub)
            bounds = [(None, None), (None, None), (0, None)]
            
            try:
                res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
                if res.status == 0:
                    D = res.x[2]
                    cx = res.x[0]
                    cy = res.x[1]
                    S = (D + APOTHEM) / APOTHEM
                    
                    if S < best_S:
                        best_S = S
                        best_beta = beta
                        best_center = (cx, cy)
                        best_pts = pts
                        
            except Exception:
                pass

    # Refine the best beta found
    if best_center is not None and best_S < float('inf'):
        # Local search around best_beta
        # Use a finer grid or minimize_scalar
        from scipy.optimize import minimize_scalar
        
        def obj_func(b):
            if b < 0 or b > math.pi/6:
                return float('inf')
            # Re-evaluate for this specific beta
            # We can reuse the logic but it's cleaner to just call a helper or inline it.
            # For simplicity and safety, let's do a small local grid search.
            # Or just trust the 1200 steps which is ~0.004 deg resolution. That's plenty.
            return best_S
            
        # Actually, let's just do a few more fine steps around the best
        # best_beta is in [0, pi/6].
        # Let's check neighbors in the betas array or just refine.
        # Given 1200 steps, the error is small. 
        # Let's just ensure we have the exact points for the output.
        
        # We need to output the inner hexagons.
        # The inner hexagons are at positions best_pts with rotation 0.
        # The outer hexagon is at best_center with side best_S and rotation best_beta.
        
        # Wait, the output format is:
        # inner: list of [x, y, angle_degrees] for each inner hexagon.
        # center: [cx, cy] for outer hexagon.
        # side: S
        # angle: rotation of outer hexagon in degrees.
        
        # The inner hexagons are regular hexagons of side 1.
        # In our coordinate system, they are "pointy-top" or "flat-top"?
        # Axial to Cartesian:
        # x = sqrt(3) * (q + r/2)
        # y = 1.5 * r
        # This corresponds to pointy-top hexagons?
        # Let's check vertices of a unit hexagon at (0,0) in this system.
        # Axial neighbors: (1,0) -> (sqrt(3), 0). (0,1) -> (sqrt(3)/2, 1.5).
        # Distance to (1,0) is sqrt(3).
        # The hexagon vertices for a unit side length hexagon centered at origin:
        # If pointy top: (0, 1), (sqrt(3)/2, 0.5), (sqrt(3)/2, -0.5), (0, -1), (-sqrt(3)/2, -0.5), (-sqrt(3)/2, 0.5).
        # Distance from center to vertex (apothem? no, radius) is 1.
        # Distance between centers of adjacent hexagons in a lattice:
        # For pointy top, horizontal distance is sqrt(3). Vertical is 1.5.
        # Our axial_to_cartesian gives exactly this.
        # So the inner hexagons are pointy-top, axis-aligned (angle 0).
        
        # The evaluator expects angle_degrees for each inner hexagon.
        # Since they are aligned with the lattice, angle is 0.
        
        inner = []
        for x, y in best_pts:
            inner.append([float(x), float(y), 0.0])
            
        cx, cy = best_center
        angle_deg = math.degrees(best_beta)
        
        # Safety check: ensure no overlap?
        # The LP ensures they are inside the outer hexagon.
        # It does NOT ensure they don't overlap each other!
        # The LP only checks against the BOUNDARY of the outer hexagon.
        # The points best_pts are fixed from the axial grid, which are non-overlapping.
        # So this is valid.
        
        return inner, [float(cx), float(cy)], float(best_S), float(angle_deg)
    
    # Fallback
    return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

