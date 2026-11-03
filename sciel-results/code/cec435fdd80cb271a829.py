import math
import numpy as np
import time
from scipy.optimize import minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0
TIME_BUDGET = 120.0

def construct():
    if time.time() - START_TIME > TIME_BUDGET - SAFETY_MARGIN:
        # Fallback: Return a known valid configuration with S=3
        # 11 points from the 19-hex block
        pts = [
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (2, 0), (0, 2), (-2, 0), (0, -2)
        ]
        # Convert to cartesian
        cart_pts = []
        for q, r in pts:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            cart_pts.append((x, y))
        
        # Calculate S
        # Center is (0,0)
        # Max projection
        max_proj = 0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = [x * nx + y * ny for x, y in cart_pts]
            max_proj = max(max_proj, max(projs), -min(projs))
        S = (max_proj + APOTHEM) / APOTHEM
        
        inner = []
        for x, y in cart_pts:
            inner.append([float(x), float(y), 0.0])
        
        return inner, [0.0, 0.0], float(S), 0.0

    # Generate the 19-point block (side 2)
    axial_19 = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q + r) <= 2:
                axial_19.append((q, r))
    
    # We need to choose 11 points from these 19.
    # To minimize S, we should keep the cluster as compact as possible.
    # The 7 central points are essential.
    # We need to pick 4 more from the outer 12.
    # The outer 12 are at distance > 1.
    
    # Let's try all combinations of 4 points from the outer 12?
    # 12 choose 4 is 495. This is small.
    
    central_7 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    outer_12 = [p for p in axial_19 if p not in central_7]
    
    best_S = float('inf')
    best_pts = None
    best_angle = 0.0
    
    import itertools
    
    # Pre-compute cartesian for all 19
    all_cart = {}
    for q, r in axial_19:
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        all_cart[(q, r)] = (x, y)
        
    # Indices of outer points
    outer_indices = range(len(outer_12))
    
    for combo in itertools.combinations(outer_indices, 4):
        if time.time() - START_TIME > TIME_BUDGET - SAFETY_MARGIN:
            break
            
        selected_axial = central_7 + [outer_12[i] for i in combo]
        
        # Get cartesian points
        pts = np.array([all_cart[p] for p in selected_axial])
        
        # Optimize rotation and center
        # Center should be centroid for symmetry, but let's optimize it.
        # Actually, for a hexagon, the optimal center is the one that minimizes max projection.
        # For a symmetric set, centroid is good.
        
        def calc_S(pts, beta, cx, cy):
            angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
            normals = np.column_stack([np.cos(angles), np.sin(angles)])
            centered = pts - np.array([cx, cy])
            projs = centered @ normals.T
            max_proj = np.max(np.abs(projs))
            return (max_proj + APOTHEM) / APOTHEM
        
        # Try a few angles
        best_local_S = float('inf')
        best_local_beta = 0.0
        best_local_center = None
        
        # Coarse search for beta
        for i in range(12):
            beta = i * (math.pi / 6) / 12
            # Optimize center for this beta
            # The optimal center minimizes max_k |proj_k|
            # This is a linear programming problem, or we can approximate.
            # For small N, we can just use the centroid as a starting point and do a few steps of gradient descent or Nelder-Mead.
            
            cx, cy = np.mean(pts[:, 0]), np.mean(pts[:, 1])
            
            # Refine center and beta with Nelder-Mead
            def objective(params):
                c_x, c_y, b = params
                return calc_S(pts, b, c_x, c_y)
                
            init = [cx, cy, beta]
            try:
                res = minimize_scalar(lambda b: (lambda cx, cy: calc_S(pts, b, cx, cy))(
                    *minimize(objective, [cx, cy, b], method='Nelder-Mead', options={'maxiter': 500}).x[:2]
                ), bounds=(beta - math.pi/12, beta + math.pi/12), method='bounded', options={'maxiter': 100})
                
                # The above is complex. Let's just use Nelder-Mead on all 3 variables.
                res = minimize(objective, init, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
                if res.fun < best_local_S:
                    best_local_S = res.fun
                    best_local_beta = res.x[2]
                    best_local_center = (res.x[0], res.x[1])
            except Exception:
                pass

        if best_local_S < best_S:
            best_S = best_local_S
            best_pts = pts
            best_angle = best_local_beta
            best_center = best_local_center

    if best_pts is None:
        # Fallback
        selected_axial = central_7 + outer_12[:4]
        best_pts = np.array([all_cart[p] for p in selected_axial])
        best_center = (0.0, 0.0)
        best_angle = 0.0
        max_proj = 0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = best_pts[:, 0] * nx + best_pts[:, 1] * ny
            max_proj = max(max_proj, np.max(projs), -np.min(projs))
        best_S = (max_proj + APOTHEM) / APOTHEM

    cx, cy = best_center
    angle_deg = math.degrees(best_angle)
    
    # Scale slightly to ensure disjointness if needed, but lattice points are disjoint.
    # The container side is best_S.
    
    inner = []
    for x, y in best_pts:
        inner.append([float(x), float(y), float(angle_deg)])

    center = [float(cx), float(cy)]
    side = float(best_S)
    rotation = float(angle_deg)

    return inner, center, side, rotation
