import math
import numpy as np
import time
from scipy.optimize import linprog, minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0
EPS = 1e-9

def construct():
    # Check if we are already close to the time limit
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Generate 11 points on a triangular lattice.
    # Axial coordinates (q, r)
    # Center: (0,0)
    # Ring 1: 6 points
    # Ring 2: 4 points (to make 11 total)
    axial = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ]
    
    # No scaling: we will ensure disjointness via a small epsilon offset in the LP constraints.
    # The theoretical minimum distance between centers of touching unit hexagons is sqrt(3).
    # We will add a small epsilon to the separation constraint to ensure strict disjointness.
    
    pts = []
    for q, r in axial:
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        pts.append((x, y))
        
    pts = np.array(pts)
    
    # We want to minimize the side length S of the outer regular hexagon.
    # The outer hexagon is defined by center (cx, cy), side length S, and rotation beta.
    # The apothem of the outer hexagon is A_outer = S * APOTHEM.
    # The condition for a point p to be inside the outer hexagon is:
    # | (p - c) . n_k | <= A_outer  for all 6 normals n_k.
    # Here n_k are the unit normals to the sides of the outer hexagon.
    # n_k = (cos(beta + pi/6 + k*pi/3), sin(beta + pi/6 + k*pi/3))
    
    # The inner hexagons are also regular hexagons of side 1.
    # We assume they are rotated by the same angle beta as the outer hexagon.
    # This aligns their normals with the outer normals.
    # The support function of a unit hexagon in the direction of its normal is APOTHEM.
    # So, the condition for the inner hexagon centered at p_i to be inside the outer hexagon is:
    # (p_i - c) . n_k + APOTHEM <= S * APOTHEM
    # (p_i - c) . n_k <= S * APOTHEM - APOTHEM
    # Let D = S * APOTHEM - APOTHEM.
    # Then | (p_i - c) . n_k | <= D.
    # We minimize D.
    # S = (D + APOTHEM) / APOTHEM
    
    def calc_S_for_beta(beta):
        # Check time limit
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), None, None
        
        normals = []
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))
        
        # Variables: cx, cy, D
        # Minimize D
        c_obj = np.array([0.0, 0.0, 1.0])
        
        A_ub = []
        b_ub = []
        
        for nx, ny in normals:
            for px, py in pts:
                # Constraint 1: (px - cx)nx + (py - cy)ny <= D
                # px nx - cx nx + py ny - cy ny - D <= 0
                # -nx cx - ny cy - D <= -(px nx + py ny)
                A_ub.append([-nx, -ny, -1.0])
                b_ub.append(-(px * nx + py * ny))
                
                # Constraint 2: -((px - cx)nx + (py - cy)ny) <= D
                # (cx - px)nx + (cy - py)ny - D <= 0
                # nx cx + ny cy - D <= px nx + py ny
                A_ub.append([nx, ny, -1.0])
                b_ub.append(px * nx + py * ny)
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        # Bounds
        # D must be >= 0
        bounds = [(None, None), (None, None), (0, None)]
        
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                D = res.x[2]
                cx = res.x[0]
                cy = res.x[1]
                S = (D + APOTHEM) / APOTHEM
                return S, (cx, cy), D
            else:
                return float('inf'), None, None
        except Exception:
            return float('inf'), None, None

    # Minimize S over beta
    best_S = float('inf')
    best_beta = 0.0
    best_center = None
    
    # Sample beta first to find a good starting point
    # The symmetry suggests beta in [0, pi/6]
    for i in range(13):
        beta = i * (math.pi / 6) / 12
        S, center, D = calc_S_for_beta(beta)
        if S < best_S:
            best_S = S
            best_beta = beta
            best_center = center
    
    # Refine with minimize_scalar
    try:
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            res = minimize_scalar(lambda b: calc_S_for_beta(b)[0], bounds=(0, math.pi / 6), method='bounded',
                                 options={'maxiter': 100, 'xatol': 1e-8})
            if res.fun < best_S:
                best_S = res.fun
                best_beta = res.x
                _, best_center, _ = calc_S_for_beta(best_beta)
    except Exception:
        pass
    
    if best_center is None or best_S == float('inf'):
        # Fallback: Calculate a valid S using the centroid and a simple bound
        best_center = (float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1])))
        best_beta = 0.0
        # Calculate the maximum distance from center to any vertex of any inner hexagon
        # plus the apothem of the inner hexagon (approximation for worst-case alignment)
        max_proj = 0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = pts[:, 0] * nx + pts[:, 1] * ny
            max_proj = max(max_proj, np.max(projs), -np.min(projs))
        # S must be large enough such that S * APOTHEM >= max_proj + APOTHEM
        # D = S * APOTHEM. D >= max_proj + APOTHEM.
        # S >= (max_proj + APOTHEM) / APOTHEM
        best_S = (max_proj + APOTHEM) / APOTHEM

    # Prepare output
    # Inner hexagons: (x, y, angle_degrees)
    # The angle of the inner hexagons should be the same as the outer one to align normals.
    angle_deg = math.degrees(best_beta)
    
    # To ensure strict disjointness, we need to add a small perturbation to the inner hexagon positions.
    # Since the LP finds the tightest fit, adjacent hexagons may be touching.
    # We shift the entire configuration slightly away from the center to create gaps.
    # Alternatively, we can scale the positions by a tiny factor > 1.
    # But scaling changes the shape. Shifting is safer but might break symmetry.
    # Let's try a small radial expansion from the centroid.
    
    cx, cy = best_center
    inner = []
    for x, y in pts:
        # Small radial expansion
        dx = x - cx
        dy = y - cy
        dist = math.hypot(dx, dy)
        if dist > 1e-12:
            # Expand by a tiny amount
            # The gap needed is roughly EPS.
            # If we expand radially by delta, the distance between two adjacent centers increases by ~ 2*delta*cos(theta)
            # For safety, let's just scale the whole cluster by 1+1e-8
            pass
            
    # Actually, scaling the points by (1+eps) is the simplest and most robust way to ensure disjointness
    # without worrying about direction.
    # However, this changes the optimal S slightly.
    # Let's re-run the LP with scaled points to get the correct S for the scaled configuration.
    # This is computationally cheap (one more LP).
    
    SCALE = 1.0 + 1e-6
    pts_scaled = []
    for x, y in pts:
        # Scale relative to the center found by the LP
        # This ensures the configuration remains centered
        sx = cx + (x - cx) * SCALE
        sy = cy + (y - cy) * SCALE
        pts_scaled.append((sx, sy))
    
    pts_scaled = np.array(pts_scaled)
    
    # Re-calculate S for the scaled points with the same beta and center
    # We can just compute the max projection directly since the center is fixed.
    max_proj = 0
    for k in range(6):
        ang = best_beta + math.pi / 6.0 + k * math.pi / 3.0
        nx, ny = math.cos(ang), math.sin(ang)
        projs = (pts_scaled[:, 0] - cx) * nx + (pts_scaled[:, 1] - cy) * ny
        max_proj = max(max_proj, np.max(projs), -np.min(projs))
    
    D_scaled = max_proj
    S_scaled = (D_scaled + APOTHEM) / APOTHEM
    
    inner = []
    for x, y in pts_scaled:
        inner.append([float(x), float(y), float(angle_deg)])
        
    center = [float(cx), float(cy)]
    side = float(S_scaled)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation
