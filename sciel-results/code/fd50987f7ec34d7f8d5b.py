import math
import numpy as np
import time
from scipy.optimize import linprog

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0
EPS = 1e-9

def construct():
    """
    Returns (inner, center, side, angle) where:
    - inner: list of [x, y, angle_degrees] for N unit hexagons
    - center: [cx, cy] of the outer hexagon
    - side: side length of the outer hexagon
    - angle: rotation angle (degrees) of the outer hexagon
    """
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        # Fallback: simple grid arrangement
        pts = [(0, 0), (2, 0), (-2, 0), (1, 1.732), (-1, 1.732), (0, -3.464),
               (2, 1.732), (-2, 1.732), (3, -1.732), (-3, -1.732), (0, 5.196)]
        pts = np.array(pts)
        cx, cy = float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))
        max_proj = 0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = (pts[:, 0] - cx) * nx + (pts[:, 1] - cy) * ny
            max_proj = max(max_proj, np.max(projs), -np.min(projs))
        side = (max_proj + APOTHEM) / APOTHEM
        inner = [[float(x), float(y), 0.0] for x, y in pts]
        return inner, [cx, cy], float(side), 0.0

    # Axial to cartesian conversion for unit hexagons
    def axial_to_cartesian(q, r):
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        return x, y

    # Generate candidate axial coordinate sets
    def generate_candidates():
        candidates = []
        
        # 1. Standard hexagonal packing (radius 2 hexagon)
        # This is the densest packing for small numbers
        hex_grid = []
        for q in range(-3, 4):
            for r in range(-3, 4):
                s = -q - r
                if max(abs(q), abs(r), abs(s)) <= 2:
                    hex_grid.append((q, r))
        
        # Remove the center and some others to get exactly 11, keeping compactness
        # Actually, let's just take the first 11 from a spiral or specific patterns
        
        # Pattern 1: Center + ring 1 (7 hexes) + 4 from ring 2
        # Ring 1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
        # Ring 2 directions: (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1)
        
        base = [(0,0)]
        ring1 = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
        ring2 = [(2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1)]
        
        # Add center and ring 1 (7 hexes)
        cand1 = base + ring1
        # Add 4 from ring 2 to make 11. Choose symmetrically?
        # (2,0), (0,2), (-2,0), (0,-2) -> Diamond shape
        cand1 += [(2,0), (0,2), (-2,0), (0,-2)]
        candidates.append(cand1)
        
        # Add 4 from ring 2. Choose (2,0), (1,1), (0,2), (-1,2) -> Arc
        cand2 = base + ring1 + [(2,0), (1,1), (0,2), (-1,2)]
        candidates.append(cand2)
        
        # Add 4 from ring 2. Choose (2,0), (-2,0), (0,2), (0,-2) is already cand1.
        # Try (2,0), (1,1), (-2,0), (-1,-1)
        cand3 = base + ring1 + [(2,0), (1,1), (-2,0), (-1,-1)]
        candidates.append(cand3)
        
        # Try a linear-ish arrangement
        cand4 = [(0,0), (1,0), (2,0), (3,0), (4,0),
                 (0,1), (1,1), (2,1),
                 (0,2), (1,2), (2,2)]
        candidates.append(cand4)
        
        cand5 = [(0,0), (1,0),
                 (0,1), (1,1), (2,1),
                 (0,2), (1,2), (2,2),
                 (0,3), (1,3), (2,3)]
        candidates.append(cand5)
        
        cand6 = [(0,0), (1,0), (2,0),
                 (0,1), (1,1), (2,1),
                 (0,2), (1,2), (2,2),
                 (1,3), (2,3)]
        candidates.append(cand6)
        
        cand7 = [(0,0), (1,0), (2,0), (3,0),
                 (0,1), (1,1), (2,1),
                 (1,2), (2,2), (3,2)]
        candidates.append(cand7)
        
        cand8 = [(0,0), (1,0),
                 (0,1), (1,1),
                 (0,2), (1,2),
                 (0,3), (1,3),
                 (0,4), (1,4),
                 (0,5)]
        candidates.append(cand8)
        
        # Compact "blob"
        cand9 = [(0,0), (1,0), (-1,0),
                 (0,1), (1,1), (-1,1),
                 (0,2), (1,2), (-1,2),
                 (0,3), (1,3)]
        candidates.append(cand9)
        
        cand10 = [(0,0), (1,0), (2,0),
                  (1,1), (2,1),
                  (0,2), (1,2), (2,2),
                  (0,3), (1,3), (2,3)]
        candidates.append(cand10)
        
        # Check for duplicates in axial coords
        unique_cands = []
        seen = set()
        for c in candidates:
            # Normalize: sort and freeze
            key = tuple(sorted(c))
            if key not in seen:
                seen.add(key)
                unique_cands.append(c)
        return unique_cands

    candidates = generate_candidates()
    
    best_S = float('inf')
    best_pts = None
    best_beta = 0.0
    best_center = None

    # For each candidate, find optimal orientation (beta) and center (cx, cy)
    for axial in candidates:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
            
        # Convert to cartesian
        pts = np.array([axial_to_cartesian(q, r) for q, r in axial])
        
        # Optimize orientation beta in [0, pi/6)
        # The width of the hexagon cluster depends on beta.
        # We want to minimize the circumradius of the cluster when projected onto the 6 normals of the outer hexagon.
        
        # Coarse search over beta
        best_local_S = float('inf')
        best_local_beta = 0.0
        best_local_center = None
        
        for i in range(60):
            beta = i * (math.pi / 6.0) / 60.0
            
            # Compute projections for this beta
            # Normals of outer hexagon
            normals = []
            for k in range(6):
                ang = beta + math.pi / 6.0 + k * math.pi / 3.0
                normals.append((math.cos(ang), math.sin(ang)))
            normals = np.array(normals)
            
            # For a fixed beta, the optimal center (cx, cy) minimizes the maximum projection.
            # This is a linear programming problem:
            # Minimize D
            # Subject to:
            #   cx*nx + cy*ny + D >= px*nx + py*ny  for all points (p) and normals (n)
            #   cx*nx + cy*ny - D >= px*nx + py*ny  for all points (p) and normals (n)
            #   D >= 0
            
            c_obj = np.array([0.0, 0.0, 1.0])
            A_ub = []
            b_ub = []
            
            for nx, ny in normals:
                for px, py in pts:
                    # Constraint 1: -cx*nx - cy*ny - D <= -(px*nx + py*ny)
                    A_ub.append([-nx, -ny, -1.0])
                    b_ub.append(-(px * nx + py * ny))
                    
                    # Constraint 2: cx*nx + cy*ny - D <= px*nx + py*ny
                    A_ub.append([nx, ny, -1.0])
                    b_ub.append(px * nx + py * ny)
            
            A_ub = np.array(A_ub)
            b_ub = np.array(b_ub)
            bounds = [(None, None), (None, None), (0, None)]
            
            try:
                res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
                if res.status == 0:
                    D = res.x[2]
                    S = (D + APOTHEM) / APOTHEM
                    if S < best_local_S:
                        best_local_S = S
                        best_local_beta = beta
                        best_local_center = (res.x[0], res.x[1])
            except:
                pass
        
        if best_local_S < best_S:
            best_S = best_local_S
            best_pts = pts
            best_beta = best_local_beta
            best_center = best_local_center
            
        # Refine beta around the best found so far
        if best_local_beta > 0 and best_local_beta < math.pi/6.0:
            try:
                from scipy.optimize import minimize_scalar
                def obj_beta(b):
                    normals = np.array([(math.cos(b + math.pi / 6.0 + k * math.pi / 3.0),
                                         math.sin(b + math.pi / 6.0 + k * math.pi / 3.0))
                                        for k in range(6)])
                    c_obj = np.array([0.0, 0.0, 1.0])
                    A_ub = []
                    b_ub = []
                    for nx, ny in normals:
                        for px, py in pts:
                            A_ub.append([-nx, -ny, -1.0])
                            b_ub.append(-(px * nx + py * ny))
                            A_ub.append([nx, ny, -1.0])
                            b_ub.append(px * nx + py * ny)
                    A_ub = np.array(A_ub)
                    b_ub = np.array(b_ub)
                    bounds = [(None, None), (None, None), (0, None)]
                    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
                    if res.status == 0:
                        return (res.x[2] + APOTHEM) / APOTHEM
                    return 1e10
                
                # Search in a small window around best_local_beta
                lower = max(0, best_local_beta - 0.02)
                upper = min(math.pi/6.0, best_local_beta + 0.02)
                
                res = minimize_scalar(obj_beta, bounds=(lower, upper), method='bounded', options={'xatol': 1e-8})
                if res.fun < best_local_S:
                    S, _, _ = obj_beta(res.x), None, None # Need to recompute center
                    # Recompute center for optimal beta
                    beta_opt = res.x
                    normals = np.array([(math.cos(beta_opt + math.pi / 6.0 + k * math.pi / 3.0),
                                         math.sin(beta_opt + math.pi / 6.0 + k * math.pi / 3.0))
                                        for k in range(6)])
                    c_obj = np.array([0.0, 0.0, 1.0])
                    A_ub = []
                    b_ub = []
                    for nx, ny in normals:
                        for px, py in pts:
                            A_ub.append([-nx, -ny, -1.0])
                            b_ub.append(-(px * nx + py * ny))
                            A_ub.append([nx, ny, -1.0])
                            b_ub.append(px * nx + py * ny)
                    A_ub = np.array(A_ub)
                    b_ub = np.array(b_ub)
                    bounds = [(None, None), (None, None), (0, None)]
                    res_l = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
                    if res_l.status == 0:
                        D = res_l.x[2]
                        S = (D + APOTHEM) / APOTHEM
                        if S < best_S:
                            best_S = S
                            best_pts = pts
                            best_beta = beta_opt
                            best_center = (res_l.x[0], res_l.x[1])
            except:
                pass

    # Fallback if no valid solution found
    if best_center is None:
        # Use a simple grid
        pts = np.array([(0,0), (math.sqrt(3), 0), (0, 1.5), (math.sqrt(3)/2, 1.5), (-math.sqrt(3)/2, 1.5),
                        (0, -1.5), (math.sqrt(3)/2, -1.5), (-math.sqrt(3)/2, -1.5),
                        (2*math.sqrt(3), 0), (-2*math.sqrt(3), 0), (math.sqrt(3)/2, 3)])
        cx, cy = float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))
        max_proj = 0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = (pts[:, 0] - cx) * nx + (pts[:, 1] - cy) * ny
            max_proj = max(max_proj, np.max(projs), -np.min(projs))
        best_S = (max_proj + APOTHEM) / APOTHEM
        best_pts = pts
        best_beta = 0.0
        best_center = (cx, cy)

    # Finalize result
    cx, cy = best_center
    angle_deg = math.degrees(best_beta)
    
    # Scale points by 1+eps to ensure they are strictly inside (if needed, though LP ensures <=)
    # The LP ensures projection <= D, so distance <= D.
    # To be safe against floating point errors in evaluation, we can scale slightly.
    # But scaling changes the distance. It's better to just use the center and side calculated.
    # The evaluator checks if inner hexagons are inside outer.
    # Inner hexagons are unit size. Their centers are at pts.
    # The "radius" of a unit hexagon is 1 (circumradius) or APOTHEM (inradius).
    # The distance from center to any vertex of an inner hexagon is 1.
    # So the effective radius of an inner hexagon is 1.
    # Wait, the LP minimized D such that |p - c|_proj <= D.
    # The condition for a unit hexagon centered at p to be inside a hexagon centered at c with apothem A_out
    # is that the maximum projection of (p-c) onto any normal + 1 (circumradius of inner) <= A_out?
    # No. The Minkowski sum of two hexagons.
    # If inner hexagons are aligned with outer, the condition is:
    # max_proj(inner_centers - outer_center) + APOTHEM_inner <= APOTHEM_outer
    # Since they are all rotated by the same angle (relative to outer? No, inner can rotate independently).
    # The problem says "each inner hexagon may translate and rotate".
    # However, in our solution, we set all inner hexagons to have the same angle as the outer (or 0 relative to outer?).
    # The `inner` list returns [x, y, angle_degrees].
    # The `outer` hexagon has rotation `angle_deg`.
    # If we set inner angle to 0 and outer angle to `angle_deg`, they are NOT aligned.
    # To maximize packing density, they should be aligned.
    # So we should set inner angle = outer angle.
    
    inner = []
    for x, y in best_pts:
        inner.append([float(x), float(y), float(angle_deg)])
        
    center = [float(cx), float(cy)]
    side = float(best_S)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation
