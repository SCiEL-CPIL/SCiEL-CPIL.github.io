import math
import numpy as np
import time
from scipy.optimize import linprog, minimize, minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 3.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Define candidate point sets (axial coordinates)
    # Focus on compact shapes, particularly truncated hexagons and rings.
    candidates = []

    # axial_1: Center + Ring 1 + 4 from Ring 2
    # Ring 2 points: (2,0), (2,-1), (2,-2), (1,-2), (0,-2), (-1,-2), (-2,-2), (-2,-1), (-2,0), (-2,1), (-2,2), (-1,2), (0,2), (1,2), (2,2)
    # Wait, axial coords for hexagon radius 2:
    # q=2: r in [-2, 0] -> (2,0), (2,-1), (2,-2)
    # q=1: r in [-2, 1] -> (1,0), (1,-1), (1,-2), (1,1)
    # q=0: r in [-2, 2] -> (0,0), (0,1), (0,-1), (0,2), (0,-2)
    # q=-1: r in [-1, 2] -> (-1,0), (-1,1), (-1,2), (-1,-1)
    # q=-2: r in [0, 2] -> (-2,0), (-2,1), (-2,2)
    # Corners: (2,0), (0,2), (-2,2), (-2,0), (0,-2), (2,-2)
    # Let's try a "truncated hexagon" shape: Remove corners from radius 2 hexagon.
    # Radius 2 has 19 points. Remove 6 corners -> 13 points. Remove 2 more to get 11.
    # Which 2 to remove? Maybe the ones that are most "pointy" or redundant.
    # Let's try removing (2,-1) and (-2,1) which are adjacent to corners.
    # Or maybe just pick a symmetric subset.
    
    # Candidate A: Truncated Hexagon (11 points)
    # Points:
    # (0,0)
    # Ring 1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Ring 2 (non-corners):
    # (2,-1), (1,-2), (-1,-2), (-2,-1), (-2,1), (-1,2), (1,2)
    # Total: 1 + 6 + 7 = 14. Too many.
    # We need 11.
    # 1 center + 6 ring 1 = 7. Need 4 more from ring 2.
    # To be symmetric, pick 4 from ring 2 that are spaced out.
    # E.g., (2,-1), (-1,-2), (-2,1), (1,2)?
    # Let's check symmetry.
    # (2,-1) rotated 120 deg:
    # Axial rotation 120: (q,r) -> (-r, q+r).
    # (2,-1) -> (1, 1). (1,1) is in Ring 1? No, (1,1) is in Ring 1?
    # |1|<=1, |1|<=1, |2|<=1? No. (1,1) is distance 1?
    # dist^2 = q^2 + qr + r^2.
    # (1,1): 1+1+1=3. dist=sqrt(3). Ring 1 has dist=1. So (1,1) is Ring 2.
    # So (2,-1) is Ring 2.
    # If we pick (2,-1), its rotations are (1,1), (-2,2), (-1,-1), (2,-2)?
    # Let's list Ring 2 points again carefully.
    # Hexagon radius 2 points:
    # (2,0), (2,-1), (2,-2)
    # (1,1), (1,0), (1,-1), (1,-2)
    # (0,2), (0,1), (0,0), (0,-1), (0,-2)
    # (-1,2), (-1,1), (-1,0), (-1,-1)
    # (-2,2), (-2,1), (-2,0)
    
    # Ring 1 (dist 1): (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Ring 2 (dist sqrt(3) or 2):
    # dist 2: (2,0), (0,2), (-2,2), (-2,0), (0,-2), (2,-2) [Corners]
    # dist sqrt(3): (1,1), (1,-2), (-2,-1), (-1,2), (2,-1), (-1,-1)
    # Wait, (1,1) dist^2 = 1+1+1=3.
    # (1,-2) dist^2 = 1-2+4=3.
    # (2,-1) dist^2 = 4-2+1=3.
    # So there are 6 points at dist sqrt(3) and 6 at dist 2.
    # Total 12 in Ring 2.
    # If we take Center (1) + Ring 1 (6) = 7.
    # We need 4 more.
    # If we take 4 from the "dist sqrt(3)" set, we get a nice shape.
    # The "dist sqrt(3)" points are:
    # (1,1), (1,-2), (-2,-1), (-1,2), (2,-1), (-1,-1)
    # Let's pick 4 of these. To maintain some symmetry, maybe pick opposite pairs?
    # (1,1) and (-1,-1) are opposite?
    # (1,1) + (-1,-1) = (0,0). Yes.
    # (1,-2) and (-1,2) are opposite? Yes.
    # So pick (1,1), (-1,-1), (1,-2), (-1,2).
    # This gives 11 points.
    # Let's call this "Candidate A".
    
    cand_A = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, -1), (1, -2), (-1, 2)
    ]
    candidates.append(cand_A)

    # Candidate B: Different 4 from Ring 2
    # Pick (2,-1), (-2,1), (-1,-2), (1,2)?
    # (2,-1) opposite is (-2,1).
    # (-1,-2) opposite is (1,2).
    cand_B = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, -1), (-2, 1), (-1, -2), (1, 2)
    ]
    candidates.append(cand_B)

    # Candidate C: Pick (2,-1), (2,-2) is corner, so avoid corners if possible for compactness?
    # Actually corners might be good for packing if rotated.
    # Let's try a shape with 3 from Ring 2 and 1 from Ring 1 removed?
    # No, keep Ring 1 full.
    
    # Candidate D: "Star" shape?
    # Center + 6 Ring 1 + 4 Ring 2 (dist 2 corners?)
    # (2,0), (0,2), (-2,2), (-2,0)
    cand_D = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (0, 2), (-2, 2), (-2, 0)
    ]
    candidates.append(cand_D)

    # Candidate E: Mixed
    # (2,-1), (1,1), (-1,-1), (-2,1)
    cand_E = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, -1), (1, 1), (-1, -1), (-2, 1)
    ]
    candidates.append(cand_E)

    # Let's also add the original candidates from the seed to be safe, 
    # but the seed had many. I'll add a few good ones from the seed list.
    
    # Seed axial_1
    seed_1 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ]
    candidates.append(seed_1)

    # Seed axial_2
    seed_2 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(seed_2)

    # Seed axial_7
    seed_7 = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(seed_7)

    # Seed axial_13
    seed_13 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (1, 2)
    ]
    candidates.append(seed_13)

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    def calc_S_for_beta(pts, beta):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), None, None
        
        normals = []
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))
        
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

    def calc_S_exact(pts, cx, cy, beta):
        normals = np.array([
            (math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0))
            for k in range(6)
        ])
        pts_centered = pts - np.array([cx, cy])
        projs = pts_centered @ normals.T
        max_val = np.max(np.abs(projs))
        S = (max_val + APOTHEM) / APOTHEM
        return S, max_val

    def optimize_shape(pts):
        best_S = float('inf')
        best_beta = 0.0
        best_center = None
        
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), 0.0, None
        
        # Coarse grid search over beta
        # 0 to pi/6 is the fundamental domain for hexagon rotation
        for i in range(73):
            beta = i * (math.pi / 6) / 72
            S, center, D = calc_S_for_beta(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_center = center
        
        # Refine beta with bounded scalar optimization
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                res = minimize_scalar(lambda b: calc_S_for_beta(pts, b)[0], bounds=(0, math.pi / 6), method='bounded',
                                     options={'maxiter': 500, 'xatol': 1e-14})
                if res.fun < best_S:
                    best_S = res.fun
                    best_beta = res.x
                    _, best_center, _ = calc_S_for_beta(pts, best_beta)
        except Exception:
            pass
        
        # Final polish with Nelder-Mead on (cx, cy, beta)
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN and best_center is not None:
                def objective(params):
                    cx, cy, beta = params
                    beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
                    S, _ = calc_S_exact(pts, cx, cy, beta)
                    return S
                
                cx0, cy0 = best_center
                beta0 = best_beta
                
                # Try a few perturbations
                starts = [
                    [cx0, cy0, beta0],
                    [cx0 + 0.01, cy0, beta0],
                    [cx0 - 0.01, cy0, beta0],
                    [cx0, cy0 + 0.01, beta0],
                    [cx0, cy0 - 0.01, beta0],
                    [cx0, cy0, beta0 + 0.01],
                    [cx0, cy0, beta0 - 0.01],
                ]
                
                for init in starts:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    res = minimize(objective, init, method='Nelder-Mead',
                                 options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12})
                    if res.fun < best_S:
                        best_S = res.fun
                        best_center = (res.x[0], res.x[1])
                        best_beta = ((res.x[2] % (math.pi/6)) + math.pi/6) % (math.pi/6)
        except Exception:
            pass
            
        return best_S, best_beta, best_center

    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None

    # Evaluate all candidates
    for axial in candidates:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        pts = axial_to_cartesian(axial)
        S, beta, center = optimize_shape(pts)
        if S < best_global_S:
            best_global_S = S
            best_global_pts = pts
            best_global_beta = beta
            best_global_center = center

    if best_global_center is None or best_global_S == float('inf'):
        # Fallback
        best_global_pts = axial_to_cartesian(candidates[0])
        best_global_center = (float(np.mean(best_global_pts[:, 0])), float(np.mean(best_global_pts[:, 1])))
        best_global_beta = 0.0
        max_proj = 0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = best_global_pts[:, 0] * nx + best_global_pts[:, 1] * ny
            max_proj = max(max_proj, np.max(projs), -np.min(projs))
        best_global_S = (max_proj + APOTHEM) / APOTHEM
    
    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)
    
    # Scale by 1.0 + epsilon to ensure strict containment if needed, 
    # but usually exact is fine. The evaluator checks disjointness and containment.
    # If we scale up slightly, we might violate containment if S was exact.
    # If we scale down, we are safe but S is smaller? No, S is determined by the outer hexagon.
    # The inner hexagons must fit inside.
    # If we calculate S based on the points, and then output those points, they fit.
    # The "side" returned is the outer hexagon side.
    # The evaluator checks if the inner hexagons are inside the outer hexagon of that side.
    # So we should NOT scale the points. We should output the points that fit in S.
    
    inner = []
    for x, y in best_global_pts:
        inner.append([float(x), float(y), float(angle_deg)])
        
    center = [float(cx), float(cy)]
    side = float(best_global_S)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation
