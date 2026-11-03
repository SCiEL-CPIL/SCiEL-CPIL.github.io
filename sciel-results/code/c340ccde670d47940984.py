import math
import numpy as np
import time
from scipy.optimize import minimize_scalar, linprog

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0 # Increased safety margin

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Compact configurations of 11 hexagons
    # Axial coordinates (q, r)
    candidates = []

    # 1. 3x3 block (9) + 2 attached to one side
    # 3x3: q in 0..2, r in 0..2
    c1 = [(q, r) for q in range(3) for r in range(3)]
    c1.append((3, 0))
    c1.append((3, 1))
    candidates.append(c1)

    # 2. 3x3 block (9) + 2 attached to corner
    c2 = [(q, r) for q in range(3) for r in range(3)]
    c2.append((3, 0))
    c2.append((0, 3))
    candidates.append(c2)

    # 3. 2x5 strip (10) + 1 in middle
    c3 = [(q, r) for q in range(2) for r in range(5)]
    c3.append((2, 2))
    candidates.append(c3)

    # 4. Hex(2) (7) + 4
    # Hex(2) centered at (0,0)
    hex2 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    # Add 4 more
    c4 = hex2 + [(2,0), (2,-1), (-2,1), (-1,2)] # Just a guess, need to check overlap
    # Better: add to the "flat" sides
    c4 = hex2 + [(2,0), (-2,0), (0,2), (0,-2)]
    candidates.append(c4)

    # 5. 4x3 block minus 1 (11)
    # 4x3 is 12. Remove one corner.
    c5 = [(q, r) for q in range(4) for r in range(3)]
    c5.pop() # remove last (3,2)
    candidates.append(c5)

    # 6. 3x4 block minus 1 (11)
    c6 = [(q, r) for q in range(3) for r in range(4)]
    c6.pop()
    candidates.append(c6)

    # 7. "Plus" shape extended?
    # Center (0,0)
    # Arms of length 2?
    # (0,0)
    # (1,0), (2,0), (-1,0), (-2,0)
    # (0,1), (0,2), (0,-1), (0,-2)
    # That's 9. Add (1,1) and (-1,-1)?
    c7 = [(0,0), (1,0), (2,0), (-1,0), (-2,0), (0,1), (0,2), (0,-1), (0,-2), (1,1), (-1,-1)]
    candidates.append(c7)

    # 8. Another 3x3 + 2
    c8 = [(q, r) for q in range(3) for r in range(3)]
    c8.append((3, 2))
    c8.append((3, 3))
    candidates.append(c8)

    # 9. 2x6 strip (12) minus 1
    c9 = [(q, r) for q in range(2) for r in range(6)]
    c9.pop()
    candidates.append(c9)

    # 10. Hex(2) + 4 in a different way
    c10 = hex2 + [(2,1), (1,2), (-1,2), (-2,1)]
    candidates.append(c10)

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

    def optimize_shape(pts):
        best_S = float('inf')
        best_beta = 0.0
        best_center = None
        
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), 0.0, None
        
        # Coarse grid search over beta
        # Symmetry of hexagon is pi/3, but we can restrict to pi/6 due to inner/outer?
        # Actually, rotating the container by pi/6 swaps the "flat" and "pointy" sides relative to the inner grid?
        # The inner grid is fixed. The container is regular.
        # The function S(beta) is periodic with period pi/3.
        # We search in [0, pi/3]
        for i in range(61):
            beta = i * (math.pi / 3) / 60
            S, center, D = calc_S_for_beta(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_center = center
        
        # Refine beta
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                res = minimize_scalar(lambda b: calc_S_for_beta(pts, b)[0], bounds=(0, math.pi / 3), method='bounded',
                                     options={'maxiter': 300, 'xatol': 1e-12})
                if res.fun < best_S:
                    best_S = res.fun
                    best_beta = res.x
                    _, best_center, _ = calc_S_for_beta(pts, best_beta)
        except Exception:
            pass
            
        return best_S, best_beta, best_center

    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None

    for axial in candidates:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        # Check validity: distances between centers must be >= sqrt(3) - epsilon
        pts = axial_to_cartesian(axial)
        valid = True
        for i in range(N):
            for j in range(i+1, N):
                dist = np.linalg.norm(pts[i] - pts[j])
                if dist < math.sqrt(3) - 1e-5:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            continue
            
        S, beta, center = optimize_shape(pts)
        if S < best_global_S:
            best_global_S = S
            best_global_pts = pts
            best_global_beta = beta
            best_global_center = center

    if best_global_center is None or best_global_S == float('inf'):
        # Fallback to a known valid config
        axial_fallback = [(q, r) for q in range(3) for r in range(3)] + [(3, 0), (3, 1)]
        best_global_pts = axial_to_cartesian(axial_fallback)
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
    
    # Ensure no overlap by scaling slightly? No, the config is valid.
    # Just return the points.
    inner = []
    for x, y in best_global_pts:
        inner.append([float(x), float(y), 0.0])
        
    center = [float(cx), float(cy)]
    side = float(best_global_S)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation
