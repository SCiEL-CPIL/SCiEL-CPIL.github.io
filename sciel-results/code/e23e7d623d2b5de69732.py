import math
import numpy as np
import time
from scipy.optimize import minimize_scalar, minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    def is_adjacent(q1, r1, q2, r2):
        # Check if two axial coords are neighbors
        dq = q1 - q2
        dr = r1 - r2
        # Neighbors are: (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)
        return (abs(dq) <= 1 and abs(dr) <= 1 and abs(dq + dr) <= 1 and (dq != 0 or dr != 0))
        # Actually, the condition for distance 1 is:
        # max(|dq|, |dr|, |dq+dr|) == 1? No.
        # Distance d = (|dq| + |dr| + |dq+dr|) / 2.
        # d == 1 iff |dq| + |dr| + |dq+dr| == 2.
        return (abs(dq) + abs(dr) + abs(dq + dr)) == 2

    def generate_candidates():
        candidates = []
        seen = set()
        
        # Core flower: 7 hexagons
        core = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
        # Ring 2: 12 hexagons
        ring = [(2,0), (1,1), (0,2), (-1,2), (-2,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1), (2,-1)] 
        # Wait, (2,-1) is listed twice? No.
        # Let's verify the 12 points.
        # (2,0), (1,1), (0,2), (-1,2), (-2,2), (-2,1)
        # (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1), (2,-1) -> (2,-1) is same as last?
        # Let's list systematically.
        # q from -2 to 2.
        # If q=2, r can be -1, 0. (2,-1), (2,0).
        # If q=1, r can be -2, -1, 0, 1. (1,-2), (1,-1), (1,0), (1,1).
        #   (1,-1) and (1,0) are in core?
        #   (1,0) is core. (1,-1) is core.
        #   So (1,-2) and (1,1) are ring.
        # If q=0, r can be -2, -1, 0, 1, 2.
        #   (0,-1), (0,0), (0,1) are core.
        #   So (0,-2) and (0,2) are ring.
        # If q=-1, r can be -1, 0, 1, 2.
        #   (-1,0), (-1,1) are core. (-1,-1) is ring?
        #   d(-1,-1) = (1+1+0)/2 = 1. Wait.
        #   (-1,-1) is adjacent to (0,0)?
        #   d = (1+1+0)/2 = 1. Yes, (-1,-1) is a neighbor of (0,0).
        #   So (-1,-1) is CORE.
        #   My core list was: (0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1).
        #   Missing (-1,-1)?
        #   Neighbors of (0,0) in axial:
        #   (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1).
        #   Let's check (-1,-1).
        #   q=-1, r=-1.
        #   Is it a neighbor?
        #   d = (|-1| + |-1| + |-1-1|)/2 = (1+1+2)/2 = 2.
        #   So (-1,-1) is distance 2. It is RING.
        #   Okay, my core list is correct.
        
        # Let's re-list ring carefully.
        # Points with d=2.
        # (2,0), (1,1), (0,2), (-1,2), (-2,2), (-2,1)
        # (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1), (2,-1) -> (2,-1) is distinct from (1,1)?
        # (2,-1): x = sqrt(3)(2-0.5) = 1.5sqrt(3), y = -1.5.
        # (1,1): x = sqrt(3)(1.5) = 1.5sqrt(3), y = 1.5.
        # They are distinct.
        # So the 12 points are:
        ring = [
            (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 2), (-2, 1),
            (-2, 0), (-1, -1), (0, -2), (1, -2), (2, -1), (2, -1) # Duplicate in my thought, let's use code to generate
        ]
        
        # Better: Generate all (q,r) in range [-3,3] with d=2.
        ring_pts = []
        for q in range(-3, 4):
            for r in range(-3, 4):
                d = (abs(q) + abs(r) + abs(q+r)) / 2.0
                if d == 2.0:
                    ring_pts.append((q,r))
        # ring_pts should have 12 elements.
        
        # Generate all subsets of size 4 from ring_pts
        from itertools import combinations
        for combo in combinations(range(len(ring_pts)), 4):
            selected = [ring_pts[i] for i in combo]
            # Check if any pair is adjacent
            valid = True
            for i in range(4):
                for j in range(i+1, 4):
                    if is_adjacent(selected[i][0], selected[i][1], selected[j][0], selected[j][1]):
                        valid = False
                        break
                if not valid: break
            if valid:
                full_set = core + selected
                # Normalize
                # Try all 12 symmetries
                # Rotations: (q,r) -> (r, -q-r) -> (-q-r, q) -> ...
                # Reflections: (q,r) -> (r,q) -> ...
                
                def rotate(q, r, k):
                    # k steps of 60 deg
                    for _ in range(k):
                        q, r = r, -q - r
                    return (q, r)
                
                def reflect(q, r):
                    return (r, q)
                    
                min_norm = None
                for k in range(6):
                    for refl in [False, True]:
                        new_set = []
                        for q, r in full_set:
                            if refl:
                                q, r = reflect(q, r)
                            q, r = rotate(q, r, k)
                            new_set.append((q, r))
                        new_set.sort()
                        if min_norm is None or new_set < min_norm:
                            min_norm = new_set
                            
                if min_norm not in seen:
                    seen.add(min_norm)
                    candidates.append(min_norm)
                    
        # Add other shapes if time permits
        # 1x11 line
        line = [(i, 0) for i in range(11)]
        # Center it
        line_centered = [(i-5, 0) for i in range(11)]
        # Normalize and add
        # ... (skipping detailed normalization for brevity in thought, will add in code)
        
        return candidates

    def calc_S_for_beta(pts, beta):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), None, None

        cx0 = np.mean(pts[:, 0])
        cy0 = np.mean(pts[:, 1])
        pts_centered = pts - np.array([cx0, cy0])

        normals = []
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))

        max_proj = 0.0
        for nx, ny in normals:
            projs = pts_centered[:, 0] * nx + pts_centered[:, 1] * ny
            max_proj = max(max_proj, np.max(projs), -np.min(projs))

        D = max_proj
        S = (D + APOTHEM) / APOTHEM
        return S, (cx0, cy0), D

    def optimize_shape(pts):
        best_S = float('inf')
        best_beta = 0.0
        best_center = None

        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), 0.0, None

        for i in range(73):
            beta = i * (math.pi / 6) / 72
            S, center, D = calc_S_for_beta(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_center = center

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

        return best_S, best_beta, best_center

    candidates = generate_candidates()
    
    # Fallback if generation fails or is empty
    if not candidates:
        candidates = [[(0,0)]]

    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None

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
        # Fallback to a simple valid config
        axial = [(0,0), (1,0), (2,0), (3,0), (4,0), (5,0), (6,0), (7,0), (8,0), (9,0), (10,0)]
        best_global_pts = axial_to_cartesian(axial)
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

    pts_scaled = []
    for x, y in best_global_pts:
        sx = cx + (x - cx) * 1.0
        sy = cy + (y - cy) * 1.0
        pts_scaled.append((sx, sy))

    pts_scaled = np.array(pts_scaled)

    max_proj = 0
    for k in range(6):
        ang = best_global_beta + math.pi / 6.0 + k * math.pi / 3.0
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
