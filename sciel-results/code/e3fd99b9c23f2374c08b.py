import math
import numpy as np
import time
from scipy.optimize import linprog, minimize, minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # For aligned hexagons (all same orientation), the problem is:
    # Find center (cx, cy) and minimize S such that for all 6 directions,
    # max_i |dot(p_i - c, n_k)| + APOTHEM <= S * APOTHEM
    # i.e., max_i |dot(p_i - c, n_k)| <= (S - 1) * APOTHEM for all k
    
    # This is equivalent to: find c that minimizes max over all (i,k) of |dot(p_i - c, n_k)|
    # Then S = 1 + max_val / APOTHEM

    # Candidate configurations in axial coordinates
    candidates = []

    # Hexagonal number patterns and variations
    # 1 center + 6 ring + 4 more
    
    # Pattern 1: 3-4-3-1 (rows of 3,4,3,1)
    axial_1 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3)
    ]
    candidates.append(axial_1)

    # Pattern 2: 1-3-4-3 (rows of 1,3,4,3)
    axial_2 = [
        (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_2)

    # Pattern 3: 2-3-4-2
    axial_3 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (0, 3), (1, 3)
    ]
    candidates.append(axial_3)

    # Pattern 4: 2-4-3-2
    axial_4 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3)
    ]
    candidates.append(axial_4)

    # Pattern 5: 3-3-3-2
    axial_5 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3)
    ]
    candidates.append(axial_5)

    # Pattern 6: 4-3-3-1
    axial_6 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3)
    ]
    candidates.append(axial_6)

    # Pattern 7: 1-4-3-3
    axial_7 = [
        (1, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_7)

    # Pattern 8: 3-4-2-2
    axial_8 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3)
    ]
    candidates.append(axial_8)

    # Pattern 9: 2-2-4-3
    axial_9 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_9)

    # Pattern 10: 4-2-3-2
    axial_10 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3)
    ]
    candidates.append(axial_10)

    # Pattern 11: 2-3-3-3
    axial_11 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_11)

    # Pattern 12: 3-2-4-2
    axial_12 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (0, 3), (1, 3)
    ]
    candidates.append(axial_12)

    # Pattern 13: 2-4-2-3
    axial_13 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_13)

    # Pattern 14: 3-3-2-3
    axial_14 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_14)

    # Pattern 15: 3-2-3-3
    axial_15 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_15)

    # Pattern 16: 2-3-2-4
    axial_16 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3), (2, 3), (3, 3)
    ]
    candidates.append(axial_16)

    # Pattern 17: 4-3-2-2
    axial_17 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3)
    ]
    candidates.append(axial_17)

    # Pattern 18: 1-3-4-3 variant
    axial_18 = [
        (0, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_18)

    # Pattern 19: 3-4-3-1 variant
    axial_19 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3)
    ]
    candidates.append(axial_19)

    # Pattern 20: 2-5-2-2
    axial_20 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3)
    ]
    candidates.append(axial_20)

    # Pattern 21: 2-2-5-2
    axial_21 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2), (3, 2), (4, 2),
        (0, 3), (1, 3)
    ]
    candidates.append(axial_21)

    # Pattern 22: 5-2-2-2
    axial_22 = [
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3)
    ]
    candidates.append(axial_22)

    # Pattern 23: 2-2-2-5
    axial_23 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3), (2, 3), (3, 3), (4, 3)
    ]
    candidates.append(axial_23)

    # Pattern 24: 4-4-3 (3 rows)
    axial_24 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2)
    ]
    candidates.append(axial_24)

    # Pattern 25: 3-4-4
    axial_25 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2)
    ]
    candidates.append(axial_25)

    # Pattern 26: 4-4-3 variant
    axial_26 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (1, 2), (2, 2), (3, 2)
    ]
    candidates.append(axial_26)

    # Pattern 27: 3-3-5
    axial_27 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2), (3, 2), (4, 2)
    ]
    candidates.append(axial_27)

    # Pattern 28: 5-3-3
    axial_28 = [
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2)
    ]
    candidates.append(axial_28)

    # Pattern 29: 4-3-4
    axial_29 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2), (3, 2)
    ]
    candidates.append(axial_29)

    # Pattern 30: 3-4-3-1 with different offset
    axial_30 = [
        (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3)
    ]
    candidates.append(axial_30)

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    def compute_S_aligned(pts, cx, cy):
        """Compute the required outer hexagon side length for aligned hexagons."""
        # For aligned hexagons, the 6 normal directions are fixed
        normals = []
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))
        
        max_proj = 0.0
        for nx, ny in normals:
            projs = (pts[:, 0] - cx) * nx + (pts[:, 1] - cy) * ny
            max_proj = max(max_proj, np.max(projs), -np.min(projs))
        
        S = (max_proj + APOTHEM) / APOTHEM
        return S

    def optimize_center_aligned(pts):
        """Find the optimal center for aligned hexagons using LP."""
        # Minimize t subject to:
        # For each normal n_k and each point p_i:
        #   (p_i - c) · n_k <= t
        #   -(p_i - c) · n_k <= t  i.e., (c - p_i) · n_k <= t
        #   t >= 0
        
        # Variables: cx, cy, t
        c_obj = np.array([0.0, 0.0, 1.0])
        A_ub = []
        b_ub = []
        
        normals = []
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))
        
        for nx, ny in normals:
            for px, py in pts:
                # (px - cx)*nx + (py - cy)*ny <= t
                # -nx*cx - ny*cy - t <= -px*nx - py*ny
                A_ub.append([-nx, -ny, -1.0])
                b_ub.append(-(px * nx + py * ny))
                # (cx - px)*nx + (cy - py)*ny <= t
                # nx*cx + ny*cy - t <= px*nx + py*ny
                A_ub.append([nx, ny, -1.0])
                b_ub.append(px * nx + py * ny)
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        bounds = [(None, None), (None, None), (0, None)]
        
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                cx = res.x[0]
                cy = res.x[1]
                t = res.x[2]
                S = (t + APOTHEM) / APOTHEM
                return S, (cx, cy), t
            else:
                return float('inf'), None, None
        except Exception:
            return float('inf'), None, None

    def optimize_with_rotation(pts, beta):
        """Optimize center for given rotation angle beta."""
        # Rotate points by beta
        cos_b = math.cos(beta)
        sin_b = math.sin(beta)
        
        rotated_pts = np.zeros_like(pts)
        for i in range(len(pts)):
            x, y = pts[i]
            rotated_pts[i] = [x * cos_b - y * sin_b, x * sin_b + y * cos_b]
        
        # For rotated hexagons, the normals are also rotated
        # But we can just compute the required S for the rotated configuration
        # The normals for the outer hexagon are at angles beta + pi/6 + k*pi/3
        
        normals = []
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))
        
        # LP to find optimal center
        c_obj = np.array([0.0, 0.0, 1.0])
        A_ub = []
        b_ub = []
        
        for nx, ny in normals:
            for px, py in rotated_pts:
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
                cx_rot = res.x[0]
                cy_rot = res.x[1]
                t = res.x[2]
                S = (t + APOTHEM) / APOTHEM
                
                # Convert center back to original coordinates
                cx = cx_rot * cos_b + cy_rot * sin_b
                cy = -cx_rot * sin_b + cy_rot * cos_b
                
                return S, (cx, cy), t
            else:
                return float('inf'), None, None
        except Exception:
            return float('inf'), None, None

    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None

    # First pass: try aligned (beta=0) for all candidates
    for idx, axial in enumerate(candidates):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        pts = axial_to_cartesian(axial)
        S, center, t = optimize_center_aligned(pts)
        if S < best_global_S:
            best_global_S = S
            best_global_pts = pts
            best_global_beta = 0.0
            best_global_center = center

    # Second pass: try small rotations for the top candidates
    if best_global_pts is not None:
        # Try rotations for the best few candidates
        top_candidates = []
        for idx, axial in enumerate(candidates):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            pts = axial_to_cartesian(axial)
            S_aligned, _, _ = optimize_center_aligned(pts)
            top_candidates.append((S_aligned, idx, pts))
        
        top_candidates.sort(key=lambda x: x[0])
        
        # Try rotations for top 10 candidates
        for _, idx, pts in top_candidates[:10]:
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            
            # Try a range of rotation angles
            for beta in np.linspace(0, math.pi/6, 13):  # 0 to 30 degrees
                if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                    break
                S, center, t = optimize_with_rotation(pts, beta)
                if S < best_global_S:
                    best_global_S = S
                    best_global_pts = pts
                    best_global_beta = beta
                    best_global_center = center

    # Third pass: refine the best solution with finer rotation search
    if best_global_pts is not None:
        # Fine rotation search around best_beta
        beta_start = best_global_beta
        for beta in np.linspace(max(0, beta_start - 0.05), min(math.pi/6, beta_start + 0.05), 21):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            S, center, t = optimize_with_rotation(best_global_pts, beta)
            if S < best_global_S:
                best_global_S = S
                best_global_pts = best_global_pts
                best_global_beta = beta
                best_global_center = center

    # Final: if no valid solution found, use a default
    if best_global_center is None or best_global_S == float('inf'):
        # Fallback: use the first candidate with center at centroid
        pts = axial_to_cartesian(candidates[0])
        cx = float(np.mean(pts[:, 0]))
        cy = float(np.mean(pts[:, 1]))
        best_global_pts = pts
        best_global_center = (cx, cy)
        best_global_beta = 0.0
        best_global_S = compute_S_aligned(pts, cx, cy)

    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)
    
    # Apply a tiny scale to ensure no numerical overlap
    SCALE = 1.0 + 1e-10
    pts_scaled = []
    for x, y in best_global_pts:
        sx = cx + (x - cx) * SCALE
        sy = cy + (y - cy) * SCALE
        pts_scaled.append((sx, sy))
    
    pts_scaled = np.array(pts_scaled)
    
    # Recompute S with the scaled points
    normals = []
    for k in range(6):
        ang = best_global_beta + math.pi / 6.0 + k * math.pi / 3.0
        normals.append((math.cos(ang), math.sin(ang)))
    
    max_proj = 0.0
    for nx, ny in normals:
        projs = (pts_scaled[:, 0] - cx) * nx + (pts_scaled[:, 1] - cy) * ny
        max_proj = max(max_proj, np.max(projs), -np.min(projs))
    
    S_final = (max_proj + APOTHEM) / APOTHEM
    
    inner = []
    for x, y in pts_scaled:
        inner.append([float(x), float(y), float(angle_deg)])
        
    center = [float(cx), float(cy)]
    side = float(S_final)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation
