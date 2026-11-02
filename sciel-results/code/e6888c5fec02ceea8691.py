import math
import numpy as np
import time
from scipy.optimize import linprog

N = 11
APOTHEM = math.sqrt(3) / 2.0
SQRT3 = math.sqrt(3)
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Candidate point sets in axial coordinates
    candidates = []
    axial_1 = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(1,1),(-1,2),(-2,1),(2,-1)]
    candidates.append(axial_1)
    axial_2 = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)]
    candidates.append(axial_2)
    axial_3 = [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,1),(3,1)]
    candidates.append(axial_3)
    axial_4 = [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2)]
    candidates.append(axial_4)
    axial_5 = [(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)]
    candidates.append(axial_5)
    axial_6 = [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3)]
    candidates.append(axial_6)
    axial_7 = [(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(2,0),(-2,0),(0,2),(0,-2)]
    candidates.append(axial_7)
    axial_8 = [(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(1,1),(2,1),(3,1),(4,1),(5,1)]
    candidates.append(axial_8)
    axial_9 = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,1),(-2,2),(-1,3),(1,-2)]
    candidates.append(axial_9)
    axial_10 = [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3)]
    candidates.append(axial_10)
    axial_11 = [(0,0),(1,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2),(1,3)]
    candidates.append(axial_11)
    axial_12 = [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(0,2),(1,2),(0,3),(1,3),(0,4)]
    candidates.append(axial_12)
    axial_13 = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(1,2)]
    candidates.append(axial_13)
    axial_14 = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,-1)]
    candidates.append(axial_14)
    axial_15 = [(0,0),(1,0),(2,0),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,1),(3,1),(0,-1)]
    candidates.append(axial_15)
    axial_16 = [(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(3,2)]
    candidates.append(axial_16)
    axial_17 = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,1),(1,2),(-1,2),(0,3)]
    candidates.append(axial_17)
    axial_18 = [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(1,2),(2,2),(3,2)]
    candidates.append(axial_18)
    axial_19 = [(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(3,3)]
    candidates.append(axial_19)
    axial_20 = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,1)]
    candidates.append(axial_20)
    axial_21 = [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(1,2),(2,2),(3,2)]
    candidates.append(axial_21)
    axial_22 = [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(1,3)]
    candidates.append(axial_22)
    axial_23 = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)]
    candidates.append(axial_23)
    axial_24 = [(0,0),(1,0),(0,1),(1,1),(0,2),(1,2),(2,2),(0,3),(1,3),(0,4)]
    candidates.append(axial_24)
    axial_25 = [(0,0),(1,0),(2,0),(0,1),(1,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)]
    candidates.append(axial_25)
    axial_26 = [(0,0),(1,0),(2,0),(3,0),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2),(1,3),(2,3)]
    candidates.append(axial_26)
    axial_27 = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(1,-2)]
    candidates.append(axial_27)
    axial_28 = [(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(0,3),(1,3),(2,3),(1,4),(2,4)]
    candidates.append(axial_28)
    axial_29 = [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(0,3)]
    candidates.append(axial_29)
    axial_30 = [(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3),(0,4)]
    candidates.append(axial_30)
    axial_31 = [(0,0),(1,0),(2,0),(0,1),(1,1),(0,2),(1,2),(2,2),(0,3),(1,3),(0,4),(1,4)]
    candidates.append(axial_31)
    axial_32 = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)]
    candidates.append(axial_32)
    axial_33 = [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(0,2),(1,2),(0,3),(1,3)]
    candidates.append(axial_33)
    axial_34 = [(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)]
    candidates.append(axial_34)
    axial_35 = [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)]
    candidates.append(axial_35)

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = SQRT3 * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    def solve_packing_lp(pts, beta):
        """
        Solve the joint packing LP for fixed orientation beta.
        Variables: [x_0..x_{n-1}, y_0..y_{n-1}, cx, cy, D] (2n + 3 variables)
        Minimize D (outer hexagon inradius offset).
        Constraints:
          1. For each pair (i,j) and each of 6 directions:
             (p_i - p_j) . d_k <= 2 * APOTHEM  (non-overlap via Minkowski sum)
          2. For each point i and each of 6 normals:
             (p_i - c) . n_k <= D  and  -(p_i - c) . n_k <= D
        """
        n_pts = len(pts)
        n_vars = 2 * n_pts + 3
        # Objective: minimize D (last variable)
        c_obj = np.zeros(n_vars)
        c_obj[-1] = 1.0

        A_ub_rows = []
        b_ub_rows = []

        # Non-overlap constraints: for each pair (i,j) with i<j, 6 directions
        # (p_i - p_j) . d_k <= 2*APOTHEM
        # => x_i - x_j <= 2*APOTHEM*cos(d_k)  and  y_i - y_j <= 2*APOTHEM*sin(d_k)
        # We need both signs: |p_i - p_j|_hex <= 2*APOTHEM
        # For regular hexagons with same orientation, the Minkowski sum is a hexagon
        # with inradius 2*APOTHEM. The constraint is:
        # For each of 6 directions d_k (unit vectors along hexagon edges):
        #   (p_i - p_j) . d_k <= 2*APOTHEM
        # The 6 directions for a flat-top hexagon are at angles 0, 60, 120, 180, 240, 300
        # But since we have both +d and -d, we only need 3 unique directions
        # Actually for a regular hexagon, the 6 edge normals are at 0, 60, ..., 300
        # The Minkowski sum of two identical regular hexagons is a regular hexagon
        # with side length 2 (if unit hexagons), so inradius = 2*APOTHEM
        
        # The 6 normal directions for the Minkowski sum hexagon (same as inner hexagons)
        # For a flat-top hexagon, normals are at 30, 90, 150, 210, 270, 330 degrees
        # i.e., pi/6 + k*pi/3 for k=0..5
        # But wait - the inner hexagons all have the SAME orientation (angle_deg = beta)
        # So the Minkowski sum is a hexagon with the same orientation
        # The constraint (p_i - p_j) . n_k <= 2*APOTHEM for all 6 normals n_k
        
        # Normals for the Minkowski sum (same orientation as inner hexagons)
        # For a flat-top regular hexagon with side 1, the outward normals to the 6 faces
        # are at angles 0, pi/3, 2pi/3, pi, 4pi/3, 5pi/3 (if flat top means flat side on top)
        # Actually, let me think carefully. A regular hexagon with side 1:
        # If we define it with vertices at angles 0, 60, 120, ... from center
        # The face normals are at 30, 90, 150, ... (offset by 30 degrees)
        # The inradius (apothem) is sqrt(3)/2
        
        # For two identical regular hexagons with the same orientation, the Minkowski sum
        # is a regular hexagon with the same orientation and side length 2 (inradius 2*sqrt(3)/2 = sqrt(3))
        # The non-overlap condition is that the distance between centers in the hexagonal
        # metric is at least 2*APOTHEM, which means:
        # (p_i - p_j) . n_k <= 2*APOTHEM for all 6 face normals n_k of the unit hexagon
        
        # Face normals for a regular hexagon oriented with flat top:
        # The 6 faces have outward normals at angles 0, pi/3, 2pi/3, pi, 4pi/3, 5pi/3
        # Wait, I need to be careful about the hexagon orientation.
        # Let's use the same convention as the enclosing hexagon.
        # The enclosing hexagon has normals at beta + pi/6 + k*pi/3.
        # The inner hexagons have the same orientation (angle_deg = beta degrees).
        # So the inner hexagon face normals are also at beta + pi/6 + k*pi/3.
        
        # Actually, for the Minkowski sum constraint, I need the face normals of the
        # UNIT hexagon (the shape of each inner hexagon). Since all inner hexagons
        # have the same orientation beta, the Minkowski sum has face normals at
        # beta + pi/6 + k*pi/3 (same as the enclosing hexagon, since they're all
        # regular hexagons with the same orientation).
        
        # So the non-overlap constraint is:
        # (p_i - p_j) . n_k <= 2*APOTHEM for k=0..5
        # where n_k = (cos(beta + pi/6 + k*pi/3), sin(beta + pi/6 + k*pi/3))
        
        for i in range(n_pts):
            for j in range(i + 1, n_pts):
                for k in range(6):
                    ang = beta + math.pi / 6.0 + k * math.pi / 3.0
                    nx = math.cos(ang)
                    ny = math.sin(ang)
                    # (p_i - p_j) . n <= 2*APOTHEM
                    # x_i*nx + y_i*ny - x_j*nx - y_j*ny <= 2*APOTHEM
                    row = np.zeros(n_vars)
                    row[i] = nx
                    row[n_pts + i] = ny
                    row[j] = -nx
                    row[n_pts + j] = -ny
                    A_ub_rows.append(row)
                    b_ub_rows.append(2.0 * APOTHEM)

        # Enclosing hexagon constraints:
        # For each point i and each normal n_k:
        #   (p_i - c) . n_k <= D
        #   -(p_i - c) . n_k <= D
        # i.e., p_i . n_k - c . n_k <= D  and  -p_i . n_k + c . n_k <= D
        
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            nx = math.cos(ang)
            ny = math.sin(ang)
            for i in range(n_pts):
                # p_i . n - c . n <= D  =>  nx*x_i + ny*y_i - nx*cx - ny*cy - D <= 0
                row = np.zeros(n_vars)
                row[i] = nx
                row[n_pts + i] = ny
                row[2 * n_pts] = -nx
                row[2 * n_pts + 1] = -ny
                row[2 * n_pts + 2] = -1.0
                A_ub_rows.append(row)
                b_ub_rows.append(0.0)
                
                # -p_i . n + c . n <= D  =>  -nx*x_i - ny*y_i + nx*cx + ny*cy - D <= 0
                row = np.zeros(n_vars)
                row[i] = -nx
                row[n_pts + i] = -ny
                row[2 * n_pts] = nx
                row[2 * n_pts + 1] = ny
                row[2 * n_pts + 2] = -1.0
                A_ub_rows.append(row)
                b_ub_rows.append(0.0)

        A_ub = np.array(A_ub_rows)
        b_ub = np.array(b_ub_rows)
        bounds = [(None, None)] * n_vars
        bounds[-1] = (0, None)  # D >= 0

        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                D = res.x[-1]
                cx = res.x[2 * n_pts]
                cy = res.x[2 * n_pts + 1]
                positions = []
                for i in range(n_pts):
                    positions.append((res.x[i], res.x[n_pts + i]))
                S = (D + APOTHEM) / APOTHEM
                return S, positions, (cx, cy)
            else:
                return float('inf'), None, None
        except Exception:
            return float('inf'), None, None

    def optimize_shape_joint(pts):
        """Optimize over beta using the joint LP formulation."""
        best_S = float('inf')
        best_beta = 0.0
        best_positions = None
        best_center = None

        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), 0.0, None, None

        # Grid search over beta in [0, pi/6]
        n_grid = 360
        for i in range(n_grid + 1):
            beta = i * (math.pi / 6) / n_grid
            S, positions, center = solve_packing_lp(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_positions = positions
                best_center = center

        # Refine with finer grid around best
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            for i in range(-100, 101):
                beta = best_beta + i * (math.pi / 6) / (n_grid * 10)
                if beta < 0:
                    beta += math.pi / 6
                if beta > math.pi / 6:
                    beta -= math.pi / 6
                S, positions, center = solve_packing_lp(pts, beta)
                if S < best_S:
                    best_S = S
                    best_beta = beta
                    best_positions = positions
                    best_center = center

        # Even finer polish
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            for i in range(-50, 51):
                beta = best_beta + i * (math.pi / 6) / (n_grid * 100)
                if beta < 0:
                    beta += math.pi / 6
                if beta > math.pi / 6:
                    beta -= math.pi / 6
                S, positions, center = solve_packing_lp(pts, beta)
                if S < best_S:
                    best_S = S
                    best_beta = beta
                    best_positions = positions
                    best_center = center

        return best_S, best_beta, best_positions, best_center

    best_global_S = float('inf')
    best_global_beta = 0.0
    best_global_positions = None
    best_global_center = None
    best_global_pts = None

    for axial in candidates:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        pts = axial_to_cartesian(axial)
        S, beta, positions, center = optimize_shape_joint(pts)
        if S < best_global_S:
            best_global_S = S
            best_global_beta = beta
            best_global_positions = positions
            best_global_center = center
            best_global_pts = pts

    if best_global_center is None or best_global_S == float('inf'):
        # Fallback: use first candidate with fixed orientation
        best_global_pts = axial_to_cartesian(candidates[0])
        n_pts = len(best_global_pts)
        best_global_center = (float(np.mean(best_global_pts[:, 0])),
                              float(np.mean(best_global_pts[:, 1])))
        best_global_beta = 0.0
        max_proj = 0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = best_global_pts[:, 0] * nx + best_global_pts[:, 1] * ny
            max_proj = max(max_proj, float(np.max(projs)), float(-np.min(projs)))
        best_global_S = (max_proj + APOTHEM) / APOTHEM
        best_global_positions = list(zip(best_global_pts[:, 0], best_global_pts[:, 1]))

    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)

    # Small perturbation search to see if we can improve
    if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN - 5.0 and best_global_positions is not None:
        pts_perturbed = np.array(best_global_positions, dtype=float)
        improved = True
        step = 0.002
        max_iter = 100
        for it in range(max_iter):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN - 5.0:
                break
            if not improved:
                break
            improved = False
            rng = np.random.default_rng(12345 + it)
            for trial in range(30):
                idx = rng.integers(0, len(pts_perturbed))
                dx = rng.normal(0, step)
                dy = rng.normal(0, step)
                pts_trial = pts_perturbed.copy()
                pts_trial[idx, 0] += dx
                pts_trial[idx, 1] += dy
                S_trial, pos_trial, center_trial = solve_packing_lp(pts_trial, best_global_beta)
                if S_trial < best_global_S - 1e-10:
                    pts_perturbed = pts_trial
                    best_global_S = S_trial
                    best_global_positions = pos_trial
                    best_global_center = center_trial
                    improved = True
                    step *= 0.7
                    break

    # Final output
    if best_global_positions is not None:
        pts_final = np.array(best_global_positions, dtype=float)
    else:
        pts_final = best_global_pts

    # Scale slightly to ensure strict containment
    SCALE = 1.0 + 1e-10
    pts_scaled = []
    for x, y in pts_final:
        sx = cx + (x - cx) * SCALE
        sy = cy + (y - cy) * SCALE
        pts_scaled.append((sx, sy))
    pts_scaled = np.array(pts_scaled)

    # Recompute S from scaled points
    max_proj = 0
    for k in range(6):
        ang = best_global_beta + math.pi / 6.0 + k * math.pi / 3.0
        nx, ny = math.cos(ang), math.sin(ang)
        projs = (pts_scaled[:, 0] - cx) * nx + (pts_scaled[:, 1] - cy) * ny
        max_proj = max(max_proj, float(np.max(projs)), float(-np.min(projs)))

    D_scaled = max_proj
    S_scaled = (D_scaled + APOTHEM) / APOTHEM

    inner = []
    for x, y in pts_scaled:
        inner.append([float(x), float(y), float(angle_deg)])

    center = [float(cx), float(cy)]
    side = float(S_scaled)
    rotation = float(angle_deg)

    return inner, center, side, rotation
