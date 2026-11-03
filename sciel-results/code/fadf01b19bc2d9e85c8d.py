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

def _estimate_compactness(axial_list):
    """Rough estimate of bounding hexagon side for a set of axial coords (lower = more compact)."""
    pts = []
    for q, r in axial_list:
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        pts.append((x, y))
    if not pts:
        return 99.0
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    # For a regular hexagon with flat-top orientation, the 6 support values are:
    # n0=(0,1), n1=(-sqrt(3)/2,-1/2), n2=(sqrt(3)/2,-1/2), n3=(0,-1), n4=(sqrt(3)/2,1/2), n5=(-sqrt(3)/2,1/2)
    normals = [
        (0.0, 1.0),
        (-math.sqrt(3)/2, -0.5),
        (math.sqrt(3)/2, -0.5),
        (0.0, -1.0),
        (math.sqrt(3)/2, 0.5),
        (-math.sqrt(3)/2, 0.5)
    ]
    max_proj = 0.0
    for nx, ny in normals:
        for x, y in pts:
            proj = abs(x * nx + y * ny)
            if proj > max_proj:
                max_proj = proj
    return (max_proj + APOTHEM) / APOTHEM

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # --- Candidate point sets (axial coordinates) ---
    candidates = []

    axial_1 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ]
    candidates.append(axial_1)

    axial_2 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_2)

    axial_3 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 1), (3, 1)
    ]
    candidates.append(axial_3)

    axial_4 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2)
    ]
    candidates.append(axial_4)

    axial_5 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_5)

    axial_6 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3)
    ]
    candidates.append(axial_6)

    axial_7 = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_7)

    axial_8 = [
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)
    ]
    candidates.append(axial_8)

    axial_9 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (-2, 2), (-1, 3), (1, -2)
    ]
    candidates.append(axial_9)

    axial_10 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3)
    ]
    candidates.append(axial_10)

    axial_11 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (1, 3)
    ]
    candidates.append(axial_11)

    axial_12 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3),
        (0, 4)
    ]
    candidates.append(axial_12)

    axial_13 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (1, 2)
    ]
    candidates.append(axial_13)

    axial_14 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (1, 1), (-1, -1)
    ]
    candidates.append(axial_14)

    axial_15 = [
        (0, 0), (1, 0), (2, 0),
        (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 1), (3, 1), (0, -1)
    ]
    candidates.append(axial_15)

    axial_16 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3), (3, 2)
    ]
    candidates.append(axial_16)

    axial_17 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (1, 2), (-1, 2), (0, 3)
    ]
    candidates.append(axial_17)

    axial_18 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2), (3, 2)
    ]
    candidates.append(axial_18)

    axial_19 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3), (3, 3)
    ]
    candidates.append(axial_19)

    axial_20 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (1, 1), (-1, 1)
    ]
    candidates.append(axial_20)

    axial_21 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (1, 2), (2, 2), (3, 2)
    ]
    candidates.append(axial_21)

    axial_22 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3)
    ]
    candidates.append(axial_22)

    axial_23 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_23)

    axial_24 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3),
        (0, 4)
    ]
    candidates.append(axial_24)

    axial_25 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_25)

    axial_26 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (1, 3), (2, 3)
    ]
    candidates.append(axial_26)

    axial_27 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (1, -2)
    ]
    candidates.append(axial_27)

    axial_28 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3), (2, 3),
        (1, 4), (2, 4)
    ]
    candidates.append(axial_28)

    # More diverse compact configurations
    axial_40 = [
        (0, 0), (1, 0), (2, 0),
        (1, 1), (2, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3), (2, 3),
        (1, 4)
    ]
    candidates.append(axial_40)

    axial_41 = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1),
        (2, 1), (-2, 2), (1, -2), (-1, -2), (0, 3)
    ]
    candidates.append(axial_41)

    axial_42 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3),
        (0, 4), (1, 4)
    ]
    candidates.append(axial_42)

    axial_43 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3),
        (1, 4), (2, 4)
    ]
    candidates.append(axial_43)

    axial_44 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (1, 2), (-1, -2), (0, 3)
    ]
    candidates.append(axial_44)

    axial_45 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_45)

    # More diverse compact configurations
    axial_46 = [
        (0, 0), (1, 0), (2, 0),
        (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_46)

    axial_47 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3),
        (0, 4), (1, 4)
    ]
    candidates.append(axial_47)

    axial_48 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2), (3, 2),
        (0, 3), (1, 3)
    ]
    candidates.append(axial_48)

    axial_49 = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1),
        (2, 0), (-2, 0), (0, 2), (0, -2), (1, 2)
    ]
    candidates.append(axial_49)

    axial_50 = [
        (0, 0), (1, 0), (2, 0),
        (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_50)

    axial_51 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_51)

    axial_52 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3), (3, 2)
    ]
    candidates.append(axial_52)

    axial_53 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (1, 2), (-1, -2), (0, 3)
    ]
    candidates.append(axial_53)

    axial_54 = [
        (0, 0), (1, 0), (2, 0),
        (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3)
    ]
    candidates.append(axial_54)

    # Additional corner/side-filling templates
    axial_29 = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1),
        (2, 1), (-2, 2), (1, -2), (-1, -2)
    ]
    candidates.append(axial_29)

    axial_30 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3),
        (1, 4)
    ]
    candidates.append(axial_30)

    axial_31 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3),
        (2, 3), (3, 2)
    ]
    candidates.append(axial_31)

    axial_32 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (-2, 2), (1, -2), (-1, -2)
    ]
    candidates.append(axial_32)

    # Additional diverse configurations
    axial_33 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (1, 3), (2, 3), (3, 3)
    ]
    candidates.append(axial_33)

    axial_34 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3), (3, 3)
    ]
    candidates.append(axial_34)

    axial_35 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (1, 2), (-1, -2)
    ]
    candidates.append(axial_35)

    axial_36 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3), (3, 2)
    ]
    candidates.append(axial_36)

    axial_37 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (1, 3)
    ]
    candidates.append(axial_37)

    axial_38 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (-2, 2), (-1, 3), (1, -2)
    ]
    candidates.append(axial_38)

    axial_39 = [
        (0, 0), (1, 0), (2, 0),
        (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (1, 3), (2, 3)
    ]
    candidates.append(axial_39)

    # --- Geometry helpers ---

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    def get_normals(beta):
        """Return 6 unit normals of the containing hexagon for rotation beta."""
        return np.array([
            (math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0))
            for k in range(6)
        ])

    def compute_D(pts, center, normals):
        """Exact: D = max over points and normals of |(p - c) . n|."""
        centered = pts - np.array(center)
        projs = centered @ normals.T  # (n_pts, 6)
        return float(np.max(np.abs(projs)))

    def calc_S_for_beta(pts, beta, center=None):
        """LP over center for fixed beta. Returns (S, center, D)."""
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), None, float('inf')

        normals = get_normals(beta)
        c_obj = np.array([0.0, 0.0, 1.0])

        # Vectorized: for each normal (nx,ny) and each point (px,py):
        #   -nx*cx - ny*cy - D <= -(px*nx + py*ny)   =>  (nx*cx + ny*cy + D) >= px*nx + py*ny
        #   nx*cx + ny*cy - D <= px*nx + py*ny       =>  (-nx*cx - ny*cy + D) >= -px*nx - py*ny
        # Rearranged to A_ub @ x <= b_ub form:
        #   -nx*cx - ny*cy - D <= -(px*nx + py*ny)
        #    nx*cx + ny*cy - D <=  px*nx + py*ny

        n_pts = len(pts)
        # Build A_ub as (2*6*n_pts, 3) and b_ub as (2*6*n_pts,)
        # For normal k with (nx, ny):
        #   row 1: [-nx, -ny, -1] <= -(px*nx + py*ny)
        #   row 2: [ nx,  ny, -1] <=  (px*nx + py*ny)

        A_ub_list = []
        b_ub_list = []
        for nx, ny in normals:
            dots = pts[:, 0] * nx + pts[:, 1] * ny  # (n_pts,)
            # Row type 1: -nx*cx - ny*cy - D <= -dot
            A_ub_list.append(np.column_stack([-nx * np.ones(n_pts), -ny * np.ones(n_pts), -np.ones(n_pts)]))
            b_ub_list.append(-dots)
            # Row type 2: nx*cx + ny*cy - D <= dot
            A_ub_list.append(np.column_stack([nx * np.ones(n_pts), ny * np.ones(n_pts), -np.ones(n_pts)]))
            b_ub_list.append(dots)

        A_ub = np.vstack(A_ub_list)
        b_ub = np.concatenate(b_ub_list)
        bounds = [(None, None), (None, None), (0, None)]

        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                D = float(res.x[2])
                cx, cy = float(res.x[0]), float(res.x[1])
                S = (D + APOTHEM) / APOTHEM
                return S, (cx, cy), D
            else:
                return float('inf'), None, float('inf')
        except Exception:
            return float('inf'), None, float('inf')

    def smooth_D(pts, center, normals, temperature):
        """Smooth surrogate for D via log-sum-exp soft-max over |projections|."""
        centered = pts - np.array(center)
        projs = centered @ normals.T  # (n_pts, 6)
        all_projs = np.concatenate([projs.ravel(), (-projs).ravel()])
        max_val = float(np.max(all_projs))
        shifted = (all_projs - max_val) / temperature
        shifted = np.clip(shifted, -60.0, 60.0)
        log_sum = np.log(np.sum(np.exp(shifted))) + max_val / temperature
        return temperature * log_sum

    def smooth_calc_S(pts, beta, center, temperature):
        normals = get_normals(beta)
        D = smooth_D(pts, center, normals, temperature)
        S = (D + APOTHEM) / APOTHEM
        return S

    def exact_objective_3d(pts, params):
        """Exact objective over (cx, cy, beta) — for final polish."""
        cx, cy, beta = params
        beta = ((beta % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
        normals = get_normals(beta)
        D = compute_D(pts, (cx, cy), normals)
        return (D + APOTHEM) / APOTHEM

    def newton_polish_center(pts, beta, center, max_iter=50, tol=1e-14):
        """High-precision 2D Newton polish for center at fixed beta.
        Uses subgradient descent on D(cx,cy) = max_k |n_k . (p - c)|.
        Returns improved center and D."""
        normals = get_normals(beta)
        cx, cy = center
        D_old = compute_D(pts, (cx, cy), normals)
        
        for _ in range(max_iter):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            # Find active constraints (those achieving max |proj|)
            centered = pts - np.array([cx, cy])
            projs = centered @ normals.T  # (n_pts, 6)
            abs_projs = np.abs(projs)
            D_cur = float(np.max(abs_projs))
            
            if abs(D_cur - D_old) < tol:
                break
            
            # Collect active normals with their signs
            active_normals = []
            active_signs = []
            for k in range(6):
                col = abs_projs[:, k]
                active_mask = np.abs(col - D_cur) < tol
                if np.any(active_mask):
                    sign = np.sign(projs[np.argmax(col), k])
                    if sign == 0:
                        sign = 1.0
                    active_normals.append(normals[k])
                    active_signs.append(sign)
            
            if not active_normals:
                break
            
            # Subgradient: sum of active signed normals
            subgrad = np.zeros(2)
            for n, s in zip(active_normals, active_signs):
                subgrad += s * n
            
            norm_subgrad = np.linalg.norm(subgrad)
            if norm_subgrad < 1e-15:
                break
            
            # Newton step: move center in direction that reduces D
            # D is convex in center, so we do gradient descent on D
            # Use adaptive step size
            step_size = 0.1 / norm_subgrad
            cx_new = cx + step_size * subgrad[0]
            cy_new = cy + step_size * subgrad[1]
            
            D_new = compute_D(pts, (cx_new, cy_new), normals)
            if D_new < D_cur:
                cx, cy = cx_new, cy_new
                D_old = D_new
            else:
                # Halve step size
                step_size *= 0.5
                cx_new = cx + step_size * subgrad[0]
                cy_new = cy + step_size * subgrad[1]
                D_new = compute_D(pts, (cx_new, cy_new), normals)
                if D_new < D_cur:
                    cx, cy = cx_new, cy_new
                    D_old = D_new
                else:
                    break
        
        return (cx, cy), compute_D(pts, (cx, cy), normals)

    def optimize_shape(pts):
        """Optimize (center, rotation) for a fixed point set.
        Strategy:
          1. Fast vertex-based coarse scan over beta.
          2. LP refinement at top candidate betas.
          3. Bounded scalar refine around best beta.
          4. Smooth-surrogate Nelder-Mead in (cx, cy, beta) with annealing.
          5. Exact high-precision polish from multiple starts.
          6. Newton polish for center at final beta.
        """
        best_S = float('inf')
        best_beta = 0.0
        best_center = (float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1])))

        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), 0.0, best_center

        # --- Phase 1: Fast vertex-based coarse scan over beta ---
        # For each beta, compute D via LP (center optimization) but use a quick
        # vertex-based estimate to rank betas, then LP only top ones.
        n_coarse = 121
        coarse_results = []
        for i in range(n_coarse):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            beta = i * (math.pi / 6) / (n_coarse - 1)
            normals = get_normals(beta)
            # Quick D estimate: center at centroid
            centered = pts - np.array(best_center)
            projs = centered @ normals.T
            D_est = float(np.max(np.abs(projs)))
            S_est = (D_est + APOTHEM) / APOTHEM
            coarse_results.append((S_est, beta))

        # Sort by estimated S, take top 15 for LP refinement
        coarse_results.sort(key=lambda x: x[0])
        top_betas = [b for _, b in coarse_results[:15]]

        for beta in top_betas:
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            S, center, D = calc_S_for_beta(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                if center is not None:
                    best_center = center

        # --- Phase 2: Bounded scalar refine around best beta (two passes) ---
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                lo = max(0.0, best_beta - math.pi / 48)
                hi = min(math.pi / 6, best_beta + math.pi / 48)
                res = minimize_scalar(lambda b: calc_S_for_beta(pts, b)[0],
                                      bounds=(lo, hi), method='bounded',
                                      options={'maxiter': 300, 'xatol': 1e-12})
                if res.fun < best_S:
                    best_S = float(res.fun)
                    best_beta = float(res.x)
                    _, best_center, _ = calc_S_for_beta(pts, best_beta)
        except Exception:
            pass

        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                lo = max(0.0, best_beta - math.pi / 200)
                hi = min(math.pi / 6, best_beta + math.pi / 200)
                res = minimize_scalar(lambda b: calc_S_for_beta(pts, b)[0],
                                      bounds=(lo, hi), method='bounded',
                                      options={'maxiter': 500, 'xatol': 1e-15})
                if res.fun < best_S:
                    best_S = float(res.fun)
                    best_beta = float(res.x)
                    _, best_center, _ = calc_S_for_beta(pts, best_beta)
        except Exception:
            pass

        # --- Phase 3: Smooth-surrogate Nelder-Mead with annealing ---
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                temperatures = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]
                start_betas = [best_beta, best_beta + 0.03, best_beta - 0.03,
                               0.0, math.pi / 12, math.pi / 6,
                               best_beta + math.pi / 24, best_beta - math.pi / 24]

                for temp in temperatures:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    for sb in start_betas:
                        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                            break

                        def smooth_obj(params, t=temp):
                            cx, cy, beta = params
                            beta = ((beta % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                            return smooth_calc_S(pts, beta, (cx, cy), t)

                        init = [best_center[0], best_center[1], sb]
                        try:
                            res = minimize(smooth_obj, init, method='Nelder-Mead',
                                           options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12})
                            if res.fun < 1e10:
                                cx_r, cy_r, beta_r = res.x
                                beta_r = ((beta_r % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                                S_exact, c_exact, _ = calc_S_for_beta(pts, beta_r)
                                if S_exact < best_S:
                                    best_S = S_exact
                                    best_beta = beta_r
                                    if c_exact is not None:
                                        best_center = c_exact
                        except Exception:
                            pass
        except Exception:
            pass

        # --- Phase 4: Exact high-precision polish ---
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                cx0, cy0 = best_center
                polish_starts = [
                    [cx0, cy0, best_beta],
                    [cx0, cy0, best_beta + 0.01],
                    [cx0, cy0, best_beta - 0.01],
                    [cx0 + 0.01, cy0, best_beta],
                    [cx0, cy0 + 0.01, best_beta],
                    [cx0 - 0.01, cy0, best_beta],
                    [cx0, cy0 - 0.01, best_beta],
                    [cx0 + 0.005, cy0 + 0.005, best_beta],
                    [cx0 - 0.005, cy0 - 0.005, best_beta],
                    [cx0 + 0.02, cy0 + 0.01, best_beta + 0.005],
                    [cx0 - 0.02, cy0 - 0.01, best_beta - 0.005],
                    [cx0 + 0.01, cy0 - 0.02, best_beta + 0.01],
                    [cx0 - 0.01, cy0 + 0.02, best_beta - 0.01],
                ]

                for init in polish_starts:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    # Nelder-Mead polish
                    try:
                        res = minimize(exact_objective_3d, init, method='Nelder-Mead',
                                       options={'maxiter': 20000, 'xatol': 1e-15, 'fatol': 1e-15})
                        if res.fun < best_S:
                            best_S = float(res.fun)
                            best_center = (float(res.x[0]), float(res.x[1]))
                            best_beta = ((float(res.x[2]) % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                    except Exception:
                        pass

                    # SLSQP polish from current best
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        continue
                    try:
                        res2 = minimize(exact_objective_3d,
                                        [best_center[0], best_center[1], best_beta],
                                        method='SLSQP',
                                        options={'maxiter': 2000, 'ftol': 1e-16})
                        if res2.fun < best_S:
                            best_S = float(res2.fun)
                            best_center = (float(res2.x[0]), float(res2.x[1]))
                            best_beta = ((float(res2.x[2]) % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                    except Exception:
                        pass

                    # Second SLSQP pass from the SLSQP result for extra tightening
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        continue
                    try:
                        res3 = minimize(exact_objective_3d,
                                        [best_center[0], best_center[1], best_beta],
                                        method='SLSQP',
                                        options={'maxiter': 2000, 'ftol': 1e-16})
                        if res3.fun < best_S:
                            best_S = float(res3.fun)
                            best_center = (float(res3.x[0]), float(res3.x[1]))
                            best_beta = ((float(res3.x[2]) % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                    except Exception:
                        pass
        except Exception:
            pass

        # --- Phase 5: Tight LP-based beta refinement around the best found beta ---
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                # Fine grid around best_beta with LP center solve (exact, fast)
                n_fine = 200
                span = math.pi / 6 / 40
                for i in range(n_fine):
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    b = best_beta + (i / (n_fine - 1) - 0.5) * 2 * span
                    b = ((b % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                    S, center, D = calc_S_for_beta(pts, b)
                    if S < best_S:
                        best_S = S
                        best_beta = b
                        if center is not None:
                            best_center = center

                # Bounded scalar refinement using the exact LP (tight window)
                if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                    lo = max(0.0, best_beta - span / 3)
                    hi = min(math.pi / 6, best_beta + span / 3)
                    try:
                        res = minimize_scalar(lambda b: calc_S_for_beta(pts, b)[0],
                                              bounds=(lo, hi), method='bounded',
                                              options={'maxiter': 500, 'xatol': 1e-16})
                        if res.fun < best_S:
                            best_S = float(res.fun)
                            best_beta = float(res.x)
                            _, best_center, _ = calc_S_for_beta(pts, best_beta)
                    except Exception:
                        pass

                # Second bounded scalar refinement (slightly wider)
                if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                    lo2 = max(0.0, best_beta - span / 1.5)
                    hi2 = min(math.pi / 6, best_beta + span / 1.5)
                    try:
                        res2 = minimize_scalar(lambda b: calc_S_for_beta(pts, b)[0],
                                               bounds=(lo2, hi2), method='bounded',
                                               options={'maxiter': 500, 'xatol': 1e-16})
                        if res2.fun < best_S:
                            best_S = float(res2.fun)
                            best_beta = float(res2.x)
                            _, best_center, _ = calc_S_for_beta(pts, best_beta)
                    except Exception:
                        pass
        except Exception:
            pass

        # --- Phase 6: Newton polish for center at final beta ---
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                polished_center, D_polished = newton_polish_center(pts, best_beta, best_center)
                S_polished = (D_polished + APOTHEM) / APOTHEM
                if S_polished < best_S:
                    best_S = S_polished
                    best_center = polished_center
        except Exception:
            pass

        return best_S, best_beta, best_center

    # --- Sort candidates by estimated compactness (most compact first) ---
    scored_candidates = []
    for axial in candidates:
        est = _estimate_compactness(axial)
        scored_candidates.append((est, axial))
    scored_candidates.sort(key=lambda x: x[0])

    # --- Main loop over candidates (most compact first) ---
    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None

    for est, axial in scored_candidates:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        # Skip if estimate is already worse than current best
        if est >= best_global_S and best_global_S < float('inf'):
            continue
        pts = axial_to_cartesian(axial)
        S, beta, center = optimize_shape(pts)
        if S < best_global_S:
            best_global_S = S
            best_global_pts = pts
            best_global_beta = beta
            best_global_center = center

    # --- Fallback ---
    if best_global_center is None or best_global_S == float('inf'):
        best_global_pts = axial_to_cartesian(candidates[0])
        best_global_center = (float(np.mean(best_global_pts[:, 0])),
                              float(np.mean(best_global_pts[:, 1])))
        best_global_beta = 0.0
        normals = get_normals(best_global_beta)
        max_proj = 0.0
        for nx, ny in normals:
            projs = best_global_pts[:, 0] * nx + best_global_pts[:, 1] * ny
            max_proj = max(max_proj, float(np.max(projs)), float(-np.min(projs)))
        best_global_S = (max_proj + APOTHEM) / APOTHEM

    # --- Final Newton polish on the global best ---
    try:
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            polished_center, D_polished = newton_polish_center(best_global_pts, best_global_beta, best_global_center)
            S_polished = (D_polished + APOTHEM) / APOTHEM
            if S_polished < best_global_S:
                best_global_S = S_polished
                best_global_center = polished_center
    except Exception:
        pass

    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)

    # --- Final: recompute exact S with the chosen (center, beta), add tiny scale for safety ---
    SCALE = 1.0 + 1e-12
    pts_scaled = []
    for x, y in best_global_pts:
        sx = cx + (x - cx) * SCALE
        sy = cy + (y - cy) * SCALE
        pts_scaled.append((sx, sy))
    pts_scaled = np.array(pts_scaled)

    normals = get_normals(best_global_beta)
    max_proj = 0.0
    for nx, ny in normals:
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
