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
HEX_PERIOD = math.pi / 6.0

def _hex_vertices(cx, cy, angle_deg, s=1.0):
    """Vertices of a regular hexagon with center (cx,cy), side s, rotation angle_deg."""
    a = math.radians(angle_deg)
    verts = []
    for k in range(6):
        theta = a + k * math.pi / 3.0
        verts.append((cx + s * math.cos(theta), cy + s * math.sin(theta)))
    return verts

def _point_in_hex(px, py, cx, cy, angle_deg, s=1.0):
    """Check if point (px,py) is inside hexagon with center (cx,cy), side s, rotation angle_deg."""
    a = math.radians(angle_deg)
    dx, dy = px - cx, py - cy
    # Transform to hexagon's local frame
    ca, sa = math.cos(a), math.sin(a)
    lx = ca * dx + sa * dy
    ly = -sa * dx + ca * dy
    # Hexagon with flat top (vertices at 0, 60, 120, 180, 240, 300 degrees)
    # In local frame, the hexagon has vertices at angles 0, 60, ...
    # The hexagon can be described by |ly| <= APOTHEM and |lx + ly/2| <= APOTHEM and |lx - ly/2| <= APOTHEM
    # Actually for a regular hexagon with vertices at 0, 60, 120, 180, 240, 300:
    # The sides are at angles 30, 90, 150, 210, 270, 330
    # The apothem direction is at 30, 90, 150, 210, 270, 330
    # So constraints: |(lx,ly) dot (cos(30+k*60), sin(30+k*60))| <= APOTHEM for k=0..5
    for k in range(6):
        theta = math.pi / 6.0 + k * math.pi / 3.0
        proj = lx * math.cos(theta) + ly * math.sin(theta)
        if abs(proj) > s * APOTHEM + 1e-9:
            return False
    return True

def _hexagon_overlap(h1, h2, tol=1e-7):
    """Check if two hexagons (each as list of 6 vertices) overlap.
    Uses separating axis theorem for two convex polygons."""
    def get_axes(poly):
        axes = []
        for i in range(len(poly)):
            x1, y1 = poly[i]
            x2, y2 = poly[(i+1) % len(poly)]
            dx, dy = x2-x1, y2-y1
            length = math.sqrt(dx*dx + dy*dy)
            if length > 1e-12:
                axes.append((-dy/length, dx/length))
        return axes
    
    axes = get_axes(h1) + get_axes(h2)
    for ax, ay in axes:
        # Project both polygons
        proj1 = [vx*ax + vy*ay for vx, vy in h1]
        proj2 = [vx*ax + vy*ay for vx, vy in h2]
        min1, max1 = min(proj1), max(proj1)
        min2, max2 = min(proj2), max(proj2)
        if max1 < min2 + tol or max2 < min1 + tol:
            return False  # Separated
    return True  # Overlapping

def _check_disjoint(inner_hexes, tol=1e-7):
    """Check all pairs of inner hexagons are disjoint."""
    n = len(inner_hexes)
    for i in range(n):
        for j in range(i+1, n):
            if _hexagon_overlap(inner_hexes[i], inner_hexes[j], tol):
                return False
    return True

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

    # Additional compact and diverse configurations
    axial_46 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3),
        (1, 4)
    ]
    candidates.append(axial_46)

    axial_47 = [
        (0, 0), (1, 0), (2, 0),
        (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3),
        (0, 4), (1, 4)
    ]
    candidates.append(axial_47)

    axial_48 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (1, 1), (-1, -1), (0, 2)
    ]
    candidates.append(axial_48)

    axial_49 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3),
        (0, 4)
    ]
    candidates.append(axial_49)

    axial_50 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3),
        (0, 4)
    ]
    candidates.append(axial_50)

    axial_51 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3),
        (1, 4)
    ]
    candidates.append(axial_51)

    axial_52 = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, 1), (-1, 1), (2, -1),
        (1, -1), (-1, -1), (0, 2), (0, -2)
    ]
    candidates.append(axial_52)

    axial_53 = [
        (0, 0), (1, 0), (2, 0),
        (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (1, 3), (2, 3), (3, 3)
    ]
    candidates.append(axial_53)

    axial_54 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3),
        (1, 4), (2, 4)
    ]
    candidates.append(axial_54)

    axial_55 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3),
        (0, 4), (1, 4)
    ]
    candidates.append(axial_55)

    # Additional diverse candidates - compact hexagonal clusters
    axial_36 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_36)

    axial_37 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3),
        (0, 4), (1, 4),
        (0, 5)
    ]
    candidates.append(axial_37)

    axial_38 = [
        (0, 0), (1, 0), (2, 0),
        (1, 1), (2, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_38)

    axial_39 = [
        (0, 0),
        (1, 0), (-1, 0),
        (0, 1), (1, 1), (-1, 1),
        (0, 2), (1, 2),
        (0, 3), (-1, 3),
        (0, 4)
    ]
    candidates.append(axial_39)

    axial_40 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3),
        (0, 4), (1, 4)
    ]
    candidates.append(axial_40)

    axial_41 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3),
        (0, 4), (1, 4)
    ]
    candidates.append(axial_41)

    axial_42 = [
        (0, 0), (1, 0), (2, 0),
        (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3)
    ]
    candidates.append(axial_42)

    axial_43 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (-2, 2), (1, -2)
    ]
    candidates.append(axial_43)

    axial_44 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3),
        (1, 4)
    ]
    candidates.append(axial_44)

    axial_45 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3),
        (1, 4), (2, 4)
    ]
    candidates.append(axial_45)

    # The 11-hexagon hexagonal ring: center + 6 neighbors + 6 at distance 2 (corners)
    # This is the most compact 11-hex cluster (fits in a hex of side 2)
    axial_ring11 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2), (2, -2), (-2, 2)
    ]
    candidates.append(axial_ring11)

    # Additional diverse candidates
    axial_29 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1),
        (0, 2), (1, 2)
    ]
    candidates.append(axial_29)

    axial_30 = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, 1), (-1, 1), (2, -1),
        (1, -1), (-1, -1), (0, 2)
    ]
    candidates.append(axial_30)

    axial_31 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3)
    ]
    candidates.append(axial_31)

    axial_32 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (-1, -1), (0, -2)
    ]
    candidates.append(axial_32)

    axial_33 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3)
    ]
    candidates.append(axial_33)

    axial_34 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (-2, 2), (1, -2), (-1, -2)
    ]
    candidates.append(axial_34)

    axial_35 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (0, 3), (1, 3)
    ]
    candidates.append(axial_35)

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
        """LP over center for fixed beta. Returns (S, center, D). Vectorized."""
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), None, float('inf')

        normals = get_normals(beta)
        c_obj = np.array([0.0, 0.0, 1.0])

        # Vectorized: for each normal (nx,ny), constraints are:
        #   (nx, ny, -1) . (cx, cy, D) <= px*nx + py*ny   for each point
        #  (-nx,-ny, -1) . (cx, cy, D) <= -(px*nx + py*ny) for each point
        n_pts = len(pts)
        # Build A_ub: shape (2*6*n_pts, 3)
        # For normal k: rows are [nx, ny, -1] and [-nx, -ny, -1] repeated n_pts times
        A_ub_blocks = []
        b_ub_blocks = []
        for nx, ny in normals:
            proj = pts[:, 0] * nx + pts[:, 1] * ny
            A1 = np.tile([nx, ny, -1.0], (n_pts, 1))
            A2 = np.tile([-nx, -ny, -1.0], (n_pts, 1))
            A_ub_blocks.append(A1)
            A_ub_blocks.append(A2)
            b_ub_blocks.append(proj)
            b_ub_blocks.append(-proj)

        A_ub = np.vstack(A_ub_blocks)
        b_ub = np.concatenate(b_ub_blocks)
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

    def optimize_shape(pts):
        """Optimize (center, rotation) for a fixed point set.
        Strategy:
          1. Coarse grid search over beta (with LP center solve).
          2. Fine grid + bounded scalar refine around best beta.
          3. Smooth-surrogate Nelder-Mead in (cx, cy, beta) with annealing.
          4. Exact high-precision polish from multiple starts.
        """
        best_S = float('inf')
        best_beta = 0.0
        best_center = (float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1])))

        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), 0.0, best_center

        # --- Phase 1: Coarse grid over beta ---
        n_coarse = 61
        for i in range(n_coarse):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            beta = i * (math.pi / 6) / (n_coarse - 1)
            S, center, D = calc_S_for_beta(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                if center is not None:
                    best_center = center

        # --- Phase 2: Fine grid around best beta ---
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                n_fine = 201
                fine_lo = max(0.0, best_beta - 2.0 * math.pi / 6 / (n_coarse - 1))
                fine_hi = min(math.pi / 6, best_beta + 2.0 * math.pi / 6 / (n_coarse - 1))
                for i in range(n_fine):
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    beta = fine_lo + i * (fine_hi - fine_lo) / (n_fine - 1)
                    S, center, D = calc_S_for_beta(pts, beta)
                    if S < best_S:
                        best_S = S
                        best_beta = beta
                        if center is not None:
                            best_center = center
        except Exception:
            pass

        # --- Phase 3: Bounded scalar refine around best beta ---
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                lo = max(0.0, best_beta - math.pi / 600)
                hi = min(math.pi / 6, best_beta + math.pi / 600)
                res = minimize_scalar(lambda b: calc_S_for_beta(pts, b)[0],
                                      bounds=(lo, hi), method='bounded',
                                      options={'maxiter': 500, 'xatol': 1e-14})
                if res.fun < best_S:
                    best_S = float(res.fun)
                    best_beta = float(res.x)
                    _, best_center, _ = calc_S_for_beta(pts, best_beta)
        except Exception:
            pass

        # --- Phase 4: Smooth-surrogate Nelder-Mead with annealing ---
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                temperatures = [0.3, 0.15, 0.08, 0.04, 0.02, 0.01]
                start_betas = [best_beta, best_beta + 0.02, best_beta - 0.02,
                               0.0, math.pi / 12, math.pi / 6,
                               best_beta + math.pi / 36, best_beta - math.pi / 36]

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

        # --- Phase 5: Exact high-precision polish ---
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                cx0, cy0 = best_center
                polish_starts = [
                    [cx0, cy0, best_beta],
                    [cx0, cy0, best_beta + 0.003],
                    [cx0, cy0, best_beta - 0.003],
                    [cx0 + 0.003, cy0, best_beta],
                    [cx0, cy0 + 0.003, best_beta],
                    [cx0 - 0.003, cy0, best_beta],
                    [cx0, cy0 - 0.003, best_beta],
                    [cx0 + 0.001, cy0 + 0.001, best_beta],
                    [cx0 - 0.001, cy0 - 0.001, best_beta],
                    [cx0 + 0.001, cy0, best_beta + 0.001],
                    [cx0 - 0.001, cy0, best_beta - 0.001],
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
                            best_beta = ((float(res.x[2]) % HEX_PERIOD) + HEX_PERIOD) % HEX_PERIOD
                    except Exception:
                        pass

                    # SLSQP polish from current best
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        continue
                    try:
                        res2 = minimize(exact_objective_3d,
                                        [best_center[0], best_center[1], best_beta],
                                        method='SLSQP',
                                        options={'maxiter': 3000, 'ftol': 1e-16})
                        if res2.fun < best_S:
                            best_S = float(res2.fun)
                            best_center = (float(res2.x[0]), float(res2.x[1]))
                            best_beta = ((float(res2.x[2]) % HEX_PERIOD) + HEX_PERIOD) % HEX_PERIOD
                    except Exception:
                        pass
        except Exception:
            pass

        # --- Phase 6: Final 1D scalar polish over beta (LP center each time) ---
        # For a fixed point set, S(beta) = (D(beta, c*(beta)) + APOTHEM) / APOTHEM
        # where c*(beta) is the LP-optimal center. S(beta) is piecewise-linear in beta
        # (the active constraint set changes at breakpoints), so it is not smooth.
        # A final bounded scalar minimization can find a tighter minimum.
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                # Multiple narrow windows around the current best and symmetric points
                windows = []
                for w_center in [best_beta, (best_beta + HEX_PERIOD / 2) % HEX_PERIOD]:
                    for w_half in [HEX_PERIOD / 2000, HEX_PERIOD / 1000, HEX_PERIOD / 500]:
                        lo = max(0.0, w_center - w_half)
                        hi = min(HEX_PERIOD, w_center + w_half)
                        if hi > lo:
                            windows.append((lo, hi))

                for (lo, hi) in windows:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    try:
                        res = minimize_scalar(
                            lambda b: calc_S_for_beta(pts, b)[0],
                            bounds=(lo, hi), method='bounded',
                            options={'maxiter': 2000, 'xatol': 1e-16})
                        if res.fun < best_S:
                            best_S = float(res.fun)
                            best_beta = float(res.x)
                            _, best_center, _ = calc_S_for_beta(pts, best_beta)
                    except Exception:
                        pass
        except Exception:
            pass

        return best_S, best_beta, best_center

    # --- Main loop over candidates ---
    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None

    for idx_axial, axial in enumerate(candidates):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        pts = axial_to_cartesian(axial)
        # Quick bounding-box estimate to skip obviously worse candidates
        x_range = float(np.max(pts[:, 0]) - np.min(pts[:, 0]))
        y_range = float(np.max(pts[:, 1]) - np.min(pts[:, 1]))
        rough_S = max(x_range, y_range * 2.0 / math.sqrt(3)) / (2.0 * APOTHEM) + 1.0
        if best_global_S < float('inf') and rough_S > best_global_S * 1.15:
            # Skip this candidate — bounding box already too large
            continue
        S, beta, center = optimize_shape(pts)
        if S < best_global_S:
            best_global_S = S
            best_global_pts = pts
            best_global_beta = beta
            best_global_center = center

    # --- Post-processing: verify disjointness and attempt local perturbation ---
    def verify_and_refine(pts, beta, center, S):
        """Verify the configuration is valid and try to improve via local perturbation."""
        cx, cy = center
        angle_deg = math.degrees(beta)
        
        # Build inner hexagons
        inner_hexes = []
        for x, y in pts:
            hex_verts = _hex_vertices(x, y, angle_deg)
            inner_hexes.append(hex_verts)
        
        # Check disjointness
        if not _check_disjoint(inner_hexes):
            return None  # Invalid configuration
        
        # Try small perturbations of positions to see if we can reduce S
        # (This is a heuristic; the LP already optimizes center, but individual
        #  hexagons might benefit from non-lattice adjustments)
        best_S_local = S
        best_pts_local = pts.copy()
        best_center_local = center
        best_beta_local = beta
        
        # Try small random perturbations
        rng = np.random.default_rng(42)
        for attempt in range(15):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            # Perturb a random subset of points
            n_perturb = max(1, len(pts) // 5)
            indices = rng.choice(len(pts), size=n_perturb, replace=False)
            perturbed = pts.copy()
            for idx in indices:
                perturbed[idx, 0] += rng.uniform(-0.03, 0.03)
                perturbed[idx, 1] += rng.uniform(-0.03, 0.03)
            
            # Re-optimize center and rotation for this perturbed set
            S_try, beta_try, center_try = optimize_shape(perturbed)
            if S_try < best_S_local:
                # Verify this is valid
                inner_hexes_try = []
                for x, y in perturbed:
                    hex_verts = _hex_vertices(x, y, math.degrees(beta_try))
                    inner_hexes_try.append(hex_verts)
                if _check_disjoint(inner_hexes_try):
                    best_S_local = S_try
                    best_pts_local = perturbed
                    best_center_local = center_try
                    best_beta_local = beta_try
        
        return best_S_local, best_pts_local, best_beta_local, best_center_local

    # Apply verification and refinement to the best candidate
    if best_global_pts is not None and best_global_center is not None:
        result = verify_and_refine(best_global_pts, best_global_beta, best_global_center, best_global_S)
        if result is not None:
            best_global_S, best_global_pts, best_global_beta, best_global_center = result

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
