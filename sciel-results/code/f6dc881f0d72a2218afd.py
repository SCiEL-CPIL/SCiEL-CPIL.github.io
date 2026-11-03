import math
import numpy as np
import time
from scipy.optimize import linprog, minimize_scalar, minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 8.0
EPS = 1e-9

_HEX_VERTS = np.array([
    [1.0, 0.0],
    [0.5, math.sqrt(3) / 2],
    [-0.5, math.sqrt(3) / 2],
    [-1.0, 0.0],
    [-0.5, -math.sqrt(3) / 2],
    [0.5, -math.sqrt(3) / 2]
])


def _time_ok():
    return time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN


def _axial_to_cartesian(axial_list):
    pts = []
    for q, r in axial_list:
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        pts.append((x, y))
    return np.array(pts, dtype=float)


def _check_disjoint(axial_list, tol=1e-6):
    """Check that all pairs of hexagons are non-overlapping.
    For unit hexagons on a hex grid, centers must be at distance >= 2*APOTHEM*2 = 2*sqrt(3).
    Actually for flat-top hexagons with side 1, the minimum center distance for non-overlap
    is 2 * APOTHEM * 2? No.
    
    For two unit regular hexagons (side length 1), the minimum center-to-center distance
    for non-overlap depends on relative orientation. For parallel orientations, it's 2*APOTHEM
    in the normal direction... Actually the minimum distance is sqrt(3) (the distance between
    centers of adjacent hexagons in a honeycomb).
    
    In a honeycomb tiling with side-1 hexagons, adjacent centers are sqrt(3) apart.
    So the constraint is: for all pairs, |p_i - p_j| >= sqrt(3) - tol.
    """
    pts = _axial_to_cartesian(axial_list)
    n = len(pts)
    min_dist = math.sqrt(3) - tol
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < min_dist:
                return False
    return True


def _lp_enclosure(pts, beta):
    """Given fixed rotation beta, find optimal center and minimal S via LP.
    Returns (S, cx, cy, D) where D is max projection distance from center.
    """
    n = len(pts)
    # For a hexagon with apothem A and rotation beta, the 6 face normals are:
    # n_k = (cos(beta + pi/6 + k*pi/3), sin(beta + pi/6 + k*pi/3)) for k=0..5
    # Constraint: |p_i . n_k - c . n_k| <= D for all i, k
    # i.e., c . n_k - p_i . n_k <= D AND p_i . n_k - c . n_k <= D
    # Minimize D subject to these constraints, with D >= 0.
    # Variables: cx, cy, D
    
    normals = np.array([
        (math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
         math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0))
        for k in range(6)
    ])
    
    c_obj = np.array([0.0, 0.0, 1.0])
    A_ub = []
    b_ub = []
    
    for nx, ny in normals:
        for i in range(n):
            px, py = pts[i]
            # c.n - p.n <= D  =>  nx*cx + ny*cy - D <= px*nx + py*ny
            A_ub.append([nx, ny, -1.0])
            b_ub.append(px * nx + py * ny)
            # p.n - c.n <= D  =>  -nx*cx - ny*cy - D <= -(px*nx + py*ny)
            A_ub.append([-nx, -ny, -1.0])
            b_ub.append(-(px * nx + py * ny))
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    bounds = [(None, None), (None, None), (0, None)]
    
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    if res.status == 0:
        D = float(res.x[2])
        cx = float(res.x[0])
        cy = float(res.x[1])
        S = (D + APOTHEM) / APOTHEM
        return S, cx, cy, D
    return float('inf'), 0.0, 0.0, float('inf')


def _objective_beta(pts, beta):
    """Objective: S for given beta (center optimized via LP)."""
    S, _, _, _ = _lp_enclosure(pts, beta)
    return S


def _optimize_rotation(pts):
    """Optimize the rotation angle beta in [0, pi/6) using multi-start bounded minimization."""
    best_S = float('inf')
    best_beta = 0.0
    best_center = (0.0, 0.0)
    
    # Coarse grid: 60 points
    n_grid = 60
    for i in range(n_grid + 1):
        if not _time_ok():
            break
        beta = i * (math.pi / 6.0) / n_grid
        S, cx, cy, _ = _lp_enclosure(pts, beta)
        if S < best_S:
            best_S = S
            best_beta = beta
            best_center = (cx, cy)
    
    # Refine: bounded minimization from best grid point and several others
    refine_starts = [best_beta, 0.0, math.pi / 12.0, math.pi / 24.0, math.pi / 8.0, 5.0 * math.pi / 24.0]
    for rb in refine_starts:
        if not _time_ok():
            break
        lo = max(0.0, rb - 0.15)
        hi = min(math.pi / 6.0, rb + 0.15)
        if hi - lo < 1e-10:
            continue
        try:
            res = minimize_scalar(_objective_beta, bounds=(lo, hi), args=(pts,),
                                  method='bounded', options={'maxiter': 500, 'xatol': 1e-14})
            if res.fun < best_S:
                best_S = res.fun
                best_beta = float(res.x)
                _, cx, cy, _ = _lp_enclosure(pts, best_beta)
                best_center = (cx, cy)
        except Exception:
            pass
    
    # Ultra-fine polish around best
    if _time_ok():
        lo = max(0.0, best_beta - 0.01)
        hi = min(math.pi / 6.0, best_beta + 0.01)
        if hi - lo > 1e-12:
            try:
                res = minimize_scalar(_objective_beta, bounds=(lo, hi), args=(pts,),
                                      method='bounded', options={'maxiter': 1000, 'xatol': 1e-16})
                if res.fun < best_S:
                    best_S = res.fun
                    best_beta = float(res.x)
                    _, cx, cy, _ = _lp_enclosure(pts, best_beta)
                    best_center = (cx, cy)
            except Exception:
                pass
    
    return best_S, best_beta, best_center


def _local_search_axial(axial_list, n_iter=200):
    """Try to improve an axial configuration by moving individual hexagons to nearby axial positions.
    Returns list of (axial_list, S) for valid configurations found.
    """
    current = list(axial_list)
    current_S, _, _ = _optimize_rotation(_axial_to_cartesian(current))
    best = (list(current), current_S)
    
    # Generate candidate moves: for each hex, try moving to all nearby axial positions
    def neighbors(pos, radius=1):
        q, r = pos
        result = []
        for dq in range(-radius, radius + 1):
            for dr in range(-radius, radius + 1):
                if dq == 0 and dr == 0:
                    continue
                if abs(dq + dr) <= radius:
                    result.append((q + dq, r + dr))
        return result
    
    for it in range(n_iter):
        if not _time_ok():
            break
        improved = False
        # Try moving each hexagon to a nearby position
        for i in range(len(current)):
            if not _time_ok():
                break
            for new_pos in neighbors(current[i], 1):
                if new_pos in current:
                    continue  # collision
                trial = list(current)
                trial[i] = new_pos
                if _check_disjoint(trial):
                    pts = _axial_to_cartesian(trial)
                    S, _, _ = _optimize_rotation(pts)
                    if S < best[1] - 1e-10:
                        current = trial
                        best = (list(current), S)
                        improved = True
                        break
        if not improved:
            break
    
    return best


def construct():
    if not _time_ok():
        # Fallback: trivial configuration
        axial = [(0, 0)]
        pts = _axial_to_cartesian(axial)
        S, cx, cy, _ = _lp_enclosure(pts, 0.0)
        inner = [[float(x), float(y), 0.0] for x, y in pts]
        return inner, [float(cx), float(cy)], float(S), 0.0

    # Generate diverse candidate configurations
    candidates = []
    
    # 1. Hexagonal ring (center + 6 neighbors) + 4 extended
    ring_ext1 = [(0,0), (1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1), (2,0),(-2,0),(0,2),(0,-2)]
    candidates.append(ring_ext1)
    
    # 2. Ring + diagonal extensions
    ring_ext2 = [(0,0), (1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1), (1,1),(-1,2),(-2,1),(2,-1)]
    candidates.append(ring_ext2)
    
    # 3. Compact 3x3-ish
    compact1 = [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,1),(3,1)]
    candidates.append(compact1)
    
    # 4. Row-based
    row1 = [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2)]
    candidates.append(row1)
    
    # 5. Diamond
    diamond = [(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(2,0),(-2,0),(0,2),(0,-2)]
    candidates.append(diamond)
    
    # 6. Two rows of 5 + 1
    two_rows = [(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(1,1),(2,1),(3,1),(4,1),(5,1)]
    candidates.append(two_rows)
    
    # 7. T-shape
    t_shape = [(0,0),(1,0),(-1,0),(2,0),(-2,0),(0,1),(1,1),(-1,1),(0,2),(1,2),(0,-1)]
    candidates.append(t_shape)
    
    # 8. L-shape
    l_shape = [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3)]
    candidates.append(l_shape)
    
    # 9. Cross extension
    cross_ext = [(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(2,1),(1,-2),(2,-1),(-1,-1)]
    candidates.append(cross_ext)
    
    # 10. Stacked
    stacked = [(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)]
    candidates.append(stacked)
    
    # 11. Wide
    wide = [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(0,3)]
    candidates.append(wide)
    
    # 12. Zigzag
    zigzag = [(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(3,2),(1,3),(2,3)]
    candidates.append(zigzag)
    
    # 13. Another ring variant
    ring3 = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,1),(1,2),(-1,2),(0,3)]
    candidates.append(ring3)
    
    # 14. Balanced
    balanced = [(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(1,-1),(-1,-1),(2,1),(2,-1)]
    candidates.append(balanced)
    
    # 15. Compact cluster
    compact2 = [(0,0),(1,0),(2,0),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(0,1),(3,1)]
    candidates.append(compact2)
    
    # Verify all candidates are valid (disjoint)
    valid_candidates = []
    for c in candidates:
        if len(c) == N and len(set(c)) == N and _check_disjoint(c):
            valid_candidates.append(c)
    
    if not valid_candidates:
        valid_candidates = [ring_ext1]
    
    # Score all candidates
    best_S = float('inf')
    best_beta = 0.0
    best_center = (0.0, 0.0)
    best_axial = valid_candidates[0]
    
    for axial in valid_candidates:
        if not _time_ok():
            break
        pts = _axial_to_cartesian(axial)
        S, beta, center = _optimize_rotation(pts)
        if S < best_S:
            best_S = S
            best_beta = beta
            best_center = center
            best_axial = axial
    
    # Local search from the best configuration
    if _time_ok():
        improved_axial, improved_S = _local_search_axial(best_axial, n_iter=50)
        if improved_S < best_S:
            best_S = improved_S
            best_axial = improved_axial
            pts = _axial_to_cartesian(best_axial)
            best_S, best_beta, best_center = _optimize_rotation(pts)
    
    # Additional local search from second-best
    if _time_ok() and len(valid_candidates) > 1:
        # Find second best
        scored = []
        for axial in valid_candidates:
            if axial == best_axial:
                continue
            pts = _axial_to_cartesian(axial)
            S, beta, center = _optimize_rotation(pts)
            scored.append((S, axial))
        if scored:
            scored.sort()
            if scored[0][0] < best_S + 0.1:  # Close enough to be worth trying
                improved_axial, improved_S = _local_search_axial(scored[0][1], n_iter=30)
                if improved_S < best_S:
                    best_S = improved_S
                    best_axial = improved_axial
                    pts = _axial_to_cartesian(best_axial)
                    best_S, best_beta, best_center = _optimize_rotation(pts)
    
    # Final: recompute with best configuration
    pts = _axial_to_cartesian(best_axial)
    best_S, best_beta, best_center = _optimize_rotation(pts)
    
    cx, cy = best_center
    angle_deg = math.degrees(best_beta)
    
    inner = []
    for x, y in pts:
        inner.append([float(x), float(y), float(angle_deg)])
    
    center = [float(cx), float(cy)]
    side = float(best_S)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation


def _get_trivial_solution():
    """Fallback trivial solution."""
    axial = [(0, 0)]
    pts = _axial_to_cartesian(axial)
    S, cx, cy, _ = _lp_enclosure(pts, 0.0)
    inner = [[float(x), float(y), 0.0] for x, y in pts]
    return inner, [float(cx), float(cy)], float(S), 0.0
