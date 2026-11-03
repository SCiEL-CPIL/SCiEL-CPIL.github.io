import math
import numpy as np
import time
from scipy.optimize import linprog, minimize, minimize_scalar

START_TIME = time.time()
TIME_BUDGET = 120.0
SAFETY_MARGIN = 10.0
APOTHEM = math.sqrt(3) / 2.0


def construct():
    if time.time() - START_TIME > TIME_BUDGET - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Generate all cells in a side-3 hexagonal block (19 cells)
    # Cell (q, r) is in the block if max(|q|, |r|, |q+r|) <= 2
    all_cells = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if max(abs(q), abs(r), abs(q + r)) <= 2:
                all_cells.append((q, r))

    # We need to choose 11 cells out of 19 that minimize the enclosing hexagon side
    # Try compact subsets: all cells within distance d from center, plus nearby cells
    
    def axial_to_cartesian(q, r):
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        return (x, y)

    def enclosing_hexagon_side(pts, beta):
        """Compute the side length of the smallest enclosing regular hexagon with rotation beta."""
        if len(pts) == 0:
            return float('inf'), (0.0, 0.0)
        
        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        pts_arr = np.array(pts)
        n_pts = len(pts_arr)
        
        A_ub_list = []
        b_ub_list = []
        for k in range(6):
            nx, ny = normals[k]
            projs = pts_arr[:, 0] * nx + pts_arr[:, 1] * ny
            A_ub_list.append(np.column_stack([-np.full(n_pts, nx), -np.full(n_pts, ny), -np.ones(n_pts)]))
            b_ub_list.append(-projs)
            A_ub_list.append(np.column_stack([np.full(n_pts, nx), np.full(n_pts, ny), -np.ones(n_pts)]))
            b_ub_list.append(projs)
        
        A_ub = np.vstack(A_ub_list)
        b_ub = np.concatenate(b_ub_list)
        c_obj = np.array([0.0, 0.0, 1.0])
        bounds = [(None, None), (None, None), (0, None)]
        
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                D = res.x[2]
                cx = res.x[0]
                cy = res.x[1]
                S = (D + APOTHEM) / APOTHEM
                return S, (cx, cy)
            else:
                return float('inf'), (0.0, 0.0)
        except Exception:
            return float('inf'), (0.0, 0.0)

    def optimize_for_subset(cells):
        """Optimize the enclosing hexagon for a given set of axial cells."""
        pts = [axial_to_cartesian(q, r) for q, r in cells]
        pts_arr = np.array(pts)
        
        best_S = float('inf')
        best_beta = 0.0
        best_center = (0.0, 0.0)
        
        # Discrete search over rotation angles
        for i in range(72):
            if time.time() - START_TIME > TIME_BUDGET - SAFETY_MARGIN:
                return best_S, best_beta, best_center
            beta = i * (math.pi / 6.0) / 72
            S, center = enclosing_hexagon_side(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_center = center
        
        # Continuous refinement
        try:
            if time.time() - START_TIME < TIME_BUDGET - SAFETY_MARGIN:
                res = minimize_scalar(
                    lambda b: enclosing_hexagon_side(pts, b)[0],
                    bounds=(0, math.pi / 6.0),
                    method='bounded',
                    options={'maxiter': 200, 'xatol': 1e-10}
                )
                if res.fun < best_S:
                    best_S = res.fun
                    best_beta = res.x
                    _, best_center = enclosing_hexagon_side(pts, best_beta)
        except Exception:
            pass
        
        return best_S, best_beta, best_center

    # Generate candidate subsets of 11 cells
    # Strategy: start with the 7 innermost cells (center + 6 neighbors), then add 4 more
    center_idx = all_cells.index((0, 0))
    
    # Sort cells by distance from center
    def cell_dist(c):
        q, r = c
        return max(abs(q), abs(r), abs(q + r))
    
    sorted_cells = sorted(range(len(all_cells)), key=lambda i: cell_dist(all_cells[i]))
    
    # Generate candidate subsets
    candidate_subsets = []
    
    # Subset 1: 7 inner + 4 next closest
    inner_7 = sorted_cells[:7]
    next_12 = sorted_cells[7:19]
    candidate_subsets.append(set(inner_7 + next_12[:4]))
    
    # Subset 2: different selection of outer cells
    for start in range(0, 12, 3):
        outer_4 = next_12[start:start+4]
        if len(outer_4) == 4:
            candidate_subsets.append(set(inner_7 + outer_4))
    
    # Subset 3: all cells with dist <= 1 (7 cells) + specific outer cells
    # Try all combinations of 4 from the 12 outer cells (too many, so use heuristics)
    
    # Also try: 11 cells that form a "hexagonal" shape
    # A side-2 hexagon has 7 cells. Add 4 more to make 11.
    
    # Try: center + 6 neighbors + 4 cells at distance 2
    dist2_cells = [i for i in range(len(all_cells)) if cell_dist(all_cells[i]) == 2]
    for i in range(len(dist2_cells)):
        for j in range(i+1, len(dist2_cells)):
            for k in range(j+1, len(dist2_cells)):
                for l in range(k+1, len(dist2_cells)):
                    if time.time() - START_TIME > TIME_BUDGET - SAFETY_MARGIN:
                        break
                    candidate_subsets.append(set(inner_7 + [dist2_cells[i], dist2_cells[j], dist2_cells[k], dist2_cells[l]]))
                if time.time() - START_TIME > TIME_BUDGET - SAFETY_MARGIN:
                    break
            if time.time() - START_TIME > TIME_BUDGET - SAFETY_MARGIN:
                break
        if time.time() - START_TIME > TIME_BUDGET - SAFETY_MARGIN:
            break

    # Evaluate all candidate subsets
    best_global_S = float('inf')
    best_global_cells = None
    best_global_beta = 0.0
    best_global_center = (0.0, 0.0)

    for subset in candidate_subsets:
        if time.time() - START_TIME > TIME_BUDGET - SAFETY_MARGIN:
            break
        cells = [all_cells[i] for i in subset]
        if len(cells) != 11:
            continue
        S, beta, center = optimize_for_subset(cells)
        if S < best_global_S:
            best_global_S = S
            best_global_cells = cells
            best_global_beta = beta
            best_global_center = center

    # Fallback if no valid subset found
    if best_global_cells is None:
        best_global_cells = [all_cells[i] for i in sorted_cells[:11]]
        best_global_S, best_global_beta, best_global_center = optimize_for_subset(best_global_cells)

    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)

    # Build the inner hexagon positions
    inner = []
    for q, r in best_global_cells:
        x, y = axial_to_cartesian(q, r)
        inner.append([float(x), float(y), float(angle_deg)])

    center = [float(cx), float(cy)]
    side = float(best_global_S)
    rotation = float(angle_deg)

    return inner, center, side, rotation
