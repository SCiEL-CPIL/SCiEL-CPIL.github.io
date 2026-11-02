"""Circle packing in perimeter-4 rectangle using Delaunay-based LP."""
import numpy as np
from scipy.optimize import linprog
from scipy.spatial import Delaunay
import itertools

N = 21


def _solve_radii_lp(centers, W, H):
    """
    Solve LP to maximize sum(r_i) subject to:
    - r_i >= 0
    - r_i <= min(x_i, W - x_i, y_i, H - y_i) for each i
    - r_i + r_j <= dist(i,j) for all pairs i,j
    """
    n = len(centers)
    centers = np.asarray(centers)
    
    # Compute all pairwise distances
    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dists[i, j] = np.linalg.norm(centers[i] - centers[j])
    
    # Upper bound for each radius from walls
    wall_ub = np.minimum(np.minimum(centers[:, 0], W - centers[:, 0]),
                         np.minimum(centers[:, 1], H - centers[:, 1]))
    # Clip to non-negative
    wall_ub = np.maximum(wall_ub, 0.0)
    
    # Objective: maximize sum(r_i) -> minimize -sum(r_i)
    c = -np.ones(n)
    
    # Inequality constraints: A_ub @ r <= b_ub
    # Constraints:
    # 1. r_i <= wall_ub[i] for each i -> r_i <= wall_ub[i]
    # 2. r_i + r_j <= dists[i,j] for all i < j
    
    A_ub_rows = []
    b_ub_rows = []
    
    # Wall constraints
    for i in range(n):
        row = np.zeros(n)
        row[i] = 1.0
        A_ub_rows.append(row)
        b_ub_rows.append(wall_ub[i])
    
    # Pairwise non-overlap
    for i in range(n):
        for j in range(i + 1, n):
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub_rows.append(row)
            b_ub_rows.append(dists[i, j])
    
    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_rows)
    
    # Bounds: 0 <= r_i <= wall_ub[i]
    bounds = [(0.0, wall_ub[i]) for i in range(n)]
    
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if res.status == 0:
        return res.x
    else:
        # Fallback: use half the minimum constraint
        r = np.zeros(n)
        for i in range(n):
            r[i] = wall_ub[i] * 0.5
        return r


def _try_configuration(W, H, centers):
    """Try a specific rectangle and center configuration, return sum of radii."""
    centers = np.asarray(centers)
    # Check centers are within [0, W] x [0, H]
    if np.any(centers[:, 0] < -1e-9) or np.any(centers[:, 0] > W + 1e-9) or \
       np.any(centers[:, 1] < -1e-9) or np.any(centers[:, 1] > H + 1e-9):
        return -np.inf
    
    radii = _solve_radii_lp(centers, W, H)
    if radii is None or np.any(radii < -1e-9):
        return -np.inf
    radii = np.maximum(radii, 0.0)
    return np.sum(radii)


def _grid_centers(n_cols, n_rows, W, H, margin=0.0):
    """Generate grid centers within [margin, W-margin] x [margin, H-margin]."""
    if n_cols < 1 or n_rows < 1:
        return None
    x_positions = np.linspace(margin, W - margin, n_cols) if n_cols > 1 else [W / 2]
    y_positions = np.linspace(margin, H - margin, n_rows) if n_rows > 1 else [H / 2]
    centers = []
    for y in y_positions:
        for x in x_positions:
            centers.append([x, y])
    return centers


def construct():
    best_value = 0.0
    best_result = None
    
    # Try various aspect ratios and grid configurations
    # Perimeter = 4, so W + H = 2
    aspect_ratios = [
        (1.0, 1.0),       # square
        (0.8, 1.2),
        (1.2, 0.8),
        (0.7, 1.3),
        (1.3, 0.7),
        (0.6, 1.4),
        (1.4, 0.6),
        (0.5, 1.5),
        (1.5, 0.5),
        (0.9, 1.1),
        (1.1, 0.9),
    ]
    
    # Grid configurations: (n_cols, n_rows) that give at least 21 cells
    grid_configs = [
        (5, 5),    # 25 cells
        (4, 6),    # 24 cells
        (6, 4),    # 24 cells
        (3, 7),    # 21 cells
        (7, 3),    # 21 cells
        (4, 5),    # 20 cells - need to add one
        (5, 4),    # 20 cells
        (3, 8),    # 24 cells
        (8, 3),    # 24 cells
        (2, 11),   # 22 cells
        (11, 2),   # 22 cells
        (1, 21),   # 21 cells
        (21, 1),   # 21 cells
    ]
    
    for W, H in aspect_ratios:
        for n_cols, n_rows in grid_configs:
            total_cells = n_cols * n_rows
            if total_cells < N:
                continue
            
            # Try with small margin to allow circles near edges
            for margin in [0.0, 0.01, 0.02]:
                centers = _grid_centers(n_cols, n_rows, W, H, margin)
                if centers is None:
                    continue
                centers = np.array(centers)
                
                # If we have more than N centers, we need to select N of them
                # For simplicity, take the first N (or center them better)
                if len(centers) > N:
                    # Select N centers - try to keep them spread out
                    # Simple approach: take a subset
                    # We'll just use the first N for now
                    centers = centers[:N]
                
                if len(centers) < N:
                    continue
                
                value = _try_configuration(W, H, centers)
                if value > best_value:
                    best_value = value
                    radii = _solve_radii_lp(centers, W, H)
                    radii = np.maximum(radii, 0.0)
                    best_result = np.column_stack([centers, radii])
    
    # If no valid configuration found, use a fallback
    if best_result is None:
        # Fallback: simple grid in 1x1 square
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 1e-3
                cells.append([x, y, r])
        best_result = np.array(cells[:N])
        best_value = np.sum(best_result[:, 2])
    
    # Verify and return
    result = best_result
    # Ensure radii are non-negative
    result[:, 2] = np.maximum(result[:, 2], 0.0)
    
    return result

