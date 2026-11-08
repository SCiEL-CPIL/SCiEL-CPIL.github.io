import time
import numpy as np
from itertools import combinations
from scipy.optimize import minimize

N = 11
_H = np.sqrt(3) / 2.0

# Precompute triangle indices for vectorization
_combos = list(combinations(range(N), 3))
_idx_i = np.array([c[0] for c in _combos])
_idx_j = np.array([c[1] for c in _combos])
_idx_k = np.array([c[2] for c in _combos])

# Precompute pair indices for distance objective
_pair_combos = list(combinations(range(N), 2))
_pidx_i = np.array([c[0] for c in _pair_combos])
_pidx_j = np.array([c[1] for c in _pair_combos])


def _project_to_triangle(pts):
    """
    Project points to the interior of the equilateral triangle 
    with vertices (0,0), (1,0), (0.5, sqrt(3)/2).
    Uses orthogonal projection onto the edges if outside.
    """
    pts = np.array(pts, dtype=float)
    # Vertices
    v0 = np.array([0.0, 0.0])
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, _H])
    
    # Edges
    # e0: v0-v1 (y=0)
    # e1: v1-v2 (x/2 + y/sqrt(3) = 1/2? No. Line from (1,0) to (0.5, H). Slope -sqrt(3). y = -sqrt(3)(x-1). sqrt(3)x + y = sqrt(3))
    # e2: v2-v0 (y = sqrt(3)x)
    
    for _ in range(5): # Iterate a few times for stability
        # Check y >= 0
        mask_y0 = pts[:, 1] < 0
        if np.any(mask_y0):
            pts[mask_y0, 1] = 0.0
            
        # Check y <= sqrt(3) * x  =>  sqrt(3)*x - y >= 0
        # Line: sqrt(3)x - y = 0. Normal: (sqrt(3), -1). Norm^2 = 4.
        # dist = (sqrt(3)x - y) / 2
        # proj = p - dist * normal
        val2 = _H * 2.0 * pts[:, 0] - pts[:, 1]
        mask_e2 = val2 < 0
        if np.any(mask_e2):
            dist = -val2[mask_e2] / 4.0
            pts[mask_e2, 0] -= dist * (_H * 2.0)
            pts[mask_e2, 1] += dist * 1.0
            
        # Check sqrt(3)*x + y <= sqrt(3)  =>  sqrt(3)*x + y - sqrt(3) <= 0
        # Line: sqrt(3)x + y - sqrt(3) = 0. Normal: (sqrt(3), 1). Norm^2 = 4.
        # dist = (sqrt(3)x + y - sqrt(3)) / 2
        val3 = _H * 2.0 * pts[:, 0] + pts[:, 1] - _H * 2.0
        mask_e3 = val3 > 0
        if np.any(mask_e3):
            dist = val3[mask_e3] / 4.0
            pts[mask_e3, 0] -= dist * (_H * 2.0)
            pts[mask_e3, 1] -= dist * 1.0

        # Corner cases might need re-checking, but the iterative loop handles it.
        
    return pts


def _min_triangle_area(points):
    """Compute minimum area of all C(11,3) triangles."""
    pi = points[_idx_i]
    pj = points[_idx_j]
    pk = points[_idx_k]
    v1 = pj - pi
    v2 = pk - pi
    # Area = 0.5 * |cross product|
    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    return np.min(areas)


def _min_dist_sq(points):
    """Compute minimum squared distance between any pair."""
    pi = points[_pidx_i]
    pj = points[_pidx_j]
    diff = pi - pj
    dists = np.sum(diff * diff, axis=1)
    return np.min(dists)


def _generate_grid_start():
    """Generate a triangular grid of 10 points (side 4) + centroid."""
    # Grid of side 4 has points (i, j) with i>=0, j>=0, i+j<=3
    # Mapping to triangle coordinates:
    # p = (i/4) * v1 + (j/4) * v2 + (1 - i/4 - j/4) * v0
    # v0=(0,0), v1=(1,0), v2=(0.5, H)
    pts = []
    for i in range(4):
        for j in range(4 - i):
            x = i * 0.25 * 1.0 + j * 0.25 * 0.5
            y = j * 0.25 * _H
            pts.append([x, y])
    # Add centroid
    pts.append([0.5, _H / 3.0])
    return np.array(pts)


def _generate_hex_start():
    """Generate a hexagonal packing start."""
    # This is a rough heuristic.
    # Center
    pts = [[0.5, _H/3]]
    # Ring 1
    r1 = 0.2
    for k in range(6):
        angle = np.pi/3 * k
        x = 0.5 + r1 * np.cos(angle)
        y = _H/3 + r1 * np.sin(angle)
        pts.append([x, y])
    # Ring 2 (partial)
    r2 = 0.4
    for k in range(6):
        angle = np.pi/3 * k + np.pi/6
        x = 0.5 + r2 * np.cos(angle)
        y = _H/3 + r2 * np.sin(angle)
        # Only keep if inside triangle (rough check)
        if y > 0 and y < _H * 2 * x and _H * 2 * x + y < _H * 2:
             pts.append([x, y])
    
    # We need exactly 11 points. If we have more, take first 11. If less, pad with random.
    pts = np.array(pts)
    if len(pts) < N:
        rng = np.random.default_rng(42)
        rand_pts = rng.random((N - len(pts), 2)) * [1.0, _H]
        pts = np.vstack([pts, rand_pts])
    elif len(pts) > N:
        pts = pts[:N]
        
    return _project_to_triangle(pts)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_points = None
    best_score = -np.inf
    
    # Objective function for the main optimization (maximize min area)
    def objective_main(flat):
        pts = _project_to_triangle(flat.reshape(N, 2))
        return -_min_triangle_area(pts)
    
    # Objective function for initialization (maximize min distance)
    def objective_dist(flat):
        pts = _project_to_triangle(flat.reshape(N, 2))
        return -_min_dist_sq(pts)

    rng = np.random.default_rng(42)
    
    init_candidates = []
    
    # 1. Structured Starts
    init_candidates.append(_generate_grid_start().flatten())
    init_candidates.append(_generate_hex_start().flatten())
    
    # 2. Symmetric Start
    # 3 vertices, 3 midpoints, 3 on medians, 1 centroid, 1 more?
    # Let's just add a few random symmetric perturbations to the grid
    grid_pts = _generate_grid_start()
    for i in range(2):
        perturbed = grid_pts + rng.normal(0, 0.05, grid_pts.shape)
        init_candidates.append(_project_to_triangle(perturbed).flatten())

    # 3. Random Starts
    for i in range(5):
        if time.time() > deadline - 20: break
        rand_pts = rng.random((N, 2)) * [1.0, _H]
        rand_pts = _project_to_triangle(rand_pts)
        init_candidates.append(rand_pts.flatten())

    # Run main optimization on each candidate
    for init_flat in init_candidates:
        if time.time() > deadline - 5: break
        
        try:
            # First, refine the start with distance optimization to spread points
            # This helps avoid degenerate configurations where points are too close
            res_dist = minimize(objective_dist, init_flat, method='Nelder-Mead', 
                       options={'maxiter': 100, 'xatol': 1e-5, 'fatol': 1e-5})
            
            # Then, optimize for area
            res_area = minimize(objective_main, res_dist.x, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
            
            final_pts = _project_to_triangle(res_area.x.reshape(N, 2))
            score = _min_triangle_area(final_pts)
            
            if score > best_score:
                best_score = score
                best_points = final_pts
                
        except Exception:
            continue

    if best_points is None:
        # Fallback: Random points
        pts = rng.random((N, 2)) * [1.0, _H]
        best_points = _project_to_triangle(pts)
        
    return best_points

# EVOLVE-BLOCK-END
