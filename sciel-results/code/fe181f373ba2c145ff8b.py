import numpy as np
from itertools import combinations

N = 11

def construct():
    # Lattice: (i,j) -> cartesian (sqrt(3)*i + (sqrt(3)/2)*j, (3/2)*j)
    # a0 = i + j/2, a1 = i + 2j, a2 = -i + j
    # S_fixed = 1 + 0.5 * max(range(a0), range(a1), range(a2))
    
    sqrt3 = np.sqrt(3.0)
    
    # Generate pool
    pool = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            a0 = i + j/2.0
            a1 = i + 2*j
            a2 = -i + j
            if max(abs(a0), abs(a1), abs(a2)) <= 4.0:
                x = sqrt3*i + (sqrt3/2)*j
                y = (3.0/2)*j
                pool.append((i, j, a0, a1, a2, x, y))
    
    # Find best 11-point subset minimizing max_range (fixed orientation)
    # Use backtracking with pruning
    best_max_range = float('inf')
    best_subset = None
    pool_arr = np.array([(p[2], p[3], p[4]) for p in pool])  # a0,a1,a2
    
    n_pool = len(pool)
    
    # Backtracking
    selected = []
    def backtrack(idx, current_min, current_max):
        nonlocal best_max_range, best_subset
        if len(selected) == N:
            mr = max(current_max[0]-current_min[0], current_max[1]-current_min[1], current_max[2]-current_min[2])
            if mr < best_max_range:
                best_max_range = mr
                best_subset = list(selected)
            return
        # Prune: even if we add the best possible points, can we beat best?
        # Lower bound: current max_range
        cur_mr = max(current_max[0]-current_min[0], current_max[1]-current_min[1], current_max[2]-current_min[2])
        if cur_mr >= best_max_range:
            return
        remaining = n_pool - idx
        need = N - len(selected)
        if remaining < need:
            return
        for k in range(idx, n_pool):
            a = pool_arr[k]
            new_min = [min(current_min[d], a[d]) for d in range(3)]
            new_max = [max(current_max[d], a[d]) for d in range(3)]
            new_mr = max(new_max[0]-new_min[0], new_max[1]-new_min[1], new_max[2]-new_min[2])
            if new_mr < best_max_range:
                selected.append(k)
                backtrack(k+1, new_min, new_max)
                selected.pop()
    
    # Initialize with a heuristic to get a good best_max_range early
    # Greedy: start with center, add points that minimize max_range
    # ... (skip for now, just run backtracking; the first complete solution sets best)
    
    # To get a good initial bound, let's just run backtracking; it will find the first solution quickly
    # But the first solution might be bad. Let me seed with a known good config.
    # The 7-point hexagon + 4 points: let me just let backtracking find it.
    
    # Actually, let me do a quick greedy first
    # ...
    
    backtrack(0, [float('inf')]*3, [float('-inf')]*3)
    
    # Now best_subset has indices into pool
    # Compute optimal rotation for this subset
    pts = np.array([(pool[k][5], pool[k][6]) for k in best_subset])
    
    # Minimize S over rotation theta in [0, 60 deg)
    def S_for_theta(theta):
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        pts_r = pts @ R.T
        # 6 directions
        ks = np.arange(6)
        angles = np.pi/3 * ks  # 0, 60, 120, ...
        dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        projections = pts_r @ dirs.T  # (11, 6)
        widths = projections.max(axis=0) - projections.min(axis=0)
        return 1.0 + (1.0/sqrt3) * widths.max()
    
    # Sample theta
    thetas = np.linspace(0, np.pi/3, 180, endpoint=False)
    S_vals = [S_for_theta(t) for t in thetas]
    best_i = int(np.argmin(S_vals))
    best_theta = thetas[best_i]
    best_S = S_vals[best_i]
    
    # Refine around best_theta
    fine_thetas = np.linspace(best_theta - np.pi/3/180, best_theta + np.pi/3/180, 181)
    S_fine = [S_for_theta(t) for t in fine_thetas]
    best_i2 = int(np.argmin(S_fine))
    best_theta = fine_thetas[best_i2]
    best_S = S_fine[best_i2]
    
    # Outer center: centroid of pts (rotated)
    c, s = np.cos(best_theta), np.sin(best_theta)
    R = np.array([[c, -s], [s, c]])
    pts_r = pts @ R.T
    center = pts_r.mean(axis=0)
    
    # Inner hexagons: centers at pts_r, angle = best_theta (degrees)
    inner = [[float(pts_r[k][0]), float(pts_r[k][1]), float(np.degrees(best_theta))] for k in range(N)]
    outer_center = (float(center[0]), float(center[1]))
    outer_side = float(best_S)
    outer_angle = float(np.degrees(best_theta))
    
    return inner, outer_center, outer_side, outer_angle
