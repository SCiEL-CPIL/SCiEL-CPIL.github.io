"""
Circle packing in perimeter-4 rectangle.
Combines:
1. Physics-based initialization and relaxation (from current program logic, improved).
2. Smooth L-BFGS-B optimization with softplus penalties for variable radii (from inspiration).
3. Linear Programming (LP) final polish for exact radii given fixed geometry (from inspiration).
"""
import numpy as np
import time
from scipy.optimize import minimize, linprog
from scipy.spatial import Delaunay

N = 21
DEADLINE = 150.0 # Seconds
SAFETY_MARGIN = 10.0

def _softplus(x, beta=1.0):
    """Stable softplus approximation."""
    return np.where(x > 20, x, np.log1p(np.exp(x / beta)) * beta)

def _delaunay_edges(xs, ys):
    """Return list of (i, j) edges from Delaunay triangulation."""
    points = np.column_stack((xs, ys))
    try:
        tri = Delaunay(points)
    except Exception:
        return None
    edges = set()
    for simplex in tri.simplices:
        edges.add(tuple(sorted((simplex[0], simplex[1]))))
        edges.add(tuple(sorted((simplex[1], simplex[2]))))
        edges.add(tuple(sorted((simplex[0], simplex[2]))))
    return list(edges)

def _solve_lp_radii(xs, ys, w, h):
    """
    Solve LP for per-circle radii given fixed centers and rectangle.
    Maximize sum(r) s.t. r_i + r_j <= d_ij (for Delaunay edges)
                         r_i <= min(x_i, w-x_i, y_i, h-y_i)
                         r_i >= 0
    """
    limits = np.minimum(np.minimum(xs, w - xs), np.minimum(ys, h - ys))
    if np.any(limits < 0):
        return np.zeros(N)

    A_list = []
    b_list = []
    edges = _delaunay_edges(xs, ys)
    
    # We should really use ALL pairs for strict non-overlap in the LP, 
    # but Delaunay edges are usually sufficient for the "contact graph" 
    # if the packing is dense. However, to be safe and exact for validity,
    # we must consider all pairs if we want to guarantee no overlap.
    # But N=21 means 210 pairs. LP with 210 constraints is trivial.
    # Let's use all pairs to be strictly correct.
    for i in range(N):
        for j in range(i + 1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            row = np.zeros(N)
            row[i] = 1
            row[j] = 1
            A_list.append(row)
            b_list.append(dist)

    A_bound = np.eye(N)
    if A_list:
        A_ub = np.vstack([A_list, A_bound])
        b_ub = np.concatenate([b_list, limits])
    else:
        A_ub = A_bound
        b_ub = limits

    c = -np.ones(N)
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=(0, None), method='highs')
        if res.success:
            return np.maximum(0.0, res.x)
    except Exception:
        pass
    return np.maximum(0.0, limits * 0.9)

def _physics_relax(pos, rad, w, h, steps=200, dt=0.02):
    """
    Simple force-directed relaxation to remove overlaps.
    """
    n = len(pos)
    pos = pos.copy().astype(float)
    rad = rad.copy().astype(float)
    
    for _ in range(steps):
        force = np.zeros_like(pos)
        
        # Pairwise repulsion
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[i][0] - pos[j][0]
                dy = pos[i][1] - pos[j][1]
                dist_sq = dx*dx + dy*dy
                min_dist = rad[i] + rad[j]
                if dist_sq < min_dist * min_dist:
                    dist = np.sqrt(dist_sq) + 1e-6
                    overlap = min_dist - dist
                    f = 10.0 * overlap / dist
                    force[i][0] += f * dx
                    force[i][1] += f * dy
                    force[j][0] -= f * dx
                    force[j][1] -= f * dy
        
        # Boundary forces
        for i in range(n):
            if pos[i][0] < rad[i]:
                force[i][0] += 10.0 * (rad[i] - pos[i][0])
            if pos[i][0] > w - rad[i]:
                force[i][0] -= 10.0 * (pos[i][0] - (w - rad[i]))
            if pos[i][1] < rad[i]:
                force[i][1] += 10.0 * (rad[i] - pos[i][1])
            if pos[i][1] > h - rad[i]:
                force[i][1] -= 10.0 * (pos[i][1] - (h - rad[i]))
        
        # Update positions
        pos += force * dt
        pos[:, 0] = np.clip(pos[:, 0], rad, w - rad)
        pos[:, 1] = np.clip(pos[:, 1], rad, h - rad)
        
    return pos, rad

def _smooth_optimize(w, h, xs, ys, rs, deadline):
    """
    L-BFGS-B optimization of positions and radii.
    Variables: [w, x0..x20, y0..y20, r0..r20]
    Objective: -sum(r) + Penalty(Overlaps) + Penalty(Bounds)
    """
    if time.time() > deadline - 10.0:
        return w, h, xs, ys, rs

    v0 = np.zeros(1 + 3 * N)
    v0[0] = w
    v0[1:1 + N] = xs
    v0[1 + N:1 + 2 * N] = ys
    v0[1 + 2 * N:1 + 3 * N] = rs

    idx_i, idx_j = np.triu_indices(N, k=1)

    def objective(v):
        w_val = v[0]
        h_val = 2.0 - w_val
        x = v[1:1 + N]
        y = v[1 + N:1 + 2 * N]
        r = v[1 + 2 * N:1 + 3 * N]

        if w_val < 0.1 or h_val < 0.1 or np.any(r < 0):
            return 1e6

        # Base objective: minimize -sum(r)
        obj = -np.sum(r)
        
        # Penalty parameters
        # We want strong penalties for violations
        lam_overlap = 100.0
        lam_bound = 100.0
        beta = 1e-3 # softness
        
        # Overlap penalties: max(0, r_i + r_j - dist_ij)
        # Using softplus: softplus((r_i + r_j - dist_ij) / beta) * beta
        overlap_pen = 0.0
        for k in range(len(idx_i)):
            i = idx_i[k]
            j = idx_j[k]
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy) + 1e-6
            viol = (r[i] + r[j] - dist)
            overlap_pen += _softplus(viol / beta, beta)
        
        obj += lam_overlap * overlap_pen

        # Boundary penalties: max(0, r_i - x_i), etc.
        bound_pen = 0.0
        # Left
        viol_l = r - x
        bound_pen += np.sum(_softplus(viol_l / beta, beta))
        # Right
        viol_r = r - (w_val - x)
        bound_pen += np.sum(_softplus(viol_r / beta, beta))
        # Bottom
        viol_b = r - y
        bound_pen += np.sum(_softplus(viol_b / beta, beta))
        # Top
        viol_t = r - (h_val - y)
        bound_pen += np.sum(_softplus(viol_t / beta, beta))
        
        obj += lam_bound * bound_pen
        
        return obj

    # Bounds for variables
    # w in [0.5, 1.5] (since w+h=2, w=1 is square)
    # x, y in [0, 2]
    # r in [0, 1]
    bounds = [(0.5, 1.5)] + [(0.0, 2.0)] * (2 * N) + [(0.0, 1.0)] * N

    try:
        res = minimize(objective, v0, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8, 'disp': False})
        
        if res.success:
            w_r = res.x[0]
            h_r = 2.0 - w_r
            xs_r = res.x[1:1 + N]
            ys_r = res.x[1 + N:1 + 2 * N]
            rs_r = res.x[1 + 2 * N:1 + 3 * N]
            
            # Ensure radii are non-negative
            rs_r = np.maximum(0, rs_r)
            
            return w_r, h_r, xs_r, ys_r, rs_r
    except Exception:
        pass
        
    return w, h, xs, ys, rs

def _check_validity(pos, rad, w, h):
    """Check if solution is valid. Returns True if valid."""
    n = len(pos)
    for i in range(n):
        if pos[i][0] < rad[i] - 1e-6 or pos[i][0] > w - rad[i] + 1e-6:
            return False
        if pos[i][1] < rad[i] - 1e-6 or pos[i][1] > h - rad[i] + 1e-6:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i][0] - pos[j][0]
            dy = pos[i][1] - pos[j][1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < rad[i] + rad[j] - 1e-6:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY_MARGIN
    
    best_score = 0
    best_result = None

    # Initial configurations
    configs = []
    
    # 1. Grid 5x5 (21 circles)
    # 5x5 grid has 25 spots. We take 21.
    # Aspect ratio 1.0
    w, h = 1.0, 1.0
    xs = []
    ys = []
    for i in range(5):
        for j in range(5):
            if len(xs) < N:
                xs.append((i + 0.5) * w / 5.0)
                ys.append((j + 0.5) * h / 5.0)
    configs.append((w, h, np.array(xs), np.array(ys)))
    
    # 2. Hexagonal packing approximation
    # Rows: 5, 5, 5, 5, 1? No, 5+5+5+5+1 = 21.
    # Let's try 4 rows: 5, 6, 5, 5? 
    # Standard hex packing for N=21.
    # Let's just use a dense random set and let physics sort it.
    
    # 3. Random
    rng = np.random.RandomState(42)
    for _ in range(5):
        w = 1.0
        h = 1.0
        xs = rng.uniform(0.1, 0.9, N) * w
        ys = rng.uniform(0.1, 0.9, N) * h
        configs.append((w, h, xs, ys))

    # 4. Different Aspect Ratios with Grid
    for aspect in [0.8, 1.25, 0.7, 1.4]:
        h = 2.0 / (aspect + 1.0)
        w = aspect * h
        xs = []
        ys = []
        # Fit grid into w x h
        cols = int(np.ceil(np.sqrt(N * aspect)))
        rows = int(np.ceil(N / cols))
        for i in range(rows):
            for j in range(cols):
                if len(xs) < N:
                    xs.append((j + 0.5) * w / cols)
                    ys.append((i + 0.5) * h / rows)
        if len(xs) == N:
            configs.append((w, h, np.array(xs), np.array(ys)))

    for w, h, xs, ys in configs:
        if time.time() > deadline:
            break
        
        # Initial radii: small, uniform
        # Estimate max radius based on area or min distance
        # For now, start small
        r_est = 0.05
        rs = np.full(N, r_est)
        
        pos = np.column_stack((xs, ys))
        
        # Stage 1: Physics Relaxation
        pos, rs = _physics_relax(pos, rs, w, h, steps=100)
        
        # Stage 2: Smooth Optimization
        w_opt, h_opt, xs_opt, ys_opt, rs_opt = _smooth_optimize(w, h, pos[:, 0], pos[:, 1], rs, deadline)
        
        # Stage 3: LP Polish
        # Use the optimized positions and aspect ratio
        # Solve LP for max radii
        rs_lp = _solve_lp_radii(xs_opt, ys_opt, w_opt, h_opt)
        
        # Check validity of LP solution
        pos_lp = np.column_stack((xs_opt, ys_opt))
        if _check_validity(pos_lp, rs_lp, w_opt, h_opt):
            score = np.sum(rs_lp)
            if score > best_score:
                best_score = score
                best_result = np.column_stack([pos_lp, rs_lp])
        
        # Also check validity of the smooth optimization result (in case LP failed or was worse)
        if _check_validity(pos, rs, w_opt, h_opt): # Note: pos from physics might be outdated if smooth moved them
             pass # We already checked LP which uses smooth positions. 
             # Actually, smooth_optimize returns new positions. 
             # Let's check the smooth result directly too.
             # The smooth result might have small violations due to softplus.
             # If valid, it's a candidate.
             pass

    # Fallback if no valid solution found
    if best_result is None:
        # Return a trivial valid solution (very small circles)
        w, h = 1.0, 1.0
        xs = np.linspace(0.1, 0.9, N)
        ys = np.linspace(0.1, 0.9, N)
        rs = np.full(N, 0.01)
        best_result = np.column_stack([np.column_stack((xs, ys)), rs])
        # Ensure this is valid
        # With r=0.01, spacing 0.8/20 = 0.04 > 0.02. Valid.

    return best_result

# EVOLVE-BLOCK-END
