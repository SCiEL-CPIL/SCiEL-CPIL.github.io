"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize, LinearConstraint
from scipy.spatial import cKDTree

N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _pairwise_distances(pos):
    """pos: (n,2) -> (n,n) distance matrix"""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    return np.sqrt(np.sum(diff**2, axis=-1))

def _feasible_radius(pos, W, H, tol=1e-9):
    """Given positions in [0,W]x[0,H], compute max radius such that all circles fit."""
    n = len(pos)
    d = _pairwise_distances(pos)
    # constraint: r_i + r_j <= d_ij, and r_i <= min(pos_i[0], W-pos_i[0], pos_i[1], H-pos_i[1])
    # We want max sum r_i subject to these constraints.
    # This is a linear program! Let's use scipy linprog.
    from scipy.optimize import linprog
    
    # Variables: r_0, ..., r_{n-1}
    # Maximize sum(r) => minimize -sum(r)
    c = -np.ones(n)
    
    # Constraints:
    # 1) r_i + r_j <= d_ij for all i<j
    # 2) r_i <= min(pos_i[0], W-pos_i[0], pos_i[1], H-pos_i[1]) for all i
    # 3) r_i >= 0
    
    A_ub = []
    b_ub = []
    
    # Pairwise constraints
    for i in range(n):
        for j in range(i+1, n):
            row = np.zeros(n)
            row[i] = 1
            row[j] = 1
            A_ub.append(row)
            b_ub.append(d[i, j] - tol)
    
    # Boundary constraints
    for i in range(n):
        max_r = min(pos[i, 0], W - pos[i, 0], pos[i, 1], H - pos[i, 1])
        row = np.zeros(n)
        row[i] = 1
        A_ub.append(row)
        b_ub.append(max_r - tol)
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    bounds = [(0, None)] * n
    
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if res.success:
        return -res.fun, res.x
    else:
        # Fallback: use min of half distances and boundary
        r = np.zeros(n)
        for i in range(n):
            min_dist = min(pos[i, 0], W - pos[i, 0], pos[i, 1], H - pos[i, 1])
            for j in range(n):
                if i != j:
                    min_dist = min(min_dist, d[i, j] / 2)
            r[i] = min_dist
        return np.sum(r), r

def _optimize_positions(W, H, pos0, deadline, verbose=False):
    """Optimize positions to maximize sum of radii. Uses SLSQP with smooth objective."""
    n = len(pos0)
    best_sum = 0
    best_pos = pos0.copy()
    
    def objective(pos_flat):
        pos = pos_flat.reshape(-1, 2)
        # Clip to stay in bounds
        pos = np.clip(pos, 1e-6, [W - 1e-6, H - 1e-6])
        
        d = _pairwise_distances(pos)
        # For each circle, max radius is min of boundary dist and half of min pairwise dist
        # But we want to maximize sum, so use a smooth approximation
        # Actually, let's just compute the LP result - but that's slow.
        # Use a simpler objective: sum of min(boundary, min_j d_ij/2)
        radii = np.zeros(n)
        for i in range(n):
            boundary = min(pos[i, 0], W - pos[i, 0], pos[i, 1], H - pos[i, 1])
            if i < n - 1:
                min_pair = np.min(d[i, i+1:])
            else:
                min_pair = np.inf
            radii[i] = min(boundary, min_pair / 2)
        return -np.sum(radii)
    
    # Constraints for bounds
    bounds = []
    for i in range(n):
        bounds.append((1e-6, W - 1e-6))
        bounds.append((1e-6, H - 1e-6))
    
    # Try a few optimization attempts
    start = time.time()
    for attempt in range(3):
        if time.time() > deadline - 5:
            break
        
        pos_flat = best_pos.flatten()
        try:
            res = minimize(objective, pos_flat, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6})
            if res.success and -res.fun > best_sum:
                best_sum = -res.fun
                best_pos = res.x.reshape(-1, 2)
                if verbose:
                    print(f"  Attempt {attempt}: sum_r = {best_sum:.6f}")
        except Exception:
            pass
    
    return best_pos, best_sum

def _solve_for_aspect(W, H, deadline, verbose=False):
    """Solve for a given W x H rectangle (perimeter 2(W+H)=4 => W+H=2)."""
    n = N
    # Initial placement: grid
    side = int(np.ceil(np.sqrt(n)))
    pos = np.array([[(i + 0.5) * W / side, (j + 0.5) * H / side] 
                    for i in range(side) for j in range(side)])[:n]
    
    # Optimize positions
    pos_opt, sum_r = _optimize_positions(W, H, pos, deadline, verbose)
    
    # Now compute the actual optimal radii via LP
    sum_r_lp, radii = _feasible_radius(pos_opt, W, H)
    
    # Polish: try to improve with more local search
    if time.time() < deadline - 10:
        # Small perturbations and re-optimize
        for _ in range(5):
            if time.time() > deadline - 10:
                break
            pos_perturbed = pos_opt + np.random.normal(0, 0.01, pos_opt.shape)
            pos_perturbed = np.clip(pos_perturbed, 1e-6, [W - 1e-6, H - 1e-6])
            pos_new, sum_new = _optimize_positions(W, H, pos_perturbed, deadline - 10, verbose=False)
            sum_new_lp, _ = _feasible_radius(pos_new, W, H)
            if sum_new_lp > sum_r_lp:
                sum_r_lp = sum_new_lp
                pos_opt = pos_new
    
    return sum_r_lp, pos_opt, radii

def construct():
    deadline = time.time() + 110  # Safety margin from 120s
    
    best_sum = 0
    best_result = None
    best_W = 1.0
    best_H = 1.0
    
    # Search over aspect ratios: W + H = 2, so H = 2 - W
    # W in [0.5, 1.5]
    # Start with square, then try other aspects
    
    aspects = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.55, 1.8]
    
    for W in aspects:
        if time.time() > deadline - 15:
            break
        H = 2.0 - W
        if H <= 0 or W <= 0:
            continue
        
        sum_r, pos, radii = _solve_for_aspect(W, H, deadline, verbose=False)
        
        if sum_r > best_sum:
            best_sum = sum_r
            best_W = W
            best_H = H
            # Construct result: (x, y, r)
            result = np.column_stack([pos, radii])
            best_result = result
    
    # Fallback if no valid result
    if best_result is None:
        W, H = 1.0, 1.0
        side = int(np.ceil(np.sqrt(N)))
        pos = np.array([[(i + 0.5) * W / side, (j + 0.5) * H / side] 
                        for i in range(side) for j in range(side)])[:N]
        _, radii = _feasible_radius(pos, W, H)
        best_result = np.column_stack([pos, radii])
        best_sum = np.sum(radii)
    
    # Verify perimeter
    # Bounding box of circles: min/max of (x-r, x+r, y-r, y+r)
    # But the task says "bounding rectangle has perimeter at most 4"
    # The rectangle is the one we designed (W x H), so perimeter = 2(W+H) = 4.
    # We need to ensure circles are within [0,W]x[0,H].
    
    # Clip radii to ensure circles fit
    for i in range(len(best_result)):
        x, y, r = best_result[i]
        max_r = min(x, best_W - x, y, best_H - y)
        if r > max_r:
            best_result[i, 2] = max_r
    
    # Verify disjointness and fix if needed
    for i in range(len(best_result)):
        for j in range(i+1, len(best_result)):
            d = np.sqrt((best_result[i,0]-best_result[j,0])**2 + (best_result[i,1]-best_result[j,1])**2)
            if best_result[i,2] + best_result[j,2] > d:
                # Reduce radii
                avg_r = d / 2 * 0.999
                best_result[i, 2] = min(best_result[i, 2], avg_r)
                best_result[j, 2] = min(best_result[j, 2], avg_r)
    
    return best_result
# EVOLVE-BLOCK-END
