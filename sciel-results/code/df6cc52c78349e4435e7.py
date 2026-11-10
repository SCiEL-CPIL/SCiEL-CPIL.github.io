# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _is_valid(centers, sides, tol=1e-7):
    """Check all constraints: inside unit square, positive sides, non-overlapping."""
    n = len(sides)
    if n != 10:
        return False
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
    # Inside [-tol, 1+tol]^2
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if cx - s/2 < -tol or cx + s/2 > 1.0 + tol:
            return False
        if cy - s/2 < -tol or cy + s/2 > 1.0 + tol:
            return False
    # Non-overlap: for i < j, must not have |cx_i-cx_j| < t AND |cy_i-cyj| < t
    # where t = (s_i + s_j)/2 - tol
    for i in range(n):
        for j in range(i+1, n):
            t = (sides[i] + sides[j]) / 2.0 - tol
            dx = abs(centers[i,0] - centers[j,0])
            dy = abs(centers[i,1] - centers[j,1])
            if dx < t and dy < t:
                return False
    return True


def _objective(centers, sides):
    """Sum of side lengths."""
    return float(np.sum(sides))


def _clamp_to_unit(centers, sides):
    """Clamp centers so squares stay inside [0,1]^2."""
    for i in range(len(sides)):
        half = sides[i] / 2.0
        centers[i, 0] = max(half, min(1.0 - half, centers[i, 0]))
        centers[i, 1] = max(half, min(1.0 - half, centers[i, 1]))
    return centers


def _resolve_overlap(centers, sides, tol=1e-7):
    """
    Attempt to resolve overlaps by pushing squares apart.
    Returns modified centers, sides.
    """
    n = len(sides)
    for iteration in range(200):
        moved = False
        for i in range(n):
            for j in range(i+1, n):
                t = (sides[i] + sides[j]) / 2.0 - tol
                dx = centers[j, 0] - centers[i, 0]
                dy = centers[j, 1] - centers[i, 1]
                if abs(dx) < t and abs(dy) < t:
                    # Overlapping: push apart along the axis of smaller gap
                    if abs(dx) < abs(dy):
                        # Push horizontally
                        push = (t - abs(dx)) / 2.0 + tol
                        if dx >= 0:
                            centers[j, 0] += push
                            centers[i, 0] -= push
                        else:
                            centers[j, 0] -= push
                            centers[i, 0] += push
                    else:
                        # Push vertically
                        push = (t - abs(dy)) / 2.0 + tol
                        if dy >= 0:
                            centers[j, 1] += push
                            centers[i, 1] -= push
                        else:
                            centers[j, 1] -= push
                            centers[i, 1] += push
                    moved = True
        if not moved:
            break
    return centers


def _grow_square_max(centers, sides, i, tol=1e-7):
    """Find the maximum side length for square i given all other positions/sizes (analytic)."""
    cx, cy = centers[i]
    # Bound from unit square: s <= 2*min(cx, 1-cx, cy, 1-cy)
    max_s = min(2.0 * cx, 2.0 * (1.0 - cx), 2.0 * cy, 2.0 * (1.0 - cy))
    if max_s < 0:
        max_s = 0.0
    # For each other square j, constraint: NOT(|cx-cxj| < (s+sj)/2 - tol AND |cy-cyj| < (s+sj)/2 - tol)
    # i.e., s <= 2*(dx+tol) - sj OR s <= 2*(dy+tol) - sj
    # The binding constraint for square i is: s <= min over j of max(2*(dx_j+tol)-sj, 2*(dy_j+tol)-sj)
    n = len(sides)
    for j in range(n):
        if j == i:
            continue
        dx = abs(cx - centers[j, 0])
        dy = abs(cy - centers[j, 1])
        # To avoid overlap with j: s <= 2*(dx + tol) - sides[j] OR s <= 2*(dy + tol) - sides[j]
        # The tighter of these two is the constraint
        c1 = 2.0 * (dx + tol) - sides[j]
        c2 = 2.0 * (dy + tol) - sides[j]
        c = max(c1, c2)
        if c < max_s:
            max_s = c
    if max_s < sides[i]:
        return sides[i]
    return max_s


def _grow_all_to_convergence(centers, sides, tol=1e-7, max_iters=200):
    """Iteratively grow all squares until convergence (simultaneous growth)."""
    n = len(sides)
    for _ in range(max_iters):
        improved = False
        for i in range(n):
            new_s = _grow_square_max(centers, sides, i, tol)
            if new_s > sides[i] + 1e-10:
                sides[i] = new_s
                improved = True
        if not improved:
            break
    return sides


def _smooth_penalty(centers, sides, tol=1e-7, beta=50.0):
    """
    Compute a smooth penalty for constraint violations.
    Returns (penalty_value, gradient).
    Penalty includes:
    - Overlap violations (soft max of violation amounts)
    - Boundary violations
    """
    n = len(sides)
    penalty = 0.0
    grad_c = np.zeros_like(centers)
    grad_s = np.zeros_like(sides)
    
    # Boundary constraints: s/2 <= min(cx, 1-cx, cy, 1-cy)
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        half = s / 2.0
        # Violations: half - cx, half - (1-cx), half - cy, half - (1-cy)
        v1 = half - cx
        v2 = half - (1.0 - cx)
        v3 = half - cy
        v4 = half - (1.0 - cy)
        # Soft max of these
        max_v = max(v1, v2, v3, v4)
        if max_v > 0:
            # Smooth approximation
            exp_vals = [np.exp(beta * v1), np.exp(beta * v2), np.exp(beta * v3), np.exp(beta * v4)]
            sum_exp = sum(exp_vals)
            soft_max = max_v + np.log(sum_exp) / beta
            penalty += soft_max
            # Gradient
            weights = [e / sum_exp for e in exp_vals]
            grad_c[i, 0] += weights[0] * (-0.5) + weights[1] * (0.5)
            grad_c[i, 1] += weights[2] * (-0.5) + weights[3] * (0.5)
            grad_s[i] += (weights[0] + weights[1] + weights[2] + weights[3]) * 0.5
    
    # Overlap constraints: for i < j, NOT(|dx| < t AND |dy| < t) where t = (si+sj)/2 - tol
    for i in range(n):
        for j in range(i+1, n):
            t = (sides[i] + sides[j]) / 2.0 - tol
            dx = centers[j, 0] - centers[i, 0]
            dy = centers[j, 1] - centers[i, 1]
            adx = abs(dx)
            ady = abs(dy)
            
            # Violation if adx < t AND ady < t
            # Define violation amount: max(0, t - adx) * max(0, t - ady) / t^2 (normalized)
            # Or use: max(0, min(t - adx, t - ady))
            viol_x = t - adx
            viol_y = t - ady
            
            # Use soft product: if both positive, penalty
            if viol_x > 0 and viol_y > 0:
                # Smooth penalty: use log-sum-exp style
                # Penalty = -log(exp(-beta*viol_x) + exp(-beta*viol_y)) / beta (this is soft-min)
                # Actually we want penalty when both are violated
                # Use: max(0, min(viol_x, viol_y)) with smoothing
                m = min(viol_x, viol_y)
                if m > 0:
                    penalty += m
                    # Gradient: if viol_x < viol_y, gradient w.r.t. x; else w.r.t. y
                    if viol_x <= viol_y:
                        if dx >= 0:
                            grad_c[i, 0] += 0.5
                            grad_c[j, 0] -= 0.5
                        else:
                            grad_c[i, 0] -= 0.5
                            grad_c[j, 0] += 0.5
                        grad_s[i] += 0.5
                        grad_s[j] += 0.5
                    else:
                        if dy >= 0:
                            grad_c[i, 1] += 0.5
                            grad_c[j, 1] -= 0.5
                        else:
                            grad_c[i, 1] -= 0.5
                            grad_c[j, 1] += 0.5
                        grad_s[i] += 0.5
                        grad_s[j] += 0.5
    
    return penalty, grad_c, grad_s


def _smooth_optimize(centers, sides, tol=1e-7, max_iters=1000, beta=50.0):
    """
    Optimize positions and sizes using smooth penalty minimization.
    Objective: minimize penalty - lambda * sum(sides)
    """
    n = len(sides)
    lambda_w = 1.0  # Weight for maximizing sum of sides
    
    def objective_and_grad(x):
        c = x[:2*n].reshape(n, 2)
        s = x[2*n:]
        # Ensure positive sides
        s = np.maximum(s, 1e-6)
        pen, gc, gs = _smooth_penalty(c, s, tol, beta)
        obj = pen - lambda_w * np.sum(s)
        g = np.zeros_like(x)
        g[:2*n] = gc.flatten()
        g[2*n:] = gs - lambda_w
        return obj, g
    
    x0 = np.concatenate([centers.flatten(), sides])
    result = minimize(objective_and_grad, x0, method='L-BFGS-B', 
                      options={'maxiter': max_iters, 'ftol': 1e-12, 'gtol': 1e-10})
    
    new_c = result.x[:2*n].reshape(n, 2)
    new_s = np.maximum(result.x[2*n:], 1e-6)
    return new_c, new_s


def _final_polish(centers, sides, tol=1e-7, deadline=None):
    """
    Final high-precision polish: jointly optimize positions and sizes.
    """
    if deadline is not None and time.time() > deadline - 30:
        return centers, sides
    
    n = len(sides)
    best_c = centers.copy()
    best_s = sides.copy()
    best_obj = _objective(best_c, best_s)
    
    # Try smooth optimization from current point
    try:
        new_c, new_s = _smooth_optimize(centers, sides, tol, max_iters=500, beta=50.0)
        # Clamp to valid range
        new_c = _clamp_to_unit(new_c, new_s)
        new_s = _grow_all_to_convergence(new_c, new_s, tol)
        if _is_valid(new_c, new_s):
            obj = _objective(new_c, new_s)
            if obj > best_obj + 1e-9:
                best_c = new_c
                best_s = new_s
                best_obj = obj
    except Exception:
        pass
    
    # Try with different beta values
    for beta in [20.0, 100.0]:
        if deadline is not None and time.time() > deadline - 20:
            break
        try:
            new_c, new_s = _smooth_optimize(best_c, best_s, tol, max_iters=300, beta=beta)
            new_c = _clamp_to_unit(new_c, new_s)
            new_s = _grow_all_to_convergence(new_c, new_s, tol)
            if _is_valid(new_c, new_s):
                obj = _objective(new_c, new_s)
                if obj > best_obj + 1e-9:
                    best_c = new_c
                    best_s = new_s
                    best_obj = obj
        except Exception:
            pass
    
    return best_c, best_s


def _try_push_to_corner(centers, sides, i, tol=1e-7):
    """
    Try to push square i toward each of the 4 corners and edges.
    Returns (new_centers, new_sides, improved) if a better packing is found.
    """
    n = len(sides)
    best_obj = _objective(centers, sides)
    best_c = centers.copy()
    best_s = sides.copy()
    
    half = sides[i] / 2.0
    
    # Candidate target positions: corners and edge midpoints
    targets = [
        (half, half),              # bottom-left corner
        (1.0 - half, half),        # bottom-right
        (half, 1.0 - half),        # top-left
        (1.0 - half, 1.0 - half),  # top-right
        (0.5, half),               # bottom edge mid
        (0.5, 1.0 - half),         # top edge mid
        (half, 0.5),               # left edge mid
        (1.0 - half, 0.5),         # right edge mid
    ]
    
    for tx, ty in targets:
        test_c = centers.copy()
        test_c[i, 0] = tx
        test_c[i, 1] = ty
        if not _is_valid(test_c, sides, tol):
            continue
        # Try growing all squares
        trial_s = sides.copy()
        for k in range(n):
            new_s = _grow_square_max(test_c, trial_s, k, tol)
            if new_s > trial_s[k] + 1e-9:
                trial_s[k] = new_s
        if _is_valid(test_c, trial_s, tol):
            obj = _objective(test_c, trial_s)
            if obj > best_obj + 1e-9:
                best_obj = obj
                best_c = test_c.copy()
                best_s = trial_s.copy()
    
    return best_c, best_s


def _local_optimize(centers, sides, tol=1e-7, max_rounds=50, deadline=None):
    """
    Local optimization: alternate between simultaneous growth, repositioning, and perturbations.
    """
    n = len(sides)
    best_sides = sides.copy()
    best_centers = centers.copy()
    best_obj = _objective(best_centers, best_sides)
    
    for rnd in range(max_rounds):
        if deadline is not None and time.time() > deadline - 10:
            break
            
        # Step 1: Simultaneous growth to convergence
        new_sides = sides.copy()
        new_sides = _grow_all_to_convergence(centers, new_sides, tol)
        
        if _is_valid(centers, new_sides, tol):
            sides = new_sides
            obj = _objective(centers, sides)
            if obj > best_obj + 1e-9:
                best_obj = obj
                best_sides = sides.copy()
                best_centers = centers.copy()
        
        # Step 2: Try pushing each square to corners/edges
        for i in range(n):
            if deadline is not None and time.time() > deadline - 10:
                break
            new_c, new_s = _try_push_to_corner(centers, sides, i, tol)
            if _is_valid(new_c, new_s, tol):
                obj = _objective(new_c, new_s)
                if obj > best_obj + 1e-9:
                    best_obj = obj
                    best_centers = new_c.copy()
                    best_sides = new_s.copy()
                    centers = new_c.copy()
                    sides = new_s.copy()
        
        # Step 3: Small random perturbations with growth
        rng = np.random.default_rng(seed=rnd * 1000 + 42)
        for i in range(n):
            if deadline is not None and time.time() > deadline - 10:
                break
            for _ in range(5):
                dx = rng.uniform(-0.04, 0.04)
                dy = rng.uniform(-0.04, 0.04)
                new_c = centers.copy()
                new_c[i, 0] = np.clip(centers[i, 0] + dx, sides[i]/2, 1.0 - sides[i]/2)
                new_c[i, 1] = np.clip(centers[i, 1] + dy, sides[i]/2, 1.0 - sides[i]/2)
                if _is_valid(new_c, sides, tol):
                    trial_s = sides.copy()
                    trial_s = _grow_all_to_convergence(new_c, trial_s, tol)
                    if _is_valid(new_c, trial_s, tol):
                        obj = _objective(new_c, trial_s)
                        if obj > best_obj + 1e-9:
                            best_obj = obj
                            best_centers = new_c.copy()
                            best_sides = trial_s.copy()
                            centers = new_c.copy()
                            sides = trial_s.copy()
        
        # Convergence check
        if rnd > 5:
            can_improve = False
            for i in range(n):
                max_s = _grow_square_max(centers, sides, i, tol)
                if max_s > sides[i] + 1e-9:
                    can_improve = True
                    break
            if not can_improve:
                break
    
    return best_centers, best_sides


def _init_grid_packing():
    """Baseline 3x3 grid packing."""
    n = 10
    k = 3
    side = 1.0 / k
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    cx = (cols + 0.5) * side
    cy = (rows + 0.5) * side
    centers = np.stack([cx, cy], axis=1).astype(np.float64)
    sides = np.full(n, side, dtype=np.float64)
    return centers, sides


def _init_heterogeneous_1():
    """
    One large square (side 0.5) in bottom-left, rest in top-right region.
    """
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)
    centers[0] = [0.25, 0.25]
    sides[0] = 0.5
    side_t = 1.0 / 3.0
    idx = 1
    for row in range(2):
        for col in range(3):
            centers[idx, 0] = (col + 0.5) * side_t
            centers[idx, 1] = 0.5 + (row + 0.5) * side_t
            sides[idx] = side_t
            idx += 1
    side_r = 1.0 / 3.0
    for col in range(3):
        centers[idx, 0] = 0.5 + (col + 0.5) * side_r
        centers[idx, 1] = 0.25
        sides[idx] = side_r
        idx += 1
    return centers, sides


def _init_heterogeneous_2():
    """
    Two large squares (side 0.4) and 8 small ones.
    """
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)
    centers[0] = [0.2, 0.2]
    sides[0] = 0.4
    centers[1] = [0.8, 0.2]
    sides[1] = 0.4
    side_s = 0.25
    idx = 2
    for row in range(2):
        for col in range(4):
            centers[idx, 0] = (col + 0.5) * side_s
            centers[idx, 1] = 0.4 + (row + 0.5) * side_s
            sides[idx] = side_s
            idx += 1
    return centers, sides


def _init_heterogeneous_3():
    """
    One very large square (side 0.5) and 9 small squares filling the rest.
    """
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)
    centers[0] = [0.25, 0.25]
    sides[0] = 0.5
    side_s = 1.0/6.0
    idx = 1
    for col in range(6):
        centers[idx, 0] = (col + 0.5) * side_s
        centers[idx, 1] = 0.5 + 0.5 * side_s
        sides[idx] = side_s
        idx += 1
    for col in range(3):
        centers[idx, 0] = 0.5 + (col + 0.5) * side_s
        centers[idx, 1] = 0.5 * side_s
        sides[idx] = side_s
        idx += 1
    return centers, sides


def _init_heterogeneous_4():
    """
    1 large (0.5) + 2 medium (0.25) + 7 small, placed in feasible positions.
    """
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)
    centers[0] = [0.25, 0.25]
    sides[0] = 0.5
    centers[1] = [0.125, 0.625]
    sides[1] = 0.25
    centers[2] = [0.875, 0.625]
    sides[2] = 0.25
    side_s = 0.125
    idx = 3
    positions = [
        [0.375, 0.625], [0.625, 0.625],
        [0.125, 0.875], [0.375, 0.875], [0.625, 0.875], [0.875, 0.875],
        [0.75, 0.25],
    ]
    for pos in positions:
        centers[idx] = pos
        sides[idx] = side_s
        idx += 1
    return centers, sides


def _init_heterogeneous_5():
    """
    Two large squares (0.45) in opposite corners, 8 smaller filling the rest.
    """
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)
    centers[0] = [0.225, 0.225]
    sides[0] = 0.45
    centers[1] = [0.775, 0.775]
    sides[1] = 0.45
    side_s = 0.15
    idx = 2
    positions = [
        [0.225, 0.575], [0.225, 0.875],
        [0.525, 0.225], [0.825, 0.225],
        [0.525, 0.525],
        [0.05, 0.575], [0.05, 0.875],
        [0.825, 0.525],
    ]
    for pos in positions:
        centers[idx] = pos
        sides[idx] = side_s
        idx += 1
    return centers, sides


def _init_heterogeneous_6():
    """
    One large (0.5) in corner, 3 medium (0.25) in other corners/edges, 6 small.
    """
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)
    centers[0] = [0.25, 0.25]
    sides[0] = 0.5
    centers[1] = [0.625, 0.125]
    sides[1] = 0.25
    centers[2] = [0.875, 0.625]
    sides[2] = 0.25
    centers[3] = [0.125, 0.875]
    sides[3] = 0.25
    side_s = 0.125
    idx = 4
    positions = [
        [0.625, 0.375], [0.875, 0.375],
        [0.375, 0.625], [0.375, 0.875],
        [0.625, 0.875], [0.875, 0.875],
    ]
    for pos in positions:
        centers[idx] = pos
        sides[idx] = side_s
        idx += 1
    return centers, sides


def _init_heterogeneous_7():
    """
    Three large squares (0.4) in three corners, 7 smaller filling the rest.
    """
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)
    centers[0] = [0.2, 0.2]
    sides[0] = 0.4
    centers[1] = [0.8, 0.2]
    sides[1] = 0.4
    centers[2] = [0.2, 0.8]
    sides[2] = 0.4
    side_s = 0.2
    idx = 3
    positions = [
        [0.5, 0.5], [0.8, 0.5],
        [0.5, 0.8], [0.8, 0.8],
        [0.65, 0.65], [0.65, 0.35],
        [0.35, 0.65],
    ]
    for pos in positions:
        centers[idx] = pos
        sides[idx] = side_s
        idx += 1
    return centers, sides


def _init_heterogeneous_8():
    """
    Four medium squares (0.3) in a 2x2 grid, 6 small filling gaps.
    """
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)
    # 2x2 grid of side 0.3: positions at (0.15,0.15), (0.65,0.15), (0.15,0.65), (0.65,0.65)
    positions_large = [(0.15, 0.15), (0.65, 0.15), (0.15, 0.65), (0.65, 0.65)]
    for i, (cx, cy) in enumerate(positions_large):
        centers[i] = [cx, cy]
        sides[i] = 0.3
    # 6 small squares of side 0.1 in remaining gaps
    side_s = 0.1
    idx = 4
    positions = [
        [0.4, 0.15], [0.4, 0.65],  # middle vertical
        [0.15, 0.4], [0.65, 0.4],  # middle horizontal
        [0.4, 0.4],                # center
        [0.85, 0.85],              # top-right corner
    ]
    for pos in positions:
        centers[idx] = pos
        sides[idx] = side_s
        idx += 1
    return centers, sides


def _init_heterogeneous_9():
    """
    One very large (0.55) in corner, 4 medium (0.2), 5 small (0.1).
    """
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)
    centers[0] = [0.275, 0.275]
    sides[0] = 0.55
    # 4 medium squares of side 0.2
    positions_med = [(0.65, 0.1), (0.85, 0.3), (0.1, 0.65), (0.3, 0.85)]
    for i, (cx, cy) in enumerate(positions_med):
        centers[1+i] = [cx, cy]
        sides[1+i] = 0.2
    # 5 small squares of side 0.1
    side_s = 0.1
    idx = 5
    positions = [
        [0.55, 0.55], [0.75, 0.55],
        [0.55, 0.75], [0.75, 0.75],
        [0.9, 0.9],
    ]
    for pos in positions:
        centers[idx] = pos
        sides[idx] = side_s
        idx += 1
    return centers, sides


def _init_heterogeneous_10():
    """
    Spiral-like arrangement: large in center, decreasing sizes outward.
    """
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)
    # Center: side 0.25
    centers[0] = [0.5, 0.5]
    sides[0] = 0.25
    # Ring 1: 4 squares of side 0.2
    positions = [(0.3, 0.3), (0.7, 0.3), (0.3, 0.7), (0.7, 0.7)]
    for i, (cx, cy) in enumerate(positions):
        centers[1+i] = [cx, cy]
        sides[1+i] = 0.2
    # Ring 2: 5 squares of side 0.15
    side_s = 0.15
    idx = 5
    positions = [
        [0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9],
        [0.5, 0.1],
    ]
    for pos in positions:
        centers[idx] = pos
        sides[idx] = side_s
        idx += 1
    return centers, sides


def squares_in_square10() -> tuple:
    """
    Erdos "squares in a square" problem for n=10.
    Returns (centers, sides) maximizing sum of side lengths.
    """
    start_time = time.time()
    deadline = start_time + 570.0  # 570s safety margin
    
    # Generate multiple initial packings
    inits = [
        _init_grid_packing(),
        _init_heterogeneous_1(),
        _init_heterogeneous_2(),
        _init_heterogeneous_3(),
        _init_heterogeneous_4(),
        _init_heterogeneous_5(),
        _init_heterogeneous_6(),
        _init_heterogeneous_7(),
        _init_heterogeneous_8(),
        _init_heterogeneous_9(),
        _init_heterogeneous_10(),
    ]
    
    best_centers = None
    best_sides = None
    best_obj = -1.0
    
    for centers, sides in inits:
        if time.time() > deadline - 60:
            break
        
        # First, resolve any overlaps
        centers = centers.copy()
        sides = sides.copy()
        centers = _clamp_to_unit(centers, sides)
        centers = _resolve_overlap(centers, sides)
        centers = _clamp_to_unit(centers, sides)
        
        if not _is_valid(centers, sides):
            # Try harder to fix
            for _ in range(10):
                centers = _resolve_overlap(centers, sides)
                centers = _clamp_to_unit(centers, sides)
                if _is_valid(centers, sides):
                    break
        
        if not _is_valid(centers, sides):
            continue
        
        # Local optimization
        opt_centers, opt_sides = _local_optimize(centers, sides, deadline=deadline)
        
        if _is_valid(opt_centers, opt_sides):
            obj = _objective(opt_centers, opt_sides)
            if obj > best_obj:
                best_obj = obj
                best_centers = opt_centers.copy()
                best_sides = opt_sides.copy()
        
        # Also try the original (pre-optimization) in case optimizer degraded
        orig_obj = _objective(centers, sides)
        if orig_obj > best_obj:
            best_obj = orig_obj
            best_centers = centers.copy()
            best_sides = sides.copy()
    
    # If no valid solution found, fall back to grid
    if best_centers is None:
        centers, sides = _init_grid_packing()
        best_centers = centers
        best_sides = sides
    else:
        # Final validation
        if not _is_valid(best_centers, best_sides):
            centers, sides = _init_grid_packing()
            best_centers = centers
            best_sides = sides
        
        # Final polish: high-precision smooth optimization
        if time.time() < deadline - 30:
            best_centers, best_sides = _final_polish(best_centers, best_sides, deadline=deadline)
        
        # Final polish: high-precision simultaneous growth
        final_sides = best_sides.copy()
        final_sides = _grow_all_to_convergence(best_centers, final_sides, tol=1e-7, max_iters=500)
        if _is_valid(best_centers, final_sides):
            if _objective(best_centers, final_sides) > _objective(best_centers, best_sides):
                best_sides = final_sides
    
    return best_centers, best_sides


# EVOLVE-BLOCK-END
