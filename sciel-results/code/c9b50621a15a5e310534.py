"""Circle packing in unit square: pack 26 circles, maximize sum of radii."""
import numpy as np
from scipy.optimize import linprog
import time

N = 26
DEADLINE = 150.0  # wall-clock seconds
SAFETY_MARGIN = 10.0  # return 10s before deadline


def _check_validity(circles, tol=1e-9):
    """Check if circles are valid (inside unit square, disjoint interiors)."""
    for x, y, r in circles:
        if r <= 0:
            return False
        if x - r < -tol or x + r > 1 + tol:
            return False
        if y - r < -tol or y + r > 1 + tol:
            return False
    n = len(circles)
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, ri = circles[i]
            xj, yj, rj = circles[j]
            dist = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
            if dist < ri + rj - tol:
                return False
    return True


def _solve_lp_radii(positions, tol=1e-12):
    """
    Given fixed center positions, solve LP to maximize sum of radii
    subject to:
      - r_i >= 0
      - x_i - r_i >= 0, x_i + r_i <= 1, y_i - r_i >= 0, y_i + r_i <= 1
      - r_i + r_j <= dist(i,j) for all pairs i,j
    
    Returns optimal radii vector or None if infeasible.
    """
    n = len(positions)
    # Variables: r_0, r_1, ..., r_{n-1}
    # Maximize sum(r_i) => minimize -sum(r_i)
    c = -np.ones(n)
    
    # Inequality constraints: A_ub @ r <= b_ub
    # Boundary constraints:
    #   -r_i <= x_i  =>  r_i >= -x_i  (but r_i >= 0 dominates if x_i >= 0)
    #   r_i <= x_i
    #   r_i <= 1 - x_i
    #   r_i <= y_i
    #   r_i <= 1 - y_i
    # Pair constraints:
    #   r_i + r_j <= dist_ij
    
    A_ub_rows = []
    b_ub_rows = []
    
    for i in range(n):
        x, y = positions[i]
        # r_i <= x
        row = np.zeros(n)
        row[i] = 1.0
        A_ub_rows.append(row)
        b_ub_rows.append(x)
        # r_i <= 1 - x
        row = np.zeros(n)
        row[i] = 1.0
        A_ub_rows.append(row)
        b_ub_rows.append(1.0 - x)
        # r_i <= y
        row = np.zeros(n)
        row[i] = 1.0
        A_ub_rows.append(row)
        b_ub_rows.append(y)
        # r_i <= 1 - y
        row = np.zeros(n)
        row[i] = 1.0
        A_ub_rows.append(row)
        b_ub_rows.append(1.0 - y)
    
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi = positions[i]
            xj, yj = positions[j]
            dist = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub_rows.append(row)
            b_ub_rows.append(dist)
    
    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_rows)
    
    bounds = [(0, None)] * n
    
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if res.status != 0:
        return None
    
    radii = res.x
    # Clamp tiny negatives
    radii = np.maximum(radii, 0.0)
    return radii


def _gradient_ascent_optimize(positions, radii, max_iters=500, step_size=0.01, deadline=None):
    """
    Gradient ascent on positions to maximize sum of radii,
    where radii are recomputed via LP at each step.
    """
    n = len(positions)
    best_positions = positions.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline - SAFETY_MARGIN:
            break
        
        # Compute gradient of LP objective w.r.t. positions
        # Use finite differences or analytical gradient from LP dual
        # For simplicity, use a small perturbation approach
        grad = np.zeros((n, 2))
        eps = 0.001
        
        for i in range(n):
            for d in range(2):
                pos_plus = positions.copy()
                pos_plus[i, d] += eps
                r_plus = _solve_lp_radii(pos_plus)
                if r_plus is None:
                    grad[i, d] = -1.0  # infeasible, penalize
                else:
                    grad[i, d] = (np.sum(r_plus) - best_score) / eps
                
                pos_minus = positions.copy()
                pos_minus[i, d] -= eps
                r_minus = _solve_lp_radii(pos_minus)
                if r_minus is None:
                    grad[i, d] = -1.0
                else:
                    grad[i, d] = (np.sum(r_plus) - np.sum(r_minus)) / (2 * eps) if r_plus is not None else -1.0
        
        # Update positions
        positions_new = positions + step_size * grad
        # Keep positions within [0,1]
        positions_new = np.clip(positions_new, 0.0, 1.0)
        
        # Solve LP for new positions
        radii_new = _solve_lp_radii(positions_new)
        if radii_new is None:
            break
        
        score_new = np.sum(radii_new)
        if score_new > best_score:
            best_positions = positions_new.copy()
            best_radii = radii_new.copy()
            best_score = score_new
            positions = positions_new
            radii = radii_new
        else:
            # Reduce step size
            step_size *= 0.9
        
        if step_size < 1e-6:
            break
    
    return best_positions, best_radii


def _random_positions(n, seed=None):
    """Generate random positions in [0.05, 0.95]^2."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0.05, 0.95, size=(n, 2))


def _initial_grid_positions(n):
    """Generate grid-based initial positions."""
    side = int(np.ceil(np.sqrt(n)))
    positions = []
    for i in range(side):
        for j in range(side):
            if len(positions) >= n:
                break
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            positions.append([x, y])
    return np.array(positions[:n])


def construct():
    start_time = time.time()
    
    # Start with grid positions
    positions = _initial_grid_positions(N)
    
    # Solve LP for initial radii
    radii = _solve_lp_radii(positions)
    if radii is None:
        # Fallback: very small circles
        positions = _initial_grid_positions(N)
        radii = np.full(N, 0.01)
    
    best_positions = positions.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    # Try multiple random restarts with gradient optimization
    rng = np.random.default_rng(42)
    num_restarts = 20
    
    for restart in range(num_restarts):
        if time.time() > DEADLINE - SAFETY_MARGIN:
            break
        
        # New random positions
        if restart == 0:
            pos = _initial_grid_positions(N)
        else:
            pos = _random_positions(N, seed=rng.integers(0, 2**31))
        
        # Solve LP
        rad = _solve_lp_radii(pos)
        if rad is None:
            continue
        
        score = np.sum(rad)
        if score > best_score:
            best_positions = pos.copy()
            best_radii = rad.copy()
            best_score = score
        
        # Gradient ascent from this point
        pos_opt, rad_opt = _gradient_ascent_optimize(
            pos.copy(), rad.copy(),
            max_iters=100,
            step_size=0.02,
            deadline=DEADLINE
        )
        
        if time.time() > DEADLINE - SAFETY_MARGIN:
            break
        
        score_opt = np.sum(rad_opt)
        if score_opt > best_score:
            best_positions = pos_opt.copy()
            best_radii = rad_opt.copy()
            best_score = score_opt
    
    # Final polish: try to improve with smaller steps
    if time.time() < DEADLINE - SAFETY_MARGIN:
        pos_polish, rad_polish = _gradient_ascent_optimize(
            best_positions.copy(), best_radii.copy(),
            max_iters=50,
            step_size=0.005,
            deadline=DEADLINE
        )
        score_polish = np.sum(rad_polish)
        if score_polish > best_score:
            best_positions = pos_polish.copy()
            best_radii = rad_polish.copy()
            best_score = score_polish
    
    # Build result
    result = np.column_stack([best_positions, best_radii])
    
    # Ensure validity: clip radii to be non-negative and check constraints
    for i in range(N):
        x, y, r = result[i]
        # Ensure circle is inside unit square
        r = min(r, x, 1.0 - x, y, 1.0 - y)
        result[i, 2] = max(r, 0.0)
    
    # Final validity check and fix
    for i in range(N):
        for j in range(i + 1, N):
            xi, yi, ri = result[i]
            xj, yj, rj = result[j]
            dist = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
            if ri + rj > dist + 1e-9:
                # Reduce radii to make them fit
                scale = dist / (ri + rj) if (ri + rj) > 0 else 1.0
                result[i, 2] = ri * scale
                result[j, 2] = rj * scale
    
    return result
