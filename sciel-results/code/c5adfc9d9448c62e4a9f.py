"""Circle packing in unit square n=32 using gradient-based optimization."""
import numpy as np
from scipy.optimize import minimize
import time

N = 32

# EVOLVE-BLOCK-START

def _compute_forces(x, y, r):
    """Compute repulsive forces between circles and wall constraints.
    Returns (force_x, force_y, force_r, penalty_value)."""
    n = len(x)
    fx = np.zeros(n)
    fy = np.zeros(n)
    fr = np.zeros(n)
    penalty = 0.0
    
    # Pairwise repulsion
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = r[i] + r[j]
            
            if dist < min_dist:
                # Overlap penalty: push apart
                if dist > 1e-12:
                    overlap = min_dist - dist
                    # Force proportional to overlap, direction along separation
                    fx[i] += overlap * dx / dist
                    fy[i] += overlap * dy / dist
                    fx[j] -= overlap * dx / dist
                    fy[j] -= overlap * dy / dist
                    # Also push radii apart slightly
                    fr[i] += 0.5 * overlap
                    fr[j] += 0.5 * overlap
                penalty += 0.5 * overlap * overlap
    
    # Wall constraints
    for i in range(n):
        # Left wall
        if x[i] < r[i]:
            overlap = r[i] - x[i]
            fx[i] += overlap
            fr[i] += 0.5 * overlap
            penalty += 0.5 * overlap * overlap
        # Right wall
        if x[i] + r[i] > 1.0:
            overlap = x[i] + r[i] - 1.0
            fx[i] -= overlap
            fr[i] += 0.5 * overlap
            penalty += 0.5 * overlap * overlap
        # Bottom wall
        if y[i] < r[i]:
            overlap = r[i] - y[i]
            fy[i] += overlap
            fr[i] += 0.5 * overlap
            penalty += 0.5 * overlap * overlap
        # Top wall
        if y[i] + r[i] > 1.0:
            overlap = y[i] + r[i] - 1.0
            fy[i] -= overlap
            fr[i] += 0.5 * overlap
            penalty += 0.5 * overlap * overlap
    
    return fx, fy, fr, penalty


def _objective_and_grad(vars):
    """Objective: maximize sum of radii, with penalty for violations.
    vars = [x_0, ..., x_{n-1}, y_0, ..., y_{n-1}, r_0, ..., r_{n-1}]"""
    n = N
    x = vars[:n]
    y = vars[n:2*n]
    r = vars[2*n:3*n]
    
    # Ensure positive radii
    r = np.maximum(r, 1e-6)
    
    fx, fy, fr, penalty = _compute_forces(x, y, r)
    
    # Objective: -sum(r) + penalty (we minimize, so negate sum of radii)
    obj = -np.sum(r) + 10.0 * penalty
    
    # Gradient
    gx = -fx
    gy = -fy
    gr = -np.ones(n) + 10.0 * fr
    
    grad = np.concatenate([gx, gy, gr])
    
    return obj, grad


def _is_valid(x, y, r, tol=1e-9):
    """Check if configuration is valid (no overlaps, all inside unit square)."""
    n = len(x)
    
    # Check radii positive
    if np.any(r < 0):
        return False
    
    # Check inside unit square
    if np.any(x < r - tol) or np.any(x + r > 1.0 + tol):
        return False
    if np.any(y < r - tol) or np.any(y + r > 1.0 + tol):
        return False
    
    # Check pairwise non-overlap
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - tol:
                return False
    
    return True


def _initial_grid():
    """Create initial grid placement of circles."""
    # Use a hexagonal-like packing for better initial density
    # Start with a 6x6 grid, take first 32
    side = 6
    cells = []
    for i in range(side):
        for j in range(side):
            # Offset every other row for hexagonal packing
            offset = 0.5 / side if j % 2 == 1 else 0
            cx = (i + 0.5) / side + offset * (1.0 / side)
            cy = (j + 0.5) / side
            r = 0.5 / side
            cells.append([cx, cy, r])
    
    # Take first N
    cells = cells[:N]
    
    x = np.array([c[0] for c in cells])
    y = np.array([c[1] for c in cells])
    r = np.array([c[2] for c in cells])
    
    return x, y, r


def _initial_random():
    """Create random initial placement."""
    np.random.seed(42)
    x = np.random.uniform(0.1, 0.9, N)
    y = np.random.uniform(0.1, 0.9, N)
    r = np.random.uniform(0.03, 0.08, N)
    return x, y, r


def _optimize_from(x0, y0, r0, deadline):
    """Run optimization from given initial state. Returns best (x, y, r, sum_r)."""
    n = N
    vars0 = np.concatenate([x0, y0, r0])
    
    # Use L-BFGS-B with bounds
    bounds = []
    for i in range(n):
        # x in [0, 1]
        bounds.append((0.0, 1.0))
    for i in range(n):
        # y in [0, 1]
        bounds.append((0.0, 1.0))
    for i in range(n):
        # r in [0, 0.5] (max possible radius in unit square)
        bounds.append((1e-6, 0.5))
    
    start_time = time.time()
    
    try:
        result = minimize(
            _objective_and_grad,
            vars0,
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={
                'maxiter': 500,
                'ftol': 1e-12,
                'gtol': 1e-8,
                'maxls': 50,
                'maxfun': 10000,
            }
        )
    except Exception as e:
        return x0, y0, r0, np.sum(r0)
    
    elapsed = time.time() - start_time
    
    # Extract result
    if result.success or result.fun < _objective_and_grad(vars0)[0]:
        vars = result.x
        x = vars[:n]
        y = vars[n:2*n]
        r = vars[2*n:3*n]
        r = np.maximum(r, 1e-6)
        
        # Clip to valid range
        x = np.clip(x, 0.0, 1.0)
        y = np.clip(y, 0.0, 1.0)
        r = np.clip(r, 1e-6, 0.5)
        
        sum_r = np.sum(r)
        return x, y, r, sum_r
    else:
        return x0, y0, r0, np.sum(r0)


def _polish(x, y, r, deadline):
    """Final high-precision polish pass."""
    n = N
    vars0 = np.concatenate([x, y, r])
    
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0))
    for i in range(n):
        bounds.append((0.0, 1.0))
    for i in range(n):
        bounds.append((1e-6, 0.5))
    
    start_time = time.time()
    
    try:
        result = minimize(
            _objective_and_grad,
            vars0,
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={
                'maxiter': 200,
                'ftol': 1e-15,
                'gtol': 1e-10,
                'maxls': 30,
                'maxfun': 5000,
            }
        )
    except Exception:
        return x, y, r
    
    elapsed = time.time() - start_time
    if time.time() > deadline - 5:
        return x, y, r
    
    if result.fun < _objective_and_grad(vars0)[0]:
        vars = result.x
        x_new = np.clip(vars[:n], 0.0, 1.0)
        y_new = np.clip(vars[n:2*n], 0.0, 1.0)
        r_new = np.clip(vars[2*n:3*n], 1e-6, 0.5)
        return x_new, y_new, r_new
    else:
        return x, y, r


def construct():
    """Construct circle packing for n=32 in unit square."""
    deadline = time.time() + 140.0  # Safety margin before 150s limit
    
    # Try multiple initializations
    best_sum = 0.0
    best_x = best_y = best_r = None
    
    # Initialization 1: Grid
    x0, y0, r0 = _initial_grid()
    x1, y1, r1, sum1 = _optimize_from(x0, y0, r0, deadline)
    if sum1 > best_sum:
        best_sum = sum1
        best_x, best_y, best_r = x1, y1, r1
    
    if time.time() > deadline - 20:
        pass  # Not enough time for more restarts
    else:
        # Initialization 2: Random
        x0, y0, r0 = _initial_random()
        x2, y2, r2, sum2 = _optimize_from(x0, y0, r0, deadline)
        if sum2 > best_sum:
            best_sum = sum2
            best_x, best_y, best_r = x2, y2, r2
    
    if time.time() > deadline - 15:
        pass
    else:
        # Initialization 3: Another random with different seed
        np.random.seed(123)
        x0 = np.random.uniform(0.1, 0.9, N)
        y0 = np.random.uniform(0.1, 0.9, N)
        r0 = np.random.uniform(0.04, 0.07, N)
        x3, y3, r3, sum3 = _optimize_from(x0, y0, r0, deadline)
        if sum3 > best_sum:
            best_sum = sum3
            best_x, best_y, best_r = x3, y3, r3
    
    # Polish the best solution
    if best_x is not None and time.time() < deadline - 5:
        best_x, best_y, best_r = _polish(best_x, best_y, best_r, deadline)
    
    # Final validation and clipping
    best_x = np.clip(best_x, 0.0, 1.0)
    best_y = np.clip(best_y, 0.0, 1.0)
    best_r = np.clip(best_r, 1e-6, 0.5)
    
    # If invalid, fall back to grid
    if not _is_valid(best_x, best_y, best_r, tol=1e-8):
        # Try to fix by slightly shrinking radii
        best_r *= 0.99
        if not _is_valid(best_x, best_y, best_r, tol=1e-8):
            # Use grid as fallback
            x0, y0, r0 = _initial_grid()
            best_x, best_y, best_r = x0, y0, r0
    
    result = np.column_stack([best_x, best_y, best_r])
    return result

# EVOLVE-BLOCK-END
