"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 21
DEADLINE = 115.0  # safety margin before 120s limit

def _valid_solution(circles):
    """Check if circles are valid: positive radii, within [0, W]x[0, H] with W+H=2 (perimeter 4), disjoint."""
    if len(circles) != N:
        return False
    for i in range(N):
        x, y, r = circles[i]
        if r <= 0:
            return False
        if x < -1e-6 or x > 2 + 1e-6 or y < -1e-6 or y > 2 + 1e-6:
            return False
    # Check disjoint interiors
    for i in range(N):
        for j in range(i + 1, N):
            xi, yi, ri = circles[i]
            xj, yj, rj = circles[j]
            dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
            if dist < ri + rj - 1e-9:
                return False
    return True

def _score(circles):
    """Sum of radii."""
    return np.sum(circles[:, 2])

def _construct_grid(n, W, H):
    """Create a grid of n circles in rectangle W x H."""
    # Try different grid arrangements
    best_score = -1
    best_circles = None
    
    # Try various splits
    for rows in range(1, n + 1):
        for cols in range(1, n + 1):
            if rows * cols < n:
                continue
            # Place circles in a grid pattern
            circles = []
            count = 0
            for i in range(rows):
                for j in range(cols):
                    if count >= n:
                        break
                    x = (j + 0.5) * W / cols
                    y = (i + 0.5) * H / rows
                    r = min(W / (2 * cols), H / (2 * rows)) * 0.95
                    circles.append([x, y, r])
                    count += 1
                if count >= n:
                    break
            if len(circles) == n:
                circles = np.array(circles)
                if _valid_solution(circles):
                    s = _score(circles)
                    if s > best_score:
                        best_score = s
                        best_circles = circles
    
    return best_circles if best_circles is not None else None

def _optimize_layout(circles, W, H, max_iters=1000, tol=1e-8):
    """
    Optimize circle positions and radii to maximize sum of radii,
    subject to non-overlap and boundary constraints.
    
    Uses a penalty-based approach with scipy minimize.
    """
    n = len(circles)
    x0 = circles.flatten()
    
    def objective_and_constraints(params):
        """Compute objective (negative sum of radii) and constraint violations."""
        params = np.array(params)
        xs = params[:n]
        ys = params[n:2*n]
        rs = params[2*n:3*n]
        
        # Objective: negative sum of radii
        obj = -np.sum(rs)
        
        # Constraint violations
        violations = 0.0
        
        # Boundary constraints
        for i in range(n):
            # x in [r_i, W - r_i]
            if xs[i] < rs[i]:
                violations += (rs[i] - xs[i])**2
            if xs[i] > W - rs[i]:
                violations += (xs[i] - (W - rs[i]))**2
            # y in [r_i, H - r_i]
            if ys[i] < rs[i]:
                violations += (rs[i] - ys[i])**2
            if ys[i] > H - rs[i]:
                violations += (ys[i] - (H - rs[i]))**2
        
        # Non-overlap constraints
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2)
                min_dist = rs[i] + rs[j]
                if dist < min_dist:
                    violations += (min_dist - dist)**2
        
        return obj, violations
    
    def total_cost(params):
        obj, viol = objective_and_constraints(params)
        return obj + 10.0 * viol
    
    def total_cost_grad(params):
        # Numerical gradient
        eps = 1e-7
        grad = np.zeros_like(params)
        base_cost = total_cost(params)
        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += eps
            grad[i] = (total_cost(params_plus) - base_cost) / eps
        return grad
    
    # Use L-BFGS-B
    result = minimize(total_cost, x0, method='L-BFGS-B', 
                      jac=total_cost_grad,
                      options={'maxiter': max_iters, 'ftol': tol, 'gtol': 1e-6})
    
    return result.x.reshape(n, 3)

def _local_search(circles, W, H, max_iters=500):
    """
    Local search: try to increase radii while maintaining validity.
    Iteratively expand radii and re-optimize positions.
    """
    circles = circles.copy()
    
    for _ in range(max_iters):
        # Try to increase all radii by a small amount
        factor = 1.001
        new_circles = circles.copy()
        new_circles[:, 2] *= factor
        
        # Re-optimize positions
        optimized = _optimize_layout(new_circles, W, H, max_iters=200, tol=1e-6)
        
        if _valid_solution(optimized):
            circles = optimized
        else:
            # Try smaller increments
            found = False
            for f in [1.0005, 1.0001]:
                new_circles = circles.copy()
                new_circles[:, 2] *= f
                optimized = _optimize_layout(new_circles, W, H, max_iters=100, tol=1e-6)
                if _valid_solution(optimized):
                    circles = optimized
                    found = True
                    break
            if not found:
                break
    
    return circles

def construct():
    start_time = time.time()
    
    # Initial guess: square rectangle W=H=1 (perimeter = 4)
    W = 1.0
    H = 1.0
    
    # Create initial grid
    best_circles = _construct_grid(N, W, H)
    
    if best_circles is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        count = 0
        for i in range(side):
            for j in range(side):
                if count >= N:
                    break
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 1e-3
                cells.append([x, y, r])
                count += 1
            if count >= N:
                break
        best_circles = np.array(cells[:N])
    
    best_score = _score(best_circles)
    
    # Optimization loop
    while time.time() - start_time < DEADLINE:
        # Optimize layout
        optimized = _optimize_layout(best_circles, W, H, max_iters=300, tol=1e-7)
        
        if _valid_solution(optimized):
            opt_score = _score(optimized)
            if opt_score > best_score:
                best_circles = optimized
                best_score = opt_score
        
        # Local search to expand radii
        if time.time() - start_time < DEADLINE - 10:
            expanded = _local_search(best_circles, W, H, max_iters=200)
            if _valid_solution(expanded):
                exp_score = _score(expanded)
                if exp_score > best_score:
                    best_circles = expanded
                    best_score = exp_score
        
        # Check if we should stop
        if time.time() - start_time > DEADLINE - 5:
            break
    
    return best_circles
