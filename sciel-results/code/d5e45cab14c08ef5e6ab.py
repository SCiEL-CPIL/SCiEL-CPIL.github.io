"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _compute_score(circles):
    """Compute the sum of radii if valid, else 0."""
    n = len(circles)
    if n == 0:
        return 0.0
    # Check bounds: circle must be inside [0,1]^2
    for i in range(n):
        x, y, r = circles[i]
        if r <= 0:
            return 0.0
        if x - r < -1e-9 or x + r > 1 + 1e-9 or y - r < -1e-9 or y + r > 1 + 1e-9:
            return 0.0
    # Check pairwise disjointness
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, ri = circles[i]
            xj, yj, rj = circles[j]
            dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
            if dist < ri + rj - 1e-9:
                return 0.0
    return float(np.sum(circles[:, 2]))


def _make_initial_grid(n):
    """Create an initial grid layout of n circles."""
    side = int(np.ceil(np.sqrt(n)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) >= n:
                break
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side - 0.01
            cells.append([x, y, r])
    return np.array(cells)


def _penalty_objective(params, n):
    """
    Objective function: minimize negative sum of radii + penalty for violations.
    params: (3*n,) array of [x1,y1,r1, x2,y2,r2, ...]
    """
    circles = params.reshape(n, 3)
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    
    # Negative sum of radii (we minimize, so negate)
    obj = -np.sum(r)
    
    # Penalty for boundary violations
    boundary_pen = 0.0
    for i in range(n):
        if r[i] <= 0:
            boundary_pen += 100.0 * (1.0 - r[i])
        else:
            # Distance to boundary
            d_left = x[i] - r[i]
            d_right = (1.0 - x[i]) - r[i]
            d_bottom = y[i] - r[i]
            d_top = (1.0 - y[i]) - r[i]
            min_d = min(d_left, d_right, d_bottom, d_top)
            if min_d < 0:
                boundary_pen += 100.0 * (-min_d)
    
    # Penalty for overlap
    overlap_pen = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
            required = r[i] + r[j]
            if dist < required:
                overlap_pen += 100.0 * (required - dist)
    
    return obj + boundary_pen + overlap_pen


def _gradient_objective(params, n):
    """
    Compute objective and gradient for L-BFGS-B.
    """
    circles = params.reshape(n, 3)
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    
    obj = -np.sum(r)
    grad = np.zeros_like(params)
    
    # Gradient of -sum(r)
    for i in range(n):
        grad[3*i + 2] = -1.0
    
    # Boundary penalty gradient
    for i in range(n):
        if r[i] <= 0:
            # Penalty: 100*(1-r), gradient w.r.t r: -100
            grad[3*i + 2] += -100.0
        else:
            d_left = x[i] - r[i]
            d_right = (1.0 - x[i]) - r[i]
            d_bottom = y[i] - r[i]
            d_top = (1.0 - y[i]) - r[i]
            min_d = min(d_left, d_right, d_bottom, d_top)
            if min_d < 0:
                # Find which constraint is violated
                if d_left == min_d:
                    # Penalty: 100*(-d_left) = 100*(r - x)
                    # grad w.r.t x: -100, w.r.t r: 100
                    grad[3*i + 0] += -100.0
                    grad[3*i + 2] += 100.0
                elif d_right == min_d:
                    # Penalty: 100*(x + r - 1)
                    # grad w.r.t x: 100, w.r.t r: 100
                    grad[3*i + 0] += 100.0
                    grad[3*i + 2] += 100.0
                elif d_bottom == min_d:
                    # Penalty: 100*(r - y)
                    # grad w.r.t y: -100, w.r.t r: 100
                    grad[3*i + 1] += -100.0
                    grad[3*i + 2] += 100.0
                elif d_top == min_d:
                    # Penalty: 100*(y + r - 1)
                    # grad w.r.t y: 100, w.r.t r: 100
                    grad[3*i + 1] += 100.0
                    grad[3*i + 2] += 100.0
    
    # Overlap penalty gradient
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy) + 1e-12
            required = r[i] + r[j]
            if dist < required:
                # Penalty: 100*(required - dist)
                # grad w.r.t x[i]: 100 * (-dx/dist)
                # grad w.r.t y[i]: 100 * (-dy/dist)
                # grad w.r.t x[j]: 100 * (dx/dist)
                # grad w.r.t y[j]: 100 * (dy/dist)
                # grad w.r.t r[i]: 100
                # grad w.r.t r[j]: 100
                factor = 100.0 / dist
                grad[3*i + 0] += -factor * dx
                grad[3*i + 1] += -factor * dy
                grad[3*j + 0] += factor * dx
                grad[3*j + 1] += factor * dy
                grad[3*i + 2] += 100.0
                grad[3*j + 2] += 100.0
                obj += 100.0 * (required - dist)
    
    return obj, grad


def _optimize_from_initial(circles, deadline):
    """Optimize from initial circles using L-BFGS-B with gradient."""
    n = len(circles)
    params0 = circles.flatten()
    
    # Bounds: x,y in [0,1], r in [0, 0.5]
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0))  # x
        bounds.append((0.0, 1.0))  # y
        bounds.append((0.0, 0.5))  # r
    
    best_circles = circles.copy()
    best_score = _compute_score(circles)
    
    start_time = time.time()
    
    try:
        result = minimize(
            _penalty_objective,
            params0,
            args=(n,),
            method='L-BFGS-B',
            jac=_gradient_objective,
            bounds=bounds,
            options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 5000}
        )
        
        if result.success or result.fun < _penalty_objective(params0, n):
            new_circles = result.x.reshape(n, 3)
            new_score = _compute_score(new_circles)
            if new_score > best_score:
                best_circles = new_circles
                best_score = new_score
        
        # Try multiple restarts if time permits
        while time.time() < deadline - 10:
            # Perturb best solution
            perturbed = best_circles.copy()
            perturbed[:, 0] += np.random.normal(0, 0.05, n)
            perturbed[:, 1] += np.random.normal(0, 0.05, n)
            perturbed[:, 2] += np.random.normal(0, 0.02, n)
            perturbed[:, 0] = np.clip(perturbed[:, 0], 0.05, 0.95)
            perturbed[:, 1] = np.clip(perturbed[:, 1], 0.05, 0.95)
            perturbed[:, 2] = np.clip(perturbed[:, 2], 0.01, 0.45)
            
            params_perturbed = perturbed.flatten()
            result2 = minimize(
                _penalty_objective,
                params_perturbed,
                args=(n,),
                method='L-BFGS-B',
                jac=_gradient_objective,
                bounds=bounds,
                options={'maxiter': 3000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 3000}
            )
            
            new_circles2 = result2.x.reshape(n, 3)
            new_score2 = _compute_score(new_circles2)
            if new_score2 > best_score:
                best_circles = new_circles2
                best_score = new_score2
                
    except Exception as e:
        pass
    
    return best_circles


def construct():
    n = N
    deadline = time.time() + 130  # Leave 20s safety margin
    
    # Start with grid
    circles = _make_initial_grid(n)
    
    # Optimize
    circles = _optimize_from_initial(circles, deadline)
    
    # Final validation and cleanup
    final_score = _compute_score(circles)
    if final_score <= 0:
        # Fallback to grid
        circles = _make_initial_grid(n)
    
    return circles
# EVOLVE-BLOCK-END
