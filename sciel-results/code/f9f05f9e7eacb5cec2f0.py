"""Circle packing solver for n=32 in unit square using L-BFGS-B with penalty method."""
N = 32

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _hex_init(n, offset=0.0, rng_seed=0):
    """Generate hexagonal packing initialization for n circles."""
    rng = np.random.RandomState(rng_seed)
    points = []
    for q in range(-12, 13):
        for r in range(-12, 13):
            x = 1.5 * q
            y = np.sqrt(3) * (r + 0.5 * (q & 1))
            x += rng.uniform(-0.1, 0.1)
            y += rng.uniform(-0.1, 0.1)
            points.append([x + offset, y + offset])
    points = np.array(points)
    
    center = np.array([0.5, 0.5])
    points += (center - np.mean(points, axis=0))
    
    dists = np.sum((points - center)**2, axis=1)
    sorted_idx = np.argsort(dists)
    init_points = points[sorted_idx[:n]]
    
    radii = np.full(n, 0.08)
    return init_points, radii

def _random_init(n, seed=0):
    """Generate random initialization with non-overlapping circles."""
    rng = np.random.RandomState(seed)
    points = []
    radii = []
    
    attempts = 0
    while len(points) < n and attempts < 20000:
        attempts += 1
        x = rng.uniform(0.05, 0.95)
        y = rng.uniform(0.05, 0.95)
        r = rng.uniform(0.04, 0.12)
        
        ok = True
        for i in range(len(points)):
            dx = x - points[i][0]
            dy = y - points[i][1]
            d = np.sqrt(dx*dx + dy*dy)
            if d < r + radii[i] + 0.005:
                ok = False
                break
        
        if ok:
            points.append([x, y])
            radii.append(r)
    
    while len(points) < n:
        side = int(np.ceil(np.sqrt(len(points) + 1)))
        i = len(points)
        px = (i % side + 0.5) / side
        py = (i // side + 0.5) / side
        points.append([px, py])
        radii.append(0.07)
    
    return np.array(points), np.array(radii)

def _grid_init(n):
    """Generate grid initialization."""
    side = int(np.ceil(np.sqrt(n)))
    points = []
    radii = []
    for i in range(side):
        for j in range(side):
            if len(points) < n:
                px = (i + 0.5) / side
                py = (j + 0.5) / side
                r = 0.5 / side * 0.95
                points.append([px, py])
                radii.append(r)
    return np.array(points), np.array(radii)

def _objective(vars, C=500.0):
    """Compute objective: -sum(radii) + penalty for violations."""
    x = vars[0::3]
    y = vars[1::3]
    r = vars[2::3]
    
    obj = -np.sum(r)
    
    # Pairwise overlap penalties
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx*dx + dy*dy)
            r_sum = r[i] + r[j]
            if d < r_sum:
                obj += C * (r_sum - d) ** 2
    
    # Boundary penalties
    for i in range(N):
        if x[i] < r[i]:
            obj += C * (r[i] - x[i]) ** 2
        if x[i] + r[i] > 1.0:
            obj += C * (x[i] + r[i] - 1.0) ** 2
        if y[i] < r[i]:
            obj += C * (r[i] - y[i]) ** 2
        if y[i] + r[i] > 1.0:
            obj += C * (y[i] + r[i] - 1.0) ** 2
    
    return obj

def _gradient(vars, C=500.0):
    """Compute gradient of objective."""
    grad = np.zeros(3 * N)
    x = vars[0::3]
    y = vars[1::3]
    r = vars[2::3]
    
    # Gradient of -sum(r)
    grad[2::3] -= 1.0
    
    # Pairwise overlap penalties
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d_sq = dx*dx + dy*dy
            r_sum = r[i] + r[j]
            
            if d_sq < 1e-14:
                dx = 1e-7
                d = 1e-7
            else:
                d = np.sqrt(d_sq)
            
            if d < r_sum:
                V = r_sum - d
                coeff = 2.0 * C * V
                inv_d = 1.0 / d
                
                grad[3*i + 0] += coeff * (-dx * inv_d)
                grad[3*i + 1] += coeff * (-dy * inv_d)
                grad[3*i + 2] += coeff * 1.0
                
                grad[3*j + 0] += coeff * (dx * inv_d)
                grad[3*j + 1] += coeff * (dy * inv_d)
                grad[3*j + 2] += coeff * 1.0
    
    # Boundary penalties
    for i in range(N):
        if x[i] < r[i]:
            V = r[i] - x[i]
            coeff = 2.0 * C * V
            grad[3*i + 0] += coeff * (-1.0)
            grad[3*i + 2] += coeff * 1.0
        
        if x[i] + r[i] > 1.0:
            V = x[i] + r[i] - 1.0
            coeff = 2.0 * C * V
            grad[3*i + 0] += coeff * 1.0
            grad[3*i + 2] += coeff * 1.0
        
        if y[i] < r[i]:
            V = r[i] - y[i]
            coeff = 2.0 * C * V
            grad[3*i + 1] += coeff * (-1.0)
            grad[3*i + 2] += coeff * 1.0
        
        if y[i] + r[i] > 1.0:
            V = y[i] + r[i] - 1.0
            coeff = 2.0 * C * V
            grad[3*i + 1] += coeff * 1.0
            grad[3*i + 2] += coeff * 1.0
    
    return grad

def _push_apart(vars, num_iters=50):
    """Iteratively push overlapping circles apart to get a feasible starting point."""
    vars = vars.copy()
    
    for _ in range(num_iters):
        moved = False
        for i in range(N):
            for j in range(i + 1, N):
                x_i, y_i, r_i = vars[3*i], vars[3*i+1], vars[3*i+2]
                x_j, y_j, r_j = vars[3*j], vars[3*j+1], vars[3*j+2]
                
                dx = x_i - x_j
                dy = y_i - y_j
                d = np.sqrt(dx*dx + dy*dy)
                r_sum = r_i + r_j
                
                if d < r_sum:
                    if d < 1e-10:
                        dx, dy = 1e-5, 0
                        d = 1e-5
                    overlap = (r_sum - d) / 2.0 + 0.001
                    nx, ny = dx / d, dy / d
                    
                    vars[3*i] += nx * overlap
                    vars[3*i+1] += ny * overlap
                    vars[3*j] -= nx * overlap
                    vars[3*j+1] -= ny * overlap
                    moved = True
        
        # Keep inside bounds
        for i in range(N):
            vars[3*i] = np.clip(vars[3*i], 0.0, 1.0)
            vars[3*i+1] = np.clip(vars[3*i+1], 0.0, 1.0)
        
        if not moved:
            break
    
    return vars

def _is_valid(x, y, r, tol=1e-7):
    """Check if configuration is valid."""
    for i in range(N):
        if x[i] < r[i] - tol or x[i] + r[i] > 1.0 + tol:
            return False
        if y[i] < r[i] - tol or y[i] + r[i] > 1.0 + tol:
            return False
    for i in range(N):
        for j in range(i+1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx*dx + dy*dy)
            if d < r[i] + r[j] - tol:
                return False
    return True

def _optimize(vars0, deadline, C=500.0):
    """Run L-BFGS-B optimization."""
    # First push apart to get feasible starting point
    vars_feasible = _push_apart(vars0, num_iters=30)
    
    # Set bounds
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))  # x
        bounds.append((0.0, 1.0))  # y
        bounds.append((0.005, 0.3))  # radius
    
    # Run L-BFGS-B
    try:
        result = minimize(
            _objective,
            vars_feasible,
            jac=_gradient,
            args=(C,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if time.time() > deadline:
            return vars_feasible
        return result.x
    except Exception:
        return vars_feasible

def _polish(vars, deadline, C=2000.0):
    """High-precision polish with higher penalty."""
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
        bounds.append((0.005, 0.3))
    
    try:
        result = minimize(
            _objective,
            vars,
            jac=_gradient,
            args=(C,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        return result.x
    except Exception:
        return vars

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    best_vars = None
    best_sum = 0.0
    best_penalty = float('inf')
    
    # Multiple restarts with different initializations
    num_restarts = 10
    
    for restart in range(num_restarts):
        if time.time() > deadline - 20:
            break
        
        # Different initialization strategies
        if restart == 0:
            points, radii = _hex_init(N, offset=0.0, rng_seed=0)
        elif restart == 1:
            points, radii = _hex_init(N, offset=0.03, rng_seed=1)
        elif restart == 2:
            points, radii = _random_init(N, seed=100)
        elif restart == 3:
            points, radii = _random_init(N, seed=200)
        elif restart == 4:
            points, radii = _hex_init(N, offset=-0.02, rng_seed=2)
        elif restart == 5:
            points, radii = _random_init(N, seed=300)
        elif restart == 6:
            points, radii = _grid_init(N)
        elif restart == 7:
            points, radii = _random_init(N, seed=400)
        elif restart == 8:
            points, radii = _hex_init(N, offset=0.01, rng_seed=3)
        else:
            points, radii = _random_init(N, seed=500)
        
        # Build initial variables
        x0 = np.zeros(3 * N)
        x0[0::3] = points[:, 0]
        x0[1::3] = points[:, 1]
        x0[2::3] = radii
        
        # Optimize
        vars_opt = _optimize(x0, deadline)
        
        # Evaluate
        x_opt = vars_opt[0::3]
        y_opt = vars_opt[1::3]
        r_opt = vars_opt[2::3]
        
        # Compute penalty to check validity
        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dx = x_opt[i] - x_opt[j]
                dy = y_opt[i] - y_opt[j]
                d = np.sqrt(dx*dx + dy*dy)
                r_sum = r_opt[i] + r_opt[j]
                if d < r_sum:
                    penalty += (r_sum - d) ** 2
        for i in range(N):
            if x_opt[i] < r_opt[i]:
                penalty += (r_opt[i] - x_opt[i]) ** 2
            if x_opt[i] + r_opt[i] > 1.0:
                penalty += (x_opt[i] + r_opt[i] - 1.0) ** 2
            if y_opt[i] < r_opt[i]:
                penalty += (r_opt[i] - y_opt[i]) ** 2
            if y_opt[i] + r_opt[i] > 1.0:
                penalty += (y_opt[i] + r_opt[i] - 1.0) ** 2
        
        val = np.sum(r_opt)
        
        if penalty < best_penalty - 1e-8 or (abs(penalty - best_penalty) < 1e-8 and val > best_sum):
            best_sum = val
            best_vars = vars_opt.copy()
            best_penalty = penalty
    
    if best_vars is None:
        # Fallback to grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    # Polish the best result
    if time.time() < deadline - 10:
        best_vars = _polish(best_vars, deadline)
    
    # Final validation and repair
    x_best = best_vars[0::3]
    y_best = best_vars[1::3]
    r_best = best_vars[2::3]
    
    if not _is_valid(x_best, y_best, r_best, tol=1e-6):
        r_best = r_best * 0.98
        if not _is_valid(x_best, y_best, r_best, tol=1e-6):
            r_best = r_best * 0.95
            if not _is_valid(x_best, y_best, r_best, tol=1e-6):
                side = int(np.ceil(np.sqrt(N)))
                cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
                return np.array(cells[:N])
    
    result = np.column_stack((x_best, y_best, r_best))
    return result

# EVOLVE-BLOCK-END
