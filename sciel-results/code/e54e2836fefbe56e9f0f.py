"""Circle packing solver for n=32 in unit square using L-BFGS-B with smooth surrogate."""
N = 32

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, linprog

def _hex_init(n, offset=0.0, rng_seed=0):
    """Generate hexagonal packing initialization for n circles."""
    rng = np.random.RandomState(rng_seed)
    points = []
    for q in range(-15, 16):
        for r in range(-15, 16):
            x = 1.5 * q
            y = np.sqrt(3) * (r + 0.5 * (q & 1))
            x += rng.uniform(-0.05, 0.05)
            y += rng.uniform(-0.05, 0.05)
            points.append([x + offset, y + offset])
    points = np.array(points)
    
    center = np.array([0.5, 0.5])
    points += (center - np.mean(points, axis=0))
    
    dists = np.sum((points - center)**2, axis=1)
    sorted_idx = np.argsort(dists)
    init_points = points[sorted_idx[:n]]
    
    # Estimate radii based on local spacing
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
        x = rng.uniform(0.08, 0.92)
        y = rng.uniform(0.08, 0.92)
        r = rng.uniform(0.05, 0.12)
        
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
            if len(points) >= n:
                break
            px = (i + 0.5) / side
            py = (j + 0.5) / side
            points.append([px, py])
            radii.append(0.5 / side - 0.005)
    return np.array(points), np.array(radii)

def _objective_and_grad(vars, alpha=50.0):
    """
    Objective: maximize sum of radii.
    Use smooth surrogate for constraints:
    - Pairwise: soft penalty for overlap using smooth max
    - Boundary: soft penalty for going out of bounds
    """
    x = vars[0::3]
    y = vars[1::3]
    r = vars[2::3]
    
    n = len(r)
    obj = -np.sum(r)
    grad = np.zeros(3 * n)
    grad[2::3] = -1.0
    
    # Pairwise smooth overlap penalty
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d_sq = dx*dx + dy*dy
            d = np.sqrt(d_sq + 1e-12)
            r_sum = r[i] + r[j]
            
            # Smooth max(0, r_sum - d)
            # Use softplus-like: (log(1 + exp(alpha * (r_sum - d))) / alpha)
            z = alpha * (r_sum - d)
            if z > 50:
                smooth_max = z / alpha
                dsd = 1.0
                drs = 1.0
            elif z < -50:
                smooth_max = 0.0
                dsd = 0.0
                drs = 0.0
            else:
                smooth_max = np.log1p(np.exp(z)) / alpha
                dsd = 1.0 / (1.0 + np.exp(-z))
                drs = 1.0 / (1.0 + np.exp(-z))
            
            penalty = 100.0 * smooth_max
            obj += penalty
            
            # Gradient of penalty w.r.t. x[i], y[i], r[i], x[j], y[j], r[j]
            # d(penalty)/d(x[i]) = 100 * dsd * (-dx/d)
            # d(penalty)/d(r[i]) = 100 * (drs * 1 - dsd * (-1)) ... wait, let me redo
            
            # penalty = 100 * f(r_sum - d) where f(z) = smooth_max
            # d(penalty)/d(x[i]) = 100 * f'(r_sum - d) * (-dx/d)
            # d(penalty)/d(r[i]) = 100 * f'(r_sum - d) * (1)
            
            if d < 1e-10:
                inv_d = 0.0
            else:
                inv_d = 1.0 / d
            
            coeff = 100.0 * dsd
            grad[3*i + 0] += coeff * (-dx * inv_d)
            grad[3*i + 1] += coeff * (-dy * inv_d)
            grad[3*i + 2] += coeff * 1.0
            
            grad[3*j + 0] += coeff * (dx * inv_d)
            grad[3*j + 1] += coeff * (dy * inv_d)
            grad[3*j + 2] += coeff * 1.0
    
    # Boundary penalties
    for i in range(n):
        # Left: x[i] < r[i]
        z = alpha * (r[i] - x[i])
        if z > 50:
            sm = z / alpha
            ds = 1.0
        elif z < -50:
            sm = 0.0
            ds = 0.0
        else:
            sm = np.log1p(np.exp(z)) / alpha
            ds = 1.0 / (1.0 + np.exp(-z))
        
        obj += 100.0 * sm
        grad[3*i + 0] += 100.0 * ds * (-1.0)
        grad[3*i + 2] += 100.0 * ds * 1.0
        
        # Right: x[i] + r[i] > 1
        z = alpha * (x[i] + r[i] - 1.0)
        if z > 50:
            sm = z / alpha
            ds = 1.0
        elif z < -50:
            sm = 0.0
            ds = 0.0
        else:
            sm = np.log1p(np.exp(z)) / alpha
            ds = 1.0 / (1.0 + np.exp(-z))
        
        obj += 100.0 * sm
        grad[3*i + 0] += 100.0 * ds * 1.0
        grad[3*i + 2] += 100.0 * ds * 1.0
        
        # Bottom: y[i] < r[i]
        z = alpha * (r[i] - y[i])
        if z > 50:
            sm = z / alpha
            ds = 1.0
        elif z < -50:
            sm = 0.0
            ds = 0.0
        else:
            sm = np.log1p(np.exp(z)) / alpha
            ds = 1.0 / (1.0 + np.exp(-z))
        
        obj += 100.0 * sm
        grad[3*i + 1] += 100.0 * ds * (-1.0)
        grad[3*i + 2] += 100.0 * ds * 1.0
        
        # Top: y[i] + r[i] > 1
        z = alpha * (y[i] + r[i] - 1.0)
        if z > 50:
            sm = z / alpha
            ds = 1.0
        elif z < -50:
            sm = 0.0
            ds = 0.0
        else:
            sm = np.log1p(np.exp(z)) / alpha
            ds = 1.0 / (1.0 + np.exp(-z))
        
        obj += 100.0 * sm
        grad[3*i + 1] += 100.0 * ds * 1.0
        grad[3*i + 2] += 100.0 * ds * 1.0
    
    return obj, grad

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

def _linprog_init(points, radii, deadline):
    """
    Use linear programming to get a better initial estimate of radii.
    Linearize separation constraints at current positions.
    """
    if time.time() > deadline:
        return radii.copy()
    
    x = points[:, 0]
    y = points[:, 1]
    r = radii.copy()
    
    # Variables: r[0], r[1], ..., r[n-1]
    # Maximize: sum(r)
    # Constraints:
    #   r[i] + r[j] <= d_ij - epsilon for all pairs
    #   r[i] <= x[i], r[i] <= 1-x[i], r[i] <= y[i], r[i] <= 1-y[i]
    #   r[i] >= 0
    
    n = N
    c = -np.ones(n)  # minimize -sum(r)
    
    A_ub = []
    b_ub = []
    
    # Pairwise constraints
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx*dx + dy*dy)
            # r[i] + r[j] <= d - small margin
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(max(d - 0.001, 0.01))
    
    # Boundary constraints
    for i in range(n):
        for bound_val in [x[i], 1.0 - x[i], y[i], 1.0 - y[i]]:
            row = np.zeros(n)
            row[i] = 1.0
            A_ub.append(row)
            b_ub.append(max(bound_val - 0.001, 0.01))
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    bounds = [(0.01, 0.5)] * n
    
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if result.success:
            r_new = result.x
            # Clip to reasonable range
            r_new = np.clip(r_new, 0.01, 0.5)
            return r_new
    except:
        pass
    
    return r

def _optimize(vars0, deadline, max_iter=2000):
    """Run L-BFGS-B optimization."""
    if time.time() > deadline:
        return vars0.copy()
    
    best_vars = vars0.copy()
    best_obj = _objective_and_grad(vars0)[0]
    
    for attempt in range(3):
        if time.time() > deadline:
            break
        
        try:
            result = minimize(
                _objective_and_grad,
                vars0,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': max_iter, 'maxfun': 5000, 'maxls': 20}
            )
            
            if result.success or result.fun < best_obj:
                if result.fun < best_obj:
                    best_obj = result.fun
                    best_vars = result.x.copy()
            
            # Use result as starting point for next attempt
            vars0 = result.x.copy()
        except:
            break
    
    return best_vars

def _polish(vars, deadline):
    """Polish with tighter constraints."""
    if time.time() > deadline:
        return vars
    
    # Increase alpha for tighter constraint enforcement
    global _alpha_override
    _alpha_override = 100.0
    
    try:
        result = minimize(
            _objective_and_grad,
            vars,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 1000, 'maxfun': 3000, 'maxls': 20}
        )
        if result.success:
            return result.x
    except:
        pass
    
    return vars

_alpha_override = 50.0

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    best_vars = None
    best_sum = 0.0
    
    # Multiple initializations
    inits = []
    
    # Hexagonal
    pts, rad = _hex_init(N, offset=0.0, rng_seed=0)
    inits.append((pts, rad))
    
    pts, rad = _hex_init(N, offset=0.03, rng_seed=1)
    inits.append((pts, rad))
    
    pts, rad = _hex_init(N, offset=-0.02, rng_seed=2)
    inits.append((pts, rad))
    
    # Random
    pts, rad = _random_init(N, seed=100)
    inits.append((pts, rad))
    
    pts, rad = _random_init(N, seed=200)
    inits.append((pts, rad))
    
    pts, rad = _random_init(N, seed=300)
    inits.append((pts, rad))
    
    # Grid
    pts, rad = _grid_init(N)
    inits.append((pts, rad))
    
    for idx, (points, radii) in enumerate(inits):
        if time.time() > deadline:
            break
        
        # Try to improve radii with LP
        if time.time() < deadline - 20:
            radii_lp = _linprog_init(points, radii, deadline)
            if np.sum(radii_lp) > np.sum(radii):
                radii = radii_lp
        
        # Build initial variables
        x0 = np.zeros(3 * N)
        x0[0::3] = points[:, 0]
        x0[1::3] = points[:, 1]
        x0[2::3] = radii
        
        # Optimize
        vars_opt = _optimize(x0, deadline)
        
        x_opt = vars_opt[0::3]
        y_opt = vars_opt[1::3]
        r_opt = vars_opt[2::3]
        
        sum_r = np.sum(r_opt)
        
        if sum_r > best_sum:
            best_sum = sum_r
            best_vars = vars_opt.copy()
    
    if best_vars is None:
        # Fallback to grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    # Polish
    if time.time() < deadline - 10:
        best_vars = _polish(best_vars, deadline)
    
    x_best = best_vars[0::3]
    y_best = best_vars[1::3]
    r_best = best_vars[2::3]
    
    # Validate and repair
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
