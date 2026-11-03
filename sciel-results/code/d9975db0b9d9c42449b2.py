"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, LinearConstraint
from scipy.spatial.distance import cdist

# Deadline guard: kill before 120s wall-clock with margin
DEADLINE = 115.0  # seconds
_start_time = time.time()

def _elapsed():
    return time.time() - _start_time

def _remaining():
    return DEADLINE - _elapsed()


def _hex_initial(n, seed=0):
    """Generate a hexagonal-ish initial packing for n circles in unit square."""
    rng = np.random.default_rng(seed)
    # Start with a grid that's slightly perturbed
    side = int(np.ceil(np.sqrt(n)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) >= n:
                break
            x = (i + 0.5) / side + rng.uniform(-0.02, 0.02)
            y = (j + 0.5) / side + rng.uniform(-0.02, 0.02)
            r = 0.5 / side - 1e-4
            cells.append([x, y, r])
    return np.array(cells[:n])


def _check_validity(circles, tol=1e-7):
    """Check if all circles are disjoint and inside unit square."""
    n = len(circles)
    for i in range(n):
        x, y, r = circles[i]
        if x < -tol or x > 1 + tol or y < -tol or y > 1 + tol or r < -tol:
            return False
        if x - r < -tol or x + r > 1 + tol or y - r < -tol or y + r > 1 + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, ri = circles[i]
            xj, yj, rj = circles[j]
            dist = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
            if dist < ri + rj - tol:
                return False
    return True


def _objective(params, n):
    """Objective: negative sum of radii (we minimize)."""
    circles = params.reshape(-1, 3)
    # Penalty for going outside unit square
    penalty = 0.0
    for i in range(n):
        x, y, r = circles[i]
        if x < 0: penalty += (100 * abs(x)) ** 2
        if x > 1: penalty += (100 * abs(x - 1)) ** 2
        if y < 0: penalty += (100 * abs(y)) ** 2
        if y > 1: penalty += (100 * abs(y - 1)) ** 2
        if r < 0: penalty += (100 * abs(r)) ** 2
    # Penalty for overlapping circles
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, ri = circles[i]
            xj, yj, rj = circles[j]
            dist = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
            min_dist = ri + rj
            if dist < min_dist:
                penalty += (100 * (min_dist - dist)) ** 2
    return -np.sum(circles[:, 2]) + penalty


def _grad_objective(params, n):
    """Gradient of the objective."""
    circles = params.reshape(-1, 3)
    grad = np.zeros_like(params)
    
    # Gradient of -sum(radii)
    grad[2::3] = -1.0
    
    # Gradient of boundary penalties
    for i in range(n):
        x, y, r = circles[i]
        if x < 0:
            grad[3*i] += 200 * abs(x)
        elif x > 1:
            grad[3*i] -= 200 * abs(x - 1)
        if y < 0:
            grad[3*i+1] += 200 * abs(y)
        elif y > 1:
            grad[3*i+1] -= 200 * abs(y - 1)
        if r < 0:
            grad[3*i+2] += 200 * abs(r)
    
    # Gradient of overlap penalties
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, ri = circles[i]
            xj, yj, rj = circles[j]
            dx = xi - xj
            dy = yi - yj
            dist = np.sqrt(dx**2 + dy**2)
            min_dist = ri + rj
            if dist < min_dist and dist > 1e-10:
                overlap = min_dist - dist
                factor = 200 * overlap / dist
                grad[3*i] += factor * dx
                grad[3*i+1] += factor * dy
                grad[3*i+2] += 200 * overlap
                grad[3*j] -= factor * dx
                grad[3*j+1] -= factor * dy
                grad[3*j+2] += 200 * overlap
    
    return grad


def _optimize_from_initial(init_circles, n, max_iter=500, timeout=30):
    """Run L-BFGS-B optimization from an initial configuration."""
    start_t = time.time()
    params = init_circles.flatten().copy()
    
    # Bounds: x, y in [0.01, 0.99], r in [0.005, 0.25]
    bounds = []
    for i in range(n):
        bounds.append((0.01, 0.99))  # x
        bounds.append((0.01, 0.99))  # y
        bounds.append((0.005, 0.25))  # r
    
    try:
        result = minimize(
            _objective,
            params,
            jac=_grad_objective,
            args=(n,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'maxfun': max_iter * 10, 'ftol': 1e-10, 'gtol': 1e-8}
        )
    except Exception:
        return init_circles, 0.0
    
    if time.time() - start_t > timeout:
        return init_circles, 0.0
    
    best_circles = result.x.reshape(-1, 3)
    # Clip to valid range
    best_circles[:, 0] = np.clip(best_circles[:, 0], 0.005, 0.995)
    best_circles[:, 1] = np.clip(best_circles[:, 1], 0.005, 0.995)
    best_circles[:, 2] = np.clip(best_circles[:, 2], 0.001, 0.25)
    
    score = np.sum(best_circles[:, 2])
    return best_circles, score


def construct():
    global _start_time
    _start_time = time.time()
    
    n = N
    best_circles = None
    best_score = 0.0
    
    # Multiple random restarts with different initial configurations
    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    
    for seed in seeds:
        if _remaining() < 10:
            break
        
        # Generate initial configuration
        init = _hex_initial(n, seed)
        
        # Add some randomness to break symmetry
        rng = np.random.default_rng(seed + 100)
        init[:, 0] += rng.uniform(-0.03, 0.03, n)
        init[:, 1] += rng.uniform(-0.03, 0.03, n)
        init[:, 0] = np.clip(init[:, 0], 0.02, 0.98)
        init[:, 1] = np.clip(init[:, 1], 0.02, 0.98)
        # Slightly vary radii
        init[:, 2] *= rng.uniform(0.9, 1.1, n)
        
        # Optimize
        circles, score = _optimize_from_initial(init, n, max_iter=300, timeout=8)
        
        if score > best_score:
            best_score = score
            best_circles = circles.copy()
        
        # Also try with slightly different starting radii
        init2 = init.copy()
        init2[:, 2] = 0.5 / np.sqrt(n) * rng.uniform(0.85, 1.15, n)
        circles2, score2 = _optimize_from_initial(init2, n, max_iter=300, timeout=8)
        if score2 > best_score:
            best_score = score2
            best_circles = circles2.copy()
    
    # Final refinement with more iterations on the best solution
    if best_circles is not None and _remaining() > 15:
        refined, refined_score = _optimize_from_initial(
            best_circles, n, max_iter=1000, timeout=_remaining() - 5
        )
        if refined_score > best_score:
            best_circles = refined
            best_score = refined_score
    
    if best_circles is None:
        # Fallback to grid
        side = int(np.ceil(np.sqrt(n)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        best_circles = np.array(cells[:n])
    
    return best_circles
# EVOLVE-BLOCK-END
