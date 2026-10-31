"""Circle packing in unit square: pack 26 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _validity_check(circles):
    """Check if circles are valid: inside unit square, disjoint interiors."""
    n = len(circles)
    if n != N:
        return False
    for i in range(n):
        x, y, r = circles[i]
        if r <= 0:
            return False
        if x - r < -1e-9 or x + r > 1 + 1e-9 or y - r < -1e-9 or y + r > 1 + 1e-9:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, ri = circles[i]
            xj, yj, rj = circles[j]
            dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
            if dist < ri + rj - 1e-9:
                return False
    return True

def _objective(params, n=N):
    """Objective to minimize: negative sum of radii + large penalty for violations."""
    # params: [x0, y0, r0, x1, y1, r1, ..., x_{n-1}, y_{n-1}, r_{n-1}]
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    penalty = 0.0
    # Boundary constraints
    for i in range(n):
        if x[i] - r[i] < 0:
            penalty += (x[i] - r[i])**2
        if x[i] + r[i] > 1:
            penalty += (x[i] + r[i] - 1)**2
        if y[i] - r[i] < 0:
            penalty += (y[i] - r[i])**2
        if y[i] + r[i] > 1:
            penalty += (y[i] + r[i] - 1)**2
    
    # Overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
            min_dist = r[i] + r[j]
            if dist < min_dist:
                penalty += (min_dist - dist)**2
    
    # Radius positivity
    for i in range(n):
        if r[i] < 0:
            penalty += r[i]**2
    
    return -np.sum(r) + 100.0 * penalty

def _gradient(params, n=N):
    """Gradient of the objective."""
    x = params[0::3].copy()
    y = params[1::3].copy()
    r = params[2::3].copy()
    
    grad = np.zeros(3 * n)
    
    # Objective part: -sum(r)
    for i in range(n):
        grad[2*i + 2] = -1.0
    
    # Penalty gradient
    penalty_coeff = 100.0
    
    # Boundary constraints
    for i in range(n):
        # x[i] - r[i] < 0
        if x[i] - r[i] < 0:
            val = x[i] - r[i]
            grad[3*i + 0] += penalty_coeff * 2 * val
            grad[3*i + 2] += penalty_coeff * (-2 * val)
        # x[i] + r[i] > 1
        if x[i] + r[i] > 1:
            val = x[i] + r[i] - 1
            grad[3*i + 0] += penalty_coeff * 2 * val
            grad[3*i + 2] += penalty_coeff * 2 * val
        # y[i] - r[i] < 0
        if y[i] - r[i] < 0:
            val = y[i] - r[i]
            grad[3*i + 1] += penalty_coeff * 2 * val
            grad[3*i + 2] += penalty_coeff * (-2 * val)
        # y[i] + r[i] > 1
        if y[i] + r[i] > 1:
            val = y[i] + r[i] - 1
            grad[3*i + 1] += penalty_coeff * 2 * val
            grad[3*i + 2] += penalty_coeff * 2 * val
    
    # Overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy)
            min_dist = r[i] + r[j]
            if dist < min_dist and dist > 1e-12:
                diff = min_dist - dist
                inv_dist = 1.0 / dist
                # d/dx_i of (min_dist - dist)^2 = -2*(min_dist-dist)*(dx/dist)
                grad[3*i + 0] += penalty_coeff * (-2 * diff * dx * inv_dist)
                grad[3*j + 0] += penalty_coeff * (2 * diff * dx * inv_dist)
                grad[3*i + 1] += penalty_coeff * (-2 * diff * dy * inv_dist)
                grad[3*j + 1] += penalty_coeff * (2 * diff * dy * inv_dist)
                grad[3*i + 2] += penalty_coeff * (-2 * diff)
                grad[3*j + 2] += penalty_coeff * (-2 * diff)
    
    # Radius positivity
    for i in range(n):
        if r[i] < 0:
            grad[3*i + 2] += penalty_coeff * 2 * r[i]
    
    return grad

def _force_directed_relaxation(circles, max_iters=500, lr=0.01):
    """Apply force-directed relaxation to fix overlaps and boundary violations."""
    n = len(circles)
    circles = circles.copy()
    
    for it in range(max_iters):
        forces = np.zeros((n, 3))
        
        # Boundary forces
        for i in range(n):
            x, y, r = circles[i]
            if x - r < 0:
                forces[i, 0] += (x - r)  # push right
                forces[i, 2] -= (x - r)  # shrink
            if x + r > 1:
                forces[i, 0] += (1 - x - r)  # push left
                forces[i, 2] -= (x + r - 1)  # shrink
            if y - r < 0:
                forces[i, 1] += (y - r)
                forces[i, 2] -= (y - r)
            if y + r > 1:
                forces[i, 1] += (1 - y - r)
                forces[i, 2] -= (y + r - 1)
        
        # Overlap forces
        for i in range(n):
            for j in range(i + 1, n):
                xi, yi, ri = circles[i]
                xj, yj, rj = circles[j]
                dx = xi - xj
                dy = yi - yj
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = ri + rj
                if dist < min_dist and dist > 1e-12:
                    overlap = min_dist - dist
                    inv_dist = 1.0 / dist
                    fx = overlap * dx * inv_dist
                    fy = overlap * dy * inv_dist
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy
                    # Also shrink slightly
                    shrink = overlap * 0.1
                    forces[i, 2] -= shrink
                    forces[j, 2] -= shrink
        
        # Apply forces
        for i in range(n):
            circles[i, 0] += lr * forces[i, 0]
            circles[i, 1] += lr * forces[i, 1]
            circles[i, 2] += lr * forces[i, 2]
            # Clamp
            circles[i, 0] = np.clip(circles[i, 0], 0, 1)
            circles[i, 1] = np.clip(circles[i, 1], 0, 1)
            circles[i, 2] = max(0.001, min(circles[i, 2], 0.5))
        
        # Check convergence
        if np.max(np.abs(forces)) < 1e-6:
            break
    
    return circles

def _make_valid(circles):
    """Ensure circles are valid by shrinking radii if needed."""
    n = len(circles)
    circles = circles.copy()
    
    # First, fix boundary issues
    for i in range(n):
        x, y, r = circles[i]
        max_r = min(x, 1 - x, y, 1 - y)
        if r > max_r:
            circles[i, 2] = max_r * 0.99
    
    # Iteratively fix overlaps
    for _ in range(100):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                xi, yi, ri = circles[i]
                xj, yj, rj = circles[j]
                dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
                if dist < ri + rj and dist > 1e-12:
                    # Shrink both radii
                    overlap = (ri + rj - dist) / 2
                    circles[i, 2] = max(0.001, ri - overlap)
                    circles[j, 2] = max(0.001, rj - overlap)
                    changed = True
        if not changed:
            break
    
    # Final boundary check
    for i in range(n):
        x, y, r = circles[i]
        max_r = min(x, 1 - x, y, 1 - y)
        if r > max_r:
            circles[i, 2] = max_r * 0.99
    
    return circles

def _random_initialization(seed=None):
    """Create a random valid initial configuration."""
    if seed is not None:
        np.random.seed(seed)
    
    # Start with a grid-like arrangement and add randomness
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) >= N:
                break
            x = (i + 0.5) / side + np.random.uniform(-0.1/side, 0.1/side)
            y = (j + 0.5) / side + np.random.uniform(-0.1/side, 0.1/side)
            r = 0.5 / side * np.random.uniform(0.8, 1.0)
            cells.append([x, y, r])
    
    circles = np.array(cells[:N])
    circles = _force_directed_relaxation(circles, max_iters=200, lr=0.05)
    circles = _make_valid(circles)
    return circles

def construct():
    start_time = time.time()
    deadline = start_time + 120  # Safety margin
    
    best_circles = None
    best_value = -np.inf
    
    # Multiple restarts with different strategies
    n_restarts = 20
    
    for restart in range(n_restarts):
        if time.time() > deadline - 5:
            break
        
        # Different initialization strategies
        if restart == 0:
            # Grid-based
            side = int(np.ceil(np.sqrt(N)))
            cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                     for i in range(side) for j in range(side)]
            circles = np.array(cells[:N])
        elif restart < 5:
            # Random with grid bias
            circles = _random_initialization(seed=restart * 12345)
        elif restart < 10:
            # More random
            np.random.seed(restart * 9999)
            x = np.random.uniform(0.1, 0.9, N)
            y = np.random.uniform(0.1, 0.9, N)
            r = np.random.uniform(0.02, 0.08, N)
            circles = np.column_stack([x, y, r])
            circles = _force_directed_relaxation(circles, max_iters=300, lr=0.03)
            circles = _make_valid(circles)
        else:
            # Mix of strategies
            seed = restart * 7777
            circles = _random_initialization(seed=seed)
        
        if not _validity_check(circles):
            circles = _make_valid(circles)
        
        # Optimize using L-BFGS-B
        params = circles.flatten()
        
        try:
            result = minimize(
                _objective,
                params,
                method='L-BFGS-B',
                jac=_gradient,
                options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < best_value:
                opt_circles = result.x.reshape(-1, 3)
                opt_circles = _force_directed_relaxation(opt_circles, max_iters=100, lr=0.01)
                opt_circles = _make_valid(opt_circles)
                
                if _validity_check(opt_circles):
                    value = np.sum(opt_circles[:, 2])
                    if value > best_value:
                        best_value = value
                        best_circles = opt_circles.copy()
        except Exception:
            pass
        
        # Also try force-directed relaxation as a separate approach
        relaxed = _force_directed_relaxation(circles.copy(), max_iters=500, lr=0.02)
        relaxed = _make_valid(relaxed)
        if _validity_check(relaxed):
            value = np.sum(relaxed[:, 2])
            if value > best_value:
                best_value = value
                best_circles = relaxed.copy()
    
    # Final fallback: ensure we return something valid
    if best_circles is None or not _validity_check(best_circles):
        # Simple valid solution: small circles in a grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.4 / side) 
                 for i in range(side) for j in range(side)]
        best_circles = np.array(cells[:N])
        best_circles = _make_valid(best_circles)
    
    return best_circles
# EVOLVE-BLOCK-END
