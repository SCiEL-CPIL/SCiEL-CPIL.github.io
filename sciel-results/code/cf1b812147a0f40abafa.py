"""Circle packing in unit square - 26 circles maximizing sum of radii.
Strategy: Use scipy.optimize.minimize with SLSQP and proper constraints,
with multiple initializations and local search refinement.
"""

import numpy as np
import time
import random
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

def _is_valid(circles, tol=1e-8):
    """Check validity: inside unit square, non-overlapping, positive radii."""
    n = len(circles)
    for i in range(n):
        x, y, r = circles[i]
        if r < -tol:
            return False
        if x - r < -tol or x + r > 1.0 + tol:
            return False
        if y - r < -tol or y + r > 1.0 + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, ri = circles[i]
            xj, yj, rj = circles[j]
            dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
            if dist < ri + rj - tol:
                return False
    return True


def _post_process(circles, tol=1e-8):
    """Shrink circles slightly to ensure strict validity, clip to bounds."""
    out = []
    for x, y, r in circles:
        r = max(0.0, r - tol)
        x = min(1.0 - r, max(r, x))
        y = min(1.0 - r, max(r, y))
        out.append((x, y, r))
    return out


def _sum_radii(circles):
    return sum(c[2] for c in circles)


def _init_hexagonal():
    """Hexagonal packing initial layout for 26 circles."""
    circles = []
    r = 0.075
    rows = [5, 5, 5, 5, 6]
    row_spacing = r * np.sqrt(3)
    col_spacing = 2 * r
    
    row_idx = 0
    for count in rows:
        offset = 0 if row_idx % 2 == 0 else col_spacing / 2
        total_width = col_spacing * (count - 1) + offset
        start_x = (1.0 - total_width) / 2.0
        y = r + row_spacing * row_idx + 0.03
        for col in range(count):
            x = start_x + offset + col * col_spacing
            if len(circles) < N:
                circles.append((x, y, r))
        row_idx += 1
    
    circles = circles[:N]
    while len(circles) < N:
        circles.append((0.5, 0.5, 0.01))
    return circles


def _init_grid():
    """Simple grid initialization."""
    side = int(np.ceil(np.sqrt(N)))
    circles = []
    for k in range(N):
        i = k % side
        j = k // side
        x = (i + 0.5) / side
        y = (j + 0.5) / side
        r = 0.4 / side
        circles.append((x, y, r))
    return circles


def _init_corners():
    """Initialization with circles at corners and edges."""
    circles = []
    # Corners
    circles.append((0.08, 0.08, 0.08))
    circles.append((0.92, 0.08, 0.08))
    circles.append((0.08, 0.92, 0.08))
    circles.append((0.92, 0.92, 0.08))
    
    # Edges
    circles.append((0.5, 0.05, 0.06))
    circles.append((0.5, 0.95, 0.06))
    circles.append((0.05, 0.5, 0.06))
    circles.append((0.95, 0.5, 0.06))
    
    # Fill rest randomly
    random.seed(42)
    while len(circles) < N:
        x = random.uniform(0.15, 0.85)
        y = random.uniform(0.15, 0.85)
        r = random.uniform(0.04, 0.07)
        circles.append((x, y, r))
    return circles


def _init_random(seed=None):
    """Random initialization."""
    if seed is not None:
        random.seed(seed)
    circles = []
    for _ in range(N):
        x = random.uniform(0.1, 0.9)
        y = random.uniform(0.1, 0.9)
        r = random.uniform(0.03, 0.08)
        circles.append((x, y, r))
    return circles


def _slsqp_solve(params, bounds, constraints, maxiter=200):
    """Solve SLSQP optimization problem."""
    def objective(p):
        return -sum(p[3*i+2] for i in range(N))
    
    def jac(p):
        g = np.zeros(3*N)
        g[2::3] = -1.0
        return g
    
    try:
        result = minimize(
            objective, params, jac=jac,
            method='SLSQP', bounds=bounds, constraints=constraints,
            options={'maxiter': maxiter, 'ftol': 1e-10}
        )
        return result.x
    except Exception:
        return params


def _local_search(circles, deadline, max_iters=50):
    """Local search using SLSQP with pairwise and boundary constraints."""
    params = np.array([val for c in circles for val in c])
    
    # Bounds
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.001, 0.5)] * N
    
    # Constraints
    constraints = []
    
    # Boundary constraints: x >= r, x <= 1-r, y >= r, y <= 1-r
    for i in range(N):
        # x - r >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda p, i=i: p[3*i] - p[3*i+2],
            'jac': lambda p, i=i: np.array([1.0 if k==3*i else (-1.0 if k==3*i+2 else 0.0) for k in range(3*N)])
        })
        # 1 - x - r >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda p, i=i: 1.0 - p[3*i] - p[3*i+2],
            'jac': lambda p, i=i: np.array([(-1.0 if k==3*i else (-1.0 if k==3*i+2 else 0.0)) for k in range(3*N)])
        })
        # y - r >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda p, i=i: p[3*i+1] - p[3*i+2],
            'jac': lambda p, i=i: np.array([1.0 if k==3*i+1 else (-1.0 if k==3*i+2 else 0.0) for k in range(3*N)])
        })
        # 1 - y - r >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda p, i=i: 1.0 - p[3*i+1] - p[3*i+2],
            'jac': lambda p, i=i: np.array([(-1.0 if k==3*i+1 else (-1.0 if k==3*i+2 else 0.0)) for k in range(3*N)])
        })
    
    # Overlap constraints: dist(i,j) >= r_i + r_j
    # Equivalent to: (x_i-x_j)^2 + (y_i-y_j)^2 >= (r_i+r_j)^2
    # As inequality: (x_i-x_j)^2 + (y_i-y_j)^2 - (r_i+r_j)^2 >= 0
    for i in range(N):
        for j in range(i+1, N):
            def overlap_fun(p, i=i, j=j):
                dx = p[3*i] - p[3*j]
                dy = p[3*i+1] - p[3*j+1]
                return dx*dx + dy*dy - (p[3*i+2] + p[3*j+2])**2
            
            def overlap_jac(p, i=i, j=j):
                jac = np.zeros(3*N)
                dx = p[3*i] - p[3*j]
                dy = p[3*i+1] - p[3*j+1]
                sum_r = p[3*i+2] + p[3*j+2]
                jac[3*i] = 2*dx
                jac[3*j] = -2*dx
                jac[3*i+1] = 2*dy
                jac[3*j+1] = -2*dy
                jac[3*i+2] = -2*sum_r
                jac[3*j+2] = -2*sum_r
                return jac
            
            constraints.append({'type': 'ineq', 'fun': overlap_fun, 'jac': overlap_jac})
    
    # Optimize
    new_params = _slsqp_solve(params, bounds, constraints)
    
    # Convert back to circles
    new_circles = [(new_params[3*i], new_params[3*i+1], new_params[3*i+2]) for i in range(N)]
    
    return new_circles


def _simulated_annealing(circles, deadline, max_iters=200):
    """Simulated annealing for global optimization."""
    if time.time() > deadline - 10.0:
        return circles
    
    current = list(circles)
    best = list(circles)
    best_sum = _sum_radii(best)
    
    temp = 0.03
    cooling = 0.95
    
    for it in range(max_iters):
        if time.time() > deadline - 10.0:
            break
        
        # Generate perturbation
        perturbed = []
        for x, y, r in current:
            nx = x + random.gauss(0, temp * 0.5)
            ny = y + random.gauss(0, temp * 0.5)
            nr = r + random.gauss(0, temp * 0.3)
            nr = max(0.01, min(0.4, nr))
            nx = max(nr, min(1.0 - nr, nx))
            ny = max(nr, min(1.0 - nr, ny))
            perturbed.append((nx, ny, nr))
        
        # Local refine
        refined = _local_search(perturbed, deadline)
        new_sum = _sum_radii(refined)
        
        if new_sum > best_sum:
            best = refined
            best_sum = new_sum
            current = refined
        else:
            delta = _sum_radii(current) - new_sum
            if delta < 1e-8:
                continue
            prob = np.exp(-delta / temp)
            if random.random() < prob:
                current = refined
        
        temp *= cooling
        if temp < 0.001:
            break
    
    return best


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Multiple diverse initializations
    inits = [
        _init_hexagonal(),
        _init_grid(),
        _init_corners(),
        _init_random(42),
        _init_random(123),
        _init_random(777),
        _init_random(999),
        _init_random(1234),
    ]
    
    best_circles = inits[0]
    best_sum = _sum_radii(best_circles)
    
    # Local search on each initialization
    for init_circles in inits:
        if time.time() > deadline - 30.0:
            break
        
        # Local refine
        refined = _local_search(init_circles, deadline)
        refined_sum = _sum_radii(refined)
        
        if refined_sum > best_sum and _is_valid(refined):
            best_circles = refined
            best_sum = refined_sum
    
    # Simulated annealing for global exploration
    if time.time() < deadline - 20.0:
        sa_result = _simulated_annealing(best_circles, deadline)
        sa_sum = _sum_radii(sa_result)
        
        if sa_sum > best_sum and _is_valid(sa_result):
            best_circles = sa_result
            best_sum = sa_sum
    
    # Final polish
    if time.time() < deadline - 5.0:
        final = _local_search(best_circles, deadline)
        final_sum = _sum_radii(final)
        
        if final_sum > best_sum and _is_valid(final):
            best_circles = final
            best_sum = final_sum
    
    # Post-process
    best_circles = _post_process(best_circles)
    
    # Verify and return
    if not _is_valid(best_circles):
        # Fallback
        best_circles = _post_process(_init_grid())
    
    return np.array(best_circles)

# EVOLVE-BLOCK-END
