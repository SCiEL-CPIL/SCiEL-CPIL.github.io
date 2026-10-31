"""Circle packing in unit square: maximize sum of radii for 26 disjoint circles."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import cdist
import time
import itertools

N = 26
DEADLINE = 150.0
SAFETY_MARGIN = 8.0


def _is_valid(centers, radii, tol=1e-7):
    """Check if circles are valid (inside square, disjoint interiors)."""
    n = len(radii)
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if r < -tol:
            return False
        if x - r < -tol or x + r > 1.0 + tol:
            return False
        if y - r < -tol or y + r > 1.0 + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < radii[i] + radii[j] - tol:
                return False
    return True


def _objective(params):
    """Objective function: minimize negative sum of radii with penalty for violations.
    
    params: (2N + N) = 3N values: centers (2N) + radii (N)
    """
    n = N
    centers = params[:2 * n].reshape(n, 2)
    radii = params[2 * n:]
    
    # Penalty for boundary violations
    penalty = 0.0
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        # Boundary constraints
        if x - r < 0:
            penalty += (x - r) ** 2 * 100
        if x + r > 1:
            penalty += (x + r - 1) ** 2 * 100
        if y - r < 0:
            penalty += (y - r) ** 2 * 100
        if y + r > 1:
            penalty += (y + r - 1) ** 2 * 100
        # Radius non-negativity
        if r < 0:
            penalty += r ** 2 * 100
    
    # Penalty for overlap violations
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            violation = radii[i] + radii[j] - d
            if violation > 0:
                penalty += violation ** 2 * 100
    
    return -np.sum(radii) + penalty


def _objective_smooth(params):
    """Smooth objective using log-sum-exp for soft constraints."""
    n = N
    centers = params[:2 * n].reshape(n, 2)
    radii = params[2 * n:]
    
    # Soft boundary violations
    penalty = 0.0
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        # Use softplus-like penalty
        for val in [x - r, -(x + r - 1), y - r, -(y + r - 1), -r]:
            if val < 0:
                penalty += -val * 10
        penalty += np.sum(np.logaddexp(0, -np.array([x - r, 1 - x - r, y - r, 1 - y - r, r]) * 50) / 50) * 10
    
    # Soft overlap violations
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            violation = radii[i] + radii[j] - d
            if violation > 0:
                penalty += violation * 10
            penalty += np.logaddexp(0, violation * 50) / 50 * 10
    
    return -np.sum(radii) + penalty


def _initial_grid():
    """Generate a grid of centers with initial radii."""
    side = int(np.ceil(np.sqrt(N)))
    centers = []
    for i in range(side):
        for j in range(side):
            if len(centers) >= N:
                break
            centers.append(((i + 0.5) / side, (j + 0.5) / side))
    centers = np.array(centers[:N])
    radii = np.full(N, 0.5 / side * 0.9)
    params = np.concatenate([centers.flatten(), radii])
    return params


def _initial_hexagonal():
    """Generate hexagonal packing initial guess."""
    # Hexagonal arrangement for better packing
    rows = 5
    cols = 6
    centers = []
    for i in range(rows):
        for j in range(cols):
            if len(centers) >= N:
                break
            x = (j + 0.5) / cols + (0.5 / cols if i % 2 == 1 else 0)
            y = (i + 0.5) / rows
            x = min(max(x, 0.05), 0.95)
            y = min(max(y, 0.05), 0.95)
            centers.append((x, y))
    centers = np.array(centers[:N])
    radii = np.full(N, 0.1)
    params = np.concatenate([centers.flatten(), radii])
    return params


def _initial_corner_heavy():
    """Initial guess with circles near corners and edges."""
    centers = []
    # Corners
    for cx, cy in [(0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9)]:
        centers.append((cx, cy))
    # Edge midpoints
    for cx, cy in [(0.5, 0.1), (0.5, 0.9), (0.1, 0.5), (0.9, 0.5)]:
        centers.append((cx, cy))
    # Fill remaining with grid
    side = int(np.ceil(np.sqrt(N - 8)))
    for i in range(side):
        for j in range(side):
            if len(centers) >= N:
                break
            x = 0.25 + 0.5 * (i + 0.5) / side
            y = 0.25 + 0.5 * (j + 0.5) / side
            centers.append((x, y))
    centers = np.array(centers[:N])
    radii = np.full(N, 0.08)
    params = np.concatenate([centers.flatten(), radii])
    return params


def _initial_random(seed=42):
    """Generate random initial guess."""
    np.random.seed(seed)
    centers = np.random.uniform(0.1, 0.9, size=(N, 2))
    radii = np.random.uniform(0.05, 0.12, size=N)
    params = np.concatenate([centers.flatten(), radii])
    return params


def _post_process(centers, radii, tol=1e-9):
    """Post-process to ensure validity by shrinking overlapping circles."""
    centers = centers.copy()
    radii = radii.copy()
    
    # Multiple passes to resolve overlaps
    for _ in range(50):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(centers[i] - centers[j])
                required = radii[i] + radii[j]
                if d < required - tol:
                    # Shrink both circles proportionally
                    total_r = radii[i] + radii[j]
                    if total_r < 1e-10:
                        continue
                    scale = d / total_r
                    radii[i] *= scale
                    radii[j] *= scale
                    changed = True
        
        # Also ensure boundary constraints
        for i in range(N):
            x, y = centers[i]
            max_r = min(x, 1 - x, y, 1 - y)
            if radii[i] > max_r:
                radii[i] = max(0, max_r)
                changed = True
        
        if not changed:
            break
    
    return centers, radii


def _grow_and_push(centers, radii, max_iters=200):
    """Iteratively grow circles and push centers to resolve conflicts."""
    centers = centers.copy()
    radii = radii.copy()
    
    for it in range(max_iters):
        # Try to grow all radii slightly
        grow_factor = 1.0 + 0.005
        new_radii = radii * grow_factor
        
        # Check constraints and adjust
        valid = True
        for i in range(N):
            x, y = centers[i]
            r = new_radii[i]
            # Boundary check
            max_r = min(x, 1 - x, y, 1 - y)
            if r > max_r:
                new_radii[i] = max_r
                valid = False
        
        # Overlap check
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(centers[i] - centers[j])
                if new_radii[i] + new_radii[j] > d + 1e-9:
                    valid = False
                    # Push centers apart
                    diff = centers[i] - centers[j]
                    dist = np.linalg.norm(diff)
                    if dist < 1e-10:
                        diff = np.array([1e-5, 1e-5])
                        dist = np.linalg.norm(diff)
                    unit = diff / dist
                    overlap = new_radii[i] + new_radii[j] - d
                    push = overlap * 0.5
                    centers[i] += unit * push
                    centers[j] -= unit * push
        
        if valid:
            radii = new_radii
        else:
            # Keep old radii, centers may have been updated
            # Re-check boundary for centers
            centers = np.clip(centers, 0.01, 0.99)
    
    return centers, radii


def _polish(params, method='L-BFGS-B', maxiter=500):
    """Polish a solution using local optimization."""
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))  # x_i
        bounds.append((0.0, 1.0))  # y_i
    for i in range(N):
        bounds.append((0.0, 0.5))  # r_i
    
    try:
        result = minimize(
            _objective_smooth,
            params,
            method=method,
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if result.success or result.fun < _objective_smooth(params):
            return result.x
    except Exception:
        pass
    
    return params


def _optimize_from(params):
    """Optimize from an initial parameter set."""
    # First, post-process to get a valid starting point
    centers = params[:2 * N].reshape(N, 2)
    radii = params[2 * N:]
    centers, radii = _post_process(centers, radii)
    params = np.concatenate([centers.flatten(), radii])
    
    # Grow and push
    centers, radii = _grow_and_push(centers, radii, max_iters=100)
    params = np.concatenate([centers.flatten(), radii])
    
    # Polish with L-BFGS-B
    params = _polish(params, method='L-BFGS-B', maxiter=300)
    
    # Final post-process
    centers = params[:2 * N].reshape(N, 2)
    radii = params[2 * N:]
    centers, radii = _post_process(centers, radii)
    
    return centers, radii


def _differential_evolution_run(start_time, deadline):
    """Run differential evolution for global optimization."""
    bounds = []
    for i in range(N):
        bounds.append((0.05, 0.95))  # x_i
        bounds.append((0.05, 0.95))  # y_i
    for i in range(N):
        bounds.append((0.01, 0.25))  # r_i
    
    # Use a smaller population and fewer iterations to fit in time
    popsize = 15
    maxiter = 50
    
    def de_objective(params):
        if time.time() > deadline:
            return 1e6
        return _objective(params)
    
    try:
        result = differential_evolution(
            de_objective,
            bounds,
            maxiter=maxiter,
            popsize=popsize,
            seed=42,
            tol=1e-8,
            polish=False,  # We'll do our own polish
            init='sobol'
        )
        if result.success:
            centers = result.x[:2 * N].reshape(N, 2)
            radii = result.x[2 * N:]
            centers, radii = _post_process(centers, radii)
            return centers, radii
    except Exception:
        pass
    
    return None


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY_MARGIN
    
    best_score = -np.inf
    best_centers = None
    best_radii = None
    
    def _update_best(centers, radii):
        nonlocal best_score, best_centers, best_radii
        if not _is_valid(centers, radii):
            return
        score = np.sum(radii)
        if score > best_score:
            best_score = score
            best_centers = centers.copy()
            best_radii = radii.copy()
    
    # Strategy 1: Multiple initial guesses with local optimization
    initial_guesses = [
        _initial_grid(),
        _initial_hexagonal(),
        _initial_corner_heavy(),
        _initial_random(42),
        _initial_random(123),
        _initial_random(456),
    ]
    
    for init_params in initial_guesses:
        if time.time() > deadline:
            break
        
        centers, radii = _optimize_from(init_params)
        _update_best(centers, radii)
    
    # Strategy 2: Differential evolution if time permits
    if time.time() < deadline - 30:
        de_result = _differential_evolution_run(start_time, deadline)
        if de_result is not None:
            centers, radii = de_result
            _update_best(centers, radii)
    
    # Strategy 3: Perturbation-based multi-restart polish
    if best_centers is not None and time.time() < deadline - 10:
        base_centers = best_centers.copy()
        base_radii = best_radii.copy()
        
        for trial in range(20):
            if time.time() > deadline:
                break
            
            # Add small perturbations
            np.random.seed(1000 + trial)
            perturbed_centers = base_centers + np.random.normal(0, 0.02, size=(N, 2))
            perturbed_centers = np.clip(perturbed_centers, 0.01, 0.99)
            perturbed_radii = np.maximum(base_radii * (1 + np.random.normal(0, 0.1, size=N)), 0.01)
            
            params = np.concatenate([perturbed_centers.flatten(), perturbed_radii])
            centers, radii = _optimize_from(params)
            _update_best(centers, radii)
    
    # Final fallback
    if best_centers is None:
        side = int(np.ceil(np.sqrt(N)))
        centers = np.array([((i + 0.5) / side, (j + 0.5) / side) 
                           for i in range(side) for j in range(side)][:N])
        radii = np.full(N, 0.5 / side * 0.8)
        centers, radii = _post_process(centers, radii)
        best_centers = centers
        best_radii = radii
    
    # Final validation and adjustment
    best_centers, best_radii = _post_process(best_centers, best_radii)
    
    result = np.column_stack([best_centers, best_radii])
    return result
