"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 32

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

_DEADLINE = 115.0  # safety margin before 120s kill


def _pairwise_distances(centers):
    """Compute pairwise distances between centers (n, n) matrix."""
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    return np.sqrt(np.sum(diff**2, axis=-1))


def _objective_and_grad(params, n):
    """
    Objective: minimize -sum(radii) subject to non-overlap and boundary constraints.
    params: array of shape (2n, n) where first 2n are centers, last n are radii.
    Actually let's use a flat vector of length 3n: [x0, y0, r0, x1, y1, r1, ...]
    """
    centers = params[:2*n].reshape(n, 2)
    radii = params[2*n:]
    
    # Boundary constraints: x_i >= r_i, x_i <= 1 - r_i, y_i >= r_i, y_i <= 1 - r_i
    # Non-overlap: dist(i,j) >= r_i + r_j
    
    # We'll use a penalty approach with smooth min
    penalty = 0.0
    grad = np.zeros_like(params)
    
    # Boundary penalties (smooth)
    for i in range(n):
        x, y, r = centers[i, 0], centers[i, 1], radii[i]
        # x >= r
        viol = r - x
        if viol > 0:
            penalty += viol**2
            grad[2*i] -= 2*viol
            grad[2*n+i] += 2*viol
        # x <= 1 - r
        viol = x + r - 1
        if viol > 0:
            penalty += viol**2
            grad[2*i] += 2*viol
            grad[2*n+i] += 2*viol
        # y >= r
        viol = r - y
        if viol > 0:
            penalty += viol**2
            grad[2*i+1] -= 2*viol
            grad[2*n+i] += 2*viol
        # y <= 1 - r
        viol = y + r - 1
        if viol > 0:
            penalty += viol**2
            grad[2*i+1] += 2*viol
            grad[2*n+i] += 2*viol
    
    # Non-overlap penalties
    for i in range(n):
        for j in range(i+1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            d = np.sqrt(dx*dx + dy*dy) + 1e-12
            required = radii[i] + radii[j]
            viol = required - d
            if viol > 0:
                penalty += viol**2
                # Gradient w.r.t. centers
                grad_factor = 2*viol / d
                grad[2*i] += grad_factor * dx
                grad[2*i+1] += grad_factor * dy
                grad[2*j] -= grad_factor * dx
                grad[2*j+1] -= grad_factor * dy
                # Gradient w.r.t. radii
                grad[2*n+i] += 2*viol
                grad[2*n+j] += 2*viol
    
    obj = -np.sum(radii) + penalty
    return obj, grad


def _is_valid(centers, radii, tol=1e-7):
    """Check if the packing is valid."""
    n = len(radii)
    for i in range(n):
        x, y, r = centers[i, 0], centers[i, 1], radii[i]
        if r <= 0:
            return False
        if x < r - tol or x > 1 - r + tol:
            return False
        if y < r - tol or y > 1 - r + tol:
            return False
    for i in range(n):
        for j in range(i+1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            d = np.sqrt(dx*dx + dy*dy)
            if d < radii[i] + radii[j] - tol:
                return False
    return True


def _make_valid(centers, radii):
    """Clip radii to ensure validity, reducing if needed."""
    n = len(radii)
    for i in range(n):
        # Clip to boundary
        radii[i] = min(radii[i], centers[i, 0], 1 - centers[i, 0], centers[i, 1], 1 - centers[i, 1])
        radii[i] = max(radii[i], 1e-6)
    
    # Iterate to resolve overlaps
    for _ in range(100):
        changed = False
        for i in range(n):
            for j in range(i+1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                d = np.sqrt(dx*dx + dy*dy) + 1e-12
                required = radii[i] + radii[j]
                if d < required:
                    # Scale both radii down
                    scale = d / (required + 1e-12)
                    radii[i] *= scale
                    radii[j] *= scale
                    changed = True
        if not changed:
            break
    
    # Final boundary clip
    for i in range(n):
        radii[i] = min(radii[i], centers[i, 0], 1 - centers[i, 0], centers[i, 1], 1 - centers[i, 1])
        radii[i] = max(radii[i], 1e-6)
    
    return centers, radii


def _hex_init(n):
    """Initialize with a hexagonal-like pattern."""
    centers = np.zeros((n, 2))
    # Use a grid with slight perturbation
    side = int(np.ceil(np.sqrt(n)))
    idx = 0
    for i in range(side):
        for j in range(side):
            if idx >= n:
                break
            x = (i + 0.5) / side + 0.05 * (np.random.random() - 0.5)
            y = (j + 0.5) / side + 0.05 * (np.random.random() - 0.5)
            x = np.clip(x, 0.05, 0.95)
            y = np.clip(y, 0.05, 0.95)
            centers[idx] = [x, y]
            idx += 1
    radii = np.full(n, 0.5 / side - 0.02)
    return centers, radii


def _random_init(n):
    """Random initialization."""
    centers = np.random.uniform(0.1, 0.9, (n, 2))
    radii = np.full(n, 0.05)
    return centers, radii


def _optimize_single(centers, radii, max_iter=500, tol=1e-8):
    """Run L-BFGS-B optimization from a single starting point."""
    n = len(radii)
    params = np.concatenate([centers.flatten(), radii])
    
    # Bounds: x, y in [0, 1], r in [0, 0.5]
    bounds = []
    for i in range(n):
        bounds.append((0, 1))  # x
        bounds.append((0, 1))  # y
        bounds.append((1e-6, 0.5))  # r
    
    result = minimize(
        _objective_and_grad,
        params,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': max_iter, 'ftol': tol, 'gtol': tol, 'maxls': 50}
    )
    
    opt_params = result.x
    opt_centers = opt_params[:2*n].reshape(n, 2)
    opt_radii = opt_params[2*n:]
    
    return opt_centers, opt_radii


def _polish(centers, radii, max_iter=200, tol=1e-10):
    """High-precision polish pass."""
    n = len(radii)
    params = np.concatenate([centers.flatten(), radii])
    
    bounds = []
    for i in range(n):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((1e-6, 0.5))
    
    result = minimize(
        _objective_and_grad,
        params,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': max_iter, 'ftol': tol, 'gtol': tol, 'maxls': 100}
    )
    
    opt_params = result.x
    opt_centers = opt_params[:2*n].reshape(n, 2)
    opt_radii = opt_params[2*n:]
    
    return opt_centers, opt_radii


def construct():
    start_time = time.time()
    
    best_sum = -np.inf
    best_centers = None
    best_radii = None
    
    # Multiple restarts with different initializations
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() - start_time > _DEADLINE - 20:
            break
        
        if restart == 0:
            centers, radii = _hex_init(N)
        elif restart == 1:
            centers, radii = _random_init(N)
        else:
            # Perturbed hex init
            centers, radii = _hex_init(N)
            centers += np.random.normal(0, 0.03, (N, 2))
            centers = np.clip(centers, 0.05, 0.95)
            radii = np.clip(radii * (0.8 + 0.4 * np.random.random(N)), 0.01, 0.2)
        
        # Optimize
        opt_centers, opt_radii = _optimize_single(centers, radii, max_iter=300, tol=1e-8)
        
        # Make valid
        opt_centers, opt_radii = _make_valid(opt_centers, opt_radii)
        
        if _is_valid(opt_centers, opt_radii):
            s = np.sum(opt_radii)
            if s > best_sum:
                best_sum = s
                best_centers = opt_centers.copy()
                best_radii = opt_radii.copy()
    
    if best_centers is None:
        # Fallback to grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        arr = np.array(cells[:N])
        return arr
    
    # Polish the best solution
    if time.time() - start_time < _DEADLINE - 10:
        pol_centers, pol_radii = _polish(best_centers, best_radii, max_iter=200, tol=1e-10)
        pol_centers, pol_radii = _make_valid(pol_centers, pol_radii)
        if _is_valid(pol_centers, pol_radii):
            if np.sum(pol_radii) > best_sum:
                best_centers = pol_centers
                best_radii = pol_radii
                best_sum = np.sum(pol_radii)
    
    # Local perturbation search to escape shallow minima
    if time.time() - start_time < _DEADLINE - 5:
        for _ in range(3):
            if time.time() - start_time > _DEADLINE - 3:
                break
            pert_centers = best_centers + np.random.normal(0, 0.01, (N, 2))
            pert_centers = np.clip(pert_centers, 0.02, 0.98)
            pert_radii = best_radii.copy()
            
            pert_centers, pert_radii = _optimize_single(pert_centers, pert_radii, max_iter=200, tol=1e-9)
            pert_centers, pert_radii = _make_valid(pert_centers, pert_radii)
            
            if _is_valid(pert_centers, pert_radii):
                if np.sum(pert_radii) > best_sum:
                    best_centers = pert_centers
                    best_radii = pert_radii
                    best_sum = np.sum(pert_radii)
    
    # Final validation and clipping
    best_centers, best_radii = _make_valid(best_centers, best_radii)
    
    # Ensure strict validity with tighter tolerance
    for i in range(N):
        best_radii[i] = min(best_radii[i], 
                           best_centers[i, 0] - 1e-9, 
                           1 - best_centers[i, 0] - 1e-9,
                           best_centers[i, 1] - 1e-9, 
                           1 - best_centers[i, 1] - 1e-9)
        best_radii[i] = max(best_radii[i], 1e-8)
    
    # Final overlap resolution
    for _ in range(50):
        changed = False
        for i in range(N):
            for j in range(i+1, N):
                dx = best_centers[i, 0] - best_centers[j, 0]
                dy = best_centers[i, 1] - best_centers[j, 1]
                d = np.sqrt(dx*dx + dy*dy) + 1e-12
                required = best_radii[i] + best_radii[j] + 1e-9
                if d < required:
                    scale = d / (required + 1e-12)
                    best_radii[i] *= scale
                    best_radii[j] *= scale
                    changed = True
        if not changed:
            break
    
    # Final boundary clip
    for i in range(N):
        best_radii[i] = min(best_radii[i], 
                           best_centers[i, 0] - 1e-9, 
                           1 - best_centers[i, 0] - 1e-9,
                           best_centers[i, 1] - 1e-9, 
                           1 - best_centers[i, 1] - 1e-9)
        best_radii[i] = max(best_radii[i], 1e-8)
    
    result = np.column_stack([best_centers, best_radii])
    return result
# EVOLVE-BLOCK-END
