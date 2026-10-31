"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize, LinearConstraint
import time

N = 26
DEADLINE = 145.0  # seconds, safety margin before 150s kill


def _check_validity(circles):
    """Check if circles are valid: inside unit square and non-overlapping."""
    if circles is None or len(circles) != N:
        return False
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    if np.any(r <= 0) or np.any(r > 0.5 + 1e-6):
        return False
    if np.any(x - r < -1e-6) or np.any(x + r > 1 + 1e-6):
        return False
    if np.any(y - r < -1e-6) or np.any(y + r > 1 + 1e-6):
        return False
    # Check pairwise distances
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)
            if dist < r[i] + r[j] - 1e-6:
                return False
    return True


def _objective(params, n=N):
    """Objective to minimize: -sum(radii) + penalties for violations."""
    x = params[:n]
    y = params[n:2*n]
    r = params[2*n:3*n]
    
    # Negative sum of radii (we want to maximize)
    obj = -np.sum(r)
    
    # Boundary penalties (soft): circle must be inside [0,1]^2
    # x - r >= 0, x + r <= 1, y - r >= 0, y + r <= 1
    pen = 0.0
    pen += np.sum(np.maximum(0, -(x - r)) ** 2)
    pen += np.sum(np.maximum(0, (x + r) - 1) ** 2)
    pen += np.sum(np.maximum(0, -(y - r)) ** 2)
    pen += np.sum(np.maximum(0, (y + r) - 1) ** 2)
    
    # Overlap penalties: dist_ij >= r_i + r_j
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx * dx + dy * dy
            min_dist = r[i] + r[j]
            if min_dist > 1e-10:
                # Soft penalty: (min_dist - dist)^2 when overlapping
                dist = np.sqrt(dist_sq) if dist_sq > 0 else 1e-10
                violation = max(0, min_dist - dist)
                pen += violation ** 2
    
    return obj + 100.0 * pen


def _gradient(params, n=N):
    """Gradient of the objective."""
    x = params[:n].copy()
    y = params[n:2*n].copy()
    r = params[2*n:3*n].copy()
    
    gx = np.zeros(n)
    gy = np.zeros(n)
    gr = np.zeros(n)
    
    # d(-sum(r))/dr = -1
    gr[:] = -1.0
    
    # Boundary gradients
    # x - r >= 0: penalty = max(0, -(x-r))^2, d/dx = -2*max(0, r-x) if x<r else 0
    mask = x < r
    gx[mask] += 100.0 * (-2 * (r[mask] - x[mask]))
    gr[mask] += 100.0 * (2 * (r[mask] - x[mask]))
    
    # x + r <= 1: penalty = max(0, x+r-1)^2
    mask = x + r > 1
    gx[mask] += 100.0 * (2 * (x[mask] + r[mask] - 1))
    gr[mask] += 100.0 * (2 * (x[mask] + r[mask] - 1))
    
    # y - r >= 0
    mask = y < r
    gy[mask] += 100.0 * (-2 * (r[mask] - y[mask]))
    gr[mask] += 100.0 * (2 * (r[mask] - y[mask]))
    
    # y + r <= 1
    mask = y + r > 1
    gy[mask] += 100.0 * (2 * (y[mask] + r[mask] - 1))
    gr[mask] += 100.0 * (2 * (y[mask] + r[mask] - 1))
    
    # Overlap gradients
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx * dx + dy * dy
            min_dist = r[i] + r[j]
            if dist_sq < 1e-12:
                continue
            dist = np.sqrt(dist_sq)
            if min_dist > dist:
                violation = min_dist - dist
                factor = 100.0 * 2 * violation / dist
                gx[i] += factor * dx
                gx[j] -= factor * dx
                gy[i] += factor * dy
                gy[j] -= factor * dy
                gr[i] += 100.0 * 2 * violation
                gr[j] += 100.0 * 2 * violation
    
    return np.concatenate([gx, gy, gr])


def _initial_grid(n=N):
    """Generate initial grid of circles."""
    side = int(np.ceil(np.sqrt(n)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) >= n:
                break
            cx = (i + 0.5) / side
            cy = (j + 0.5) / side
            r = 0.5 / side - 1e-4
            cells.append([cx, cy, r])
    return np.array(cells[:n])


def _project_to_valid(circles, n=N):
    """Project circles to be valid (inside square, non-overlapping) approximately."""
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()
    
    # Ensure inside square
    r = np.clip(r, 1e-4, 0.5)
    x = np.clip(x, r, 1 - r)
    y = np.clip(y, r, 1 - r)
    
    # Try to resolve overlaps by shrinking radii
    for _ in range(50):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = r[i] + r[j]
                if dist < min_dist - 1e-8:
                    # Shrink both radii
                    excess = min_dist - dist
                    shrink = excess / 2 + 1e-6
                    r[i] = max(1e-4, r[i] - shrink)
                    r[j] = max(1e-4, r[j] - shrink)
                    changed = True
        if not changed:
            break
    
    return np.column_stack([x, y, r])


def construct():
    start_time = time.time()
    
    # Initial guess: grid
    circles = _initial_grid(N)
    best_circles = circles.copy()
    best_value = np.sum(circles[:, 2])
    
    if not _check_validity(best_circles):
        best_circles = _project_to_valid(best_circles)
        best_value = np.sum(best_circles[:, 2])
    
    # Optimization loop with multiple restarts
    n = N
    dim = 3 * n
    
    for restart in range(3):
        if time.time() - start_time > DEADLINE - 30:
            break
        
        # Perturb initial guess for restarts > 0
        if restart == 0:
            params = best_circles.flatten().copy()
        else:
            params = best_circles.flatten().copy()
            noise_scale = 0.05 / restart
            noise = np.random.normal(0, noise_scale, dim)
            params = params + noise
            # Ensure radii positive
            params[2*n:] = np.maximum(params[2*n:], 1e-3)
        
        # L-BFGS-B optimization
        bounds = [(0, 1)] * n + [(0, 1)] * n + [(1e-4, 0.5)] * n
        
        try:
            result = minimize(
                _objective,
                params,
                method='L-BFGS-B',
                jac=_gradient,
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < _objective(params):
                new_circles = result.x.reshape(3, n).T
                # Project to valid
                new_circles = _project_to_valid(new_circles)
                new_value = np.sum(new_circles[:, 2])
                
                if _check_validity(new_circles) and new_value > best_value:
                    best_circles = new_circles
                    best_value = new_value
        except Exception:
            pass
        
        if time.time() - start_time > DEADLINE - 20:
            break
    
    # Final polish: fix positions, grow radii
    if time.time() - start_time < DEADLINE - 10:
        x = best_circles[:, 0].copy()
        y = best_circles[:, 1].copy()
        r = best_circles[:, 2].copy()
        
        # Try to grow radii while maintaining non-overlap
        for _ in range(100):
            if time.time() - start_time > DEADLINE - 5:
                break
            grew = False
            for i in range(n):
                # Max radius allowed by boundaries
                max_r_bound = min(x[i], 1 - x[i], y[i], 1 - y[i])
                # Max radius allowed by overlaps
                max_r_overlap = max_r_bound
                for j in range(n):
                    if i == j:
                        continue
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx * dx + dy * dy)
                    if dist > 1e-10:
                        max_r_overlap = min(max_r_overlap, (dist - r[j]) / 1.0)
                
                new_r = min(max_r_bound, max_r_overlap) - 1e-7
                if new_r > r[i] + 1e-8:
                    r[i] = new_r
                    grew = True
            
            if not grew:
                break
        
        polished = np.column_stack([x, y, r])
        if _check_validity(polished):
            pol_value = np.sum(polished[:, 2])
            if pol_value > best_value:
                best_circles = polished
                best_value = pol_value
    
    return best_circles
