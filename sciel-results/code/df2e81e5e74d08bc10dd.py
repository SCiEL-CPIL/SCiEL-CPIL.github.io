"""Rewritten program for packing 200 equal circles in a unit square."""
import time
import numpy as np
from scipy.optimize import minimize

N = 200

def _score(coords):
    """Compute the radius for a given set of coordinates."""
    # coords shape (N, 2)
    # distance between all pairs
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    # minimum pairwise distance
    min_pair = np.min(dist[~np.eye(N, dtype=bool)])
    # minimum distance to boundary
    min_bound = np.min(np.minimum(np.minimum(0.5 - coords[:, 0], coords[:, 0] + 0.5),
                                   np.minimum(0.5 - coords[:, 1], coords[:, 1] + 0.5)))
    return min(min_pair / 2.0, min_bound)

def _neg_score(coords):
    """Negative score for minimization."""
    return -_score(coords)

def _gradient(coords):
    """Gradient of the negative score (approximate via finite differences or analytical)."""
    # For efficiency, we use a simple numerical gradient approach
    # But actually, let's use a more efficient approach: compute the score and its gradient analytically
    # The score is min(min_pair/2, min_bound)
    # We need to find which constraint is active
    
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    min_pair_idx = np.unravel_index(np.argmin(dist), dist.shape)
    min_pair_val = dist[min_pair_idx]
    
    # Distance to boundaries
    d_left = coords[:, 0] + 0.5
    d_right = 0.5 - coords[:, 0]
    d_bottom = coords[:, 1] + 0.5
    d_top = 0.5 - coords[:, 1]
    
    d_bound = np.minimum(np.minimum(d_left, d_right), np.minimum(d_bottom, d_top))
    min_bound_idx = np.argmin(d_bound)
    min_bound_val = d_bound[min_bound_idx]
    
    # Determine which is the limiting factor
    if min_pair_val / 2.0 < min_bound_val:
        # Pairwise distance is limiting
        i, j = min_pair_idx
        if i == j:
            return np.zeros_like(coords)
        diff_ij = coords[i] - coords[j]
        dist_ij = np.linalg.norm(diff_ij)
        if dist_ij < 1e-12:
            return np.zeros_like(coords)
        # Gradient of -min_pair/2 w.r.t. coords
        grad = np.zeros_like(coords)
        # d(-dist/2)/d(coords[i]) = -diff_ij / (2 * dist_ij) * (-1) ... 
        # Actually: score = dist/2, so -score = -dist/2
        # d(-dist/2)/d(coords[i]) = -1/2 * d(dist)/d(coords[i]) = -1/2 * diff_ij / dist_ij
        # d(-dist/2)/d(coords[j]) = -1/2 * d(dist)/d(coords[j]) = -1/2 * (-diff_ij) / dist_ij = diff_ij / (2 * dist_ij)
        grad[i] = -diff_ij / (2.0 * dist_ij)
        grad[j] = diff_ij / (2.0 * dist_ij)
        return grad
    else:
        # Boundary distance is limiting
        idx = min_bound_idx
        grad = np.zeros_like(coords)
        # d(-min_bound)/d(coords[idx])
        if d_bound[idx] == d_left[idx]:
            # d_left = coords[:, 0] + 0.5, d(-d_left)/d(coords[idx, 0]) = -1
            grad[idx, 0] = -1.0
        elif d_bound[idx] == d_right[idx]:
            # d_right = 0.5 - coords[:, 0], d(-d_right)/d(coords[idx, 0]) = 1
            grad[idx, 0] = 1.0
        elif d_bound[idx] == d_bottom[idx]:
            # d_bottom = coords[:, 1] + 0.5, d(-d_bottom)/d(coords[idx, 1]) = -1
            grad[idx, 1] = -1.0
        else:
            # d_top = 0.5 - coords[:, 1], d(-d_top)/d(coords[idx, 1]) = 1
            grad[idx, 1] = 1.0
        return grad

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin
    
    # Initial configuration: hexagonal lattice
    # For 200 circles, use a hexagonal arrangement
    # Hex lattice with spacing s: rows have alternating offsets
    
    def hex_initial(n_points):
        """Generate initial hexagonal lattice points."""
        points = []
        # Hex lattice: row spacing = s * sqrt(3)/2, col spacing = s
        # Try to fit n_points in a hex pattern
        # Start with a rough estimate
        s = 0.05  # initial spacing
        rows = 0
        cols = 0
        # Estimate: area per point ~ s^2 * sqrt(3)/2
        # Total area ~ 1, so s ~ sqrt(2/(n * sqrt(3)))
        s_est = np.sqrt(2.0 / (n_points * np.sqrt(3.0)))
        
        # Generate hex lattice points centered
        max_radius = 0.6
        for y_row in range(-50, 51):
            y = y_row * s_est * np.sqrt(3.0) / 2.0
            if abs(y) > max_radius:
                continue
            x_offset = (s_est / 2.0) if (y_row % 2 != 0) else 0.0
            for x_col in range(-50, 51):
                x = x_col * s_est + x_offset
                if abs(x) > max_radius or abs(y) > max_radius:
                    continue
                points.append([x, y])
        
        points = np.array(points)
        # Shuffle and take first n_points
        np.random.seed(42)
        idx = np.random.choice(len(points), n_points, replace=False)
        return points[idx]
    
    best_coords = None
    best_score = -np.inf
    
    # Try multiple initial configurations
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline - 20.0:
            break
        
        if restart == 0:
            coords = hex_initial(N)
        else:
            # Perturbed hex
            coords = hex_initial(N)
            coords += np.random.normal(0, 0.01, coords.shape)
        
        # Optimize using L-BFGS-B
        try:
            result = minimize(_neg_score, coords, method='L-BFGS-B',
                             jac=_gradient,
                             options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8,
                                     'maxfun': 5000})
            
            if result.success or result.fun < _neg_score(coords):
                score = -result.fun
                if score > best_score:
                    best_score = score
                    best_coords = result.x.copy()
        except Exception:
            pass
        
        # Check time
        if time.time() > deadline - 30.0:
            break
    
    # Final polish: if we have a good solution, do a more careful optimization
    if best_coords is not None and time.time() < deadline - 10.0:
        try:
            result = minimize(_neg_score, best_coords, method='L-BFGS-B',
                             jac=_gradient,
                             options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10,
                                     'maxfun': 10000})
            if result.fun < _neg_score(best_coords):
                best_coords = result.x.copy()
        except Exception:
            pass
    
    # Ensure coordinates are within bounds
    if best_coords is None:
        # Fallback to grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_coords = grid[:N]
    else:
        # Clamp to valid range
        best_coords = np.clip(best_coords, -0.4999, 0.4999)
    
    return best_coords
