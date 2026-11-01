"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 21
DEADLINE = 115.0  # safety margin before 120s kill

# EVOLVE-BLOCK-START

def _logsumexp_penalty(distances, margin, beta):
    """
    Smooth approximation of max(0, max_pairwise_violation) using log-sum-exp.
    distances: array of pairwise distances
    margin: required minimum distance (sum of radii)
    beta: sharpness of the log-sum-exp approximation
    Returns a scalar penalty that is ~0 when all constraints are satisfied.
    """
    violations = margin - distances
    # Only penalize positive violations (where circles overlap)
    # Use log-sum-exp for smooth max
    max_viol = np.max(violations)
    if max_viol <= 1e-10:
        return 0.0
    # Shift for numerical stability
    v = violations - max_viol
    # Clip to avoid exp overflow
    v = np.clip(v, -50, 50)
    lse = max_viol + np.log(np.sum(np.exp(beta * v))) / beta
    return max(0.0, lse)


def _compute_objective(x, radii, beta):
    """
    Objective: maximize sum of radii (minimize -sum(radii)) subject to
    - circles inside rectangle [0,w] x [0,h]
    - pairwise non-overlap
    
    x: flattened array [x1,y1,...,xn,yn]
    radii: fixed radii array
    Returns: (negative_sum_radii, constraint_violation)
    """
    n = len(radii)
    centers = x.reshape(-1, 2)
    
    # Boundary constraints: each circle must be inside [0,w] x [0,h]
    # We'll handle these by clamping in the penalty
    w = 2.0  # will be set externally; use a dummy here
    h = 2.0
    
    # Actually, we need to pass w,h. Let's restructure.
    return None


def _make_objective(centers0, radii, w, h, beta):
    """Create objective function for given radii and rectangle."""
    n = len(radii)
    
    def objective(x):
        centers = x.reshape(-1, 2)
        
        # Pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-12)
        np.fill_diagonal(dists, 1e12)
        
        # Required distances (sum of radii)
        req = radii[:, np.newaxis] + radii[np.newaxis, :]
        np.fill_diagonal(req, 0.0)
        
        # Overlap penalty: smooth max of (req - dists) for positive values
        violations = req - dists
        # Only consider upper triangle to avoid double counting
        iu = np.triu_indices(n, 1)
        viol_pairs = violations[iu]
        
        if len(viol_pairs) > 0:
            max_v = np.max(viol_pairs)
            if max_v > 1e-10:
                v = viol_pairs - max_v
                v = np.clip(v, -50, 50)
                lse = max_v + np.log(np.sum(np.exp(beta * v))) / beta
                overlap_pen = max(0.0, lse)
            else:
                overlap_pen = 0.0
        else:
            overlap_pen = 0.0
        
        # Boundary penalty: circles must be in [0,w] x [0,h]
        # Penalty for going outside
        bx = np.maximum(0, radii - centers[:, 0]) + np.maximum(0, centers[:, 0] + radii - w)
        by = np.maximum(0, radii - centers[:, 1]) + np.maximum(0, centers[:, 1] + radii - h)
        bound_pen = np.sum(bx) + np.sum(by)
        
        # Total penalty
        penalty = overlap_pen * 100.0 + bound_pen * 100.0
        
        # Objective: minimize penalty (we fix radii, optimize positions)
        return penalty
    
    return objective


def _try_configuration(radii, w, h, n_restarts=3, beta=10.0, time_limit=5.0):
    """
    Try to find valid positions for given radii in rectangle w x h.
    Returns success (bool) and best centers if successful.
    """
    n = len(radii)
    start_time = time.time()
    
    # Initial guess: grid-like placement
    side = int(np.ceil(np.sqrt(n)))
    best_centers = None
    best_penalty = np.inf
    
    for restart in range(n_restarts):
        if time.time() - start_time > time_limit:
            break
        
        # Generate initial positions
        if restart == 0:
            # Grid
            cols = side
            rows = int(np.ceil(n / cols))
            xs = np.linspace(radii[0], w - radii[0], cols) if cols > 1 else [w/2]
            ys = np.linspace(radii[0], h - radii[0], rows) if rows > 1 else [h/2]
            centers = []
            for i in range(n):
                cx = xs[i % cols]
                cy = ys[i // cols]
                centers.append([cx, cy])
            centers = np.array(centers)
        else:
            # Random
            centers = np.column_stack([
                np.random.uniform(radii, w - radii),
                np.random.uniform(radii, h - radii)
            ])
        
        x0 = centers.flatten()
        
        # Bounds
        bounds = []
        for i in range(n):
            bounds.append((radii[i] * 0.1, w - radii[i] * 0.1))
            bounds.append((radii[i] * 0.1, h - radii[i] * 0.1))
        
        obj = _make_objective(centers, radii, w, h, beta)
        
        result = minimize(obj, x0, method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
        
        if result.fun < best_penalty:
            best_penalty = result.fun
            best_centers = result.x.reshape(-1, 2)
        
        if best_penalty < 1e-6:
            break
    
    return best_penalty < 1e-4, best_centers


def _check_validity(centers, radii, w, h, tol=1e-6):
    """Check if configuration is valid."""
    n = len(radii)
    # Boundary check
    for i in range(n):
        if centers[i, 0] - radii[i] < -tol or centers[i, 0] + radii[i] > w + tol:
            return False
        if centers[i, 1] - radii[i] < -tol or centers[i, 1] + radii[i] > h + tol:
            return False
    
    # Pairwise check
    for i in range(n):
        for j in range(i+1, n):
            d = np.sqrt(np.sum((centers[i] - centers[j])**2))
            if d < radii[i] + radii[j] - tol:
                return False
    return True


def construct():
    np.random.seed(42)
    start_time = time.time()
    
    best_score = 0.0
    best_result = None
    
    # Strategy: search over rectangle aspect ratios and circle radius configurations
    # Perimeter = 2(w+h) = 4 => w+h = 2
    
    # Try different aspect ratios
    aspect_ratios = [1.0, 1.2, 0.8, 1.5, 0.67, 2.0, 0.5, 1.33, 0.75, 1.25]
    
    # For each aspect ratio, try different radius distributions
    # Key insight: equal radii may not be optimal; larger circles can be more efficient
    
    radius_configs = [
        'equal',           # all same radius
        'two_sizes',       # some large, some small
        'gradient',        # varying sizes
    ]
    
    for ar in aspect_ratios:
        if time.time() - start_time > DEADLINE - 10:
            break
        
        w = 2.0 * ar / (1 + ar)
        h = 2.0 / (1 + ar)
        
        for rc in radius_configs:
            if time.time() - start_time > DEADLINE - 10:
                break
            
            if rc == 'equal':
                # All circles same radius
                # Max radius roughly: area based
                # Total area = N * pi * r^2 <= w*h
                r_max = np.sqrt(w * h / (N * np.pi))
                # Try a range of radii
                for r_frac in [0.95, 0.9, 0.85, 0.8, 0.75, 0.7]:
                    r = r_max * r_frac
                    radii = np.full(N, r)
                    # Check if circles fit in width
                    if N * 2 * r > w * 3 or N * 2 * r > h * 3:  # rough check
                        continue
                    
                    success, centers = _try_configuration(radii, w, h, n_restarts=3, beta=15.0, time_limit=3.0)
                    if success and _check_validity(centers, radii, w, h):
                        score = np.sum(radii)
                        if score > best_score:
                            best_score = score
                            best_result = np.column_stack([centers, radii])
                        
            elif rc == 'two_sizes':
                # Some large circles, some small
                # Try: k large circles of radius R, (N-k) small of radius r
                for k in [4, 6, 8, 10]:
                    if N - k < 1:
                        continue
                    # Large circles in a grid, small ones in gaps
                    # Approximate: large circles take up most space
                    # R such that k circles of radius R fit
                    R = np.sqrt(w * h / (k * np.pi * 1.5)) * 0.8
                    r = R * 0.4
                    radii = np.concatenate([np.full(k, R), np.full(N-k, r)])
                    
                    success, centers = _try_configuration(radii, w, h, n_restarts=2, beta=15.0, time_limit=3.0)
                    if success and _check_validity(centers, radii, w, h):
                        score = np.sum(radii)
                        if score > best_score:
                            best_score = score
                            best_result = np.column_stack([centers, radii])
                        
            elif rc == 'gradient':
                # Varying radii
                for base_r in [0.15, 0.18, 0.2, 0.22]:
                    radii = np.linspace(base_r * 1.5, base_r * 0.5, N)
                    # Sort so larger circles are placed first (easier to pack)
                    radii = np.sort(radii)[::-1]
                    
                    success, centers = _try_configuration(radii, w, h, n_restarts=2, beta=15.0, time_limit=3.0)
                    if success and _check_validity(centers, radii, w, h):
                        score = np.sum(radii)
                        if score > best_score:
                            best_score = score
                            best_result = np.column_stack([centers, radii])
    
    # If no valid configuration found, use a fallback
    if best_result is None:
        # Simple grid with small circles
        side = int(np.ceil(np.sqrt(N)))
        w = 2.0
        h = 2.0
        r = 0.5 / side - 0.01
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                x = (i + 0.5) * w / side
                y = (j + 0.5) * h / side
                cells.append([x, y, r])
        best_result = np.array(cells[:N])
    
    # Verify perimeter constraint
    if best_result is not None:
        xs = best_result[:, 0]
        ys = best_result[:, 1]
        rs = best_result[:, 2]
        x_min = np.min(xs - rs)
        x_max = np.max(xs + rs)
        y_min = np.min(ys - rs)
        y_max = np.max(ys + rs)
        bw = x_max - x_min
        bh = y_max - y_min
        perim = 2 * (bw + bh)
        
        if perim > 4.0:
            # Scale down
            scale = 4.0 / perim
            best_result[:, 0] = x_min + (best_result[:, 0] - x_min) * scale
            best_result[:, 1] = y_min + (best_result[:, 1] - y_min) * scale
            best_result[:, 2] *= scale
    
    return best_result

# EVOLVE-BLOCK-END
