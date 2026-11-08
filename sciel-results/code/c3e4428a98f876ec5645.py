import numpy as np
import time
from scipy.optimize import minimize

N = 21
DEADLINE = 145.0  # seconds, safety margin before 150s limit

# Precompute pairs for O(n^2) overlap checks
_PAIRS = [(i, j) for i in range(N) for j in range(i + 1, N)]
_PAIR_I = np.array([p[0] for p in _PAIRS])
_PAIR_J = np.array([p[1] for p in _PAIRS])

def _bounding_rect(circles):
    """Return (w, h) of bounding rectangle of circles."""
    if len(circles) == 0:
        return 0.0, 0.0
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]
    x_min = np.min(xs - rs)
    x_max = np.max(xs + rs)
    y_min = np.min(ys - rs)
    y_max = np.max(ys + rs)
    return x_max - x_min, y_max - y_min

def _perimeter(circles):
    w, h = _bounding_rect(circles)
    return 2.0 * (w + h)

def _sum_radii(circles):
    return np.sum(circles[:, 2])

def _is_valid(circles, tol=1e-6):
    """Check that circles are disjoint and perimeter <= 4 + tol."""
    if _perimeter(circles) > 4.0 + tol:
        return False
    # Vectorized overlap check
    # Get positions and radii
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    
    # Calculate squared distances for all pairs
    # We only need to check pairs
    dx = x[_PAIR_I] - x[_PAIR_J]
    dy = y[_PAIR_I] - y[_PAIR_J]
    dist_sq = dx*dx + dy*dy
    min_dist_sq = (r[_PAIR_I] + r[_PAIR_J])**2
    
    # Check if any distance is less than sum of radii (squared)
    if np.any(dist_sq < min_dist_sq - tol*tol):
        # A more precise check for the failing ones to avoid FP issues
        for i, j in _PAIRS:
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < r[i] + r[j] - tol:
                return False
    return True

def _objective(params, n, W, H, penalty_weight=1000.0):
    """
    Objective for fixed bounding box W x H where W + H = 2.
    params = [x0, y0, r0, x1, y1, r1, ...]
    Bounds ensure circles are inside the box.
    We minimize -sum(r) + penalty for overlaps.
    """
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    # Radii are bounded to be positive by the optimizer bounds, 
    # but we clip for safety in calculation
    r_safe = np.maximum(r, 1e-9)
    
    obj = -np.sum(r_safe)
    
    # Overlap penalty (vectorized)
    # dx, dy, dist for all pairs
    dx = x[_PAIR_I] - x[_PAIR_J]
    dy = y[_PAIR_I] - y[_PAIR_J]
    dist = np.sqrt(dx*dx + dy*dy)
    
    sum_radii_pairs = r_safe[_PAIR_I] + r_safe[_PAIR_J]
    overlap = sum_radii_pairs - dist
    
    # Only penalize positive overlaps
    # Using a smooth approximation or just max(0, overlap)^2
    # max(0, x) is non-smooth, but L-BFGS-B handles it reasonably well.
    # To be smoother, we could use softplus, but squared penalty is standard.
    positive_overlap = np.maximum(0.0, overlap)
    obj += penalty_weight * np.sum(positive_overlap ** 2)
    
    return obj

def _grid_initialization_box(n, W, H, cols=None, rows=None, seed=42):
    """Generate grid-based initial packing in box W x H."""
    rng = np.random.RandomState(seed)
    
    if cols is None:
        aspect = W / H if H > 0 else 1.0
        if aspect > 1.0:
            cols = max(3, int(np.sqrt(n * aspect)))
        else:
            cols = max(3, int(np.sqrt(n / aspect)))
    if rows is None:
        rows = int(np.ceil(n / cols))
    
    circles = []
    for i in range(cols):
        for j in range(rows):
            if len(circles) >= n:
                break
            cx = (i + 0.5) * W / cols
            cy = (j + 0.5) * H / rows
            # Radius based on spacing, with margin
            # Leave some gap to avoid initial overlap
            r = 0.5 * min(W / cols, H / rows) * 0.8
            circles.append([cx, cy, r])
    
    return np.array(circles[:n])

def _random_initialization_box(n, W, H, seed=42):
    """Generate random initial packing with small circles in box W x H."""
    rng = np.random.RandomState(seed)
    circles = []
    max_attempts = 500
    attempt = 0
    while len(circles) < n and attempt < max_attempts:
        attempt += 1
        # Start with small radius
        r = 0.01 + 0.02 * rng.random()
        cx = rng.uniform(r, W - r)
        cy = rng.uniform(r, H - r)
        # Check overlap with existing circles
        valid = True
        for (ex, ey, er) in circles:
            dx = cx - ex
            dy = cy - ey
            if np.sqrt(dx * dx + dy * dy) < r + er + 0.001:
                valid = False
                break
        if valid:
            circles.append([cx, cy, r])
    
    # If we couldn't place enough, fill remaining with tiny circles
    while len(circles) < n:
        cx = 0.01 * len(circles) + 0.01
        cy = 0.01 * len(circles) + 0.01
        r = 0.005
        circles.append([cx, cy, r])
    
    return np.array(circles[:n])

def _run_optimization_fixed_box(initial_circles, W, H, start_time, max_time=10.0):
    """Run scipy optimization with fixed box dimensions."""
    n = len(initial_circles)
    x0 = initial_circles.ravel()
    
    # Bounds: positions in [r, W-r], [r, H-r], radii in [0.001, min(W,H)/2]
    # Note: Bounds for x depend on r, which is a variable. 
    # scipy minimize with bounds requires static bounds.
    # So we must use bounds that are valid for all r.
    # x_min >= 0, x_max <= W.
    # r_min >= 0.
    # If x is in [0, W] and r is in [0, min(W,H)/2], it's not guaranteed that x-r >= 0.
    # Therefore, we MUST include boundary penalties in the objective if we use static bounds,
    # OR we use a transformation.
    # However, the previous code used static bounds [0, W] and penalized boundary violations.
    # Let's stick to that, as it's robust.
    
    max_r = min(W, H) / 2.0
    bounds = []
    for i in range(n):
        bounds.append((0.0, W))  # x
        bounds.append((0.0, H))  # y
        bounds.append((0.001, max_r))  # r
    
    best_circles = initial_circles.copy()
    best_obj = _objective(x0, n, W, H)
    
    elapsed = time.time() - start_time
    remaining = max_time - elapsed
    if remaining <= 0.5:
        return best_circles, best_obj
    
    # We need to add boundary penalties to the objective since bounds don't enforce x-r>=0
    # Let's create a wrapper objective that includes boundary penalties
    def _objective_with_bounds(params, n, W, H, penalty_weight=1000.0):
        x = params[0::3]
        y = params[1::3]
        r = params[2::3]
        r_safe = np.maximum(r, 1e-9)
        
        obj = -np.sum(r_safe)
        
        # Boundary penalties
        left_viol = np.maximum(0.0, -(x - r_safe))
        right_viol = np.maximum(0.0, (x + r_safe) - W)
        bottom_viol = np.maximum(0.0, -(y - r_safe))
        top_viol = np.maximum(0.0, (y + r_safe) - H)
        
        obj += penalty_weight * (np.sum(left_viol**2) + np.sum(right_viol**2) + 
                                  np.sum(bottom_viol**2) + np.sum(top_viol**2))
        
        # Overlap penalty
        dx = x[_PAIR_I] - x[_PAIR_J]
        dy = y[_PAIR_I] - y[_PAIR_J]
        dist = np.sqrt(dx*dx + dy*dy)
        sum_radii_pairs = r_safe[_PAIR_I] + r_safe[_PAIR_J]
        overlap = sum_radii_pairs - dist
        positive_overlap = np.maximum(0.0, overlap)
        obj += penalty_weight * np.sum(positive_overlap ** 2)
        
        return obj

    # Update best_obj with the new objective definition
    best_obj = _objective_with_bounds(x0, n, W, H)
    
    try:
        result = minimize(
            _objective_with_bounds, x0,
            args=(n, W, H),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'maxfun': 5000, 'ftol': 1e-14, 'gtol': 1e-12}
        )
        
        if result.fun < best_obj:
            best_obj = result.fun
            best_circles = result.x.reshape(n, 3)
    except Exception:
        pass
    
    # Try a second pass with tighter tolerance or different starting point if time permits
    elapsed = time.time() - start_time
    remaining = max_time - elapsed
    if remaining > 2.0:
        try:
            # Perturb the best solution slightly to escape local minima
            rng = np.random.RandomState(int(time.time() * 1000) % 100000)
            perturbed = best_circles.copy()
            perturbed[:, 0] += rng.randn(n) * 0.01 * W
            perturbed[:, 1] += rng.randn(n) * 0.01 * H
            perturbed[:, 0] = np.clip(perturbed[:, 0], 0.0, W)
            perturbed[:, 1] = np.clip(perturbed[:, 1], 0.0, H)
            
            result2 = minimize(
                _objective_with_bounds, perturbed.ravel(),
                args=(n, W, H),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'maxfun': 2000, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            
            if result2.fun < best_obj:
                best_obj = result2.fun
                best_circles = result2.x.reshape(n, 3)
        except Exception:
            pass
    
    return best_circles, best_obj

def construct():
    start_time = time.time()
    
    best_circles = None
    best_sum_r = -np.inf
    
    # Try multiple aspect ratios
    # W + H = 2 (since perimeter = 2*(W+H) = 4)
    # Aspect ratio a = W/H, so W = 2a/(1+a), H = 2/(1+a)
    # Focus on ratios near 1.0 and some extremes
    aspect_ratios = [0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0]
    
    for a in aspect_ratios:
        if time.time() - start_time > DEADLINE - 10.0:
            break
        
        W = 2.0 * a / (1.0 + a)
        H = 2.0 / (1.0 + a)
        
        # Generate initial grid
        initial = _grid_initialization_box(N, W, H, seed=int(a * 100))
        
        # Run optimization
        circles, obj = _run_optimization_fixed_box(initial, W, H, start_time, max_time=12.0)
        
        # Check validity
        if _is_valid(circles, tol=1e-6):
            sum_r = _sum_radii(circles)
            if sum_r > best_sum_r:
                best_sum_r = sum_r
                best_circles = circles.copy()
        
        # Try random initialization for this aspect ratio
        if time.time() - start_time > DEADLINE - 8.0:
            continue
        initial_rand = _random_initialization_box(N, W, H, seed=int(a * 100) + 7)
        circles_rand, obj_rand = _run_optimization_fixed_box(initial_rand, W, H, start_time, max_time=8.0)
        if _is_valid(circles_rand, tol=1e-6):
            sum_r_rand = _sum_radii(circles_rand)
            if sum_r_rand > best_sum_r:
                best_sum_r = sum_r_rand
                best_circles = circles_rand.copy()
    
    # If we still don't have a valid solution, use a simple fallback
    if best_circles is None:
        W, H = 1.0, 1.0
        best_circles = _grid_initialization_box(N, W, H, seed=1)
        # Scale to ensure validity
        w, h = _bounding_rect(best_circles)
        perim = 2.0 * (w + h)
        if perim > 4.0:
            scale = 4.0 / perim * 0.999
            best_circles[:, 0] *= scale
            best_circles[:, 1] *= scale
            best_circles[:, 2] *= scale
    
    # Final refinement: iterative radius growth with validity checks
    if time.time() - start_time < DEADLINE - 3.0:
        try:
            improved = True
            for _ in range(200):
                if time.time() - start_time > DEADLINE - 2.0:
                    break
                if not improved:
                    break
                improved = False
                for i in range(N):
                    if time.time() - start_time > DEADLINE - 2.0:
                        break
                    test = best_circles.copy()
                    test[i, 2] *= 1.002
                    if _is_valid(test, tol=1e-7):
                        best_circles = test
                        best_sum_r = _sum_radii(best_circles)
                        improved = True
        except Exception:
            pass
    
    # Final safety: ensure perimeter <= 4 and no overlaps
    w, h = _bounding_rect(best_circles)
    perim = 2.0 * (w + h)
    if perim > 4.0:
        scale = 4.0 / perim * 0.9999
        best_circles[:, 0] *= scale
        best_circles[:, 1] *= scale
        best_circles[:, 2] *= scale
    
    # Ensure no overlaps by slight shrink if needed
    for i, j in _PAIRS:
        dx = best_circles[i, 0] - best_circles[j, 0]
        dy = best_circles[i, 1] - best_circles[j, 1]
        dist = np.sqrt(dx * dx + dy * dy)
        min_dist = best_circles[i, 2] + best_circles[j, 2]
        if dist < min_dist:
            factor = dist / min_dist * 0.999
            best_circles[i, 2] *= factor
            best_circles[j, 2] *= factor
    
    # Final perimeter check
    w, h = _bounding_rect(best_circles)
    perim = 2.0 * (w + h)
    if perim > 4.0:
        scale = 4.0 / perim * 0.9999
        best_circles[:, 0] *= scale
        best_circles[:, 1] *= scale
        best_circles[:, 2] *= scale
    
    return best_circles
