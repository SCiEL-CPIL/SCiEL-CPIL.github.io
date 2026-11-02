"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 145.0  # safety margin before 150s kill

def _softplus(x, beta=50.0):
    """Numerically stable softplus."""
    return np.where(x > 20.0 / beta, x, np.log1p(np.exp(beta * x)) / beta)

def _objective(params, aspect):
    """
    params: (n, 3) array of [cx, cy, r]
    aspect: width/height ratio (perimeter = 4 => 2*(w+h)=4 => w+h=2)
    Returns: -sum(r) + overlap_penalty + boundary_penalty
    """
    n = params.shape[0]
    cx = params[:, 0]
    cy = params[:, 1]
    r = params[:, 2]
    
    # Rectangle dimensions: w + h = 2, w/h = aspect
    h = 2.0 / (1.0 + aspect)
    w = aspect * h
    
    # Boundary violations: circle must fit in [0,w] x [0,h]
    # Left: cx - r >= 0 => violation = r - cx
    # Right: cx + r <= w => violation = cx + r - w
    # Bottom: cy - r >= 0 => violation = r - cy
    # Top: cy + r <= h => violation = cy + r - h
    bnd_viol = np.maximum(0, r - cx) + np.maximum(0, cx + r - w) + \
               np.maximum(0, r - cy) + np.maximum(0, cy + r - h)
    bnd_pen = np.sum(np.exp(10.0 * bnd_viol) - 1.0)
    
    # Overlap violations: for each pair, need dist >= ri + rj
    # violation = ri + rj - dist
    cx_i = cx[:, np.newaxis]
    cx_j = cx[np.newaxis, :]
    cy_i = cy[:, np.newaxis]
    cy_j = cy[np.newaxis, :]
    r_i = r[:, np.newaxis]
    r_j = r[np.newaxis, :]
    
    dist = np.sqrt((cx_i - cx_j)**2 + (cy_i - cy_j)**2)
    # Avoid division by zero / self-overlap
    np.fill_diagonal(dist, np.inf)
    
    viol = r_i + r_j - dist
    # Only penalize positive violations
    viol_pos = np.maximum(0, viol)
    # Use soft exponential penalty for smoothness
    overlap_pen = np.sum(np.exp(10.0 * viol_pos) - 1.0)
    
    # Objective: minimize penalty - sum(radii)
    # We want to maximize sum(r), so minimize -sum(r) + penalties
    obj = -np.sum(r) + 100.0 * bnd_pen + 100.0 * overlap_pen
    
    return obj

def _objective_grad(params, aspect):
    """Gradient of the objective."""
    n = params.shape[0]
    cx = params[:, 0]
    cy = params[:, 1]
    r = params[:, 2]
    
    h = 2.0 / (1.0 + aspect)
    w = aspect * h
    
    grad = np.zeros_like(params)
    
    # Boundary gradients
    # Left: max(0, r - cx), d/dc = -1 if active
    mask_left = (r - cx) > 0
    grad[mask_left, 0] += -100.0 * 10.0 * np.exp(10.0 * (r[mask_left] - cx[mask_left]))
    grad[mask_left, 2] += 100.0 * 10.0 * np.exp(10.0 * (r[mask_left] - cx[mask_left]))
    
    # Right: max(0, cx + r - w), d/cx = 1, d/r = 1
    mask_right = (cx + r - w) > 0
    grad[mask_right, 0] += 100.0 * 10.0 * np.exp(10.0 * (cx[mask_right] + r[mask_right] - w))
    grad[mask_right, 2] += 100.0 * 10.0 * np.exp(10.0 * (cx[mask_right] + r[mask_right] - w))
    
    # Bottom: max(0, r - cy), d/cy = -1
    mask_bot = (r - cy) > 0
    grad[mask_bot, 1] += -100.0 * 10.0 * np.exp(10.0 * (r[mask_bot] - cy[mask_bot]))
    grad[mask_bot, 2] += 100.0 * 10.0 * np.exp(10.0 * (r[mask_bot] - cy[mask_bot]))
    
    # Top: max(0, cy + r - h), d/cy = 1
    mask_top = (cy + r - h) > 0
    grad[mask_top, 1] += 100.0 * 10.0 * np.exp(10.0 * (cy[mask_top] + r[mask_top] - h))
    grad[mask_top, 2] += 100.0 * 10.0 * np.exp(10.0 * (cy[mask_top] + r[mask_top] - h))
    
    # Overlap gradients
    for i in range(n):
        for j in range(i+1, n):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            d = np.sqrt(dx*dx + dy*dy)
            if d < 1e-12:
                continue
            viol = r[i] + r[j] - d
            if viol > 0:
                pen_factor = 100.0 * 10.0 * np.exp(10.0 * viol)
                # d(viol)/d(cx[i]) = -dx/d, d(viol)/d(cx[j]) = dx/d
                grad[i, 0] += -pen_factor * dx / d
                grad[j, 0] += pen_factor * dx / d
                grad[i, 1] += -pen_factor * dy / d
                grad[j, 1] += pen_factor * dy / d
                grad[i, 2] += pen_factor
                grad[j, 2] += pen_factor
    
    # Gradient of -sum(r)
    grad[:, 2] += -1.0
    
    return grad

def _grid_init(n, aspect, jitter=0.0):
    """Initialize circles in a grid pattern."""
    side = int(np.ceil(np.sqrt(n)))
    h = 2.0 / (1.0 + aspect)
    w = aspect * h
    
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) >= n:
                break
            cx = (i + 0.5) * w / side
            cy = (j + 0.5) * h / side
            r = min(w, h) / side * 0.45
            if jitter > 0:
                cx += np.random.uniform(-jitter, jitter) * r
                cy += np.random.uniform(-jitter, jitter) * r
            cells.append([cx, cy, r])
    
    return np.array(cells[:n])

def _hex_init(n, aspect, jitter=0.0):
    """Initialize circles in a hexagonal pattern."""
    h = 2.0 / (1.0 + aspect)
    w = aspect * h
    
    # Estimate number of rows
    rows = int(np.ceil(np.sqrt(n * 2.0 / np.sqrt(3))))
    
    cells = []
    r_est = h / (rows * 1.1)
    
    for row in range(rows):
        y = (row + 0.5) * h / rows
        offset = 0 if row % 2 == 0 else r_est * 0.866
        cols = int(np.ceil(w / (2 * r_est))) + 1
        for col in range(cols):
            if len(cells) >= n:
                break
            x = (col + 0.5) * 2 * r_est + offset
            if x + r_est > w:
                continue
            r = min(r_est, h / rows * 0.45)
            if jitter > 0:
                x += np.random.uniform(-jitter, jitter) * r
                y += np.random.uniform(-jitter, jitter) * r
            cells.append([x, y, r])
    
    if len(cells) < n:
        # Fill remaining with grid
        cells = list(_grid_init(n, aspect, jitter).flatten().reshape(-1, 3))
    
    return np.array(cells[:n])

def _random_init(n, aspect):
    """Random initialization."""
    h = 2.0 / (1.0 + aspect)
    w = aspect * h
    r_est = min(w, h) / 5.0
    
    cells = []
    for _ in range(n):
        r = np.random.uniform(0.5, 1.0) * r_est
        cx = np.random.uniform(r, w - r)
        cy = np.random.uniform(r, h - r)
        cells.append([cx, cy, r])
    
    return np.array(cells)

def _check_valid(params, aspect, tol=1e-6):
    """Check if configuration is valid (no overlaps, within bounds)."""
    cx = params[:, 0]
    cy = params[:, 1]
    r = params[:, 2]
    
    h = 2.0 / (1.0 + aspect)
    w = aspect * h
    
    # Check boundaries
    for i in range(len(cx)):
        if cx[i] - r[i] < -tol or cx[i] + r[i] > w + tol or cy[i] - r[i] < -tol or cy[i] + r[i] > h + tol:
            return False
    
    # Check overlaps
    for i in range(len(cx)):
        for j in range(i+1, len(cx)):
            d = np.sqrt((cx[i]-cx[j])**2 + (cy[i]-cy[j])**2)
            if d < r[i] + r[j] - tol:
                return False
    
    return True

def _scale_to_perimeter(params, aspect):
    """Scale parameters to fit in perimeter-4 rectangle with given aspect."""
    h = 2.0 / (1.0 + aspect)
    w = aspect * h
    
    cx = params[:, 0]
    cy = params[:, 1]
    r = params[:, 2]
    
    # Find max extent
    max_x = np.max(cx + r)
    min_x = np.min(cx - r)
    max_y = np.max(cy + r)
    min_y = np.min(cy - r)
    
    span_x = max_x - min_x
    span_y = max_y - min_y
    
    if span_x <= 0 or span_y <= 0:
        return params
    
    # Scale to fit
    scale = min(w / span_x, h / span_y)
    if scale > 1.0:
        scale = 1.0
    
    cx_scaled = (cx - min_x) * scale + (w - span_x * scale) / 2
    cy_scaled = (cy - min_y) * scale + (h - span_y * scale) / 2
    r_scaled = r * scale
    
    return np.column_stack([cx_scaled, cy_scaled, r_scaled])

def construct():
    global _DEADLINE
    best_params = None
    best_value = -np.inf
    
    aspects = [1.0, 1.2, 0.8, 1.5, 0.67, 2.0, 0.5]
    
    for aspect in aspects:
        if time.time() > _DEADLINE:
            break
        
        inits = [
            ("grid", _grid_init(N, aspect, 0.0)),
            ("grid_j", _grid_init(N, aspect, 0.1)),
            ("hex", _hex_init(N, aspect, 0.0)),
            ("hex_j", _hex_init(N, aspect, 0.1)),
            ("rand", _random_init(N, aspect)),
        ]
        
        for name, init_params in inits:
            if time.time() > _DEADLINE:
                break
            
            # Optimize
            try:
                result = minimize(
                    _objective,
                    init_params,
                    args=(aspect,),
                    jac=_objective_grad,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                params = result.x
                # Ensure radii are positive
                params[:, 2] = np.maximum(params[:, 2], 1e-6)
                
                # Check validity
                if _check_valid(params, aspect):
                    value = np.sum(params[:, 2])
                    if value > best_value:
                        best_value = value
                        best_params = params.copy()
            except Exception:
                continue
    
    if best_params is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    # Scale to perimeter 4
    # Find best aspect for final scaling
    best_scaled = None
    best_scaled_value = -np.inf
    
    for aspect in aspects:
        if time.time() > _DEADLINE:
            break
        
        scaled = _scale_to_perimeter(best_params, aspect)
        if _check_valid(scaled, aspect):
            value = np.sum(scaled[:, 2])
            if value > best_scaled_value:
                best_scaled_value = value
                best_scaled = scaled
    
    if best_scaled is not None:
        return best_scaled
    return best_params
# EVOLVE-BLOCK-END
