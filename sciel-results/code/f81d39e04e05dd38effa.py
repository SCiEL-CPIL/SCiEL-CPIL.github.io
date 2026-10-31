"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

_start_time = time.time()
TIME_LIMIT = 140  # seconds, leave margin for 150s kill

def _elapsed():
    return time.time() - _start_time

def _unified_penalty(params):
    """
    Unified penalty: params = [w, x1,y1,r1, ..., x21,y21,r21]
    w is the rectangle width, h = 2 - w.
    Minimize: -sum(r_i) + C * (overlap + boundary penalties)
    """
    w = params[0]
    h = 2.0 - w
    if w <= 0.1 or h <= 0.1:
        return 1e6
    
    n = (len(params) - 1) // 3
    pts = params[1:].reshape(n, 3)
    xs = pts[:, 0]
    ys = pts[:, 1]
    rs = pts[:, 2]
    
    # Objective: -sum(r)
    obj = -np.sum(rs)
    
    # Overlap penalties
    C = 10.0
    overlap_pen = 0.0
    for i in range(n):
        for j in range(i+1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx*dx + dy*dy)
            min_dist = rs[i] + rs[j]
            if dist < min_dist:
                overlap_pen += (min_dist - dist)**2
    
    # Boundary penalties
    boundary_pen = 0.0
    for i in range(n):
        if xs[i] < rs[i]:
            boundary_pen += (rs[i] - xs[i])**2
        if xs[i] > w - rs[i]:
            boundary_pen += (xs[i] - (w - rs[i]))**2
        if ys[i] < rs[i]:
            boundary_pen += (rs[i] - ys[i])**2
        if ys[i] > h - rs[i]:
            boundary_pen += (ys[i] - (h - rs[i]))**2
    
    total = obj + C * (overlap_pen + boundary_pen)
    return total

def _init_hexagonal(w, h, n_circles, seed=0):
    """Initialize circles in a hexagonal-like pattern."""
    np.random.seed(seed)
    circles = []
    
    # Try different row/col arrangements
    best_circles = None
    best_area = 0
    
    for rows in range(3, 8):
        for cols in range(3, 8):
            if rows * cols < n_circles:
                continue
            
            # Hexagonal offset: alternate rows shifted by half cell width
            cell_w = w / cols
            cell_h = h / rows
            
            for i in range(rows):
                for j in range(cols):
                    if len(circles) >= n_circles:
                        break
                    x = (j + 0.5) * cell_w
                    y = (i + 0.5) * cell_h
                    # Hexagonal offset for odd rows
                    if i % 2 == 1:
                        x += cell_w * 0.3
                    # Clamp
                    x = np.clip(x, 0, w)
                    y = np.clip(y, 0, h)
                    r = min(cell_w, cell_h) * 0.42
                    circles.append([x, y, r])
            
            if len(circles) == n_circles:
                # Estimate quality by sum of radii
                total_r = sum(c[2] for c in circles)
                if total_r > best_area:
                    best_area = total_r
                    best_circles = circles[:]
            circles = []
    
    if best_circles is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(n_circles)))
        for i in range(side):
            for j in range(side):
                if len(circles) >= n_circles:
                    break
                x = (j + 0.5) * w / side
                y = (i + 0.5) * h / side
                r = min(w/side, h/side) * 0.42
                circles.append([x, y, r])
        best_circles = circles
    
    # Add small jitter to break symmetry
    for c in best_circles:
        c[0] += (np.random.rand() - 0.5) * 0.02 * w
        c[1] += (np.random.rand() - 0.5) * 0.02 * h
    
    return np.array(best_circles[:n_circles])

def _init_grid(w, h, n_circles, rows, cols, jitter=0.0, seed=0):
    """Initialize circles in a grid pattern."""
    np.random.seed(seed)
    circles = []
    for i in range(rows):
        for j in range(cols):
            if len(circles) >= n_circles:
                break
            x = (j + 0.5) * w / cols
            y = (i + 0.5) * h / rows
            cell_w = w / cols
            cell_h = h / rows
            r = min(cell_w, cell_h) * 0.42
            if jitter > 0:
                x += (np.random.rand() - 0.5) * jitter * cell_w
                y += (np.random.rand() - 0.5) * jitter * cell_h
            circles.append([x, y, r])
    return np.array(circles[:n_circles])

def _validate(pts, w, h, tol=1e-5):
    """Check if circles are valid (within bounds, non-overlapping)."""
    n = len(pts)
    for i in range(n):
        x, y, r = pts[i]
        if x < r - tol or x > w - r + tol or y < r - tol or y > h - r + tol:
            return False
    for i in range(n):
        for j in range(i+1, n):
            dx = pts[i,0] - pts[j,0]
            dy = pts[i,1] - pts[j,1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < pts[i,2] + pts[j,2] - tol:
                return False
    return True

def _polish(pts, w, h):
    """Polish the result to ensure validity."""
    pts = pts.copy()
    # Clamp to bounds
    for i in range(len(pts)):
        x, y, r = pts[i]
        r = min(r, x, w - x, y, h - y)
        r = max(r, 0.005)
        pts[i] = [x, y, r]
    
    # Fix overlaps by reducing radii
    for _ in range(20):
        overlap = False
        for i in range(len(pts)):
            for j in range(i+1, len(pts)):
                dx = pts[i,0] - pts[j,0]
                dy = pts[i,1] - pts[j,1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = pts[i,2] + pts[j,2]
                if dist < min_dist:
                    reduce = (min_dist - dist) / 2 + 0.001
                    pts[i,2] = max(pts[i,2] - reduce, 0.005)
                    pts[j,2] = max(pts[j,2] - reduce, 0.005)
                    overlap = True
        if not overlap:
            break
    
    # Final boundary check
    for i in range(len(pts)):
        x, y, r = pts[i]
        r = min(r, x, w - x, y, h - y)
        r = max(r, 0.005)
        pts[i,2] = r
    
    return pts

def construct():
    global _start_time
    _start_time = time.time()
    
    best_val = -np.inf
    best_result = None  # (w, h, pts)
    
    # Phase 1: Unified optimization with free aspect ratio
    # Try multiple initializations with different aspect ratios and patterns
    
    initial_w_values = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    
    for w0 in initial_w_values:
        if _elapsed() > TIME_LIMIT - 10:
            break
        h0 = 2.0 - w0
        
        # Try hexagonal and grid initializations
        for init_type in ['hex', 'grid']:
            if _elapsed() > TIME_LIMIT - 10:
                break
            
            if init_type == 'hex':
                circles = _init_hexagonal(w0, h0, 21, seed=int(w0*100))
            else:
                # Try a few grid configs
                for rows in [4, 5]:
                    for cols in [5, 6]:
                        if rows * cols >= 21:
                            circles = _init_grid(w0, h0, 21, rows, cols, jitter=0.1, seed=int(w0*100)+rows)
                            break
                    else:
                        continue
                    break
            
            if len(circles) < 21:
                continue
            
            # Build params: [w, x1,y1,r1, ..., x21,y21,r21]
            params = np.concatenate([[w0], circles.flatten()])
            
            # Bounds: w in [0.3, 1.7], positions in [0, w] or [0, h], radii in [0.005, min(w,h)/2]
            bounds = [(0.3, 1.7)]  # w
            for i in range(21):
                bounds.append((0, 2.0))   # x
                bounds.append((0, 2.0))   # y
                bounds.append((0.005, 1.0))  # r
            
            try:
                result = minimize(_unified_penalty, params, method='L-BFGS-B',
                                 bounds=bounds, options={'maxiter': 300, 'maxfun': 8000})
            except Exception:
                continue
            
            if _elapsed() > TIME_LIMIT - 5:
                break
            
            w_opt = result.x[0]
            h_opt = 2.0 - w_opt
            pts = result.x[1:].reshape(21, 3)
            
            # Validate
            if _validate(pts, w_opt, h_opt):
                val = np.sum(pts[:, 2])
                if val > best_val:
                    best_val = val
                    best_result = (w_opt, h_opt, pts.copy())
    
    # Phase 2: Local refinement around best found
    if best_result is not None:
        w_best, h_best, pts_best = best_result
        
        # Try small perturbations
        for dx in [-0.05, 0, 0.05]:
            for dy in [-0.05, 0, 0.05]:
                if _elapsed() > TIME_LIMIT - 5:
                    break
                
                w_try = w_best + dx
                h_try = 2.0 - w_try
                if w_try < 0.3 or h_try < 0.3:
                    continue
                
                pts_try = pts_best.copy()
                # Scale positions to new rectangle
                pts_try[:, 0] *= w_try / w_best
                pts_try[:, 1] *= h_try / h_best
                
                params = np.concatenate([[w_try], pts_try.flatten()])
                bounds = [(0.3, 1.7)]
                for i in range(21):
                    bounds.append((0, 2.0))
                    bounds.append((0, 2.0))
                    bounds.append((0.005, 1.0))
                
                try:
                    result = minimize(_unified_penalty, params, method='L-BFGS-B',
                                     bounds=bounds, options={'maxiter': 200, 'maxfun': 5000})
                except Exception:
                    continue
                
                w_opt = result.x[0]
                h_opt = 2.0 - w_opt
                pts = result.x[1:].reshape(21, 3)
                
                if _validate(pts, w_opt, h_opt):
                    val = np.sum(pts[:, 2])
                    if val > best_val:
                        best_val = val
                        best_result = (w_opt, h_opt, pts.copy())
    
    if best_result is None:
        # Fallback: simple square grid
        side = 5
        w = 1.0
        h = 1.0
        pts = []
        for i in range(side):
            for j in range(side):
                if len(pts) >= 21:
                    break
                x = (j + 0.5) * w / side
                y = (i + 0.5) * h / side
                r = min(w/side, h/side) * 0.40
                pts.append([x, y, r])
        pts = np.array(pts[:21])
        return pts
    
    w, h, pts = best_result
    pts = _polish(pts, w, h)
    
    return pts
# EVOLVE-BLOCK-END
