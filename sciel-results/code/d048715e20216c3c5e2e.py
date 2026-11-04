"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import scipy.optimize as opt

N = 21

# EVOLVE-BLOCK-START

def _penalty_and_value(circles, w, h, penalty_weight=1000.0):
    """
    Calculate the penalty and the objective value.
    circles: (N, 3) array [x, y, r]
    w, h: width and height of the rectangle
    """
    n = len(circles)
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]
    
    # Objective: Maximize sum of radii -> Minimize -sum(rs)
    obj = -np.sum(rs)
    
    # Penalty for overlaps
    penalty = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx*dx + dy*dy)
            min_dist = rs[i] + rs[j]
            if dist < min_dist:
                overlap = min_dist - dist
                penalty += overlap * overlap
    
    # Penalty for going out of bounds
    # x in [r, w-r], y in [r, h-r]
    for i in range(n):
        if xs[i] < rs[i]:
            penalty += (rs[i] - xs[i]) ** 2
        elif xs[i] > w - rs[i]:
            penalty += (xs[i] - (w - rs[i])) ** 2
            
        if ys[i] < rs[i]:
            penalty += (rs[i] - ys[i]) ** 2
        elif ys[i] > h - rs[i]:
            penalty += (ys[i] - (h - rs[i])) ** 2
            
    total_penalty = penalty_weight * penalty
    
    return obj + total_penalty, penalty, -np.sum(rs)

def _is_valid(circles, w, h, tol=1e-8):
    """Check if the configuration is valid."""
    n = len(circles)
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]
    
    # Check bounds
    for i in range(n):
        if xs[i] < rs[i] - tol or xs[i] > w - rs[i] + tol:
            return False
        if ys[i] < rs[i] - tol or ys[i] > h - rs[i] + tol:
            return False
            
    # Check overlaps
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < rs[i] + rs[j] - tol:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    best_value = -np.inf
    best_circles = None
    best_w = 1.0
    best_h = 1.0
    
    # Perimeter = 2(w+h) = 4 => w+h = 2
    # We can try different aspect ratios.
    # Let w = 2 * t, h = 2 * (1-t). t in (0, 1).
    # Square is t=0.5.
    
    # Initial guesses for aspect ratio
    aspect_ratios = [0.5, 0.4, 0.6, 0.3, 0.7, 0.45, 0.55]
    
    for t in aspect_ratios:
        if time.time() > deadline:
            break
            
        w = 2.0 * t
        h = 2.0 * (1.0 - t)
        
        # Initialize circles in a grid pattern
        side = int(np.ceil(np.sqrt(N)))
        # Adjust grid to fit aspect ratio roughly
        # Just use a simple grid for init
        init_circles = []
        for i in range(side):
            for j in range(side):
                if len(init_circles) >= N:
                    break
                x = (i + 0.5) * (w / side)
                y = (j + 0.5) * (h / side)
                r = min(w, h) / side * 0.4 # Initial radius estimate
                init_circles.append([x, y, r])
        
        init_circles = np.array(init_circles[:N])
        
        # Define bounds for optimization
        # x: [0, w], y: [0, h], r: [0, min(w,h)/2]
        # We will use bounds to keep them inside, but penalty is safer for strict constraints
        # L-BFGS-B handles bounds well.
        max_r = min(w, h) / 2.0
        bounds = []
        for i in range(N):
            bounds.append((0, w))
            bounds.append((0, h))
            bounds.append((1e-6, max_r))
            
        # Objective function for minimize
        def objective(c):
            c = c.reshape(N, 3)
            val, pen, raw_val = _penalty_and_value(c, w, h, penalty_weight=100.0)
            return val
        
        # Run optimization
        try:
            result = opt.minimize(
                objective,
                init_circles.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.success:
                opt_circles = result.x.reshape(N, 3)
                val, pen, raw_val = _penalty_and_value(opt_circles, w, h, penalty_weight=1000.0)
                
                # If penalty is small, it's a good candidate
                if pen < 1e-6:
                    # Refine with higher penalty weight if needed or just accept
                    if raw_val > best_value:
                        best_value = raw_val
                        best_circles = opt_circles
                        best_w = w
                        best_h = h
        except Exception:
            pass
            
        # If we still have time, try to polish the best result found so far
        # by running a local optimization with tighter constraints or higher precision
        # But for now, let's just check time and break if needed.
        
    # Final verification and adjustment
    if best_circles is not None:
        # Ensure strict validity
        # If invalid, shrink radii slightly
        if not _is_valid(best_circles, best_w, best_h, tol=1e-8):
            # Simple shrink
            for i in range(N):
                # Check bounds
                best_circles[i, 0] = np.clip(best_circles[i, 0], best_circles[i, 2], best_w - best_circles[i, 2])
                best_circles[i, 1] = np.clip(best_circles[i, 1], best_circles[i, 2], best_h - best_circles[i, 2])
            
            # Check overlaps and shrink
            changed = True
            while changed:
                changed = False
                for i in range(N):
                    for j in range(i + 1, N):
                        dx = best_circles[i, 0] - best_circles[j, 0]
                        dy = best_circles[i, 1] - best_circles[j, 1]
                        dist = np.sqrt(dx*dx + dy*dy)
                        min_dist = best_circles[i, 2] + best_circles[j, 2]
                        if dist < min_dist:
                            # Shrink both slightly
                            shrink = (min_dist - dist) * 0.5 + 1e-5
                            best_circles[i, 2] -= shrink
                            best_circles[j, 2] -= shrink
                            changed = True
                            
            # Re-check bounds after shrink
            for i in range(N):
                best_circles[i, 0] = np.clip(best_circles[i, 0], best_circles[i, 2], best_w - best_circles[i, 2])
                best_circles[i, 1] = np.clip(best_circles[i, 1], best_circles[i, 2], best_h - best_circles[i, 2])
            
            if _is_valid(best_circles, best_w, best_h, tol=1e-7):
                return best_circles
            else:
                # Fallback to seed-like solution if all else fails
                side = int(np.ceil(np.sqrt(N)))
                cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
                return np.array(cells[:N])
        else:
            return best_circles
    else:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

# EVOLVE-BLOCK-END
