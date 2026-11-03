"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii."""
import numpy as np
from scipy.optimize import minimize
import time

N = 21

# EVOLVE-BLOCK-START
def _validate(circles, w, h, tol=1e-7):
    """Check if circles are valid: inside rectangle and non-overlapping."""
    for cx, cy, r in circles:
        if r <= 0:
            return False
        if cx - r < -tol or cx + r > w + tol or cy - r < -tol or cy + r > h + tol:
            return False
    n = len(circles)
    for i in range(n):
        for j in range(i + 1, n):
            dx = circles[i][0] - circles[j][0]
            dy = circles[i][1] - circles[j][1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < circles[i][2] + circles[j][2] - tol:
                return False
    return True

def _total_radius(circles):
    return sum(c[2] for c in circles)

def _relax_circles(circles, w, h, iterations=1000, tol=1e-8):
    """
    Relaxation algorithm to resolve overlaps and boundary violations.
    This ensures the output is strictly valid.
    """
    n = len(circles)
    # Copy to float array for speed
    pos = np.array([c[:2] for c in circles])
    rad = np.array([c[2] for c in circles])
    
    for _ in range(iterations):
        moved = False
        
        # 1. Resolve boundary violations
        for i in range(n):
            if pos[i, 0] - rad[i] < 0:
                pos[i, 0] = rad[i]
                moved = True
            if pos[i, 0] + rad[i] > w:
                pos[i, 0] = w - rad[i]
                moved = True
            if pos[i, 1] - rad[i] < 0:
                pos[i, 1] = rad[i]
                moved = True
            if pos[i, 1] + rad[i] > h:
                pos[i, 1] = h - rad[i]
                moved = True
        
        # 2. Resolve overlaps
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[j, 0] - pos[i, 0]
                dy = pos[j, 1] - pos[i, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = rad[i] + rad[j]
                
                if dist < min_dist:
                    if dist < 1e-9:
                        # Random push if coincident
                        angle = np.random.uniform(0, 2 * np.pi)
                        push = (min_dist - dist) / 2.0
                        pos[i, 0] -= np.cos(angle) * push
                        pos[i, 1] -= np.sin(angle) * push
                        pos[j, 0] += np.cos(angle) * push
                        pos[j, 1] += np.sin(angle) * push
                    else:
                        push = (min_dist - dist) / 2.0
                        ux = dx / dist
                        uy = dy / dist
                        pos[i, 0] -= ux * push
                        pos[i, 1] -= uy * push
                        pos[j, 0] += ux * push
                        pos[j, 1] += uy * push
                    moved = True
        
        if not moved:
            break
        
        # Check convergence
        # (Optional: check max displacement)
        
    # Final boundary clamp to ensure validity
    for i in range(n):
        pos[i, 0] = np.clip(pos[i, 0], rad[i], w - rad[i])
        pos[i, 1] = np.clip(pos[i, 1], rad[i], h - rad[i])
        
    new_circles = [(pos[i, 0], pos[i, 1], rad[i]) for i in range(n)]
    
    # Final validation check
    if _validate(new_circles, w, h, tol=1e-6):
        return new_circles
    
    # If relaxation failed (shouldn't happen with small radii), return original or fail
    return circles

def _slsqp_optimize(circles, w, h, deadline):
    """
    Optimize radii and positions using SLSQP with hard constraints.
    """
    if time.time() > deadline - 5.0:
        return None
        
    n = len(circles)
    x0 = np.array([v for c in circles for v in c])
    
    # Objective: maximize sum of radii => minimize negative sum
    def objective(x):
        if time.time() > deadline:
            return 0.0
        r = x[2::3]
        return -np.sum(r)
    
    # Constraints
    constraints = []
    
    # Boundary constraints
    for i in range(n):
        # x >= r  => x - r >= 0
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3] - x[i*3+2]})
        # x + r <= w => w - x - r >= 0
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: w - x[i*3] - x[i*3+2]})
        # y >= r  => y - r >= 0
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3+1] - x[i*3+2]})
        # y + r <= h => h - y - r >= 0
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: h - x[i*3+1] - x[i*3+2]})
        # r > 0 => r >= 1e-6
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3+2] - 1e-6})
        
    # Overlap constraints: dist >= r_i + r_j
    for i in range(n):
        for j in range(i + 1, n):
            constraints.append({
                'type': 'ineq', 
                'fun': lambda x, i=i, j=j: np.sqrt((x[i*3]-x[j*3])**2 + (x[i*3+1]-x[j*3+1])**2) - x[i*3+2] - x[j*3+2]
            })
    
    # Bounds
    bounds = []
    for i in range(n):
        bounds.append((1e-6, w)) # x
        bounds.append((1e-6, h)) # y
        bounds.append((1e-6, min(w, h)/2)) # r
    
    try:
        res = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-9}
        )
        
        if res.success:
            new_circles = [(res.x[i*3], res.x[i*3+1], res.x[i*3+2]) for i in range(n)]
            if _validate(new_circles, w, h, tol=1e-5):
                return new_circles
    except Exception:
        pass
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    best_circles = None
    best_score = 0.0
    
    # Aspect ratios: w+h=2
    aspect_ratios = [
        (1.0, 1.0),
        (0.9, 1.1),
        (1.1, 0.9),
        (0.8, 1.2),
        (1.2, 0.8),
    ]
    
    for w, h in aspect_ratios:
        if time.time() > deadline - 10.0:
            break
            
        # Initialize with a grid of small circles
        # 21 circles: 5x5 grid minus 4 corners? Or 4x5 + 1?
        # Let's use a 5x5 grid, remove 4 corners to get 21.
        # Or just a 5x5 grid of 25 and remove 4? 
        # Simpler: 5x5 grid, take first 21.
        cols, rows = 5, 5
        circles = []
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= N:
                    break
                cx = (j + 0.5) / cols * w
                cy = (i + 0.5) / rows * h
                # Small initial radius
                r = 0.05
                circles.append((cx, cy, r))
                
        # Relax to ensure validity
        valid_circles = _relax_circles(circles, w, h, iterations=500)
        
        if _validate(valid_circles, w, h):
            score = _total_radius(valid_circles)
            if score > best_score:
                best_score = score
                best_circles = valid_circles
            
            # Optimize with SLSQP
            if time.time() < deadline - 15.0:
                optimized = _slsqp_optimize(valid_circles, w, h, deadline)
                if optimized is not None:
                    opt_score = _total_radius(optimized)
                    if opt_score > best_score:
                        best_score = opt_score
                        best_circles = optimized
    
    if best_circles is not None:
        return np.array(best_circles)
    
    # Fallback
    side = 5
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.05) 
             for i in range(side) for j in range(side)]
    return np.array(cells[:N])
# EVOLVE-BLOCK-END
