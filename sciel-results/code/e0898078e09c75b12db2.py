"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii."""
import numpy as np
from scipy.optimize import minimize
import time

N = 21

# EVOLVE-BLOCK-START

def _validate(circles, w, h, tol=1e-7):
    """Check if circles are valid: inside rectangle and non-overlapping."""
    n = len(circles)
    if n != N:
        return False
    for i in range(n):
        cx, cy, r = circles[i]
        if r <= 0:
            return False
        if cx - r < -tol or cx + r > w + tol or cy - r < -tol or cy + r > h + tol:
            return False
    
    # Vectorized overlap check
    cx = circles[:, 0]
    cy = circles[:, 1]
    r = circles[:, 2]
    for i in range(n):
        dx = cx[i] - cx[i+1:]
        dy = cy[i] - cy[i+1:]
        dist = np.sqrt(dx*dx + dy*dy)
        min_dist = r[i] + r[i+1:]
        if np.any(dist < min_dist - tol):
            return False
    return True

def _total_radius(circles):
    return float(np.sum(circles[:, 2]))

def _hexagonal_init(w, h):
    """
    Create initial hexagonal-like arrangement for 21 circles.
    Rows: 5, 4, 5, 4, 3 = 21 circles.
    """
    rows = [5, 4, 5, 4, 3]
    total_rows = len(rows)
    # We want to map this grid into [0, w] x [0, h]
    # Approximate spacing
    # In a hexagonal lattice, horizontal spacing is 2r, vertical is sqrt(3)r.
    # Here we just distribute them evenly to start, then optimize.
    
    circles = []
    for row_idx, count in enumerate(rows):
        y = (row_idx + 0.5) * h / total_rows
        for col_idx in range(count):
            # Offset odd rows by half a step
            if row_idx % 2 == 1:
                x = (col_idx + 0.5) * w / 5
            else:
                x = (col_idx + 0.5) * w / 5
            
            # Initial radius: estimate based on local density
            # Roughly, area per circle ~ w*h/N
            # r ~ sqrt(w*h/(N*pi))
            r = 0.8 * np.sqrt(w * h / (N * np.pi))
            
            # Ensure it fits locally
            r = min(r, x, w-x, y, h-y)
            r = max(r, 0.01)
            
            circles.append([x, y, r])
    
    return np.array(circles[:N])

def _penalty_objective(x, w, h, beta):
    """
    Objective function for penalty-based optimization.
    x = [x_1, ..., x_n, y_1, ..., y_n, r_1, ..., r_n]
    Minimize: -sum(r) + beta * (overlap_penalty + boundary_penalty)
    """
    n = N
    cx = x[:n]
    cy = x[n:2*n]
    r = x[2*n:]
    
    # Objective: maximize sum of radii => minimize -sum(r)
    obj = -np.sum(r)
    
    # Boundary Penalty
    pen = 0.0
    for i in range(n):
        # Left
        viol = r[i] - cx[i]
        if viol > 0:
            pen += viol**2
        # Right
        viol = r[i] - (w - cx[i])
        if viol > 0:
            pen += viol**2
        # Bottom
        viol = r[i] - cy[i]
        if viol > 0:
            pen += viol**2
        # Top
        viol = r[i] - (h - cy[i])
        if viol > 0:
            pen += viol**2
            
    # Overlap Penalty
    for i in range(n):
        dx = cx[i] - cx[i+1:]
        dy = cy[i] - cy[i+1:]
        dist = np.sqrt(dx*dx + dy*dy)
        min_dist = r[i] + r[i+1:]
        overlap = min_dist - dist
        # Only penalize if overlapping
        pos_overlap = np.maximum(0, overlap)
        pen += np.sum(pos_overlap**2)
        
    return obj + beta * pen

def _optimize_penalty(circles, w, h, deadline, beta_start=1.0, beta_end=1000.0, n_steps=10):
    """
    Optimize using L-BFGS-B with increasing penalty weight.
    """
    n = N
    x0 = np.concatenate([circles[:, 0], circles[:, 1], circles[:, 2]])
    
    # Bounds
    bounds = []
    for i in range(n):
        bounds.append((0.0, w)) # x
    for i in range(n):
        bounds.append((0.0, h)) # y
    for i in range(n):
        bounds.append((1e-6, min(w, h)/2)) # r
        
    betas = np.geomspace(beta_start, beta_end, n_steps)
    current_x = x0
    
    for beta in betas:
        if time.time() > deadline - 10.0:
            break
            
        res = minimize(
            _penalty_objective,
            current_x,
            args=(w, h, beta),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if res.success:
            current_x = res.x
        else:
            # If it fails, keep previous
            pass
            
    new_circles = np.array([
        (current_x[i], current_x[n+i], current_x[2*n+i]) for i in range(n)
    ])
    
    # Relax to ensure validity before returning
    new_circles = _relax_circles(new_circles, w, h, iterations=200)
    
    if _validate(new_circles, w, h, tol=1e-6):
        return new_circles
    
    return None

def _relax_circles(circles, w, h, iterations=500, tol=1e-8):
    """
    Relaxation algorithm to resolve overlaps and boundary violations.
    """
    n = len(circles)
    pos = np.array(circles[:, :2], dtype=float)
    rad = np.array(circles[:, 2], dtype=float)
    
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
        
    # Final boundary clamp
    for i in range(n):
        pos[i, 0] = np.clip(pos[i, 0], rad[i], w - rad[i])
        pos[i, 1] = np.clip(pos[i, 1], rad[i], h - rad[i])
        
    new_circles = np.column_stack((pos, rad))
    
    if _validate(new_circles, w, h, tol=1e-6):
        return new_circles
    
    return circles

def _slsqp_polish(circles, w, h, deadline):
    """
    Final high-precision polish using SLSQP with hard constraints.
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
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3] - x[i*3+2]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: w - x[i*3] - x[i*3+2]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3+1] - x[i*3+2]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: h - x[i*3+1] - x[i*3+2]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3+2] - 1e-6})
        
    # Overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            constraints.append({
                'type': 'ineq', 
                'fun': lambda x, i=i, j=j: np.sqrt((x[i*3]-x[j*3])**2 + (x[i*3+1]-x[j*3+1])**2) - x[i*3+2] - x[j*3+2]
            })
    
    # Bounds
    bounds = []
    for i in range(n):
        bounds.append((1e-6, w))
        bounds.append((1e-6, h))
        bounds.append((1e-6, min(w, h)/2))
    
    try:
        res = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 100, 'ftol': 1e-12}
        )
        
        if res.success:
            new_circles = np.array([(res.x[i*3], res.x[i*3+1], res.x[i*3+2]) for i in range(n)])
            if _validate(new_circles, w, h, tol=1e-7):
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
    # Hexagonal packing suggests w/h ~ 1.15
    aspect_ratios = [
        (1.0, 1.0),
        (0.95, 1.05),
        (1.05, 0.95),
        (0.9, 1.1),
        (1.1, 0.9),
        (1.15, 0.85),
        (0.85, 1.15),
        (1.2, 0.8),
        (0.8, 1.2),
    ]
    
    for w, h in aspect_ratios:
        if time.time() > deadline - 20.0:
            break
            
        # 1. Initialize with hexagonal grid
        init_circles = _hexagonal_init(w, h)
        
        # 2. Relax to ensure initial validity
        valid_init = _relax_circles(init_circles, w, h, iterations=100)
        
        if not _validate(valid_init, w, h, tol=1e-5):
            continue
            
        current_best_for_ratio = valid_init
        current_score = _total_radius(valid_init)
        
        # 3. Penalty-based optimization (L-BFGS-B)
        if time.time() < deadline - 30.0:
            opt_circles = _optimize_penalty(valid_init, w, h, deadline)
            if opt_circles is not None:
                opt_score = _total_radius(opt_circles)
                if opt_score > current_score:
                    current_best_for_ratio = opt_circles
                    current_score = opt_score
        
        # 4. SLSQP Polish
        if time.time() < deadline - 10.0:
            polished = _slsqp_polish(current_best_for_ratio, w, h, deadline)
            if polished is not None:
                pol_score = _total_radius(polished)
                if pol_score > current_score:
                    current_best_for_ratio = polished
                    current_score = pol_score
        
        if current_score > best_score:
            best_score = current_score
            best_circles = current_best_for_ratio
    
    if best_circles is not None:
        return np.array(best_circles)
    
    # Fallback: simple grid
    side = 5
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.05) 
             for i in range(side) for j in range(side)]
    return np.array(cells[:N])

# EVOLVE-BLOCK-END
