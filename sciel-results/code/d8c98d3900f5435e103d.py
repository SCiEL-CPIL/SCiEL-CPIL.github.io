import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 26

def construct():
    # We will use a hybrid approach:
    # 1. Greedy initialization using Delaunay triangulation to place circles.
    # 2. Local optimization using L-BFGS-B to maximize sum of radii.
    
    def greedy_initialize(n):
        # Start with a few large circles in the corners to seed the layout
        # This helps avoid degenerate triangulations initially.
        # A robust seed: 4 corners and 1 center? Or just a small grid.
        # Let's start with 4 circles in the corners.
        circles = [
            [0.05, 0.05, 0.05],
            [0.95, 0.05, 0.05],
            [0.05, 0.95, 0.05],
            [0.95, 0.95, 0.05]
        ]
        
        for _ in range(n - len(circles)):
            best_circle = None
            best_radius = 0.0
            
            # Grid search for the largest empty circle
            # Resolution of 20x20 is usually sufficient for initialization
            grid_res = 30
            xs = np.linspace(0, 1, grid_res)
            ys = np.linspace(0, 1, grid_res)
            
            for x in xs:
                for y in ys:
                    # Check boundary constraints
                    r_bound = min(x, 1 - x, y, 1 - y)
                    if r_bound <= best_radius:
                        continue
                    
                    # Check overlap with existing circles
                    r_max = r_bound
                    for cx, cy, cr in circles:
                        dx = x - cx
                        dy = y - cy
                        dist = np.sqrt(dx*dx + dy*dy)
                        # The new circle must not overlap with existing ones
                        # dist >= r_new + cr  => r_new <= dist - cr
                        r_allowed = dist - cr
                        if r_allowed < r_max:
                            r_max = r_allowed
                            if r_max <= best_radius:
                                break
                    
                    if r_max > best_radius:
                        best_radius = r_max
                        best_circle = [x, y, r_max]
            
            if best_circle is not None and best_radius > 1e-6:
                circles.append(best_circle)
            else:
                # Fallback: place a tiny circle at a random valid spot or center
                # Try center first
                cx, cy = 0.5, 0.5
                r_bound = min(cx, 1-cx, cy, 1-cy)
                r_max = r_bound
                for cx_, cy_, cr_ in circles:
                    dx = cx - cx_
                    dy = cy - cy_
                    dist = np.sqrt(dx*dx + dy*dy)
                    r_max = min(r_max, dist - cr_)
                
                if r_max > 1e-6:
                    circles.append([cx, cy, r_max])
                else:
                    # If even center doesn't work, add a very small circle at a random point
                    # This is a failure case for greedy, but we need N circles
                    # We'll rely on the optimizer to fix it
                    circles.append([0.5, 0.5, 0.01])

        return np.array(circles)

    # 1. Generate initial configuration using Greedy Method
    init_circles = greedy_initialize(N)
    
    # Ensure we have exactly N circles
    while len(init_circles) < N:
        init_circles = np.vstack([init_circles, [[0.5, 0.5, 0.01]]])
        
    x0 = init_circles[:, 0]
    y0 = init_circles[:, 1]
    r0 = init_circles[:, 2]
    
    # 2. Optimize using L-BFGS-B
    # Variables: x1, y1, r1, ..., xN, yN, rN
    
    def energy(params):
        x_ = params[0::3]
        y_ = params[1::3]
        r_ = params[2::3]
        
        # 1. Boundary violations
        pen_bound = 0.0
        for i in range(N):
            pen_bound += max(0, r_[i] - x_[i])**2
            pen_bound += max(0, r_[i] - (1 - x_[i]))**2
            pen_bound += max(0, r_[i] - y_[i])**2
            pen_bound += max(0, r_[i] - (1 - y_[i]))**2
            
        # 2. Overlap violations
        pen_overlap = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dist_sq = (x_[i] - x_[j])**2 + (y_[i] - y_[j])**2
                min_dist = r_[i] + r_[j]
                if dist_sq < min_dist**2:
                    overlap = min_dist - np.sqrt(dist_sq)
                    pen_overlap += overlap**2
                    
        lam = 100.0
        return -np.sum(r_) + lam * (pen_bound + pen_overlap)
        
    # We don't provide gradient, let L-BFGS-B use numerical diff or we provide it.
    # Providing gradient is faster.
    def grad_energy(params):
        # Numerical gradient is slow for 78 vars * 100 iters.
        # Let's try analytic or just let scipy handle it with '2-point' which is default?
        # Actually, let's just use the default numerical gradient in minimize for simplicity,
        # but L-BFGS-B with 78 vars is manageable.
        return None

    params0 = np.zeros(3*N)
    params0[0::3] = x0
    params0[1::3] = y0
    params0[2::3] = r0
    
    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((0, 0.5))
        
    # Run optimization
    # Using L-BFGS-B
    res = minimize(energy, params0, method='L-BFGS-B', bounds=bounds, 
                   options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-5})
    
    params_opt = res.x
    x_opt = params_opt[0::3]
    y_opt = params_opt[1::3]
    r_opt = params_opt[2::3]
    
    # Post-processing: Ensure strict validity and maximize radii locally
    # We perform a local search to adjust radii upwards if possible
    for _ in range(50):
        changed = False
        # First, fix any hard violations by shrinking
        for i in range(N):
            r_max_bound = min(x_opt[i], 1-x_opt[i], y_opt[i], 1-y_opt[i])
            if r_opt[i] > r_max_bound:
                r_opt[i] = max(0, r_max_bound)
                changed = True
                
        for i in range(N):
            for j in range(i+1, N):
                dist = np.sqrt((x_opt[i]-x_opt[j])**2 + (y_opt[i]-y_opt[j])**2)
                min_dist = r_opt[i] + r_opt[j]
                if dist < min_dist:
                    overlap = min_dist - dist
                    # Scale down both radii proportionally
                    if min_dist > 1e-10:
                        scale = dist / min_dist
                        r_opt[i] *= scale
                        r_opt[j] *= scale
                    else:
                        r_opt[i] = 0
                        r_opt[j] = 0
                    changed = True
        
        # Now, try to increase radii
        for i in range(N):
            # Max radius allowed by boundaries
            r_max_bound = min(x_opt[i], 1-x_opt[i], y_opt[i], 1-y_opt[i])
            r_max = r_max_bound
            
            # Max radius allowed by other circles
            for j in range(N):
                if i == j:
                    continue
                dist = np.sqrt((x_opt[i]-x_opt[j])**2 + (y_opt[i]-y_opt[j])**2)
                # dist >= r_i + r_j  => r_i <= dist - r_j
                r_allowed = dist - r_opt[j]
                if r_allowed < r_max:
                    r_max = r_allowed
            
            if r_max > r_opt[i] + 1e-6:
                r_opt[i] = r_max
                changed = True
            elif r_max < r_opt[i]:
                r_opt[i] = max(0, r_max)
                changed = True
        
        if not changed:
            break
            
    for i in range(N):
        r_opt[i] = min(r_opt[i], x_opt[i], 1-x_opt[i], y_opt[i], 1-y_opt[i])
        
    result = np.zeros((N, 3))
    result[:, 0] = x_opt
    result[:, 1] = y_opt
    result[:, 2] = r_opt
    
    return result
