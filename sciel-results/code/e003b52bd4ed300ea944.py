"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    best_score = -1
    best_result = None
    
    # Perimeter 4 => W + H = 2
    # We will try a range of aspect ratios.
    # Focus on near-square and slightly rectangular to accommodate hexagonal packing.
    
    aspect_ratios = [1.0, 0.95, 1.05, 0.9, 1.1, 0.85, 1.15, 0.8, 1.2, 0.75, 1.3, 0.7, 1.4]
    
    # Strategy: Continuous optimization with L-BFGS-B
    # Variables: x, y, r for each circle (3*N variables)
    # Objective: -sum(r_i)
    # Constraints:
    #   1. r_i >= 0
    #   2. x_i - r_i >= 0
    #   3. x_i + r_i <= w
    #   4. y_i - r_i >= 0
    #   5. y_i + r_i <= h
    #   6. (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
    
    # Since L-BFGS-B handles bounds easily but not non-linear constraints directly,
    # we use a penalty method or convert to an unconstrained problem with penalties.
    # Alternatively, use SLSQP which can handle non-linear constraints, but it's slower.
    # Let's try a penalty approach with L-BFGS-B as it's faster and robust for large N.
    
    def _check_validity(result, w, h, tol=1e-6):
        """Check if the result is valid."""
        if result is None:
            return False
        n = result.shape[0]
        if n != N:
            return False
        for i in range(n):
            cx, cy, r = result[i]
            if r < -tol:
                return False
            if cx - r < -tol or cx + r > w + tol or cy - r < -tol or cy + r > h + tol:
                return False
        for i in range(n):
            for j in range(i+1, n):
                dx = result[i, 0] - result[j, 0]
                dy = result[i, 1] - result[j, 1]
                dist = math.hypot(dx, dy)
                if dist < result[i, 2] + result[j, 2] - tol:
                    return False
        return True

    def _separate_circles(result, w, h, n_iters=100):
        """
        Attempt to separate overlapping circles by pushing them apart.
        This is a simple heuristic repair step.
        """
        if result is None:
            return None
        result = result.copy()
        
        for _ in range(n_iters):
            moved = False
            for i in range(N):
                for j in range(i+1, N):
                    dx = result[j, 0] - result[i, 0]
                    dy = result[j, 1] - result[i, 1]
                    dist = math.hypot(dx, dy)
                    min_dist = result[i, 2] + result[j, 2]
                    
                    if dist < min_dist:
                        # Overlap detected
                        if dist < 1e-9:
                            # If centers are same, push randomly
                            angle = np.random.uniform(0, 2*np.pi)
                            dx = math.cos(angle)
                            dy = math.sin(angle)
                            dist = 1e-9
                        else:
                            dx /= dist
                            dy /= dist
                        
                        # Push apart
                        push = (min_dist - dist) / 2.0 + 1e-4
                        result[i, 0] -= dx * push
                        result[i, 1] -= dy * push
                        result[j, 0] += dx * push
                        result[j, 1] += dy * push
                        moved = True
            
            # Project back to rectangle
            for i in range(N):
                r = result[i, 2]
                if result[i, 0] < r:
                    result[i, 0] = r
                if result[i, 0] > w - r:
                    result[i, 0] = w - r
                if result[i, 1] < r:
                    result[i, 1] = r
                if result[i, 1] > h - r:
                    result[i, 1] = h - r
            
            if not moved:
                break
                
        return result

    def _objective_and_grad(x, w, h, penalty_weight=1000.0):
        """
        Objective function with penalty for constraints.
        x: array of shape (3*N,)
        Returns: objective value, gradient
        """
        n = len(x) // 3
        cx = x[0::3]
        cy = x[1::3]
        r = x[2::3]
        
        obj = -np.sum(r)
        grad = np.zeros_like(x)
        grad[2::3] = -1.0
        
        # Penalty for boundary violations
        for i in range(n):
            # x_i - r_i >= 0
            viol = r[i] - cx[i]
            if viol > 0:
                obj += penalty_weight * viol**2
                # d/d cx: -2 * penalty * viol
                grad[i] -= 2 * penalty_weight * viol
                # d/d r: 2 * penalty * viol
                grad[i*3 + 2] += 2 * penalty_weight * viol
            
            # x_i + r_i <= w
            viol = cx[i] + r[i] - w
            if viol > 0:
                obj += penalty_weight * viol**2
                grad[i] += 2 * penalty_weight * viol
                grad[i*3 + 2] += 2 * penalty_weight * viol
            
            # y_i - r_i >= 0
            viol = r[i] - cy[i]
            if viol > 0:
                obj += penalty_weight * viol**2
                grad[i*3 + 1] -= 2 * penalty_weight * viol
                grad[i*3 + 2] += 2 * penalty_weight * viol
            
            # y_i + r_i <= h
            viol = cy[i] + r[i] - h
            if viol > 0:
                obj += penalty_weight * viol**2
                grad[i*3 + 1] += 2 * penalty_weight * viol
                grad[i*3 + 2] += 2 * penalty_weight * viol
        
        # Penalty for overlap
        for i in range(n):
            for j in range(i+1, n):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                dist_sq = dx*dx + dy*dy
                min_dist = r[i] + r[j]
                
                if dist_sq < min_dist**2:
                    dist = math.sqrt(dist_sq)
                    if dist < 1e-9:
                        # Avoid division by zero, use random direction or skip
                        # This is a singularity, handle carefully
                        continue
                    
                    # Violation: min_dist^2 - dist^2
                    viol = min_dist**2 - dist_sq
                    if viol > 0:
                        obj += penalty_weight * viol**2
                        
                        # Gradient w.r.t cx[i], cy[i]
                        # d(viol)/d cx[i] = 2*dx
                        # d(viol)/d cy[i] = 2*dy
                        # d(viol)/d cx[j] = -2*dx
                        # d(viol)/d cy[j] = -2*dy
                        # d(viol)/d r[i] = 2*min_dist
                        # d(viol)/d r[j] = 2*min_dist
                        
                        # grad_obj = 2 * penalty * viol * grad_viol
                        factor = 2 * penalty_weight * viol
                        grad[i] += factor * 2*dx
                        grad[i*3 + 1] += factor * 2*dy
                        grad[j] -= factor * 2*dx
                        grad[j*3 + 1] -= factor * 2*dy
                        grad[i*3 + 2] += factor * 2*min_dist
                        grad[j*3 + 2] += factor * 2*min_dist
        
        return obj, grad

    def _local_optimize(x0, w, h, max_iter=1000):
        """
        Run L-BFGS-B optimization.
        """
        # Bounds:
        # x_i in [0, w], y_i in [0, h], r_i in [0, min(w,h)/2]
        bounds = []
        r_max = min(w, h) / 2.0
        for _ in range(N):
            bounds.append((0, w))
            bounds.append((0, h))
            bounds.append((1e-6, r_max))
        
        options = {'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8}
        
        try:
            res = minimize(_objective_and_grad, x0, method='L-BFGS-B', 
                           jac=True, bounds=bounds, options=options)
            return res.x
        except Exception:
            return x0

    def _polish(x, w, h, n_sweeps=5):
        """
        Perform a few sweeps of local optimization with tighter tolerances.
        """
        for _ in range(n_sweeps):
            if time.time() > deadline - 10:
                break
            x = _local_optimize(x, w, h, max_iter=200)
        return x

    def _grid_initialization(w, h):
        """
        Generate a grid-based initial guess.
        """
        # Try to fit a grid of circles
        # Approximate number of circles per side
        n_side = int(math.sqrt(N))
        if n_side * n_side < N:
            n_side += 1
            
        # Create a grid of points
        xs = np.linspace(w / (2*n_side), w - w / (2*n_side), n_side)
        ys = np.linspace(h / (2*n_side), h - h / (2*n_side), n_side)
        
        points = []
        for i in range(n_side):
            for j in range(n_side):
                if len(points) >= N:
                    break
                points.append((xs[i], ys[j]))
        
        # If we don't have enough, add more in a different pattern
        # Or just use the first N points
        if len(points) < N:
            # Add points in a spiral or just random
            while len(points) < N:
                x = np.random.uniform(0.1*w, 0.9*w)
                y = np.random.uniform(0.1*h, 0.9*h)
                points.append((x, y))
        
        points = points[:N]
        
        # Estimate radius
        # Min distance between neighbors
        min_dist = float('inf')
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                dx = points[i][0] - points[j][0]
                dy = points[i][1] - points[j][1]
                dist = math.hypot(dx, dy)
                if dist < min_dist:
                    min_dist = dist
        
        r0 = min_dist / 2.0 * 0.9  # Slightly smaller to avoid initial overlap
        
        x0 = np.zeros(3*N)
        for i in range(N):
            x0[i*3] = points[i][0]
            x0[i*3 + 1] = points[i][1]
            x0[i*3 + 2] = r0
            
        return x0

    def _random_initialization(w, h):
        """
        Generate a random initial guess.
        """
        x0 = np.zeros(3*N)
        r_max = min(w, h) / 2.0
        for i in range(N):
            x0[i*3] = np.random.uniform(0, w)
            x0[i*3 + 1] = np.random.uniform(0, h)
            x0[i*3 + 2] = np.random.uniform(0.05, 0.2) # Small initial radii
        
        return x0

    def _successive_convex_relaxation(x, w, h, n_steps=5):
        """
        Successive convex relaxation: increase penalty weight gradually.
        """
        penalty_weights = [100, 500, 1000, 5000, 10000]
        for i in range(n_steps):
            if time.time() > deadline - 10:
                break
            pw = penalty_weights[min(i, len(penalty_weights)-1)]
            
            def _obj_grad_pw(x, w, h, pw):
                return _objective_and_grad(x, w, h, penalty_weight=pw)
            
            bounds = []
            r_max = min(w, h) / 2.0
            for _ in range(N):
                bounds.append((0, w))
                bounds.append((0, h))
                bounds.append((1e-6, r_max))
            
            try:
                res = minimize(_obj_grad_pw, x, method='L-BFGS-B', 
                               args=(w, h, pw), jac=True, bounds=bounds, 
                               options={'maxiter': 200, 'ftol': 1e-12})
                x = res.x
            except Exception:
                pass
        return x

    def _best_known(w, h):
        """
        Return a known good packing if available.
        """
        # For now, return a grid initialization
        return _grid_initialization(w, h)

    for w_ratio in aspect_ratios:
        if time.time() > deadline:
            break
            
        w = 2.0 * w_ratio / (1.0 + w_ratio)
        h = 2.0 - w
        
        # Ensure w, h > 0
        if w <= 0 or h <= 0:
            continue
            
        # Try multiple initializations
        inits = [
            _grid_initialization(w, h),
            _random_initialization(w, h),
            _random_initialization(w, h)
        ]
        
        for x0 in inits:
            if time.time() > deadline:
                break
            
            # Successive convex relaxation
            x = _successive_convex_relaxation(x0, w, h, n_steps=3)
            
            # Polish
            x = _polish(x, w, h, n_sweeps=2)
            
            # Convert to result array
            result = np.zeros((N, 3))
            result[:, 0] = x[0::3]
            result[:, 1] = x[1::3]
            result[:, 2] = x[2::3]
            
            # Clip radii to be non-negative
            result[:, 2] = np.maximum(result[:, 2], 0)
            
            # Separate circles if they overlap
            result = _separate_circles(result, w, h, n_iters=50)
            
            # Validate
            if _check_validity(result, w, h):
                score = np.sum(result[:, 2])
                if score > best_score:
                    best_score = score
                    best_result = result

    # Fallback: simple grid if optimization failed
    if best_result is None:
        # 5x5 grid in 1x1 square
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                cx = (i + 0.5) / side
                cy = (j + 0.5) / side
                r = 0.5 / side - 1e-4
                cells.append((cx, cy, r))
        best_result = np.array(cells[:N])
        
    return best_result
# EVOLVE-BLOCK-END
