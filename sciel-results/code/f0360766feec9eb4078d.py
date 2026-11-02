"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii."""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    def get_rect_dims(t):
        """Calculate width and height from aspect ratio parameter t."""
        # t is in (0, 1). w = 2t, h = 2(1-t). Perimeter = 2(w+h) = 4.
        t = np.clip(t, 0.05, 0.95)
        w = 2.0 * t
        h = 2.0 * (1.0 - t)
        return w, h

    def check_valid(circles, w, h):
        """Check if the configuration is strictly valid."""
        n = len(circles)
        if n != N:
            return False
        for i in range(n):
            x, y, r = circles[i]
            if r < 1e-6:
                return False
            # Boundary check with small epsilon
            if x - r < -1e-7 or x + r > w + 1e-7 or y - r < -1e-7 or y + r > h + 1e-7:
                return False
        
        # Overlap check
        for i in range(n):
            xi, yi, ri = circles[i]
            for j in range(i + 1, n):
                xj, yj, rj = circles[j]
                dist_sq = (xi - xj)**2 + (yi - yj)**2
                min_dist = ri + rj
                if dist_sq < (min_dist - 1e-7)**2:
                    return False
        return True

    def evaluate_value(circles, w, h):
        """Return sum of radii if valid, else -inf."""
        if not check_valid(circles, w, h):
            return -np.inf
        return float(np.sum(circles[:, 2]))

    def softplus(x, beta=100.0):
        """Numerically stable softplus: log(1+exp(beta*x))/beta"""
        x_beta = beta * x
        # For large positive x, softplus ≈ x
        # For large negative x, softplus ≈ 0
        # Use log1p(exp()) for stability
        result = np.where(x_beta > 0, 
                         x + np.log1p(np.exp(-x_beta)) / beta,
                         np.log1p(np.exp(x_beta)) / beta)
        return result

    def softplus_grad(x, beta=100.0):
        """Gradient of softplus: sigmoid(beta*x)"""
        # Clip to avoid overflow in exp
        x_beta = np.clip(beta * x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x_beta))

    def optimize(circles0, w, h, max_iter=300, beta=100.0, time_limit=10.0):
        """
        Optimize from initial configuration using smooth softplus penalties.
        Variables: x0 = [x1, y1, r1, ..., xN, yN, rN]
        Objective: minimize -sum(ri) + penalty
        """
        x0 = circles0.flatten()
        
        def objective_and_grad(x):
            c = x.reshape(-1, 3)
            # We want to maximize sum(ri), so minimize -sum(ri)
            val = -np.sum(c[:, 2])
            
            # Penalty terms using softplus for smoothness
            penalty = 0.0
            grad = np.zeros_like(x)
            
            # Boundary constraints:
            # 1. xi - ri >= 0  => v1 = ri - xi
            # 2. xi + ri <= w  => v2 = xi + ri - w
            # 3. yi - ri >= 0  => v3 = ri - yi
            # 4. yi + ri <= h  => v4 = yi + ri - h
            # 5. ri >= 0       => v5 = -ri
            
            for i in range(N):
                xi, yi, ri = c[i]
                
                # v1 = ri - xi
                v1 = ri - xi
                penalty += softplus(v1, beta)
                g1 = softplus_grad(v1, beta)
                # d(v1)/dxi = -1, d(v1)/dri = 1
                grad[3*i] += -g1
                grad[3*i+2] += g1
                
                # v2 = xi + ri - w
                v2 = xi + ri - w
                penalty += softplus(v2, beta)
                g2 = softplus_grad(v2, beta)
                # d(v2)/dxi = 1, d(v2)/dri = 1
                grad[3*i] += g2
                grad[3*i+2] += g2
                
                # v3 = ri - yi
                v3 = ri - yi
                penalty += softplus(v3, beta)
                g3 = softplus_grad(v3, beta)
                # d(v3)/dyi = -1, d(v3)/dri = 1
                grad[3*i+1] += -g3
                grad[3*i+2] += g3
                
                # v4 = yi + ri - h
                v4 = yi + ri - h
                penalty += softplus(v4, beta)
                g4 = softplus_grad(v4, beta)
                # d(v4)/dyi = 1, d(v4)/dri = 1
                grad[3*i+1] += g4
                grad[3*i+2] += g4
                
                # v5 = -ri
                v5 = -ri
                penalty += softplus(v5, beta)
                g5 = softplus_grad(v5, beta)
                # d(v5)/dri = -1
                grad[3*i+2] += -g5
            
            # Pairwise non-overlap: ri + rj <= dist(i,j)
            # v = ri + rj - d
            for i in range(N):
                xi, yi, ri = c[i]
                for j in range(i + 1, N):
                    xj, yj, rj = c[j]
                    dx = xi - xj
                    dy = yi - yj
                    d_sq = dx*dx + dy*dy
                    # Avoid division by zero
                    if d_sq < 1e-12:
                        d = 1e-6
                        dx_norm = 1.0
                        dy_norm = 0.0
                    else:
                        d = np.sqrt(d_sq)
                        dx_norm = dx / d
                        dy_norm = dy / d
                    
                    v = ri + rj - d
                    penalty += softplus(v, beta)
                    g = softplus_grad(v, beta)
                    
                    # d(v)/dxi = -dx/d = -dx_norm
                    # d(v)/dyi = -dy/d = -dy_norm
                    # d(v)/dri = 1
                    # d(v)/dxj = dx/d = dx_norm
                    # d(v)/dyj = dy/d = dy_norm
                    # d(v)/drj = 1
                    
                    grad[3*i] += -g * dx_norm
                    grad[3*i+1] += -g * dy_norm
                    grad[3*i+2] += g
                    grad[3*j] += g * dx_norm
                    grad[3*j+1] += g * dy_norm
                    grad[3*j+2] += g
            
            # Objective: minimize -sum(ri) + penalty
            obj = val + penalty
            
            # Gradient of -sum(ri) w.r.t. ri is -1
            for i in range(N):
                grad[3*i+2] += -1.0
            
            return obj, grad

        try:
            # Use L-BFGS-B with analytical gradient
            # Bounds: x, y can be anywhere, but r should be positive.
            # We handle r>0 via penalty, but setting bounds helps stability.
            bounds = []
            for i in range(N):
                bounds.append((None, None)) # x
                bounds.append((None, None)) # y
                bounds.append((1e-6, None)) # r (keep positive)
            
            # Check time before starting heavy optimization
            if time.time() > deadline - time_limit:
                return circles0, evaluate_value(circles0, w, h)

            result = minimize(objective_and_grad, x0, method='L-BFGS-B', jac=True,
                             bounds=bounds,
                             options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 10000})
            
            circles = result.x.reshape(-1, 3)
            val = evaluate_value(circles, w, h)
            return circles, val
            
        except Exception as e:
            # If optimization fails, return original
            return circles0, evaluate_value(circles0, w, h)

    def generate_initial_grid(aspect, seed=0):
        """
        Generate a valid initial grid packing for a given aspect ratio.
        aspect = w/h.
        w + h = 2.
        w = 2 * aspect / (1 + aspect)
        h = 2 / (1 + aspect)
        """
        rng = np.random.RandomState(seed)
        w = 2.0 * aspect / (1.0 + aspect)
        h = 2.0 / (1.0 + aspect)
        
        # Try different grid arrangements to find the best initial valid state
        best_circles = None
        best_val = -np.inf
        
        # Possible grid configurations (cols, rows) such that cols*rows >= N
        # We want to maximize the minimum cell size to get larger initial radii
        for cols in range(1, N + 1):
            rows = int(np.ceil(N / cols))
            if cols * rows < N:
                continue
            
            cell_w = w / cols
            cell_h = h / rows
            
            # Initial radius is limited by cell size
            max_r = 0.5 * min(cell_w, cell_h) - 1e-4
            if max_r < 1e-3:
                continue
            
            circles = []
            for i in range(N):
                c = i % cols
                r = i // cols
                cx = (c + 0.5) * cell_w
                cy = (r + 0.5) * cell_h
                rad = max_r
                circles.append([cx, cy, rad])
            
            circles = np.array(circles)
            val = evaluate_value(circles, w, h)
            
            if val > best_val:
                best_val = val
                best_circles = circles.copy()
        
        if best_circles is None:
            # Fallback: single row or column if grid fails (should not happen for N=21)
            # Just place them in a line with tiny radii
            best_circles = np.zeros((N, 3))
            for i in range(N):
                best_circles[i, 0] = (i + 0.5) * w / N
                best_circles[i, 1] = 0.5 * h
                best_circles[i, 2] = 0.5 * min(w/N, h) - 1e-4
            
        return best_circles, w, h, best_val

    best_value = -np.inf
    best_result = None
    best_wh = None

    # List of aspect ratios to try
    # aspect = w/h
    aspects = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.5, 0.5, 2.0, 0.4, 2.5, 0.3, 3.0, 4.0, 0.25, 1.1, 0.9]
    
    # Shuffle aspects to add variety? Or keep ordered? Ordered is fine, but we can randomize seed per aspect.
    random.seed(42)
    random.shuffle(aspects)

    for idx, aspect in enumerate(aspects):
        if time.time() > deadline - 15: # Leave 15s for final polish and check
            break
        
        # Generate initial solution
        circles0, w, h, val0 = generate_initial_grid(aspect, seed=idx)
        
        if val0 > best_value:
            best_value = val0
            best_result = circles0.copy()
            best_wh = (w, h)
        
        # Optimize
        # Allocate more time for earlier aspects? Or fixed?
        # Let's give each aspect a fixed chunk of time, decreasing as we go
        time_left = deadline - time.time()
        time_alloc = min(8.0, time_left * 0.5) # Max 8s per aspect, or half remaining
        
        if time_alloc < 1.0:
            continue
            
        circles, val = optimize(circles0, w, h, max_iter=200, beta=100.0, time_limit=time_alloc)
        
        if val > best_value:
            best_value = val
            best_result = circles.copy()
            best_wh = (w, h)
        
        # Try a second pass with different beta or higher precision if time permits
        if time.time() < deadline - 10 and val > -np.inf:
             circles2, val2 = optimize(circles, w, h, max_iter=100, beta=200.0, time_limit=min(5.0, deadline - time.time() - 5))
             if val2 > best_value:
                 best_value = val2
                 best_result = circles2.copy()
                 best_wh = (w, h)

    # Final Polish
    if best_result is not None and best_wh is not None and time.time() < deadline - 5:
        w, h = best_wh
        # High precision polish
        polished, val = polish_pass(best_result, w, h, max_iter=500, beta=500.0, time_limit=min(10.0, deadline - time.time() - 2))
        if val > best_value:
            best_value = val
            best_result = polished.copy()
            best_wh = (w, h)

    # Fallback if no valid solution found (should not happen with grid init)
    if best_result is None:
        # Create a trivial valid solution
        best_result = np.zeros((N, 3))
        w, h = 1.0, 1.0
        for i in range(N):
            best_result[i, 0] = 0.5
            best_result[i, 1] = 0.5
            best_result[i, 2] = 1e-4
        best_value = float(np.sum(best_result[:, 2]))
        best_wh = (w, h)
        
    # Ensure the best result is valid. If not, try to repair by shrinking radii.
    w_final, h_final = best_wh
    if not check_valid(best_result, w_final, h_final):
        # Repair: shrink radii until valid
        repair_state = best_result.copy()
        for _ in range(100):
            if check_valid(repair_state, w_final, h_final):
                break
            for i in range(N):
                repair_state[i, 2] *= 0.95
        if check_valid(repair_state, w_final, h_final):
            best_result = repair_state
            best_value = float(np.sum(best_result[:, 2]))
        else:
            # Last resort: tiny circles in a grid
            best_result = np.zeros((N, 3))
            side = int(np.ceil(np.sqrt(N)))
            for i in range(N):
                cx = (i % side + 0.5) * w_final / side
                cy = (i // side + 0.5) * h_final / side
                best_result[i, 0] = cx
                best_result[i, 1] = cy
                best_result[i, 2] = 1e-4

    return best_result

def polish_pass(circles0, w, h, max_iter=500, beta=500.0, time_limit=10.0):
    """High-precision polish pass."""
    # Reuse the optimize logic but with stricter parameters
    # We define it locally to avoid closure issues or just call optimize
    # Since optimize is nested in construct, we need to duplicate or refactor.
    # For simplicity and speed, let's just call a similar logic here.
    
    def objective_and_grad(x):
        c = x.reshape(-1, 3)
        val = -np.sum(c[:, 2])
        penalty = 0.0
        grad = np.zeros_like(x)
        
        for i in range(N):
            xi, yi, ri = c[i]
            v1 = ri - xi
            penalty += softplus(v1, beta)
            g1 = softplus_grad(v1, beta)
            grad[3*i] += -g1
            grad[3*i+2] += g1
            
            v2 = xi + ri - w
            penalty += softplus(v2, beta)
            g2 = softplus_grad(v2, beta)
            grad[3*i] += g2
            grad[3*i+2] += g2
            
            v3 = ri - yi
            penalty += softplus(v3, beta)
            g3 = softplus_grad(v3, beta)
            grad[3*i+1] += -g3
            grad[3*i+2] += g3
            
            v4 = yi + ri - h
            penalty += softplus(v4, beta)
            g4 = softplus_grad(v4, beta)
            grad[3*i+1] += g4
            grad[3*i+2] += g4
            
            v5 = -ri
            penalty += softplus(v5, beta)
            g5 = softplus_grad(v5, beta)
            grad[3*i+2] += -g5
            
        for i in range(N):
            xi, yi, ri = c[i]
            for j in range(i + 1, N):
                xj, yj, rj = c[j]
                dx = xi - xj
                dy = yi - yj
                d_sq = dx*dx + dy*dy
                if d_sq < 1e-12:
                    d = 1e-6
                    dx_norm = 1.0
                    dy_norm = 0.0
                else:
                    d = np.sqrt(d_sq)
                    dx_norm = dx / d
                    dy_norm = dy / d
                
                v = ri + rj - d
                penalty += softplus(v, beta)
                g = softplus_grad(v, beta)
                
                grad[3*i] += -g * dx_norm
                grad[3*i+1] += -g * dy_norm
                grad[3*i+2] += g
                grad[3*j] += g * dx_norm
                grad[3*j+1] += g * dy_norm
                grad[3*j+2] += g
        
        obj = val + penalty
        for i in range(N):
            grad[3*i+2] += -1.0
        return obj, grad

    x0 = circles0.flatten()
    bounds = []
    for i in range(N):
        bounds.append((None, None))
        bounds.append((None, None))
        bounds.append((1e-6, None))
        
    try:
        result = minimize(objective_and_grad, x0, method='L-BFGS-B', jac=True,
                         bounds=bounds,
                         options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 20000})
        circles = result.x.reshape(-1, 3)
        val = evaluate_value(circles, w, h)
        return circles, val
    except:
        return circles0, evaluate_value(circles0, w, h)

# EVOLVE-BLOCK-END
