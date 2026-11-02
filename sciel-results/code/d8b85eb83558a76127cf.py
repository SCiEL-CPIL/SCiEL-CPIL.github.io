import numpy as np
import time
from scipy.optimize import minimize

N = 21

def construct():
    start_time = time.time()
    deadline = 115.0  # safety margin before 120s
    
    def evaluate(circles, w, h):
        """Return sum of radii if valid, else -inf."""
        n = len(circles)
        for i in range(n):
            x, y, r = circles[i]
            if x - r < -1e-6 or x + r > w + 1e-6 or y - r < -1e-6 or y + r > h + 1e-6:
                return -np.inf
        for i in range(n):
            for j in range(i+1, n):
                xi, yi, ri = circles[i]
                xj, yj, rj = circles[j]
                d = np.sqrt((xi-xj)**2 + (yi-yj)**2)
                if d < ri + rj - 1e-6:
                    return -np.inf
        return float(np.sum(circles[:, 2]))
    
    def grid_solution(aspect, seed=0):
        """Generate initial grid packing for given aspect ratio."""
        rng = np.random.RandomState(seed)
        w = 2.0 * aspect / (1.0 + aspect)
        h = 2.0 / (1.0 + aspect)
        
        # Try different grid arrangements
        best = None
        for cols in range(1, N+1):
            rows = int(np.ceil(N / cols))
            if cols * rows < N:
                continue
            cell_w = w / cols
            cell_h = h / rows
            circles = []
            for i in range(N):
                c = i % cols
                r = i // cols
                cx = (c + 0.5) * cell_w
                cy = (r + 0.5) * cell_h
                rad = 0.5 * min(cell_w, cell_h) - 1e-4
                circles.append([cx, cy, rad])
            circles = np.array(circles)
            val = evaluate(circles, w, h)
            if val > (best[0] if best else -np.inf):
                best = (val, circles, w, h)
        
        if best is None:
            # Fallback: single row
            cols = N
            rows = 1
            cell_w = w / cols
            cell_h = h
            circles = []
            for i in range(N):
                cx = (i + 0.5) * cell_w
                cy = 0.5 * cell_h
                rad = 0.5 * min(cell_w, cell_h) - 1e-4
                circles.append([cx, cy, rad])
            circles = np.array(circles)
            val = evaluate(circles, w, h)
            best = (val, circles, w, h)
        
        return best[1], best[2], best[3], best[0]
    
    def softplus(x, beta=100.0):
        """Numerically stable softplus: log(1+exp(beta*x))/beta"""
        # For large positive x, softplus ≈ x
        # For large negative x, softplus ≈ 0
        x_beta = beta * x
        # Use log1p(exp()) for stability when x_beta > 0
        result = np.where(x_beta > 0, 
                         x + np.log1p(np.exp(-x_beta)) / beta,
                         np.log1p(np.exp(x_beta)) / beta)
        return result
    
    def softplus_grad(x, beta=100.0):
        """Gradient of softplus: sigmoid(beta*x)"""
        return 1.0 / (1.0 + np.exp(-beta * x))
    
    def optimize(circles0, w, h, max_iter=300, beta=100.0):
        """Optimize from initial configuration using smooth softplus penalties."""
        x0 = circles0.flatten()
        
        def objective_and_grad(x):
            c = x.reshape(-1, 3)
            val = -np.sum(c[:, 2])
            
            # Penalty terms using softplus for smoothness
            penalty = 0.0
            grad = np.zeros_like(x)
            
            # Boundary constraints: xi - ri >= 0, xi + ri <= w, yi - ri >= 0, yi + ri <= h, ri >= 0
            for i in range(N):
                xi, yi, ri = c[i]
                
                # xi - ri >= 0 => violation = ri - xi
                v1 = ri - xi
                penalty += softplus(v1, beta)
                g1 = softplus_grad(v1, beta)
                grad[3*i] += -g1
                grad[3*i+2] += g1
                
                # xi + ri <= w => violation = xi + ri - w
                v2 = xi + ri - w
                penalty += softplus(v2, beta)
                g2 = softplus_grad(v2, beta)
                grad[3*i] += g2
                grad[3*i+2] += g2
                
                # yi - ri >= 0 => violation = ri - yi
                v3 = ri - yi
                penalty += softplus(v3, beta)
                g3 = softplus_grad(v3, beta)
                grad[3*i+1] += -g3
                grad[3*i+2] += g3
                
                # yi + ri <= h => violation = yi + ri - h
                v4 = yi + ri - h
                penalty += softplus(v4, beta)
                g4 = softplus_grad(v4, beta)
                grad[3*i+1] += g4
                grad[3*i+2] += g4
                
                # ri >= 0 => violation = -ri
                v5 = -ri
                penalty += softplus(v5, beta)
                g5 = softplus_grad(v5, beta)
                grad[3*i+2] += -g5
            
            # Pairwise non-overlap: ri + rj <= dist(i,j)
            for i in range(N):
                for j in range(i+1, N):
                    xi, yi, ri = c[i]
                    xj, yj, rj = c[j]
                    dx = xi - xj
                    dy = yi - yj
                    d = np.sqrt(dx*dx + dy*dy + 1e-12)
                    
                    # violation = ri + rj - d
                    v = ri + rj - d
                    penalty += softplus(v, beta)
                    g = softplus_grad(v, beta)
                    
                    # d/dxi (ri+rj-d) = -dx/d
                    # d/dyi (ri+rj-d) = -dy/d
                    # d/dri (ri+rj-d) = 1
                    # d/dxj (ri+rj-d) = dx/d
                    # d/dyj (ri+rj-d) = dy/d
                    # d/drj (ri+rj-d) = 1
                    
                    grad[3*i] += -g * dx / d
                    grad[3*i+1] += -g * dy / d
                    grad[3*i+2] += g
                    grad[3*j] += g * dx / d
                    grad[3*j+1] += g * dy / d
                    grad[3*j+2] += g
            
            # Objective: minimize -sum(ri) + penalty
            obj = val + penalty
            # Gradient of -sum(ri) w.r.t. ri is -1
            for i in range(N):
                grad[3*i+2] += -1.0
            
            return obj, grad
        
        result = minimize(objective_and_grad, x0, method='L-BFGS-B', jac=True,
                         options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8})
        
        circles = result.x.reshape(-1, 3)
        val = evaluate(circles, w, h)
        return circles, val
    
    def polish(circles0, w, h, max_iter=500, beta=200.0):
        """High-precision polish pass with tighter tolerance."""
        return optimize(circles0, w, h, max_iter=max_iter, beta=beta)
    
    best_value = -np.inf
    best_result = None
    best_wh = None
    
    # Try multiple aspect ratios
    aspects = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.5, 0.5, 2.0, 0.4, 2.5, 0.3, 3.0, 4.0, 0.25]
    
    for aspect in aspects:
        if time.time() - start_time > deadline - 10:
            break
        
        circles0, w, h, val0 = grid_solution(aspect)
        
        if val0 > best_value:
            best_value = val0
            best_result = circles0.copy()
            best_wh = (w, h)
        
        # Optimize
        circles, val = optimize(circles0, w, h, max_iter=200)
        
        if val > best_value:
            best_value = val
            best_result = circles.copy()
            best_wh = (w, h)
    
    # Polish the best result
    if best_result is not None and best_wh is not None and time.time() - start_time < deadline - 5:
        w, h = best_wh
        polished, val = polish(best_result, w, h, max_iter=300, beta=200.0)
        if val > best_value:
            best_value = val
            best_result = polished.copy()
    
    if best_result is None:
        # Fallback
        circles, w, h, _ = grid_solution(1.0)
        best_result = circles
    
    return best_result
