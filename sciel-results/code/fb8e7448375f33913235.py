"""Circle packing in perimeter-4 rectangle using L-BFGS-B with joint geometry optimization."""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 148.0  # Safety margin for 150s limit
    
    def get_rect(W):
        """Given width W, return (W, H) such that 2(W+H) = 4 => W+H=2."""
        H = 2.0 - W
        return W, H

    def hexagonal_grid(n, W, H):
        """Generate n points in hexagonal arrangement scaled to W x H."""
        # Determine grid dimensions
        # Area per hex ~ (sqrt(3)/2) * s^2 for side s
        # We want ~ n points
        # Let's just generate a grid and take first n
        cols = int(np.ceil(np.sqrt(n * W / H))) if H > 0 else int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols)) if cols > 0 else n
        
        pts = []
        for i in range(rows):
            for j in range(cols):
                if len(pts) >= n:
                    break
                # Hex offset
                x_off = 0.5 * (j % 2)
                x = (j + x_off + 0.5) * W / cols
                y = (i + 0.5) * H / rows
                pts.append((x, y))
        
        # If we don't have enough, add random ones or fill
        while len(pts) < n:
            pts.append((random.uniform(0.1, W-0.1), random.uniform(0.1, H-0.1)))
        
        return pts[:n]

    def random_grid(n, W, H):
        pts = []
        for _ in range(n):
            pts.append((random.uniform(0.1, W-0.1), random.uniform(0.1, H-0.1)))
        return pts

    def objective_and_grad(params, W):
        """
        params: [x1, y1, r1, x2, y2, r2, ..., x21, y21, r21, logit_W]
        logit_W is used to map to W in (0.2, 1.8) to keep H=2-W positive and reasonable.
        Actually, let's just optimize W directly with bounds [0.3, 1.7].
        params: [x1, y1, r1, ..., x21, y21, r21, W]
        """
        n = 21
        pos_r = params[:3*n]
        W = params[3*n]
        H = 2.0 - W
        
        if H <= 0.1 or W <= 0.1:
            return 1e10, np.zeros_like(params)
            
        xs = pos_r[0::3]
        ys = pos_r[1::3]
        rs = pos_r[2::3]
        
        # Penalty for bounds
        pen = 0.0
        grad_pen = np.zeros_like(params)
        
        # Wall penalties
        # x in [r, W-r] => x-r >= 0, W-r-x >= 0
        # Use soft penalty: -log(x-r) or quadratic
        # Quadratic is smoother for L-BFGS
        # min_dist_x = r, max_dist_x = W-r
        # If x < r: pen += (r-x)^2, grad_x += -2(r-x)
        # If x > W-r: pen += (x-(W-r))^2, grad_x += 2(x-(W-r))
        
        for i in range(n):
            # X bounds
            if xs[i] < rs[i]:
                diff = rs[i] - xs[i]
                pen += diff * diff
                grad_pen[i*3] += -2 * diff
            elif xs[i] > W - rs[i]:
                diff = xs[i] - (W - rs[i])
                pen += diff * diff
                grad_pen[i*3] += 2 * diff
                
            # Y bounds
            if ys[i] < rs[i]:
                diff = rs[i] - ys[i]
                pen += diff * diff
                grad_pen[i*3+1] += -2 * diff
            elif ys[i] > H - rs[i]:
                diff = ys[i] - (H - rs[i])
                pen += diff * diff
                grad_pen[i*3+1] += 2 * diff
        
        # Overlap penalties
        # For each pair i,j: dist < ri+rj => pen += (ri+rj - dist)^2
        # dist = sqrt((xi-xj)^2 + (yi-yj)^2)
        # grad_xi = 2*(ri+rj-dist) * (xi-xj)/dist
        # grad_ri = 2*(ri+rj-dist)
        
        for i in range(n):
            for j in range(i+1, n):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist_sq = dx*dx + dy*dy
                dist = np.sqrt(dist_sq) if dist_sq > 1e-12 else 1e-9
                min_dist = rs[i] + rs[j]
                
                if dist < min_dist:
                    overlap = min_dist - dist
                    pen += overlap * overlap
                    
                    if dist > 1e-8:
                        factor = 2 * overlap / dist
                        grad_pen[i*3] += factor * dx
                        grad_pen[i*3+1] += factor * dy
                        grad_pen[j*3] -= factor * dx
                        grad_pen[j*3+1] -= factor * dy
                    else:
                        # Random direction
                        angle = random.uniform(0, 2*np.pi)
                        grad_pen[i*3] += 2 * overlap * np.cos(angle)
                        grad_pen[i*3+1] += 2 * overlap * np.sin(angle)
                        grad_pen[j*3] -= 2 * overlap * np.cos(angle)
                        grad_pen[j*3+1] -= 2 * overlap * np.sin(angle)
                    
                    # Radius gradients
                    grad_pen[i*3+2] += 2 * overlap
                    grad_pen[j*3+2] += 2 * overlap
        
        # Objective: Maximize sum(r) => Minimize -sum(r) + lambda * pen
        # Let's use a large lambda to enforce constraints strongly
        # Or use penalty method where we increase lambda
        # For simplicity, fixed large penalty
        lam = 10.0
        obj = -np.sum(rs) + lam * pen
        
        grad_obj = np.zeros_like(params)
        grad_obj[2::3] = -1.0  # d(-sum r)/dr = -1
        grad_obj += lam * grad_pen
        
        return obj, grad_obj

    def optimize_from(x0, W0, time_limit):
        """
        Run L-BFGS-B from initial guess.
        x0: array of positions/radii (63 elements)
        W0: initial width
        """
        params0 = np.concatenate([x0, [W0]])
        
        # Bounds
        # x, y, r >= 0.01
        # W in [0.3, 1.7]
        bounds = []
        for i in range(21):
            bounds.append((0.01, 2.0)) # x
            bounds.append((0.01, 2.0)) # y
            bounds.append((0.01, 1.0)) # r
        bounds.append((0.3, 1.7)) # W
        
        def obj_wrapper(p):
            W = p[-1]
            val, grad = objective_and_grad(p, W)
            return val, grad
            
        res = minimize(
            obj_wrapper,
            params0,
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 20}
        )
        
        return res

    def extract_result(res):
        """Extract circles and W from optimization result."""
        if not res.success and res.fun > 100:
            return None, None, None
            
        params = res.x
        W = params[-1]
        H = 2.0 - W
        
        circles = []
        for i in range(21):
            x = params[i*3]
            y = params[i*3+1]
            r = params[i*3+2]
            circles.append((x, y, r))
            
        return circles, W, H

    def is_valid(circles, W, H, tol=1e-5):
        """Check validity."""
        if not circles:
            return False
        for x, y, r in circles:
            if x < r - tol or x > W - r + tol or y < r - tol or y > H - r + tol:
                return False
        for i in range(len(circles)):
            for j in range(i+1, len(circles)):
                dx = circles[i][0] - circles[j][0]
                dy = circles[i][1] - circles[j][1]
                dist_sq = dx*dx + dy*dy
                min_dist = circles[i][2] + circles[j][2]
                if dist_sq < min_dist*min_dist - tol:
                    return False
        return True

    best_sum_r = 0.0
    best_config = None
    
    # Multi-start
    # 1. Hexagonal grid with various aspect ratios
    # 2. Random grids
    
    aspect_ratios = [0.8, 0.9, 1.0, 1.1, 1.2]
    
    for W_ratio in aspect_ratios:
        if time.time() > deadline - 10:
            break
            
        W0 = W_ratio
        H0 = 2.0 - W0
        
        # Hex init
        pts = hexagonal_grid(21, W0, H0)
        x0 = np.zeros(63)
        for i, (x, y) in enumerate(pts):
            x0[i*3] = x
            x0[i*3+1] = y
            # Initial radius: estimate based on local spacing
            # For hex grid, spacing is approx min(W/cols, H/rows)
            # Let's just use a small value
            x0[i*3+2] = 0.05
            
        res = optimize_from(x0, W0, deadline)
        circles, W, H = extract_result(res)
        
        if circles and is_valid(circles, W, H):
            s = sum(c[2] for c in circles)
            if s > best_sum_r:
                best_sum_r = s
                best_config = np.array(circles)
                
    # Random inits
    for _ in range(5):
        if time.time() > deadline - 5:
            break
        W0 = random.uniform(0.7, 1.3)
        H0 = 2.0 - W0
        pts = random_grid(21, W0, H0)
        x0 = np.zeros(63)
        for i, (x, y) in enumerate(pts):
            x0[i*3] = x
            x0[i*3+1] = y
            x0[i*3+2] = 0.04
            
        res = optimize_from(x0, W0, deadline)
        circles, W, H = extract_result(res)
        
        if circles and is_valid(circles, W, H):
            s = sum(c[2] for c in circles)
            if s > best_sum_r:
                best_sum_r = s
                best_config = np.array(circles)

    if best_config is not None:
        return best_config
        
    # Fallback
    # Simple grid
    side = 5
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) >= N:
                break
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side * 0.8
            cells.append((x, y, r))
    return np.array(cells[:N])

# EVOLVE-BLOCK-END
