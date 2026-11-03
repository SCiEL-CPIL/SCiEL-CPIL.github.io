"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 21

# EVOLVE-BLOCK-START
def _validate(w, h, circles):
    """Check if circles are valid: within bounds and non-overlapping."""
    for cx, cy, r in circles:
        if r <= 0 or cx - r < -1e-9 or cx + r > w + 1e-9 or cy - r < -1e-9 or cy + r > h + 1e-9:
            return False
    for i in range(len(circles)):
        for j in range(i + 1, len(circles)):
            dx = circles[i][0] - circles[j][0]
            dy = circles[i][1] - circles[j][1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < circles[i][2] + circles[j][2] - 1e-9:
                return False
    return True

def _total_radius(circles):
    return sum(c[2] for c in circles)

def _make_grid(w, h, n_circles, staggered=True):
    """Create a staggered grid initial placement."""
    circles = []
    rows = int(np.ceil(np.sqrt(n_circles * h / w)))
    cols = int(np.ceil(n_circles / rows))
    
    for i in range(rows):
        for j in range(cols):
            if len(circles) >= n_circles:
                break
            if staggered and i % 2 == 1:
                cx = (j + 0.5) / cols * w
            else:
                cx = (j + 0.25) / cols * w
            cy = (i + 0.5) / rows * h
            dx = w / cols
            dy = h / rows
            r = 0.45 * min(dx, dy)
            r = min(r, cx, w - cx, cy, h - cy)
            if r > 0:
                circles.append((cx, cy, r))
    
    while len(circles) < n_circles:
        idx = len(circles)
        cx = (idx + 0.5) / n_circles * w
        cy = 0.5 * h
        r = 0.02
        r = min(r, cx, w - cx, cy, h - cy)
        if r > 0:
            circles.append((cx, cy, r))
    
    return circles[:n_circles]

def _relax_circles(w, h, circles, n_iter=50, alpha=0.5):
    """Simple relaxation to resolve overlaps."""
    circles = [list(c) for c in circles]
    for _ in range(n_iter):
        for i in range(len(circles)):
            cx, cy, r = circles[i]
            # Boundary constraints
            if cx - r < 0:
                cx = r
            if cx + r > w:
                cx = w - r
            if cy - r < 0:
                cy = r
            if cy + r > h:
                cy = h - r
            circles[i][0], circles[i][1] = cx, cy
        
        for i in range(len(circles)):
            for j in range(i + 1, len(circles)):
                dx = circles[i][0] - circles[j][0]
                dy = circles[i][1] - circles[j][1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = circles[i][2] + circles[j][2]
                if dist < min_dist - 1e-9:
                    if dist < 1e-9:
                        dx, dy = 1.0, 0.0
                        dist = 1.0
                    overlap = (min_dist - dist) / 2.0 * alpha
                    nx, ny = dx / dist, dy / dist
                    circles[i][0] += overlap * nx
                    circles[i][1] += overlap * ny
                    circles[j][0] -= overlap * nx
                    circles[j][1] -= overlap * ny
        
        # Re-clamp to boundaries
        for i in range(len(circles)):
            cx, cy, r = circles[i]
            cx = max(r, min(w - r, cx))
            cy = max(r, min(h - r, cy))
            circles[i][0], circles[i][1] = cx, cy
    
    return [tuple(c) for c in circles]

def _slsqp_optimize(w, h, init_circles, deadline, max_iter=1000):
    """
    Solve the circle packing problem using SLSQP with smooth constraints.
    Variables: for each circle i, (x_i, y_i, r_i)
    Objective: maximize sum(r_i)
    Constraints:
      - r_i > 0 (lower bound)
      - x_i - r_i >= 0, x_i + r_i <= w, y_i - r_i >= 0, y_i + r_i <= h
      - (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2 for all i < j
    """
    n = N
    x0 = np.array([v for c in init_circles for v in c])
    
    def objective(x):
        r_sum = 0.0
        for i in range(n):
            r = x[i*3 + 2]
            if r > 0:
                r_sum += r
        return -r_sum
    
    # Gradient of objective
    def objective_grad(x):
        g = np.zeros_like(x)
        for i in range(n):
            g[i*3 + 2] = -1.0
        return g
    
    constraints = []
    
    # Boundary constraints: x_i - r_i >= 0
    for i in range(n):
        def bc1(x, i=i):
            return x[i*3] - x[i*3 + 2]
        def bc1_grad(x, i=i):
            g = np.zeros_like(x)
            g[i*3] = 1.0
            g[i*3 + 2] = -1.0
            return g
        constraints.append({'type': 'ineq', 'fun': bc1, 'jac': bc1_grad})
    
    # Boundary constraints: w - x_i - r_i >= 0
    for i in range(n):
        def bc2(x, i=i):
            return w - x[i*3] - x[i*3 + 2]
        def bc2_grad(x, i=i):
            g = np.zeros_like(x)
            g[i*3] = -1.0
            g[i*3 + 2] = -1.0
            return g
        constraints.append({'type': 'ineq', 'fun': bc2, 'jac': bc2_grad})
    
    # Boundary constraints: y_i - r_i >= 0
    for i in range(n):
        def bc3(x, i=i):
            return x[i*3 + 1] - x[i*3 + 2]
        def bc3_grad(x, i=i):
            g = np.zeros_like(x)
            g[i*3 + 1] = 1.0
            g[i*3 + 2] = -1.0
            return g
        constraints.append({'type': 'ineq', 'fun': bc3, 'jac': bc3_grad})
    
    # Boundary constraints: h - y_i - r_i >= 0
    for i in range(n):
        def bc4(x, i=i):
            return h - x[i*3 + 1] - x[i*3 + 2]
        def bc4_grad(x, i=i):
            g = np.zeros_like(x)
            g[i*3 + 1] = -1.0
            g[i*3 + 2] = -1.0
            return g
        constraints.append({'type': 'ineq', 'fun': bc4, 'jac': bc4_grad})
    
    # Non-overlap constraints: (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
    for i in range(n):
        for j in range(i + 1, n):
            def nc(x, i=i, j=j):
                dx = x[i*3] - x[j*3]
                dy = x[i*3 + 1] - x[j*3 + 1]
                rsum = x[i*3 + 2] + x[j*3 + 2]
                return dx*dx + dy*dy - rsum*rsum
            def nc_grad(x, i=i, j=j):
                dx = x[i*3] - x[j*3]
                dy = x[i*3 + 1] - x[j*3 + 1]
                rsum = x[i*3 + 2] + x[j*3 + 2]
                g = np.zeros_like(x)
                g[i*3] = 2*dx
                g[i*3 + 1] = 2*dy
                g[i*3 + 2] = -2*rsum
                g[j*3] = -2*dx
                g[j*3 + 1] = -2*dy
                g[j*3 + 2] = -2*rsum
                return g
            constraints.append({'type': 'ineq', 'fun': nc, 'jac': nc_grad})
    
    bounds = []
    for i in range(n):
        bounds.append((0, w))
        bounds.append((0, h))
        bounds.append((1e-6, 0.5))
    
    try:
        res = minimize(
            objective,
            x0,
            jac=objective_grad,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': max_iter, 'ftol': 1e-12, 'disp': False}
        )
        
        circles = [(res.x[i*3], res.x[i*3 + 1], res.x[i*3 + 2]) for i in range(n)]
        
        # Check validity and relax if needed
        if _validate(w, h, circles):
            return circles
        else:
            relaxed = _relax_circles(w, h, circles, n_iter=100)
            if _validate(w, h, relaxed):
                return relaxed
            return None
    except Exception:
        return None

def _solve_aspect(w, h, deadline, n_starts=3):
    """Try multiple initializations for a given aspect ratio."""
    best = None
    best_score = 0.0
    
    # Staggered grid
    init1 = _make_grid(w, h, N, staggered=True)
    init1 = _relax_circles(w, h, init1, n_iter=50)
    result = _slsqp_optimize(w, h, init1, deadline)
    if result is not None:
        score = _total_radius(result)
        if score > best_score:
            best_score = score
            best = result
    
    # Non-staggered grid
    if time.time() < deadline - 5:
        init2 = _make_grid(w, h, N, staggered=False)
        init2 = _relax_circles(w, h, init2, n_iter=50)
        result2 = _slsqp_optimize(w, h, init2, deadline)
        if result2 is not None:
            score2 = _total_radius(result2)
            if score2 > best_score:
                best_score = score2
                best = result2
    
    # Random starts
    for k in range(n_starts):
        if time.time() > deadline - 5:
            break
        np.random.seed(42 + k)
        random_circles = []
        for _ in range(N):
            cx = np.random.uniform(0.1, 0.9) * w
            cy = np.random.uniform(0.1, 0.9) * h
            r = np.random.uniform(0.03, 0.08)
            r = min(r, cx, w - cx, cy, h - cy)
            random_circles.append((cx, cy, r))
        random_circles = _relax_circles(w, h, random_circles, n_iter=50)
        result3 = _slsqp_optimize(w, h, random_circles, deadline)
        if result3 is not None:
            score3 = _total_radius(result3)
            if score3 > best_score:
                best_score = score3
                best = result3
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_circles = None
    best_score = 0.0
    
    # Try multiple aspect ratios
    aspect_ratios = [1.0, 1.2, 1.5, 0.8, 1.3, 0.9, 1.1, 2.0, 0.75]
    
    for aspect in aspect_ratios:
        if time.time() > deadline:
            break
        
        w = 2.0 * aspect / (1 + aspect)
        h = 2.0 / (1 + aspect)
        
        result = _solve_aspect(w, h, deadline, n_starts=2)
        if result is not None:
            score = _total_radius(result)
            if score > best_score:
                best_score = score
                best_circles = result
    
    # Fallback: simple grid if nothing else worked
    if best_circles is None:
        w, h = 1.0, 1.0
        best_circles = _make_grid(w, h, N, staggered=True)
        best_circles = _relax_circles(w, h, best_circles, n_iter=100)
    
    # Final polish: try to improve with more iterations if time permits
    if best_circles is not None and time.time() < deadline - 10:
        w = best_circles[0][0] + best_circles[0][2]  # rough estimate
        # Actually, we need to know w and h. Let's recompute from the best aspect.
        # For now, just try a final SLSQP with tighter tolerance on the best solution.
        # Determine w, h from the best solution's bounding box
        xs = [c[0] for c in best_circles]
        ys = [c[1] for c in best_circles]
        rs = [c[2] for c in best_circles]
        w_est = max(x + r for x, r in zip(xs, rs))
        h_est = max(y + r for y, r in zip(ys, rs))
        # Use the aspect ratio that gave this solution
        # This is a rough approach; in practice we'd track which aspect was used.
        
    return np.array(best_circles)
# EVOLVE-BLOCK-END
