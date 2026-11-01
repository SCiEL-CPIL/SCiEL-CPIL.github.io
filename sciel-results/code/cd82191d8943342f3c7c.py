"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

# EVOLVE-BLOCK-START
def construct():
    """
    Pack 21 circles in a rectangle of perimeter 4, maximizing sum of radii.
    Returns array of shape (N, 3) with columns [x, y, r].
    """
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    best_value = 0.0
    best_solution = None
    
    # We'll try several initial configurations and optimize each
    # Rectangle: width w, height h, with 2*(w+h) <= 4 => w+h <= 2
    # Circles: (x_i, y_i, r_i) with 0 <= x_i - r_i, x_i + r_i <= w, 0 <= y_i - r_i, y_i + r_i <= h
    # Disjoint: (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
    
    # Variable vector: [w, h, x_1, y_1, r_1, ..., x_N, y_N, r_N]
    # Length: 2 + 3*N
    
    def unpack(v):
        w, h = v[0], v[1]
        circles = v[2:].reshape(N, 3)
        return w, h, circles
    
    def objective(v):
        """Negative sum of radii (to minimize)"""
        w, h, circles = unpack(v)
        # Check constraints, add penalty if violated
        penalty = 0.0
        # Perimeter constraint: 2*(w+h) <= 4 => w+h <= 2
        if w + h > 2.0 + 1e-9:
            penalty += 1000.0 * (w + h - 2.0)
        if w <= 0 or h <= 0:
            penalty += 10000.0
        
        for i in range(N):
            x, y, r = circles[i]
            # Bounds
            if r <= 0:
                penalty += 100.0
            if x - r < -1e-9:
                penalty += 100.0 * (-x + r)
            if x + r > w + 1e-9:
                penalty += 100.0 * (x + r - w)
            if y - r < -1e-9:
                penalty += 100.0 * (-y + r)
            if y + r > h + 1e-9:
                penalty += 100.0 * (y + r - h)
        
        # Disjointness
        for i in range(N):
            for j in range(i+1, N):
                xi, yi, ri = circles[i]
                xj, yj, rj = circles[j]
                dist_sq = (xi - xj)**2 + (yi - yj)**2
                min_dist = ri + rj
                if dist_sq < min_dist**2:
                    # Penalty proportional to violation
                    dist = np.sqrt(dist_sq)
                    violation = min_dist - dist
                    penalty += 100.0 * violation**2
        
        sum_radii = np.sum(circles[:, 2])
        return -sum_radii + penalty
    
    def gradient(v):
        """Numerical gradient"""
        eps = 1e-7
        grad = np.zeros_like(v)
        f0 = objective(v)
        for i in range(len(v)):
            v_plus = v.copy()
            v_plus[i] += eps
            grad[i] = (objective(v_plus) - f0) / eps
        return grad
    
    def make_initial_config(w, h, n_cols, n_rows, extra=0):
        """Create initial configuration with circles in a grid"""
        total = n_cols * n_rows + extra
        if total < N:
            # Adjust
            n_rows = N // n_cols
            extra = N % n_cols
            total = n_cols * n_rows + extra
        
        circles = []
        # Place circles in a grid pattern
        placed = 0
        for i in range(n_cols):
            for j in range(n_rows):
                if placed >= N:
                    break
                cx = (i + 0.5) * w / n_cols
                cy = (j + 0.5) * h / n_rows
                # Radius limited by spacing
                rx = w / (2 * n_cols)
                ry = h / (2 * n_rows)
                r = min(rx, ry) * 0.95
                circles.append([cx, cy, r])
                placed += 1
            if placed >= N:
                break
        
        # Add remaining circles if needed
        while len(circles) < N:
            # Place in random positions
            x = np.random.uniform(0.1*w, 0.9*w)
            y = np.random.uniform(0.1*h, 0.9*h)
            r = 0.05
            circles.append([x, y, r])
        
        circles = np.array(circles[:N])
        return circles
    
    # Try multiple initializations
    initializations = []
    
    # 1. Square-ish rectangle with grid
    for aspect in [1.0, 0.8, 1.25, 0.7, 1.4]:
        h = 2.0 / (1.0 + aspect)
        w = aspect * h
        # Try different grid layouts
        for n_cols in [3, 4, 5]:
            n_rows = N // n_cols
            extra = N % n_cols
            if n_rows > 0:
                circles = make_initial_config(w, h, n_cols, n_rows, extra)
                v = np.concatenate([[w, h], circles.flatten()])
                initializations.append(v)
    
    # 2. Random initializations
    for _ in range(5):
        w = np.random.uniform(1.0, 1.8)
        h = 2.0 - w
        circles = []
        for _ in range(N):
            x = np.random.uniform(0.1*w, 0.9*w)
            y = np.random.uniform(0.1*h, 0.9*h)
            r = np.random.uniform(0.05, 0.15)
            circles.append([x, y, r])
        circles = np.array(circles)
        v = np.concatenate([[w, h], circles.flatten()])
        initializations.append(v)
    
    # Optimize each initialization
    for init_idx, v0 in enumerate(initializations):
        if time.time() > deadline - 5.0:
            break
        
        try:
            result = minimize(
                objective,
                v0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'maxfun': 1000, 'gtol': 1e-6}
            )
            
            v_opt = result.x
            w, h, circles = unpack(v_opt)
            
            # Verify solution is valid
            valid = True
            if w + h > 2.0 + 1e-6:
                valid = False
            if w <= 0 or h <= 0:
                valid = False
            
            for i in range(N):
                x, y, r = circles[i]
                if r <= 0:
                    valid = False
                    break
                if x - r < -1e-6 or x + r > w + 1e-6:
                    valid = False
                    break
                if y - r < -1e-6 or y + r > h + 1e-6:
                    valid = False
                    break
            
            if valid:
                for i in range(N):
                    for j in range(i+1, N):
                        xi, yi, ri = circles[i]
                        xj, yj, rj = circles[j]
                        dist_sq = (xi - xj)**2 + (yi - yj)**2
                        if dist_sq < (ri + rj)**2 - 1e-8:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                value = np.sum(circles[:, 2])
                if value > best_value:
                    best_value = value
                    best_solution = circles.copy()
                    
        except Exception as e:
            pass
    
    # If no valid solution found, return a simple grid
    if best_solution is None:
        # Fallback: simple grid in a 1x1 square (perimeter 4)
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        best_solution = np.array(cells[:N])
    
    return best_solution
# EVOLVE-BLOCK-END
