"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    best_score = 0
    best_result = None
    
    def smooth_penalty(violation, beta=50.0):
        """Smooth differentiable penalty for constraint violation."""
        return beta * np.log1p(np.exp(beta * violation))
    
    def objective_and_grad(params, w, h):
        """
        Objective: minimize -sum(radii) + penalty(constraints)
        params: [x1,y1,r1, x2,y2,r2, ..., xn,yn,rn]
        """
        n = N
        x = params[0::3]
        y = params[1::3]
        r = params[2::3]
        
        # Objective: maximize sum(r) => minimize -sum(r)
        obj = -np.sum(r)
        grad = np.zeros_like(params)
        grad[2::3] = -1.0  # d(-sum(r))/dr = -1
        
        # Penalty for boundary violations
        eps = 1e-8
        for i in range(n):
            # Left: x[i] >= r[i] => violation = r[i] - x[i]
            v_left = r[i] - x[i] + eps
            p_left = smooth_penalty(v_left, 50.0)
            obj += p_left
            # d/dx[i] = -50*sigmoid(50*v_left)
            sig = 1.0 / (1.0 + np.exp(-50.0 * v_left))
            grad[3*i] += -50.0 * sig
            # d/dr[i] = +50*sigmoid(50*v_left)
            grad[3*i+2] += 50.0 * sig
            
            # Right: x[i] <= w - r[i] => violation = x[i] + r[i] - w
            v_right = x[i] + r[i] - w + eps
            p_right = smooth_penalty(v_right, 50.0)
            obj += p_right
            sig = 1.0 / (1.0 + np.exp(-50.0 * v_right))
            grad[3*i] += 50.0 * sig
            grad[3*i+2] += 50.0 * sig
            
            # Bottom: y[i] >= r[i] => violation = r[i] - y[i]
            v_bottom = r[i] - y[i] + eps
            p_bottom = smooth_penalty(v_bottom, 50.0)
            obj += p_bottom
            sig = 1.0 / (1.0 + np.exp(-50.0 * v_bottom))
            grad[3*i+1] += -50.0 * sig
            grad[3*i+2] += 50.0 * sig
            
            # Top: y[i] <= h - r[i] => violation = y[i] + r[i] - h
            v_top = y[i] + r[i] - h + eps
            p_top = smooth_penalty(v_top, 50.0)
            obj += p_top
            sig = 1.0 / (1.0 + np.exp(-50.0 * v_top))
            grad[3*i+1] += 50.0 * sig
            grad[3*i+2] += 50.0 * sig
        
        # Penalty for overlap violations: dist(i,j) >= r[i] + r[j]
        for i in range(n):
            for j in range(i+1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx**2 + dy**2) + eps
                min_dist = r[i] + r[j]
                violation = min_dist - dist + eps
                p = smooth_penalty(violation, 50.0)
                obj += p
                
                sig = 1.0 / (1.0 + np.exp(-50.0 * violation))
                factor = 50.0 * sig / dist
                
                # d/dx[i]: -factor * dx
                grad[3*i] += -factor * dx
                # d/dy[i]: -factor * dy
                grad[3*i+1] += -factor * dy
                # d/dr[i]: +50*sig
                grad[3*i+2] += 50.0 * sig
                
                # d/dx[j]: +factor * dx
                grad[3*j] += factor * dx
                # d/dy[j]: +factor * dy
                grad[3*j+1] += factor * dy
                # d/dr[j]: +50*sig
                grad[3*j+2] += 50.0 * sig
        
        return obj, grad
    
    def evaluate_solution(pos, rad, w, h):
        """Check if solution is valid and compute score."""
        n = len(pos)
        for i in range(n):
            if pos[i][0] < rad[i] - 1e-5 or pos[i][0] > w - rad[i] + 1e-5:
                return 0
            if pos[i][1] < rad[i] - 1e-5 or pos[i][1] > h - rad[i] + 1e-5:
                return 0
        
        for i in range(n):
            for j in range(i+1, n):
                dx = pos[i][0] - pos[j][0]
                dy = pos[i][1] - pos[j][1]
                dist = np.sqrt(dx**2 + dy**2)
                if dist < rad[i] + rad[j] - 1e-5:
                    return 0
        
        return np.sum(rad)
    
    def hexagonal_init(w, h, n_circles):
        """Initialize circles in hexagonal pattern."""
        pos = []
        rad = []
        
        # Estimate max radius
        area_per_circle = (w * h) / n_circles
        max_r = np.sqrt(area_per_circle / np.pi) * 0.9
        
        # Hexagonal packing
        rows = int(np.ceil(np.sqrt(n_circles * 2 / 3)))
        cols = int(np.ceil(np.sqrt(n_circles * 3 / 2)))
        
        spacing = min(w / cols, h / rows) * 0.95
        r = spacing * 0.45
        
        count = 0
        for row in range(rows):
            for col in range(cols):
                if count >= n_circles:
                    break
                x = (col + 0.5) * w / cols
                if row % 2 == 1:
                    x += w / (2 * cols)
                x = min(max(x, r), w - r)
                y = (row + 0.5) * h / rows
                y = min(max(y, r), h - r)
                pos.append([x, y])
                rad.append(r)
                count += 1
        
        return np.array(pos[:n_circles]), np.array(rad[:n_circles])
    
    def grid_init(w, h, n_circles):
        """Initialize circles in grid pattern."""
        side = int(np.ceil(np.sqrt(n_circles)))
        pos = []
        rad = []
        
        for i in range(side):
            for j in range(side):
                if len(pos) >= n_circles:
                    break
                x = (i + 0.5) * w / side
                y = (j + 0.5) * h / side
                r = min(w, h) / (2 * side) * 0.9
                pos.append([x, y])
                rad.append(r)
        
        return np.array(pos[:n_circles]), np.array(rad[:n_circles])
    
    # Try different aspect ratios
    aspect_ratios = [1.0, 0.85, 1.15, 0.75, 1.25, 0.7, 1.3, 0.65, 1.4]
    
    for aspect in aspect_ratios:
        if time.time() > deadline:
            break
        
        # Perimeter = 4, so 2*(w+h) = 4 => w+h = 2
        h = 2.0 / (aspect + 1.0)
        w = aspect * h
        
        # Try multiple initializations
        inits = [
            hexagonal_init(w, h, N),
            grid_init(w, h, N)
        ]
        
        for init_pos, init_rad in inits:
            if time.time() > deadline:
                break
            
            # Build initial params
            params = np.zeros(3 * N)
            params[0::3] = init_pos[:, 0]
            params[1::3] = init_pos[:, 1]
            params[2::3] = init_rad
            
            # Optimize
            result = minimize(
                objective_and_grad,
                params,
                args=(w, h),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            # Extract solution
            opt_params = result.x
            pos = np.column_stack([opt_params[0::3], opt_params[1::3]])
            rad = opt_params[2::3]
            
            # Ensure radii are positive
            rad = np.maximum(rad, 1e-6)
            
            # Evaluate
            score = evaluate_solution(pos, rad, w, h)
            
            if score > best_score:
                best_score = score
                best_result = np.column_stack([pos, rad])
    
    if best_result is None:
        # Fallback to seed solution
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_result
# EVOLVE-BLOCK-END
