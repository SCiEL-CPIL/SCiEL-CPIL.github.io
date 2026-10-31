import numpy as np
import time
from scipy.optimize import minimize

N = 21
DEADLINE = 115.0  # Safety margin before 120s kill

def construct():
    start_time = time.time()
    best_score = -np.inf
    best_result = None
    
    # Use a fixed seed for reproducibility in the initial random parts
    np.random.seed(42)
    
    # Aspect ratios w/h such that 2(w+h)=4 => w+h=2.
    # We test a variety of aspect ratios.
    aspect_ratios = [0.8, 0.9, 1.0, 1.1, 1.25, 1.4, 1.5]
    
    for ar in aspect_ratios:
        if time.time() - start_time > DEADLINE - 15:
            break
        
        w = 2.0 / (1.0 + ar)
        h = 2.0 - w
        if h <= 0.1: # Avoid degenerate rectangles
            continue
            
        # Number of restarts per aspect ratio
        max_restarts = 4
        
        for restart in range(max_restarts):
            if time.time() - start_time > DEADLINE - 5:
                break
            
            # 1. Initialization
            # Strategy: Generate a reasonable initial packing.
            # For N=21, a grid layout is a good start.
            
            # Estimate grid dimensions
            cols = int(np.sqrt(N * w / h))
            if cols < 2: cols = 2
            rows = int(np.ceil(N / cols))
            
            # Adjust if rows*cols is too small
            while cols * rows < N:
                if cols <= rows:
                    cols += 1
                else:
                    rows += 1
            
            positions = np.zeros((N, 2))
            radii = np.zeros(N)
            
            # Fill grid
            idx = 0
            for r_idx in range(rows):
                for c_idx in range(cols):
                    if idx >= N:
                        break
                    # Center of cell
                    x = (c_idx + 0.5) * w / cols
                    y = (r_idx + 0.5) * h / rows
                    
                    # Add jitter for different restarts to escape local minima
                    if restart > 0:
                        jitter = 0.1 * min(w/cols, h/rows)
                        x += np.random.uniform(-jitter, jitter)
                        y += np.random.uniform(-jitter, jitter)
                    
                    # Clamp to ensure inside
                    x = np.clip(x, 0.01, w - 0.01)
                    y = np.clip(y, 0.01, h - 0.01)
                    
                    positions[idx] = [x, y]
                    idx += 1
            
            # Calculate initial radii
            # Min distance to neighbors and walls
            min_dist = np.inf
            for i in range(N):
                for j in range(i+1, N):
                    d = np.sqrt((positions[i,0]-positions[j,0])**2 + (positions[i,1]-positions[j,1])**2)
                    min_dist = min(min_dist, d)
            for i in range(N):
                min_dist = min(min_dist, positions[i,0], w-positions[i,0], positions[i,1], h-positions[i,1])
            
            r0 = 0.4 * min_dist
            radii = np.full(N, r0)
            
            # Flatten variables: x1, y1, r1, x2, y2, r2, ...
            x0 = np.zeros(3*N)
            for i in range(N):
                x0[3*i] = positions[i, 0]
                x0[3*i+1] = positions[i, 1]
                x0[3*i+2] = radii[i]
            
            # 2. Optimization Setup
            # Objective: Maximize sum(r) => Minimize -sum(r)
            def objective(vars_):
                r = vars_[2::3]
                return -np.sum(r)
            
            # Constraints
            # 1. Inside rectangle:
            #    x - r >= 0
            #    w - x - r >= 0
            #    y - r >= 0
            #    h - y - r >= 0
            # 2. Non-overlap:
            #    dist(i,j) - r_i - r_j >= 0
            # 3. Positive radius:
            #    r >= 0
            
            def cons_inside(vars_):
                x = vars_[0::3]
                y = vars_[1::3]
                r = vars_[2::3]
                return np.concatenate([
                    x - r,
                    w - x - r,
                    y - r,
                    h - y - r
                ])
            
            def cons_non_overlap(vars_):
                x = vars_[0::3]
                y = vars_[1::3]
                r = vars_[2::3]
                cons = []
                for i in range(N):
                    for j in range(i+1, N):
                        dx = x[i] - x[j]
                        dy = y[i] - y[j]
                        dist = np.sqrt(dx*dx + dy*dy)
                        cons.append(dist - r[i] - r[j])
                return np.array(cons)
            
            def cons_positive_r(vars_):
                r = vars_[2::3]
                return r
            
            constraints = [
                {'type': 'ineq', 'fun': cons_inside},
                {'type': 'ineq', 'fun': cons_non_overlap},
                {'type': 'ineq', 'fun': cons_positive_r}
            ]
            
            # 3. Run Optimizer
            try:
                # SLSQP is good for this kind of constrained problem
                result = minimize(
                    objective, x0, method='SLSQP',
                    constraints=constraints,
                    options={'maxiter': 300, 'ftol': 1e-9}
                )
                
                # Extract result
                x_res = result.x[0::3]
                y_res = result.x[1::3]
                r_res = result.x[2::3]
                
                # 4. Validation
                valid = True
                if np.any(r_res < 0):
                    valid = False
                else:
                    # Check boundaries
                    if np.any(x_res < r_res - 1e-6) or np.any(x_res > w - r_res + 1e-6):
                        valid = False
                    if np.any(y_res < r_res - 1e-6) or np.any(y_res > h - r_res + 1e-6):
                        valid = False
                    else:
                        # Check overlaps
                        for i in range(N):
                            for j in range(i+1, N):
                                dist = np.sqrt((x_res[i]-x_res[j])**2 + (y_res[i]-y_res[j])**2)
                                if dist < r_res[i] + r_res[j] - 1e-6:
                                    valid = False
                                    break
                            if not valid:
                                break
                
                if valid:
                    score = np.sum(r_res)
                    if score > best_score:
                        best_score = score
                        best_result = np.column_stack((x_res, y_res, r_res)).copy()
                        
            except Exception:
                continue

    # Fallback: A simple valid grid packing
    if best_result is None or best_score < 0:
        # Create a 5x5 grid in a square (1x1) -> Perimeter 4
        # 21 circles in 5x5 grid.
        # 25 spots, use 21.
        # Radius can be slightly less than 0.1 to ensure validity.
        w = 1.0
        h = 1.0
        side = 5
        x = []
        y = []
        r = []
        count = 0
        for i in range(side):
            for j in range(side):
                if count >= N:
                    break
                x.append((j + 0.5) / side)
                y.append((i + 0.5) / side)
                r.append(0.5 / side - 1e-5)
                count += 1
            if count >= N:
                break
        return np.array([x, y, r]).T
        
    return best_result
