"""Circle packing in perimeter-4 rectangle: maximize sum of radii for 21 disjoint circles."""
import numpy as np
from scipy.optimize import minimize
import time

N = 21
DEADLINE = 140.0  # safety margin before 150s kill

def construct():
    start_time = time.time()
    best_score = 0.0
    best_result = None
    
    # Parameter vector: [w, x_0, y_0, r_0, x_1, y_1, r_1, ..., x_20, y_20, r_20]
    # w = width, h = 2 - w (since perimeter = 2(w+h) = 4 => w+h=2)
    # Total params: 1 + 3*N
    
    def unpack(params):
        w = params[0]
        h = 2.0 - w
        circles = params[1:].reshape(N, 3)
        return w, h, circles
    
    def penalty_function(params):
        w = params[0]
        h = 2.0 - w
        if w <= 0 or h <= 0:
            return 1e10
        circles = params[1:].reshape(N, 3)
        x = circles[:, 0]
        y = circles[:, 1]
        r = circles[:, 2]
        
        # Objective: maximize sum of radii => minimize negative sum
        obj = -np.sum(r)
        
        # Penalty for boundary violations
        pen = 0.0
        for i in range(N):
            if r[i] < 0:
                pen += 100.0 * r[i]**2
            # Circle must be inside rectangle
            if x[i] - r[i] < 0:
                pen += 100.0 * (x[i] - r[i])**2
            if x[i] + r[i] > w:
                pen += 100.0 * (x[i] + r[i] - w)**2
            if y[i] - r[i] < 0:
                pen += 100.0 * (y[i] - r[i])**2
            if y[i] + r[i] > h:
                pen += 100.0 * (y[i] + r[i] - h)**2
        
        # Penalty for circle-circle overlap
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    pen += 100.0 * (min_dist - dist)**2
        
        return obj + pen
    
    def make_initial_guess(w_init, h_init, seed=0):
        rng = np.random.RandomState(seed)
        params = np.zeros(1 + 3*N)
        params[0] = w_init
        # Initialize circles in a grid pattern with some randomness
        side = int(np.ceil(np.sqrt(N)))
        for idx in range(N):
            i = idx % side
            j = idx // side
            base_r = 0.5 / side
            x = (i + 0.5) * w_init / side + rng.randn() * 0.01
            y = (j + 0.5) * h_init / side + rng.randn() * 0.01
            r = base_r * min(w_init, h_init) / 1.0 + rng.randn() * 0.005
            x = np.clip(x, 0, w_init)
            y = np.clip(y, 0, h_init)
            r = max(0.001, min(r, min(w_init, h_init)/2 - 0.001))
            params[1 + 3*idx] = x
            params[2 + 3*idx] = y
            params[3 + 3*idx] = r
        return params
    
    # Try multiple aspect ratios and random restarts
    aspect_ratios = [1.0, 1.5, 2.0, 0.75, 1.2, 0.8, 1.8, 2.5, 0.6]
    n_restarts = 3
    
    total_attempts = len(aspect_ratios) * n_restarts
    time_per_attempt = (DEADLINE - time.time()) / total_attempts if total_attempts > 0 else 5.0
    time_per_attempt = max(1.0, min(time_per_attempt, 15.0))
    
    for ar_idx, ar in enumerate(aspect_ratios):
        if time.time() > DEADLINE:
            break
        w_init = ar * 1.0  # w = ar * h, w + h = 2 => w = ar*2/(ar+1)
        w_init = ar * 2.0 / (ar + 1.0)
        h_init = 2.0 - w_init
        
        for restart in range(n_restarts):
            if time.time() > DEADLINE:
                break
            
            x0 = make_initial_guess(w_init, h_init, seed=ar_idx*100 + restart)
            
            # Use L-BFGS-B with bounds
            bounds = [(0.01, 1.99)]  # w bounds
            for i in range(N):
                bounds.append((0, w_init))  # x
                bounds.append((0, h_init))  # y
                bounds.append((0.001, min(w_init, h_init)/2))  # r
            
            # For simplicity, use SLSQP or L-BFGS-B without tight bounds initially
            # Then refine
            try:
                result = minimize(
                    penalty_function,
                    x0,
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'maxfun': 1000, 'ftol': 1e-10}
                )
            except Exception:
                continue
            
            if result.fun < 1e6:  # reasonable solution
                w, h, circles = unpack(result.x)
                r = circles[:, 2]
                x = circles[:, 0]
                y = circles[:, 1]
                
                # Validate: check if all constraints are approximately satisfied
                valid = True
                for i in range(N):
                    if r[i] < 0 or x[i] - r[i] < -1e-6 or x[i] + r[i] > w + 1e-6:
                        valid = False
                        break
                    if y[i] - r[i] < -1e-6 or y[i] + r[i] > h + 1e-6:
                        valid = False
                        break
                
                if valid:
                    # Check circle-circle
                    for i in range(N):
                        for j in range(i+1, N):
                            dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                            if dist < r[i] + r[j] - 1e-6:
                                valid = False
                                break
                        if not valid:
                            break
                
                if valid:
                    score = np.sum(r)
                    if score > best_score:
                        best_score = score
                        best_result = result.x.copy()
    
    # Final polish: run a few more iterations on the best solution if time permits
    if best_result is not None and time.time() < DEADLINE - 10:
        try:
            result = minimize(
                penalty_function,
                best_result,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'maxfun': 5000, 'ftol': 1e-12}
            )
            if result.fun < 1e6:
                w, h, circles = unpack(result.x)
                r = circles[:, 2]
                x = circles[:, 0]
                y = circles[:, 1]
                
                valid = True
                for i in range(N):
                    if r[i] < 0 or x[i] - r[i] < -1e-5 or x[i] + r[i] > w + 1e-5:
                        valid = False
                        break
                    if y[i] - r[i] < -1e-5 or y[i] + r[i] > h + 1e-5:
                        valid = False
                        break
                
                if valid:
                    for i in range(N):
                        for j in range(i+1, N):
                            dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                            if dist < r[i] + r[j] - 1e-5:
                                valid = False
                                break
                        if not valid:
                            break
                
                if valid:
                    score = np.sum(r)
                    if score > best_score:
                        best_score = score
                        best_result = result.x.copy()
        except Exception:
            pass
    
    # Build final result
    if best_result is not None:
        w, h, circles = unpack(best_result)
        # Clip to ensure validity
        result_array = np.zeros((N, 3))
        for i in range(N):
            x = np.clip(circles[i, 0], circles[i, 2], w - circles[i, 2])
            y = np.clip(circles[i, 1], circles[i, 2], h - circles[i, 2])
            r = max(0.001, min(circles[i, 2], min(w, h)/2 - 0.001))
            result_array[i] = [x, y, r]
        return result_array
    else:
        # Fallback: grid packing
        side = int(np.ceil(np.sqrt(N)))
        w = 1.0
        h = 1.0
        cells = []
        for idx in range(N):
            i = idx % side
            j = idx // side
            r = 0.5 / side - 0.001
            x = (i + 0.5) * w / side
            y = (j + 0.5) * h / side
            cells.append([x, y, r])
        return np.array(cells)
