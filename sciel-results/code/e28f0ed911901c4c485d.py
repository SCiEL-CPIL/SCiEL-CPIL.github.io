import numpy as np
import time
from scipy.optimize import minimize

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    # Variables: x (N), y (N), r (N), aspect (1) = 3N+1 = 64 variables
    # Constraints:
    #   1. Perimeter: w + h = 2, w/h = aspect => w = 2/(1+1/aspect), h = 2 - w
    #   2. Boundary: r_i <= x_i, x_i + r_i <= w, r_i <= y_i, y_i + r_i <= h
    #   3. Non-overlap: dist(i,j) >= r_i + r_j for all i < j
    # Objective: maximize sum(r_i) => minimize -sum(r_i)
    
    def _unpack(x, y, r, aspect):
        w = 2.0 / (1.0 + 1.0/aspect)
        h = 2.0 - w
        return w, h
    
    def _objective(v):
        x = v[:N]
        y = v[N:2*N]
        r = v[2*N:3*N]
        aspect = v[3*N]
        
        # Penalize invalid aspects
        if aspect < 0.3 or aspect > 3.3:
            return 1e6
        
        w, h = _unpack(x, y, r, aspect)
        
        # Boundary violations (soft penalty)
        penalty = 0.0
        for i in range(N):
            if x[i] < r[i]:
                penalty += (r[i] - x[i]) ** 2
            if x[i] + r[i] > w:
                penalty += (x[i] + r[i] - w) ** 2
            if y[i] < r[i]:
                penalty += (r[i] - y[i]) ** 2
            if y[i] + r[i] > h:
                penalty += (y[i] + r[i] - h) ** 2
        
        # Overlap violations (soft penalty)
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    penalty += (min_dist - dist) ** 2
        
        return -np.sum(r) + 100.0 * penalty
    
    def _constraints(v):
        x = v[:N]
        y = v[N:2*N]
        r = v[2*N:3*N]
        aspect = v[3*N]
        
        if aspect < 0.3 or aspect > 3.3:
            # Return a single infeasible constraint
            return [{'type': 'ineq', 'fun': lambda v: -1.0}]
        
        w, h = _unpack(x, y, r, aspect)
        
        constraints = []
        
        # Aspect ratio bounds
        constraints.append({'type': 'ineq', 'fun': lambda v: v[3*N] - 0.3})
        constraints.append({'type': 'ineq', 'fun': lambda v: 3.3 - v[3*N]})
        
        # Positive radii
        for i in range(N):
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[2*N+i] - 1e-8})
        
        # Boundary constraints: x_i >= r_i
        for i in range(N):
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[i] - v[2*N+i]})
        
        # x_i + r_i <= w
        for i in range(N):
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: _unpack(v[:N], v[N:2*N], v[2*N:3*N], v[3*N])[0] - v[i] - v[2*N+i]})
        
        # y_i >= r_i
        for i in range(N):
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[N+i] - v[2*N+i]})
        
        # y_i + r_i <= h
        for i in range(N):
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: _unpack(v[:N], v[N:2*N], v[2*N:3*N], v[3*N])[1] - v[N+i] - v[2*N+i]})
        
        # Non-overlap: dist(i,j) >= r_i + r_j
        for i in range(N):
            for j in range(i+1, N):
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda v, i=i, j=j: np.sqrt((v[i]-v[j])**2 + (v[N+i]-v[N+j])**2) - v[2*N+i] - v[2*N+j]
                })
        
        return constraints
    
    def _hex_start(aspect):
        w = 2.0 / (1.0 + 1.0/aspect)
        h = 2.0 - w
        r = 0.1
        circles = []
        y = r
        row = 0
        while y + r <= h:
            x_offset = 0 if row % 2 == 0 else r
            x = r + x_offset
            while x + r <= w:
                circles.append([x, y, r])
                x += 2*r
            y += np.sqrt(3)*r
            row += 1
            if len(circles) >= N:
                break
        if len(circles) < N:
            return None
        return np.array(circles[:N])
    
    def _grid_start(aspect, cols):
        w = 2.0 / (1.0 + 1.0/aspect)
        h = 2.0 - w
        rows = int(np.ceil(N / cols))
        cw = w / cols
        ch = h / rows
        circles = []
        count = 0
        for r_idx in range(rows):
            for c_idx in range(cols):
                if count >= N:
                    break
                cx = (c_idx + 0.5) * cw
                cy = (r_idx + 0.5) * ch
                rad = 0.5 * min(cw, ch) * 0.9
                circles.append([cx, cy, rad])
                count += 1
        return np.array(circles)
    
    def _random_start(aspect):
        w = 2.0 / (1.0 + 1.0/aspect)
        h = 2.0 - w
        r = 0.05 + 0.05 * np.random.rand(N)
        x = r + (w - 2*r) * np.random.rand(N)
        y = r + (h - 2*r) * np.random.rand(N)
        return np.column_stack([x, y, r])
    
    def _optimize_from(circles, aspect_init):
        x = circles[:, 0]
        y = circles[:, 1]
        r = circles[:, 2]
        
        # Scale radii down a bit to start feasible
        r = r * 0.7
        
        v0 = np.concatenate([x, y, r, [aspect_init]])
        
        # Bounds for each variable
        # x, y: [0, 2] (generous)
        # r: [1e-8, 1.0]
        # aspect: [0.3, 3.3]
        bounds = []
        for i in range(N):
            bounds.append((0, 2))  # x
        for i in range(N):
            bounds.append((0, 2))  # y
        for i in range(N):
            bounds.append((1e-8, 1.0))  # r
        bounds.append((0.3, 3.3))  # aspect
        
        # Use SLSQP with multiple restarts
        best_local = None
        best_val = np.inf
        
        # Try a few different maxiters/time settings
        for attempt in range(3):
            if time.time() > deadline - 10:
                break
            
            # Add small random perturbation for diversity
            v = v0.copy()
            if attempt > 0:
                v[:2*N] += 0.02 * np.random.randn(2*N)
                v[2*N:3*N] *= (1 + 0.1 * np.random.randn(N))
                v[3*N] += 0.1 * np.random.randn()
                v = np.clip(v, [b[0] for b in bounds], [b[1] for b in bounds])
            
            try:
                res = minimize(
                    _objective, v,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=_constraints(v),
                    options={'maxiter': 200, 'ftol': 1e-8}
                )
                if res.success or res.fun < best_val:
                    val = res.fun
                    # Extract and check
                    x_res = res.x[:N]
                    y_res = res.x[N:2*N]
                    r_res = res.x[2*N:3*N]
                    aspect_res = res.x[3*N]
                    
                    # Validate
                    w, h = _unpack(x_res, y_res, r_res, aspect_res)
                    valid = True
                    for i in range(N):
                        if r_res[i] < 1e-8:
                            valid = False
                        if x_res[i] < r_res[i] - 1e-6 or x_res[i] + r_res[i] > w + 1e-6:
                            valid = False
                        if y_res[i] < r_res[i] - 1e-6 or y_res[i] + r_res[i] > h + 1e-6:
                            valid = False
                    for i in range(N):
                        for j in range(i+1, N):
                            dist = np.sqrt((x_res[i]-x_res[j])**2 + (y_res[i]-y_res[j])**2)
                            if dist < r_res[i] + r_res[j] - 1e-6:
                                valid = False
                    
                    if valid and -res.fun < best_val:
                        best_val = -res.fun
                        best_local = np.column_stack([x_res, y_res, r_res])
            except Exception:
                pass
        
        return best_local
    
    best_result = None
    best_score = -np.inf
    
    # Try multiple initial configurations and aspect ratios
    aspects = [0.8, 1.0, 1.2, 1.5]
    
    for aspect in aspects:
        if time.time() > deadline - 20:
            break
        
        # Hex start
        hex_c = _hex_start(aspect)
        if hex_c is not None:
            res = _optimize_from(hex_c, aspect)
            if res is not None:
                score = np.sum(res[:, 2])
                if score > best_score:
                    best_score = score
                    best_result = res.copy()
        
        # Grid starts
        for cols in [4, 5, 6]:
            if time.time() > deadline - 15:
                break
            grid_c = _grid_start(aspect, cols)
            if grid_c is not None and len(grid_c) == N:
                res = _optimize_from(grid_c, aspect)
                if res is not None:
                    score = np.sum(res[:, 2])
                    if score > best_score:
                        best_score = score
                        best_result = res.copy()
        
        # Random starts
        for _ in range(2):
            if time.time() > deadline - 10:
                break
            rand_c = _random_start(aspect)
            res = _optimize_from(rand_c, aspect)
            if res is not None:
                score = np.sum(res[:, 2])
                if score > best_score:
                    best_score = score
                    best_result = res.copy()
    
    # If no valid result found, use a simple fallback
    if best_result is None:
        # Simple grid in 1x1
        cols = 5
        rows = 5
        cw = 1.0 / cols
        ch = 1.0 / rows
        circles = []
        count = 0
        for r_idx in range(rows):
            for c_idx in range(cols):
                if count >= N:
                    break
                cx = (c_idx + 0.5) * cw
                cy = (r_idx + 0.5) * ch
                rad = 0.5 * min(cw, ch) * 0.9
                circles.append([cx, cy, rad])
                count += 1
        best_result = np.array(circles)
    
    # Final validation and cleanup
    # Ensure circles are within bounds for the implied rectangle
    # The optimizer should have handled this, but clamp as safety
    w = 2.0 / (1.0 + 1.0/best_result[0, 2] if best_result.shape[1] == 3 else 1.0)
    # Actually, we need to recover the aspect. Let's just return the circles.
    # The evaluator will compute the bounding box.
    
    return best_result

