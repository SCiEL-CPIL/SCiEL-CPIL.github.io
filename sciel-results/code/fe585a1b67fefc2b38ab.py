import numpy as np
from scipy.optimize import minimize
import time

N = 21

def construct():
    start_time = time.time()
    time_limit = 110  # Leave 10s buffer
    
    def place_circle(prev_circles, W, H, init_xy=None, max_iter=80):
        """Place a single circle given previous circles and rectangle dimensions.
        Returns (x, y, r) maximizing r."""
        n_prev = len(prev_circles)
        
        def objective(params):
            x, y, r = params
            return -r
        
        def gradient(params):
            return np.array([0.0, 0.0, -1.0])
        
        constraints = []
        
        # Rectangle bounds
        constraints.append({'type': 'ineq', 'fun': lambda p: p[0] - p[2]})  # x - r >= 0
        constraints.append({'type': 'ineq', 'fun': lambda p: W - p[0] - p[2]})  # W - x - r >= 0
        constraints.append({'type': 'ineq', 'fun': lambda p: p[1] - p[2]})  # y - r >= 0
        constraints.append({'type': 'ineq', 'fun': lambda p: H - p[1] - p[2]})  # H - y - r >= 0
        
        # Non-overlap with previous circles
        for i, (px, py, pr) in enumerate(prev_circles):
            constraints.append({
                'type': 'ineq',
                'fun': lambda p, px=px, py=py, pr=pr: (p[0]-px)**2 + (p[1]-py)**2 - (p[2]+pr)**2
            })
        
        # Radius must be positive
        constraints.append({'type': 'ineq', 'fun': lambda p: p[2] - 1e-6})
        
        # Initial guess
        if init_xy is None:
            x0, y0 = W/2, H/2
        else:
            x0, y0 = init_xy
        r0 = 0.01
        
        x_start = np.array([x0, y0, r0])
        
        try:
            res = minimize(objective, x_start, jac=gradient, method='SLSQP',
                          constraints=constraints, options={'maxiter': max_iter, 'ftol': 1e-10})
            if res.success or res.fun < 0:
                x, y, r = res.x
                if r > 1e-6 and x - r >= -1e-8 and x + r <= W + 1e-8 and y - r >= -1e-8 and y + r <= H + 1e-8:
                    return (x, y, r)
        except Exception:
            pass
        
        # Fallback: small circle at center
        return (W/2, H/2, 0.01)
    
    def validate_packing(circles, W, H):
        """Validate that all circles are disjoint and within bounds."""
        for i, (x, y, r) in enumerate(circles):
            if r <= 0:
                return False
            if x - r < -1e-6 or x + r > W + 1e-6:
                return False
            if y - r < -1e-6 or y + r > H + 1e-6:
                return False
            for j in range(i+1, len(circles)):
                xj, yj, rj = circles[j]
                dist_sq = (x-xj)**2 + (y-yj)**2
                rad_sum = r + rj
                if dist_sq < rad_sum**2 - 1e-6:
                    return False
        return True
    
    def try_aspect_ratio(W, H):
        """Try packing with given aspect ratio (W + H = 2)."""
        if W <= 0 or H <= 0:
            return None, 0.0
        
        circles = []
        
        # Place first circle at center
        x, y, r = W/2, H/2, min(W, H)/2 - 0.01
        if r <= 0.01:
            r = 0.05
        circles.append((x, y, r))
        
        # Place remaining circles
        for i in range(1, N):
            if time.time() - start_time > time_limit:
                break
            
            # Try multiple initial positions
            best_circle = None
            best_r = 0
            
            # Try center
            candidates = [(W/2, H/2)]
            # Try corners
            candidates.append((0.1*W, 0.1*H))
            candidates.append((0.9*W, 0.1*H))
            candidates.append((0.1*W, 0.9*H))
            candidates.append((0.9*W, 0.9*H))
            # Try edges
            candidates.append((0.5*W, 0.1*H))
            candidates.append((0.5*W, 0.9*H))
            candidates.append((0.1*W, 0.5*H))
            candidates.append((0.9*W, 0.5*H))
            
            for cx, cy in candidates:
                circle = place_circle(circles, W, H, init_xy=(cx, cy))
                if circle[2] > best_r:
                    best_r = circle[2]
                    best_circle = circle
            
            if best_circle is None or best_r <= 0.005:
                # Can't place more circles effectively
                break
            
            circles.append(best_circle)
        
        if len(circles) < N:
            # Pad with tiny circles if needed
            while len(circles) < N:
                circles.append((0.01, 0.01, 0.005))
        
        # Validate
        if not validate_packing(circles, W, H):
            return None, 0.0
        
        total_r = sum(c[2] for c in circles)
        return circles, total_r
    
    # Try different aspect ratios
    best_circles = None
    best_score = 0.0
    best_W = 1.0
    best_H = 1.0
    
    # Aspect ratios: W from 0.5 to 1.5 (since W + H = 2, H = 2 - W)
    aspect_ratios = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    
    for W in aspect_ratios:
        if time.time() - start_time > time_limit:
            break
        
        H = 2.0 - W
        circles, score = try_aspect_ratio(W, H)
        
        if circles is not None and score > best_score:
            best_score = score
            best_circles = circles
            best_W = W
            best_H = H
    
    # If no valid packing found, use fallback
    if best_circles is None:
        # Simple grid fallback
        side = int(np.ceil(np.sqrt(N)))
        W, H = 1.0, 1.0
        circles = []
        for i in range(N):
            row = i // side
            col = i % side
            x = (col + 0.5) / side
            y = (row + 0.5) / side
            r = 0.5 / side - 0.02
            circles.append((x, y, r))
        best_score = sum(c[2] for c in circles)
    
    # Build result array
    result = np.zeros((N, 3))
    for i in range(min(N, len(best_circles))):
        result[i, 0] = best_circles[i][0]
        result[i, 1] = best_circles[i][1]
        result[i, 2] = best_circles[i][2]
    
    # Fill remaining with tiny circles if needed
    for i in range(len(best_circles), N):
        result[i, 0] = 0.01
        result[i, 1] = 0.01
        result[i, 2] = 0.005
    
    return result
