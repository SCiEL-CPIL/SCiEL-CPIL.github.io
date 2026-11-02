"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    best_value = -np.inf
    best_result = None
    best_w = 1.0
    best_h = 1.0

    def is_valid(x, w, h, tol=1e-6):
        """Check if configuration is strictly valid with tolerance."""
        n = len(x) // 3
        # Check bounds
        for i in range(n):
            xi, yi, ri = x[3*i], x[3*i+1], x[3*i+2]
            if ri < 1e-9:
                return False
            if xi - ri < -tol or xi + ri > w + tol or yi - ri < -tol or yi + ri > h + tol:
                return False
        
        # Check disjointness
        for i in range(n):
            xi, yi, ri = x[3*i], x[3*i+1], x[3*i+2]
            for j in range(i+1, n):
                xj, yj, rj = x[3*j], x[3*j+1], x[3*j+2]
                dx = xi - xj
                dy = yi - yj
                dist_sq = dx*dx + dy*dy
                min_dist = ri + rj
                # Use squared distance to avoid sqrt and be precise
                if dist_sq < (min_dist - tol)**2:
                    return False
        return True

    def check_and_update(x, w, h):
        nonlocal best_value, best_result, best_w, best_h
        if time.time() > deadline:
            return
        
        if is_valid(x, w, h, tol=1e-5):
            val = np.sum(x[2::3])
            if val > best_value:
                best_value = val
                best_result = x.copy()
                best_w = w
                best_h = h

    def softplus(x, beta=100.0):
        """Numerically stable softplus."""
        return np.where(x > 0, x + np.log1p(np.exp(-x)), np.log1p(np.exp(x)))

    def force_directed_relax(x, w, h, max_iters=500, lr=0.01):
        """
        Physics-based relaxation using repulsive forces.
        Returns improved x.
        """
        if time.time() > deadline:
            return x
        
        x = x.copy()
        # Use a cooling schedule
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            xs = x[0::3].copy()
            ys = x[1::3].copy()
            rs = x[2::3].copy()
            
            forces_x = np.zeros(N)
            forces_y = np.zeros(N)
            forces_r = np.zeros(N)
            
            # Repulsive forces between circles
            for i in range(N):
                for j in range(i+1, N):
                    dx = xs[i] - xs[j]
                    dy = ys[i] - ys[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = rs[i] + rs[j]
                    
                    if dist < 1e-10:
                        # Random nudge if exactly overlapping
                        dx = np.random.uniform(-0.01, 0.01)
                        dy = np.random.uniform(-0.01, 0.01)
                        dist = np.sqrt(dx*dx + dy*dy)
                    
                    if dist < min_dist:
                        # Strong repulsion when overlapping
                        overlap = min_dist - dist
                        force_mag = 10.0 * overlap + 1.0  # Base force + overlap penalty
                        fx = force_mag * dx / dist
                        fy = force_mag * dy / dist
                        
                        forces_x[i] += fx
                        forces_y[i] += fy
                        forces_x[j] -= fx
                        forces_y[j] -= fy
                        
                        # Push radii down slightly if overlapping
                        forces_r[i] -= 0.5 * overlap
                        forces_r[j] -= 0.5 * overlap
            
            # Boundary forces
            for i in range(N):
                # Left/Right walls
                if xs[i] - rs[i] < 0:
                    forces_x[i] += 10.0 * (xs[i] - rs[i])
                if xs[i] + rs[i] > w:
                    forces_x[i] -= 10.0 * (xs[i] + rs[i] - w)
                
                # Top/Bottom walls
                if ys[i] - rs[i] < 0:
                    forces_y[i] += 10.0 * (ys[i] - rs[i])
                if ys[i] + rs[i] > h:
                    forces_y[i] -= 10.0 * (ys[i] + rs[i] - h)
                
                # Grow radii if space allows (simple heuristic)
                # Check if this circle has room to grow
                min_margin = min(
                    xs[i] - rs[i], 
                    w - (xs[i] + rs[i]),
                    ys[i] - rs[i],
                    h - (ys[i] + rs[i])
                )
                
                # Check closest neighbor
                closest = np.inf
                for j in range(N):
                    if i == j: continue
                    dx = xs[i] - xs[j]
                    dy = ys[i] - ys[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    margin = dist - rs[i] - rs[j]
                    if margin < closest:
                        closest = margin
                
                available_space = min(min_margin, closest)
                if available_space > 0.01:
                    forces_r[i] += 1.0
                else:
                    forces_r[i] -= 0.5 * max(0, 0.01 - available_space)
            
            # Apply forces with learning rate decay
            current_lr = lr * (0.95 ** it)
            for i in range(N):
                x[3*i] += current_lr * forces_x[i]
                x[3*i+1] += current_lr * forces_y[i]
                x[3*i+2] += current_lr * forces_r[i]
                
                # Clamp to bounds
                x[3*i] = np.clip(x[3*i], 0, w)
                x[3*i+1] = np.clip(x[3*i+1], 0, h)
                x[3*i+2] = np.clip(x[3*i+2], 1e-6, min(w, h)/2)
        
        return x

    def polish(x, w, h):
        """Use gradient-based polish with softplus penalties."""
        if time.time() > deadline:
            return x
        
        def objective(params):
            xs = params[0::3]
            ys = params[1::3]
            rs = params[2::3]
            
            penalty = 0.0
            
            # Boundary penalties (softplus for smoothness)
            for i in range(N):
                if rs[i] <= 0:
                    penalty += (rs[i] + 1e-3)**2
                else:
                    penalty += softplus(-(xs[i] - rs[i]))
                    penalty += softplus(xs[i] + rs[i] - w)
                    penalty += softplus(-(ys[i] - rs[i]))
                    penalty += softplus(ys[i] + rs[i] - h)
            
            # Overlap penalties (softplus for smoothness)
            for i in range(N):
                for j in range(i+1, N):
                    dx = xs[i] - xs[j]
                    dy = ys[i] - ys[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = rs[i] + rs[j]
                    penalty += softplus(min_dist - dist)
            
            return penalty - np.sum(rs)

        bounds = []
        r_max = min(w, h) / 2.0
        for i in range(N):
            bounds.append((0, w))
            bounds.append((0, h))
            bounds.append((1e-6, r_max))
            
        try:
            res = minimize(objective, x, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8})
            if res.success and is_valid(res.x, w, h, tol=1e-6):
                if np.sum(res.x[2::3]) >= np.sum(x[2::3]) - 1e-6:
                    return res.x
        except Exception:
            pass
        return x

    # Try multiple aspect ratios and initialization strategies
    aspect_ratios = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]
    
    for w in aspect_ratios:
        if time.time() > deadline:
            break
        h = 2.0 - w
        
        # Try different grid configurations
        grid_configs = []
        # 7x3, 6x4 (with one row having fewer), 5x5 (with extras), etc.
        for a in range(3, 8):
            b = int(np.ceil(N / a))
            if a * b >= N:
                grid_configs.append((a, b))
        
        if not grid_configs:
            grid_configs = [(7, 3)]
        
        for (a, b) in grid_configs[:3]:  # Limit to 3 configs per aspect ratio
            if time.time() > deadline:
                break
            
            # Create grid initialization
            centers = np.zeros((N, 2))
            for i in range(N):
                col = i % a
                row = i // a
                centers[i][0] = (col + 0.5) * w / a
                centers[i][1] = (row + 0.5) * h / b
            
            # Estimate initial radius
            r_est = 0.5 * min(w/a, h/b) * 0.85
            r_est = min(r_est, w/2, h/2)
            
            x0 = np.zeros(3*N)
            for i in range(N):
                x0[3*i] = centers[i][0]
                x0[3*i+1] = centers[i][1]
                x0[3*i+2] = r_est
            
            # Force-directed relaxation first
            relaxed = force_directed_relax(x0, w, h, max_iters=300, lr=0.005)
            check_and_update(relaxed, w, h)
            
            # Then gradient polish
            if time.time() > deadline:
                break
            polished = polish(relaxed, w, h)
            check_and_update(polished, w, h)
    
    # Fallback if no valid result found
    if best_result is None:
        a, b = 7, 3
        w, h = 1.4, 0.6
        r = 0.09
        best_result = np.zeros(3*N)
        for i in range(a):
            for j in range(b):
                idx = i*b + j
                best_result[3*idx] = (i + 0.5) * w / a
                best_result[3*idx+1] = (j + 0.5) * h / b
                best_result[3*idx+2] = r
        best_w = w
        best_h = h

    # Final strict validation and correction
    if not is_valid(best_result, best_w, best_h, tol=1e-7):
        for i in range(N):
            best_result[3*i+2] *= 0.99
        
        if not is_valid(best_result, best_w, best_h, tol=1e-7):
            a, b = 7, 3
            w, h = 1.4, 0.6
            r = 0.05
            best_result = np.zeros(3*N)
            for i in range(a):
                for j in range(b):
                    idx = i*b + j
                    best_result[3*idx] = (i + 0.5) * w / a
                    best_result[3*idx+1] = (j + 0.5) * h / b
                    best_result[3*idx+2] = r
            best_w = w
            best_h = h

    result = np.zeros((N, 3))
    for i in range(N):
        result[i] = best_result[3*i:3*i+3]
    
    return result
# EVOLVE-BLOCK-END
