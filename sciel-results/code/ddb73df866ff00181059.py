"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 118  # Safety margin for 120s limit
    
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

    def force_directed_objective(params, w, h, k=50.0):
        """
        Force-directed relaxation objective.
        Minimizes overlap energy + boundary energy - sum(radii).
        Uses quadratic penalties for smooth gradients.
        """
        xs = params[0::3]
        ys = params[1::3]
        rs = params[2::3]
        
        # Boundary violations (quadratic)
        v_l = np.maximum(0, rs - xs)
        v_r = np.maximum(0, xs + rs - w)
        v_b = np.maximum(0, rs - ys)
        v_t = np.maximum(0, ys + rs - h)
        
        boundary_penalty = k * (np.sum(v_l**2) + np.sum(v_r**2) + np.sum(v_b**2) + np.sum(v_t**2))
        
        # Overlap violations (quadratic)
        overlap_penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx*dx + dy*dy + 1e-12)
                min_dist = rs[i] + rs[j]
                v = max(0, min_dist - dist)
                overlap_penalty += k * v**2
        
        return overlap_penalty + boundary_penalty - np.sum(rs)

    def force_directed_gradient(params, w, h, k=50.0):
        """Analytical gradient of force_directed_objective."""
        xs = params[0::3]
        ys = params[1::3]
        rs = params[2::3]
        
        grad = np.zeros_like(params)
        
        # Boundary gradients
        for i in range(N):
            # Left: v_l = max(0, r_i - x_i)
            v_l = max(0, rs[i] - xs[i])
            if v_l > 0:
                grad[3*i] += -2*k*v_l
                grad[3*i+2] += 2*k*v_l
            
            # Right: v_r = max(0, x_i + r_i - w)
            v_r = max(0, xs[i] + rs[i] - w)
            if v_r > 0:
                grad[3*i] += 2*k*v_r
                grad[3*i+2] += 2*k*v_r
            
            # Bottom: v_b = max(0, r_i - y_i)
            v_b = max(0, rs[i] - ys[i])
            if v_b > 0:
                grad[3*i+1] += -2*k*v_b
                grad[3*i+2] += 2*k*v_b
            
            # Top: v_t = max(0, y_i + r_i - h)
            v_t = max(0, ys[i] + rs[i] - h)
            if v_t > 0:
                grad[3*i+1] += 2*k*v_t
                grad[3*i+2] += 2*k*v_t
        
        # Overlap gradients
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx*dx + dy*dy + 1e-12)
                min_dist = rs[i] + rs[j]
                v = max(0, min_dist - dist)
                if v > 0:
                    # d(v^2)/dx_i = 2*v * d(min_dist - dist)/dx_i = 2*v * (dx_i/dist)
                    factor = 2*k*v / dist
                    grad[3*i] += factor * dx
                    grad[3*i+1] += factor * dy
                    grad[3*j] -= factor * dx
                    grad[3*j+1] -= factor * dy
                    grad[3*i+2] += 2*k*v
                    grad[3*j+2] += 2*k*v
        
        # Objective gradient: -sum(r) gives -1 for each radius
        for i in range(N):
            grad[3*i+2] -= 1.0
        
        return grad

    def force_directed_relax(x, w, h, max_iters=100, lr=0.01):
        """
        Gradient descent with adaptive learning rate for force-directed relaxation.
        Returns relaxed configuration.
        """
        if time.time() > deadline:
            return x
        
        x_current = x.copy()
        prev_obj = force_directed_objective(x_current, w, h)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            grad = force_directed_gradient(x_current, w, h)
            
            # Adaptive learning rate
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-8:
                break
            
            # Step size with clipping
            step = lr / (1.0 + 0.1 * grad_norm)
            x_new = x_current - step * grad
            
            # Project to bounds
            for i in range(N):
                x_new[3*i] = np.clip(x_new[3*i], 0, w)
                x_new[3*i+1] = np.clip(x_new[3*i+1], 0, h)
                x_new[3*i+2] = np.clip(x_new[3*i+2], 1e-6, min(w, h)/2)
            
            obj_new = force_directed_objective(x_new, w, h)
            
            if obj_new < prev_obj:
                x_current = x_new
                prev_obj = obj_new
            else:
                # Reduce learning rate
                lr *= 0.5
                if lr < 1e-6:
                    break
        
        return x_current

    def polish(x, w, h):
        """Polish using L-BFGS-B with analytical gradient."""
        if time.time() > deadline:
            return x
        
        bounds = []
        r_max = min(w, h) / 2.0
        for i in range(N):
            bounds.append((0, w))
            bounds.append((0, h))
            bounds.append((1e-6, r_max))
            
        try:
            res = minimize(
                lambda p: force_directed_objective(p, w, h),
                lambda p: force_directed_gradient(p, w, h),
                x, 
                jac=True,
                method='L-BFGS-B', 
                bounds=bounds, 
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            if is_valid(res.x, w, h, tol=1e-6):
                return res.x
        except Exception:
            pass
        return x

    # Try multiple aspect ratios
    widths = np.linspace(0.8, 1.4, 9)
    
    for w in widths:
        if time.time() > deadline:
            break
        h = 2.0 - w
        
        # Initialize with hexagonal-like pattern
        a = int(np.ceil(np.sqrt(N * w / h)))
        b = int(np.ceil(N / a))
        while a * b < N:
            b += 1
        
        centers = np.zeros((N, 2))
        for i in range(N):
            col = i % a
            row = i // a
            # Hexagonal offset for odd rows
            offset = 0.5 * (w/a) if row % 2 == 1 else 0
            centers[i][0] = (col + 0.5) * w/a + offset
            centers[i][1] = (row + 0.5) * h/b
            # Clip to bounds
            centers[i][0] = np.clip(centers[i][0], 0, w)
        
        r_est = 0.5 * min(w/a, h/b) * 0.85
        r_est = min(r_est, w/2, h/2)
        
        x0 = np.zeros(3*N)
        for i in range(N):
            x0[3*i] = centers[i][0]
            x0[3*i+1] = centers[i][1]
            x0[3*i+2] = r_est
        
        # Force-directed relaxation first
        x_relaxed = force_directed_relax(x0, w, h, max_iters=50, lr=0.02)
        
        # Then polish with L-BFGS-B
        x_polished = polish(x_relaxed, w, h)
        check_and_update(x_polished, w, h)
        
        # Also try with jitter
        if time.time() < deadline - 30:
            x_jitter = x0.copy()
            for i in range(N):
                x_jitter[3*i] += np.random.uniform(-0.03, 0.03)
                x_jitter[3*i+1] += np.random.uniform(-0.03, 0.03)
                x_jitter[3*i] = np.clip(x_jitter[3*i], 0, w)
                x_jitter[3*i+1] = np.clip(x_jitter[3*i+1], 0, h)
            
            x_relaxed2 = force_directed_relax(x_jitter, w, h, max_iters=50, lr=0.02)
            x_polished2 = polish(x_relaxed2, w, h)
            check_and_update(x_polished2, w, h)

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
