import numpy as np
from scipy.optimize import minimize
import time
import random

N = 21
PAIRS = [(i, j) for i in range(N) for j in range(i+1, N)]

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    def check_time():
        return time.time() > deadline

    def smooth_min(x, beta=100.0):
        """Smooth approximation of min(x) using log-sum-exp."""
        x = np.asarray(x)
        x_min = x.min()
        if x_min > 1e-10:
            return x_min
        # Use soft-min: -log(sum(exp(-beta*x)))/beta
        scaled = -beta * x
        scaled -= scaled.max()  # for numerical stability
        log_sum = np.log(np.sum(np.exp(scaled)))
        return -log_sum / beta

    def smooth_max(x, beta=100.0):
        """Smooth approximation of max(x) using log-sum-exp."""
        x = np.asarray(x)
        scaled = beta * x
        scaled -= scaled.max()
        log_sum = np.log(np.sum(np.exp(scaled)))
        return (scaled.max() + log_sum) / beta

    def pack_hexagonal(rows):
        if sum(rows) != N:
            return None
        pts = []
        for k, n_circles in enumerate(rows):
            y_k = k * np.sqrt(3)
            offset = 1.0 if k % 2 == 1 else 0.0
            for i in range(n_circles):
                x_k = i * 2.0 + offset
                pts.append((x_k, y_k))
        pts = np.array(pts)
        if len(pts) != N:
            return None
        center = pts.mean(axis=0)
        pts -= center
        xs = pts[:, 0]
        ys = pts[:, 1]
        w_centers = xs.max() - xs.min()
        h_centers = ys.max() - ys.min()
        w_total = w_centers + 2.0
        h_total = h_centers + 2.0
        scale = 2.0 / (w_total + h_total)
        r = scale
        w = scale * w_total
        h = scale * h_total
        xs_final = pts[:, 0] * scale + w/2.0
        ys_final = pts[:, 1] * scale + h/2.0
        return w, h, r, xs_final, ys_final

    def smooth_optimize(w0, h0, r0, xs0, ys0, max_iter=300, beta=50.0):
        """
        Smooth optimization: maximize sum of radii with smooth barrier for non-overlap.
        Variables: [w, r_0, ..., r_N-1, x_0, ..., x_N-1, y_0, ..., y_N-1]
        h = 2 - w (perimeter constraint)
        """
        if check_time():
            return w0, h0, r0, xs0.copy(), ys0.copy()
        
        n_vars = 1 + N + N + N  # w, radii, xs, ys
        v0 = np.zeros(n_vars)
        v0[0] = w0
        v0[1:1+N] = r0
        v0[1+N:1+2*N] = xs0
        v0[1+2*N:1+3*N] = ys0

        def objective(v):
            w_val = v[0]
            h_val = 2.0 - w_val
            r_vals = v[1:1+N]
            xs_arr = v[1+N:1+2*N]
            ys_arr = v[1+2*N:1+3*N]
            
            if w_val < 0.1 or h_val < 0.1 or np.any(r_vals < 0.001):
                return 1e6
            
            # Penalty for boundary violations
            penalty = 0.0
            lam = 100.0
            for i in range(N):
                ri = r_vals[i]
                xi = xs_arr[i]
                yi = ys_arr[i]
                # Left boundary
                if xi < ri:
                    penalty += lam * (ri - xi)**2
                # Right boundary
                if xi > w_val - ri:
                    penalty += lam * (xi - (w_val - ri))**2
                # Bottom boundary
                if yi < ri:
                    penalty += lam * (ri - yi)**2
                # Top boundary
                if yi > h_val - ri:
                    penalty += lam * (yi - (h_val - ri))**2
            
            # Smooth penalty for overlaps using soft-min of pairwise distances
            min_dist_penalty = 0.0
            for i, j in PAIRS:
                ri = r_vals[i]
                rj = r_vals[j]
                required = ri + rj
                dx = xs_arr[i] - xs_arr[j]
                dy = ys_arr[i] - ys_arr[j]
                dist_sq = dx*dx + dy*dy
                dist = np.sqrt(dist_sq + 1e-12)
                # If dist < required, penalty
                overlap = required - dist
                if overlap > 0:
                    min_dist_penalty += lam * overlap**2
            
            total_penalty = penalty + min_dist_penalty
            sum_radii = np.sum(r_vals)
            return -sum_radii + total_penalty

        def grad_objective(v):
            w_val = v[0]
            h_val = 2.0 - w_val
            r_vals = v[1:1+N].copy()
            xs_arr = v[1+N:1+2*N].copy()
            ys_arr = v[1+2*N:1+3*N].copy()
            
            if w_val < 0.1 or h_val < 0.1 or np.any(r_vals < 0.001):
                return np.ones(n_vars) * 1e6
            
            grad = np.zeros(n_vars)
            lam = 100.0
            
            # Gradient of -sum(r_i)
            grad[1:1+N] = -1.0
            
            # Boundary penalties
            for i in range(N):
                ri = r_vals[i]
                xi = xs_arr[i]
                yi = ys_arr[i]
                # Left: penalty = lam*(ri-xi)^2 if xi < ri
                if xi < ri:
                    d = ri - xi
                    grad[1+i] += 2*lam*d  # d(ri-xi)/dri = 1
                    grad[1+N+i] += -2*lam*d  # d(ri-xi)/dxi = -1
                # Right: penalty = lam*(xi-(w-ri))^2 if xi > w-ri
                if xi > w_val - ri:
                    d = xi - (w_val - ri)
                    grad[1+i] += 2*lam*d  # d(xi-w+ri)/dri = 1
                    grad[1+N+i] += 2*lam*d  # d(xi-w+ri)/dxi = 1
                    grad[0] += -2*lam*d  # d(xi-w+ri)/dw = -1
                # Bottom: penalty = lam*(ri-yi)^2 if yi < ri
                if yi < ri:
                    d = ri - yi
                    grad[1+i] += 2*lam*d
                    grad[1+2*N+i] += -2*lam*d
                # Top: penalty = lam*(yi-(h-ri))^2 if yi > h-ri
                if yi > h_val - ri:
                    d = yi - (h_val - ri)
                    grad[1+i] += 2*lam*d
                    grad[1+2*N+i] += 2*lam*d
                    grad[0] += 2*lam*d  # d(yi-h+ri)/dw = d(yi-(2-w)+ri)/dw = +1
            
            # Overlap penalties
            for i, j in PAIRS:
                ri = r_vals[i]
                rj = r_vals[j]
                required = ri + rj
                dx = xs_arr[i] - xs_arr[j]
                dy = ys_arr[i] - ys_arr[j]
                dist_sq = dx*dx + dy*dy
                dist = np.sqrt(dist_sq + 1e-12)
                overlap = required - dist
                if overlap > 0:
                    # penalty = lam * overlap^2
                    # d(overlap)/dri = 1, d(overlap)/dxi = -dx/dist, etc.
                    coeff = 2*lam*overlap
                    grad[1+i] += coeff
                    grad[1+j] += coeff
                    grad[1+N+i] += -coeff * dx/dist
                    grad[1+N+j] += coeff * dx/dist
                    grad[1+2*N+i] += -coeff * dy/dist
                    grad[1+2*N+j] += coeff * dy/dist
            
            return grad

        bounds = [(0.1, 1.9)] + [(0.001, 1.0)]*N + [(-0.5, 2.5)]*(2*N)
        
        try:
            res = minimize(objective, v0, jac=grad_objective, method='L-BFGS-B', 
                          bounds=bounds, options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False})
            w_res = res.x[0]
            h_res = 2.0 - w_res
            r_res = res.x[1:1+N]
            xs_res = res.x[1+N:1+2*N]
            ys_res = res.x[1+2*N:1+3*N]
            
            # Validate
            valid = True
            if w_res <= 0 or h_res <= 0 or np.any(r_res <= 0):
                valid = False
            else:
                for i in range(N):
                    ri = r_res[i]
                    xi = xs_res[i]
                    yi = ys_res[i]
                    if xi < ri - 1e-6 or xi > w_res - ri + 1e-6:
                        valid = False
                        break
                    if yi < ri - 1e-6 or yi > h_res - ri + 1e-6:
                        valid = False
                        break
                if valid:
                    for i, j in PAIRS:
                        dist_sq = (xs_res[i]-xs_res[j])**2 + (ys_res[i]-ys_res[j])**2
                        min_dist = r_res[i] + r_res[j]
                        if dist_sq < min_dist**2 - 1e-6:
                            valid = False
                            break
            
            if valid and np.sum(r_res) > np.sum([r0]*N):
                return w_res, h_res, r_res, xs_res, ys_res
        except Exception:
            pass
        return w0, h0, r0, xs0.copy(), ys0.copy()

    def pack_grid(rows, cols):
        pts = []
        count = 0
        for i in range(cols):
            for j in range(rows):
                if count >= N:
                    break
                pts.append((i * 2.0, j * 2.0))
                count += 1
            if count >= N:
                break
        pts = np.array(pts[:N])
        center = pts.mean(axis=0)
        pts -= center
        xs = pts[:, 0]
        ys = pts[:, 1]
        w_centers = xs.max() - xs.min()
        h_centers = ys.max() - ys.min()
        w_total = w_centers + 2.0
        h_total = h_centers + 2.0
        scale = 2.0 / (w_total + h_total)
        r = scale
        w = scale * w_total
        h = scale * h_total
        xs_final = pts[:, 0] * scale + w/2.0
        ys_final = pts[:, 1] * scale + h/2.0
        return w, h, r, xs_final, ys_final

    def physics_simulate(x0, y0, w0, h0, r0, max_steps=3000, dt=0.002, cooling_rate=0.9995):
        if check_time():
            return w0, h0, r0, x0.copy(), y0.copy()
        sigma = 2.0 * r0
        eps = 1.0
        boundary_strength = 50.0
        x = x0.copy().astype(float)
        y = y0.copy().astype(float)
        vx = np.zeros(N)
        vy = np.zeros(N)
        temperature = 0.01
        damping = 0.99
        for step in range(max_steps):
            if check_time():
                break
            fx = np.zeros(N)
            fy = np.zeros(N)
            for i in range(N):
                for j in range(i+1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist_sq = dx*dx + dy*dy
                    dist = np.sqrt(dist_sq)
                    if dist < 1e-10:
                        dx = random.uniform(-1e-6, 1e-6)
                        dy = random.uniform(-1e-6, 1e-6)
                        dist_sq = dx*dx + dy*dy
                        dist = np.sqrt(dist_sq)
                    sr6 = (sigma/dist)**6
                    sr12 = sr6*sr6
                    force_mag = 24*eps*(2*sr12 - sr6) / dist
                    fx[i] += force_mag * dx
                    fy[i] += force_mag * dy
                    fx[j] -= force_mag * dx
                    fy[j] -= force_mag * dy
            for i in range(N):
                if x[i] < r0:
                    penetration = r0 - x[i]
                    fx[i] += boundary_strength * penetration
                if x[i] > w0 - r0:
                    penetration = x[i] - (w0 - r0)
                    fx[i] -= boundary_strength * penetration
                if y[i] < r0:
                    penetration = r0 - y[i]
                    fy[i] += boundary_strength * penetration
                if y[i] > h0 - r0:
                    penetration = y[i] - (h0 - r0)
                    fy[i] -= boundary_strength * penetration
            vx = vx * damping + fx * dt
            vy = vy * damping + fy * dt
            x += vx * dt
            y += vy * dt
            temperature *= cooling_rate
            if step % 100 == 0 and temperature > 1e-4:
                x += np.random.normal(0, temperature*0.01, N)
                y += np.random.normal(0, temperature*0.01, N)
            x = np.clip(x, -0.5, w0 + 0.5)
            y = np.clip(y, -0.5, h0 + 0.5)
        xs_min = x.min() - r0
        xs_max = x.max() + r0
        ys_min = y.min() - r0
        ys_max = y.max() + r0
        w_actual = xs_max - xs_min
        h_actual = ys_max - ys_min
        if w_actual + h_actual > 0:
            scale = 2.0 / (w_actual + h_actual)
            w_final = scale * w_actual
            h_final = scale * h_actual
            r_final = scale * r0
            x_final = (x - xs_min) * scale
            y_final = (y - ys_min) * scale
        else:
            w_final = w0
            h_final = h0
            r_final = r0
            x_final = x.copy()
            y_final = y.copy()
        return w_final, h_final, r_final, x_final, y_final

    def equal_radius_smooth_optimize(w0, h0, r0, xs0, ys0, max_iter=300):
        """
        Smooth optimization for equal-radius case: maximize r with smooth barrier.
        Variables: [w, r, x_0, ..., x_N-1, y_0, ..., y_N-1]
        """
        if check_time():
            return w0, h0, r0, xs0.copy(), ys0.copy()
        
        n_vars = 2 + 2*N
        v0 = np.zeros(n_vars)
        v0[0] = w0
        v0[1] = r0
        v0[2:2+N] = xs0
        v0[2+N:2+2*N] = ys0

        def objective(v):
            w_val = v[0]
            r_val = v[1]
            h_val = 2.0 - w_val
            xs_arr = v[2:2+N]
            ys_arr = v[2+N:2+2*N]
            
            if w_val < 0.1 or h_val < 0.1 or r_val < 0.001:
                return 1e6
            
            penalty = 0.0
            lam = 100.0
            for i in range(N):
                if xs_arr[i] < r_val:
                    d = r_val - xs_arr[i]
                    penalty += lam * d*d
                if xs_arr[i] > w_val - r_val:
                    d = xs_arr[i] - (w_val - r_val)
                    penalty += lam * d*d
                if ys_arr[i] < r_val:
                    d = r_val - ys_arr[i]
                    penalty += lam * d*d
                if ys_arr[i] > h_val - r_val:
                    d = ys_arr[i] - (h_val - r_val)
                    penalty += lam * d*d
            
            for i, j in PAIRS:
                dx = xs_arr[i] - xs_arr[j]
                dy = ys_arr[i] - ys_arr[j]
                dist_sq = dx*dx + dy*dy
                min_dist = 2.0 * r_val
                if dist_sq < min_dist*min_dist:
                    dist = np.sqrt(dist_sq + 1e-12)
                    d = min_dist - dist
                    penalty += lam * d*d
            
            return -r_val + penalty

        def grad(v):
            w_val = v[0]
            r_val = v[1]
            h_val = 2.0 - w_val
            xs_arr = v[2:2+N]
            ys_arr = v[2+N:2+2*N]
            
            if w_val < 0.1 or h_val < 0.1 or r_val < 0.001:
                return np.ones(n_vars) * 1e6
            
            grad = np.zeros(n_vars)
            grad[1] = -1.0
            lam = 100.0
            
            for i in range(N):
                xi = xs_arr[i]
                yi = ys_arr[i]
                if xi < r_val:
                    d = r_val - xi
                    grad[1] += 2*lam*d
                    grad[2+i] += -2*lam*d
                if xi > w_val - r_val:
                    d = xi - (w_val - r_val)
                    grad[1] += 2*lam*d
                    grad[2+i] += 2*lam*d
                    grad[0] += -2*lam*d
                if yi < r_val:
                    d = r_val - yi
                    grad[1] += 2*lam*d
                    grad[2+N+i] += -2*lam*d
                if yi > h_val - r_val:
                    d = yi - (h_val - r_val)
                    grad[1] += 2*lam*d
                    grad[2+N+i] += 2*lam*d
                    grad[0] += 2*lam*d
            
            for i, j in PAIRS:
                dx = xs_arr[i] - xs_arr[j]
                dy = ys_arr[i] - ys_arr[j]
                dist_sq = dx*dx + dy*dy
                min_dist = 2.0 * r_val
                if dist_sq < min_dist*min_dist:
                    dist = np.sqrt(dist_sq + 1e-12)
                    d = min_dist - dist
                    coeff = 2*lam*d
                    grad[1] += 2*coeff
                    grad[2+i] += -coeff*dx/dist
                    grad[2+j] += coeff*dx/dist
                    grad[2+N+i] += -coeff*dy/dist
                    grad[2+N+j] += coeff*dy/dist
            
            return grad

        bounds = [(0.1, 1.9), (0.001, 1.0)] + [(-0.5, 2.5)]*(2*N)
        
        try:
            res = minimize(objective, v0, jac=grad, method='L-BFGS-B', 
                          bounds=bounds, options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False})
            w_res = res.x[0]
            r_res = res.x[1]
            h_res = 2.0 - w_res
            xs_res = res.x[2:2+N]
            ys_res = res.x[2+N:2+2*N]
            
            valid = True
            if w_res <= 0 or h_res <= 0 or r_res <= 0:
                valid = False
            else:
                for i in range(N):
                    if xs_res[i] < r_res - 1e-6 or xs_res[i] > w_res - r_res + 1e-6:
                        valid = False
                        break
                    if ys_res[i] < r_res - 1e-6 or ys_res[i] > h_res - r_res + 1e-6:
                        valid = False
                        break
                if valid:
                    for i, j in PAIRS:
                        dist_sq = (xs_res[i]-xs_res[j])**2 + (ys_res[i]-ys_res[j])**2
                        if dist_sq < 4*(r_res**2) - 1e-6:
                            valid = False
                            break
            
            if valid and r_res > r0:
                return w_res, h_res, r_res, xs_res, ys_res
        except Exception:
            pass
        return w0, h0, r0, xs0.copy(), ys0.copy()

    def refine_smooth(w, h, r, xs, ys, max_iter=500):
        if check_time():
            return w, h, r, xs, ys
        v0 = np.zeros(2 + 2*N)
        v0[0] = w
        v0[1] = r
        v0[2:2+N] = xs
        v0[2+N:2+2*N] = ys

        def penalty_obj(v):
            w_val = v[0]
            r_val = v[1]
            h_val = 2.0 - w_val
            xs_arr = v[2:2+N]
            ys_arr = v[2+N:2+2*N]
            if w_val < 0.1 or h_val < 0.1 or r_val < 0.001:
                return 1e6
            penalty = 0.0
            lam = 50.0
            for i in range(N):
                if xs_arr[i] < r_val:
                    d = r_val - xs_arr[i]
                    penalty += lam * d * d
                if xs_arr[i] > w_val - r_val:
                    d = xs_arr[i] - (w_val - r_val)
                    penalty += lam * d * d
                if ys_arr[i] < r_val:
                    d = r_val - ys_arr[i]
                    penalty += lam * d * d
                if ys_arr[i] > h_val - r_val:
                    d = ys_arr[i] - (h_val - r_val)
                    penalty += lam * d * d
            for i, j in PAIRS:
                dx = xs_arr[i] - xs_arr[j]
                dy = ys_arr[i] - ys_arr[j]
                dist_sq = dx*dx + dy*dy
                min_dist = 2.0 * r_val
                if dist_sq < min_dist * min_dist:
                    dist = np.sqrt(dist_sq)
                    d = min_dist - dist
                    penalty += lam * d * d
            return -r_val + penalty

        bounds = [(0.1, 1.9), (0.001, 1.0)] + [(-0.5, 2.5)] * (2*N)
        try:
            res = minimize(penalty_obj, v0, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10, 'disp': False})
            w_res = res.x[0]
            r_res = res.x[1]
            h_res = 2.0 - w_res
            xs_res = res.x[2:2+N]
            ys_res = res.x[2+N:2+2*N]
            valid = True
            if w_res <= 0 or h_res <= 0 or r_res <= 0:
                valid = False
            else:
                for i in range(N):
                    if xs_res[i] < r_res - 1e-7 or xs_res[i] > w_res - r_res + 1e-7:
                        valid = False
                        break
                    if ys_res[i] < r_res - 1e-7 or ys_res[i] > h_res - r_res + 1e-7:
                        valid = False
                        break
                if valid:
                    for i, j in PAIRS:
                        dist = np.sqrt((xs_res[i]-xs_res[j])**2 + (ys_res[i]-ys_res[j])**2)
                        if dist < 2*r_res - 1e-7:
                            valid = False
                            break
            if valid and r_res > r:
                return w_res, h_res, r_res, xs_res, ys_res
        except Exception:
            pass
        return w, h, r, xs, ys

    def refine_slsqp(w, h, r, xs, ys, max_iter=200, final_polish=False):
        if check_time():
            return w, h, r, xs, ys
        v0 = np.zeros(2 + 2*N)
        v0[0] = w
        v0[1] = r
        v0[2:2+N] = xs
        v0[2+N:2+2*N] = ys

        def objective(v):
            w_val = v[0]
            r_val = v[1]
            h_val = 2.0 - w_val
            if w_val <= 0.1 or h_val <= 0.1 or r_val <= 0:
                return 1e6
            return -r_val

        def constraints(v):
            w_val = v[0]
            r_val = v[1]
            h_val = 2.0 - w_val
            xs_arr = v[2:2+N]
            ys_arr = v[2+N:2+2*N]
            cons = []
            cons.append({'type': 'ineq', 'fun': lambda v: v[0] - 0.2})
            cons.append({'type': 'ineq', 'fun': lambda v: 2.0 - v[0] - 0.2})
            for i in range(N):
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[2+i] - v[1]})
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[0] - v[2+i] - v[1]})
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[2+N+i] - v[1]})
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: (2.0 - v[0]) - v[2+N+i] - v[1]})
            # Squared distance constraints (smoother)
            for i, j in PAIRS:
                cons.append({
                    'type': 'ineq',
                    'fun': lambda v, i=i, j=j: ((v[2+i]-v[2+j])**2 + (v[2+N+i]-v[2+N+j])**2) - 4*(v[1]**2)
                })
            return cons

        if final_polish:
            bounds = [(0.15, 1.85), (0.005, 1.0)] + [(-1.0, 3.0)] * (2*N)
            ftol = 1e-12
        else:
            bounds = [(0.2, 1.8), (0.01, 1.0)] + [(0.0, 2.0)] * (2*N)
            ftol = 1e-9

        try:
            res = minimize(objective, v0, method='SLSQP', bounds=bounds, constraints=constraints,
                           options={'maxiter': max_iter, 'ftol': ftol, 'disp': False})
            if res.success or res.fun < 0:
                w_res = res.x[0]
                r_res = res.x[1]
                h_res = 2.0 - w_res
                xs_res = res.x[2:2+N]
                ys_res = res.x[2+N:2+2*N]
                valid = True
                if w_res <= 0 or h_res <= 0 or r_res <= 0:
                    valid = False
                else:
                    for i in range(N):
                        if xs_res[i] < r_res - 1e-7 or xs_res[i] > w_res - r_res + 1e-7:
                            valid = False
                            break
                        if ys_res[i] < r_res - 1e-7 or ys_res[i] > h_res - r_res + 1e-7:
                            valid = False
                            break
                    if valid:
                        for i, j in PAIRS:
                            dist_sq = (xs_res[i]-xs_res[j])**2 + (ys_res[i]-ys_res[j])**2
                            if dist_sq < 4*(r_res**2) - 1e-7:
                                valid = False
                                break
                if valid:
                    return w_res, h_res, r_res, xs_res, ys_res
        except Exception:
            pass
        return w, h, r, xs, ys

    best_sum_r = 0
    best_config = None

    configs = []
    configs.append(pack_hexagonal([7, 7, 7]))
    configs.append(pack_hexagonal([4, 5, 4, 4, 4]))
    configs.append(pack_hexagonal([3, 4, 4, 4, 4, 2]))
    configs.append(pack_hexagonal([5, 5, 5, 3, 3]))
    configs.append(pack_hexagonal([6, 6, 5, 4]))
    configs.append(pack_grid(5, 5))
    configs.append(pack_grid(3, 7))
    configs.append(pack_grid(4, 6))
    configs.append(pack_grid(7, 3))
    configs.append(pack_grid(6, 4))

    for _ in range(6):
        if check_time():
            break
        w = 1.0
        h = 1.0
        r = 0.08
        xs = np.random.uniform(r, w-r, N)
        ys = np.random.uniform(r, h-r, N)
        configs.append((w, h, r, xs, ys))

    valid_configs = [c for c in configs if c is not None]

    for cfg in valid_configs:
        if check_time():
            break
        w, h, r, xs, ys = cfg
        sum_r = N * r

        # Physics simulation
        w_phys, h_phys, r_phys, xs_phys, ys_phys = physics_simulate(xs, ys, w, h, r)
        sum_r_phys = N * r_phys
        if sum_r_phys > best_sum_r:
            best_sum_r = sum_r_phys
            best_config = (w_phys, h_phys, np.full(N, r_phys), xs_phys, ys_phys)

        # Smooth equal-radius optimization
        if not check_time():
            w_eq, h_eq, r_eq, xs_eq, ys_eq = equal_radius_smooth_optimize(w_phys, h_phys, r_phys, xs_phys, ys_phys, max_iter=200)
            sum_r_eq = N * r_eq
            if sum_r_eq > best_sum_r:
                best_sum_r = sum_r_eq
                best_config = (w_eq, h_eq, np.full(N, r_eq), xs_eq, ys_eq)

        # Variable-radius smooth optimization (use best so far)
        if not check_time():
            cur_w, cur_h, cur_r_arr, cur_xs, cur_ys = best_config if best_config is not None else (w_phys, h_phys, np.full(N, r_phys), xs_phys, ys_phys)
            r_mean = np.mean(cur_r_arr)
            w_var, h_var, r_var_arr, xs_var, ys_var = smooth_optimize(cur_w, cur_h, r_mean, cur_xs, cur_ys, max_iter=200)
            sum_r_var = np.sum(r_var_arr)
            if sum_r_var > best_sum_r:
                best_sum_r = sum_r_var
                best_config = (w_var, h_var, r_var_arr, xs_var, ys_var)

    # Final polish
    if best_config is not None and not check_time():
        w, h, r_arr, xs, ys = best_config
        # Try equal-radius polish
        r_mean = np.mean(r_arr)
        w_eq, h_eq, r_eq, xs_eq, ys_eq = equal_radius_smooth_optimize(w, h, r_mean, xs, ys, max_iter=400)
        if N * r_eq > best_sum_r:
            best_sum_r = N * r_eq
            best_config = (w_eq, h_eq, np.full(N, r_eq), xs_eq, ys_eq)
        
        # Try variable-radius polish
        if not check_time():
            w_var, h_var, r_var_arr, xs_var, ys_var = smooth_optimize(w, h, r_mean, xs, ys, max_iter=400)
            if np.sum(r_var_arr) > best_sum_r:
                best_sum_r = np.sum(r_var_arr)
                best_config = (w_var, h_var, r_var_arr, xs_var, ys_var)

    if best_config is None:
        w, h, r, xs, ys = valid_configs[0]
        return np.column_stack([xs, ys, np.full(N, r)])

    w, h, r_arr, xs, ys = best_config
    # Scale down slightly for safety
    r_arr_final = r_arr * (1 - 1e-6)
    return np.column_stack([xs, ys, r_arr_final])
