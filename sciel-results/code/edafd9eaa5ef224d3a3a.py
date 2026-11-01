"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s limit

    best_value = 0.0
    best_result = None
    rng = np.random.default_rng(42)

    def _is_valid(cx, cy, r, W, H, tol=1e-8):
        """Check if packing is valid."""
        if W <= 0.05 or H <= 0.05:
            return False
        if np.any(cx - r < -tol) or np.any(cy - r < -tol):
            return False
        if np.any(cx + r > W + tol) or np.any(cy + r > H + tol):
            return False
        for i in range(N):
            for j in range(i + 1, N):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - tol:
                    return False
        return True

    def _sum_radii(cx, cy, r, W, H):
        if _is_valid(cx, cy, r, W, H):
            return np.sum(r)
        return -1.0

    def _cellular_init(w, h, rows, cols, rng):
        """
        Initialize circles in a grid of cells.
        Returns x, y, r arrays.
        """
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)

        total_cells = rows * cols
        if total_cells < N:
            while rows * cols < N:
                cols += 1
            total_cells = rows * cols

        for i in range(N):
            row = i // cols
            col = i % cols

            x_min = col * (w / cols)
            x_max = (col + 1) * (w / cols)
            y_min = row * (h / rows)
            y_max = (row + 1) * (h / rows)

            cx = (x_min + x_max) / 2.0
            cy = (y_min + y_max) / 2.0

            cell_w = w / cols
            cell_h = h / rows
            max_r = min(cell_w, cell_h) / 2.0

            r0 = max_r * 0.9

            jitter_x = rng.uniform(-0.01, 0.01)
            jitter_y = rng.uniform(-0.01, 0.01)

            cx = np.clip(cx + jitter_x, x_min + r0, x_max - r0)
            cy = np.clip(cy + jitter_y, y_min + r0, y_max - r0)

            x[i] = cx
            y[i] = cy
            r[i] = r0

        return x, y, r

    def _force_relax(x, y, r, w, h, max_iter=2000, time_budget=20.0):
        """Force-directed relaxation with adaptive step and jitter."""
        x = x.copy().astype(float)
        y = y.copy().astype(float)
        r = r.copy().astype(float)
        
        n = N
        dt = 0.01
        min_dt = 1e-4
        local_rng = np.random.default_rng(123 + int(time.time() * 1000) % 10000)
        
        for it in range(max_iter):
            if time.time() > time_budget:
                break
            
            fx = np.zeros(n)
            fy = np.zeros(n)
            
            # Wall repulsion
            left_viol = np.maximum(r - x, 0.0)
            right_viol = np.maximum(x + r - w, 0.0)
            bottom_viol = np.maximum(r - y, 0.0)
            top_viol = np.maximum(y + r - h, 0.0)
            
            fx -= left_viol * 50
            fx += right_viol * 50
            fy -= bottom_viol * 50
            fy += top_viol * 50
            
            # Pairwise repulsion
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            d_sq = dx * dx + dy * dy
            r_sum = r[:, None] + r[None, :]
            
            iu = np.triu_indices(n, k=1)
            d_sq_pairs = d_sq[iu]
            r_sum_pairs = r_sum[iu]
            
            d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
            overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
            
            if np.max(overlap) > 1e-6:
                f_pairs = overlap * 20.0
                dx_pairs = dx[iu]
                dy_pairs = dy[iu]
                ux = dx_pairs / d_pairs
                uy = dy_pairs / d_pairs
                
                for idx in range(len(iu[0])):
                    i_idx = iu[0][idx]
                    j_idx = iu[1][idx]
                    fx[i_idx] += ux[idx] * f_pairs[idx]
                    fy[i_idx] += uy[idx] * f_pairs[idx]
                    fx[j_idx] -= ux[idx] * f_pairs[idx]
                    fy[j_idx] -= uy[idx] * f_pairs[idx]
                    
            # Check convergence
            if np.max(overlap) < 1e-7 and np.max(np.abs(left_viol)) < 1e-7:
                break
            
            # Update positions
            x += dt * fx
            y += dt * fy
            
            # Jitter to escape local minima
            if it % 50 == 0 and it > 0:
                x += local_rng.normal(0, 0.001, n)
                y += local_rng.normal(0, 0.001, n)
                
            # Clamp positions
            x = np.clip(x, r, w - r)
            y = np.clip(y, r, h - r)
            
            # Adaptive step size
            force_mag = np.max(np.sqrt(fx*fx + fy*fy))
            if force_mag > 1.0:
                dt = max(min_dt, dt * 0.5)
            else:
                dt = min(0.05, dt * 1.1)
                
        return x, y, r

    def _slsqp_refine(x, y, r, w, h, time_budget=20.0):
        """SLSQP optimization with explicit constraints."""
        n = N
        z = np.concatenate([x, y, r])
        
        def objective(zv):
            return -np.sum(zv[2*n:3*n])
            
        constraints = []
        bounds = []
        
        for i in range(n):
            bounds.extend([(0, w), (0, h), (0.001, min(w, h)/2.0)])
            
        # Boundary constraints
        for i in range(n):
            # Left: x - r >= 0
            constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[i] - p[3*n+i]})
            # Right: w - (x + r) >= 0
            constraints.append({'type': 'ineq', 'fun': lambda p, i=i: w - p[i] - p[3*n+i]})
            # Bottom: y - r >= 0
            constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[n+i] - p[3*n+i]})
            # Top: h - (y + r) >= 0
            constraints.append({'type': 'ineq', 'fun': lambda p, i=i: h - p[n+i] - p[3*n+i]})
            
        # Overlap constraints
        for i in range(n):
            for j in range(i+1, n):
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda p, i=i, j=j: (p[i]-p[j])**2 + (p[n+i]-p[n+j])**2 - (p[3*n+i]+p[3*n+j])**2
                })
                
        try:
            res = minimize(objective, z, method='SLSQP', bounds=bounds, constraints=constraints,
                           options={'maxiter': 300, 'ftol': 1e-10, 'maxls': 100})
            if res.success:
                x_opt = res.x[:n]
                y_opt = res.x[n:2*n]
                r_opt = res.x[2*n:3*n]
                if _is_valid(x_opt, y_opt, r_opt, w, h):
                    return x_opt, y_opt, r_opt
        except Exception:
            pass
        return x, y, r

    def _expand_radii(x, y, r, w, h, max_iter=500):
        """Iteratively expand radii as much as possible."""
        r = r.copy()
        for _ in range(max_iter):
            improved = False
            for i in range(N):
                # Calculate max possible radius for i
                r_max = min(x[i], w - x[i], y[i], h - y[i])
                for j in range(N):
                    if i == j: continue
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx*dx + dy*dy)
                    r_allowed = d - r[j]
                    if r_allowed < r_max:
                        r_max = r_allowed
                r_max = max(r_max, 0.0)
                
                if r_max > r[i] + 1e-8:
                    r[i] = r_max
                    improved = True
            if not improved:
                break
        return r

    def _simulated_annealing(x, y, r, w, h, max_iter=3000, t0=0.05, t1=0.001):
        """Simulated annealing to escape local minima."""
        x = x.copy().astype(float)
        y = y.copy().astype(float)
        r = r.copy().astype(float)
        
        best_x = x.copy()
        best_y = y.copy()
        best_r = r.copy()
        best_score = _sum_radii(best_x, best_y, best_r, w, h)
        
        current_score = best_score
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            temperature = t0 * (t1 / t0) ** (it / max_iter)
            
            # Pick a random circle and perturb it
            i = np.random.randint(N)
            old_x, old_y, old_r = x[i], y[i], r[i]
            
            # Random perturbation
            new_x = old_x + np.random.uniform(-0.05, 0.05)
            new_y = old_y + np.random.uniform(-0.05, 0.05)
            new_r = old_r + np.random.uniform(-0.02, 0.02)
            
            # Clamp to valid range
            new_r = np.clip(new_r, 0.01, min(w, h) / 2.0 - 0.01)
            new_x = np.clip(new_x, new_r, w - new_r)
            new_y = np.clip(new_y, new_r, h - new_r)
            
            x[i] = new_x
            y[i] = new_y
            r[i] = new_r
            
            new_score = _sum_radii(x, y, r, w, h)
            
            if new_score > current_score:
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_x = x.copy()
                    best_y = y.copy()
                    best_r = r.copy()
            else:
                # Accept with probability exp((new_score - current_score) / temperature)
                delta = new_score - current_score
                if delta < 0 and np.random.rand() < np.exp(delta / temperature):
                    current_score = new_score
                else:
                    # Revert
                    x[i] = old_x
                    y[i] = old_y
                    r[i] = old_r
        
        if best_score > 0:
            return best_x, best_y, best_r, best_score
        return None

    # Main search
    # Define different grid layouts
    layouts = []
    for rows in [3, 4, 5]:
        for cols in [4, 5, 6, 7]:
            if rows * cols >= N:
                layouts.append((rows, cols))

    # Aspect ratios to try
    aspect_ratios = [0.8, 0.9, 1.0, 1.1, 1.2]

    for w_frac in aspect_ratios:
        w = w_frac
        h = 2.0 - w
        if w < 0.3 or h < 0.3:
            continue

        for rows, cols in layouts:
            if time.time() > deadline - 15.0:
                break
                
            # Get initialization
            x, y, r = _cellular_init(w, h, rows, cols, rng)
            
            # 1. Force relaxation
            x, y, r = _force_relax(x, y, r, w, h, time_budget=10.0)
            
            # 2. SLSQP Refinement
            x, y, r = _slsqp_refine(x, y, r, w, h, time_budget=15.0)
            
            # 3. Expand Radii
            r = _expand_radii(x, y, r, w, h)
            
            # Check validity and update best
            score = _sum_radii(x, y, r, w, h)
            if score > best_value:
                best_value = score
                best_result = np.column_stack([x, y, r])
            
            # 4. Simulated Annealing Polish
            if time.time() < deadline - 10.0:
                sa_result = _simulated_annealing(x, y, r, w, h, max_iter=2000)
                if sa_result is not None:
                    sa_x, sa_y, sa_r, sa_score = sa_result
                    if sa_score > best_value:
                        best_value = sa_score
                        best_result = np.column_stack([sa_x, sa_y, sa_r])
                        x, y, r = sa_x, sa_y, sa_r

    # Fallback
    if best_result is None:
        w, h = 1.0, 1.0
        x, y, r = _cellular_init(w, h, 5, 5, rng)
        x, y, r = _force_relax(x, y, r, w, h, time_budget=5.0)
        best_result = np.column_stack([x, y, r])

    return best_result
