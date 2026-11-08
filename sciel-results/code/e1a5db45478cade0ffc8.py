import numpy as np
import time
from scipy.optimize import minimize

N = 21

def construct():
    start_time = time.time()
    time_limit = 140.0  # Safety margin for 150s limit

    best_score = -np.inf
    best_result = None

    def _validate(xs, ys, rs, w, h, tol=1e-4):
        """Check if the configuration is valid: all circles within bounds and non-overlapping."""
        # Check bounds
        for i in range(N):
            if xs[i] < rs[i] - tol or xs[i] > w - rs[i] + tol:
                return False
            if ys[i] < rs[i] - tol or ys[i] > h - rs[i] + tol:
                return False
        # Check overlaps
        for i in range(N):
            for j in range(i + 1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < rs[i] + rs[j] - tol:
                    return False
        return True

    def _compute_radii_from_centers(xs, ys, w, h):
        """
        For given centers, compute the maximum radius for each circle such that
        it doesn't overlap with neighbors or go out of bounds.
        This is the inradius of the Voronoi cell for each point.
        """
        rs = np.zeros(N)
        for i in range(N):
            # Distance to boundaries
            r_bound = min(xs[i], w - xs[i], ys[i], h - ys[i])
            # Distance to other circles
            r_min = r_bound
            for j in range(N):
                if i == j:
                    continue
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx * dx + dy * dy)
                # The max radius for i given j is dist/2 (if they are equal)
                # But since we want disjoint interiors, the constraint is r_i + r_j <= dist
                # If we set all radii to be determined by the nearest neighbor, it's complex.
                # A simpler approximation: the radius of circle i is limited by half the distance
                # to its nearest neighbor? No, that's only if all circles have the same radius.
                # Actually, if we fix centers, the maximum radius r_i such that the circle
                # is disjoint from all others AND within bounds is:
                # r_i <= min( dist(i, j)/2 for all j )  -- this assumes r_j = r_i
                # But we can have different radii.
                # However, a common heuristic in circle packing optimization is to set
                # r_i = 0.5 * min_j dist(i, j) and also bounded by the boundary.
                # This ensures no overlap if all radii are set this way?
                # Let's check: if r_i = 0.5 * d(i,j) and r_j = 0.5 * d(i,j), then r_i + r_j = d(i,j).
                # So they are tangent. This is a valid configuration.
                # But is it optimal? Not necessarily, but it's a good feasible point.
                # Actually, we can do better. The constraint is r_i + r_j <= d(i,j).
                # If we want to maximize sum(r_i), we can't just set r_i = 0.5*d(i,j) independently.
                # However, for the purpose of this optimization step, we can use this as a
                # "feasible radius" assignment that we then refine.
                # Alternatively, we can optimize the radii directly.
                
                # Let's stick to optimizing centers and radii together, but use a smart initialization.
                # For now, let's just return a feasible radius based on nearest neighbor.
                # This will be used to initialize the radius part of the optimization.
                if dist < r_min * 2:  # If dist/2 is smaller than current r_min
                    r_min = dist / 2.0
            rs[i] = r_min
        return rs

    def _objective(params, w, h):
        """
        params: [x0, y0, r0, x1, y1, r1, ...]
        Minimize -sum(r_i) + penalty
        """
        xs = params[0::3]
        ys = params[1::3]
        rs = params[2::3]
        
        pen = 0.0
        # Boundary penalties
        for i in range(N):
            if xs[i] < rs[i]:
                pen += (rs[i] - xs[i])**2
            if xs[i] > w - rs[i]:
                pen += (xs[i] - (w - rs[i]))**2
            if ys[i] < rs[i]:
                pen += (rs[i] - ys[i])**2
            if ys[i] > h - rs[i]:
                pen += (ys[i] - (h - rs[i]))**2
        
        # Overlap penalties
        for i in range(N):
            for j in range(i + 1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = rs[i] + rs[j]
                if dist < min_dist:
                    pen += (min_dist - dist)**2
        
        return -np.sum(rs) + 100.0 * pen

    def _objective_centers_only(params, w, h):
        """
        params: [x0, y0, x1, y1, ...]
        For given centers, compute feasible radii and maximize their sum.
        """
        xs = params[0::2]
        ys = params[1::2]
        
        # Compute feasible radii
        rs = np.zeros(N)
        for i in range(N):
            r_bound = min(xs[i], w - xs[i], ys[i], h - ys[i])
            r_min = r_bound
            for j in range(N):
                if i == j:
                    continue
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx * dx + dy * dy)
                # The constraint is r_i + r_j <= dist.
                # If we assume r_j is also determined similarly, it's coupled.
                # A simple lower bound for r_i is dist/2 (assuming r_j >= 0).
                # But a better estimate: if we set r_i = r_j, then r_i = dist/2.
                # Let's use the minimum over all j of dist(i,j)/2.
                if dist / 2.0 < r_min:
                    r_min = dist / 2.0
            rs[i] = r_min
        
        # Check for any remaining overlaps (should be none by construction)
        # But just in case, add a penalty if there's still overlap due to numerical issues
        pen = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < rs[i] + rs[j]:
                    pen += (rs[i] + rs[j] - dist)**2
        
        return -np.sum(rs) + 10.0 * pen

    def _run_optimization(w, h, init_xs, init_ys, init_rs, maxiter=500):
        """Run L-BFGS-B optimization on full parameters (x, y, r)."""
        params = np.zeros(3 * N)
        params[0::3] = init_xs
        params[1::3] = init_ys
        params[2::3] = init_rs
        
        res = minimize(_objective, params, args=(w, h), method='L-BFGS-B',
                       options={'maxiter': maxiter, 'ftol': 1e-8, 'gtol': 1e-5})
        
        if res.success:
            xs = res.x[0::3]
            ys = res.x[1::3]
            rs = res.x[2::3]
            if _validate(xs, ys, rs, w, h):
                score = np.sum(rs)
                return score, np.column_stack((xs, ys, rs))
        return -np.inf, None

    def _run_optimization_centers(w, h, init_xs, init_ys, maxiter=1000):
        """Run optimization on centers only, deriving radii."""
        params = np.zeros(2 * N)
        params[0::2] = init_xs
        params[1::2] = init_ys
        
        res = minimize(_objective_centers_only, params, args=(w, h), method='L-BFGS-B',
                       options={'maxiter': maxiter, 'ftol': 1e-8, 'gtol': 1e-5})
        
        if res.success:
            xs = res.x[0::2]
            ys = res.x[1::2]
            # Compute final radii
            rs = np.zeros(N)
            for i in range(N):
                r_bound = min(xs[i], w - xs[i], ys[i], h - ys[i])
                r_min = r_bound
                for j in range(N):
                    if i == j:
                        continue
                    dx = xs[i] - xs[j]
                    dy = ys[i] - ys[j]
                    dist = np.sqrt(dx * dx + dy * dy)
                    if dist / 2.0 < r_min:
                        r_min = dist / 2.0
                rs[i] = r_min
            
            if _validate(xs, ys, rs, w, h):
                score = np.sum(rs)
                return score, np.column_stack((xs, ys, rs))
        return -np.inf, None

    # Search over aspect ratios
    aspect_ratios = np.linspace(1.0, 2.5, 12)
    
    for a in aspect_ratios:
        if time.time() - start_time > time_limit - 10:
            break
            
        w = 2 * a / (1 + a)
        h = 2 / (1 + a)
        
        # Try a few different initializations
        n_restarts = 4
        
        for r in range(n_restarts):
            if time.time() - start_time > time_limit - 5:
                break
            
            # Grid initialization
            side = int(np.ceil(np.sqrt(N)))  # 5
            xs_init = []
            ys_init = []
            idx = 0
            for i in range(side):
                for j in range(side):
                    if idx >= N:
                        break
                    x_c = (i + 0.5) * (w / side)
                    y_c = (j + 0.5) * (h / side)
                    xs_init.append(x_c)
                    ys_init.append(y_c)
                    idx += 1
            
            xs_init = np.array(xs_init)
            ys_init = np.array(ys_init)
            
            # Add jitter
            jitter_scale = 0.02 * min(w, h)
            xs_init += np.random.normal(0, jitter_scale, N)
            ys_init += np.random.normal(0, jitter_scale, N)
            
            # Compute initial radii
            rs_init = _compute_radii_from_centers(xs_init, ys_init, w, h)
            
            # Run optimization on centers only (faster)
            score1, result1 = _run_optimization_centers(w, h, xs_init, ys_init, maxiter=300)
            
            if score1 > best_score:
                best_score = score1
                best_result = result1
            
            # If we have time, also try full optimization from the best center result
            if best_result is not None and time.time() - start_time < time_limit - 15:
                xs_opt = best_result[:, 0]
                ys_opt = best_result[:, 1]
                rs_opt = best_result[:, 2]
                score2, result2 = _run_optimization(w, h, xs_opt, ys_opt, rs_opt, maxiter=200)
                
                if score2 > best_score:
                    best_score = score2
                    best_result = result2

    # Fallback to a simple grid if optimization failed
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        w = 1.0
        h = 1.0
        cells = []
        idx = 0
        for i in range(side):
            for j in range(side):
                if idx >= N:
                    break
                x_c = (i + 0.5) * (w / side)
                y_c = (j + 0.5) * (h / side)
                r_c = 0.5 / side - 1e-3
                cells.append((x_c, y_c, r_c))
                idx += 1
        return np.array(cells)
        
    return best_result
