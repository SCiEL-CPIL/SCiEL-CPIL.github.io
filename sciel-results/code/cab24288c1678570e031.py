import numpy as np
import time
from scipy.optimize import minimize

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 150s kill
    
    best_score = 0.0
    best_config = None
    
    def validity_penalty(x, y, r):
        n = len(r)
        if np.any(r < 0):
            return 1e10
        if np.any(x - r < -1e-9) or np.any(y - r < -1e-9):
            return 1e10
        if np.any(x + r > 1 + 1e-9) or np.any(y + r > 1 + 1e-9):
            return 1e10
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < r[i] + r[j] - 1e-9:
                    return 1e10
        return 0.0

    def repair_config(x, y, r):
        r_max_bound = np.minimum(
            np.minimum(x, 1 - x),
            np.minimum(y, 1 - y)
        )
        r = np.minimum(r, r_max_bound)
        r = np.maximum(r, 0.0)
        
        for _ in range(200):
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            d = np.sqrt(dx**2 + dy**2)
            np.fill_diagonal(d, np.inf)
            
            r_sum = r[:, None] + r[None, :]
            overlaps = d < r_sum - 1e-9
            np.fill_diagonal(overlaps, False)
            
            if not np.any(overlaps):
                return x, y, r
            
            # Find the most overlapping pair and shrink the larger circle
            max_overlap = 0
            max_i, max_j = 0, 0
            for i in range(N):
                for j in range(i+1, N):
                    if overlaps[i, j]:
                        ov = r_sum[i, j] - d[i, j]
                        if ov > max_overlap:
                            max_overlap = ov
                            max_i, max_j = i, j
            
            # Shrink the larger radius
            if r[max_i] >= r[max_j]:
                r[max_i] = max(0, r[max_i] - max_overlap * 0.6 - 0.001)
            else:
                r[max_j] = max(0, r[max_j] - max_overlap * 0.6 - 0.001)
        
        return x, y, r

    def optimize_gradient(x, y, r, steps=5000, start_lr=0.008):
        lr = start_lr
        lambda_overlap = 20.0
        lambda_bound = 20.0
        
        for step in range(steps):
            if time.time() > deadline:
                break
            
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            d = np.sqrt(dx**2 + dy**2)
            d_safe = np.where(d < 1e-8, 1e-8, d)
            
            r_sum = r[:, None] + r[None, :]
            overlap = np.maximum(0, r_sum - d)
            np.fill_diagonal(overlap, 0)
            
            grad_x_overlap = -2 * lambda_overlap * np.sum(overlap * (dx / d_safe), axis=1)
            grad_y_overlap = -2 * lambda_overlap * np.sum(overlap * (dy / d_safe), axis=1)
            grad_r_overlap = 2 * lambda_overlap * np.sum(overlap, axis=1)
            
            bound_x_left = np.maximum(0, r - x)
            bound_x_right = np.maximum(0, r - (1 - x))
            bound_y_bottom = np.maximum(0, r - y)
            bound_y_top = np.maximum(0, r - (1 - y))
            
            grad_x_bound = -2 * lambda_bound * bound_x_left + 2 * lambda_bound * bound_x_right
            grad_y_bound = -2 * lambda_bound * bound_y_bottom + 2 * lambda_bound * bound_y_top
            grad_r_bound = 2 * lambda_bound * (bound_x_left + bound_x_right + bound_y_bottom + bound_y_top)
            
            grad_r_obj = -np.ones(N)
            
            grad_x = grad_x_overlap + grad_x_bound
            grad_y = grad_y_overlap + grad_y_bound
            grad_r = grad_r_overlap + grad_r_bound + grad_r_obj
            
            x = x - lr * grad_x
            y = y - lr * grad_y
            r = r - lr * grad_r
            
            r = np.maximum(r, 0.001)
            x = np.clip(x, 0, 1)
            y = np.clip(y, 0, 1)
            
            lr *= 0.9995
            
        return x, y, r

    def local_search(x, y, r, max_iter=800):
        best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
        best_val = -np.sum(best_r)
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
                
            idx = np.random.randint(N)
            candidate_x = best_x.copy()
            candidate_y = best_y.copy()
            candidate_r = best_r.copy()
            
            # Adaptive scale
            progress = it / max_iter
            scale = 0.03 * (1 - progress) + 0.002
            dx = np.random.normal(0, scale)
            dy = np.random.normal(0, scale)
            dr = np.random.normal(0, scale * 0.4)
            
            candidate_x[idx] += dx
            candidate_y[idx] += dy
            candidate_r[idx] += dr
            
            candidate_x[idx] = np.clip(candidate_x[idx], 0, 1)
            candidate_y[idx] = np.clip(candidate_y[idx], 0, 1)
            candidate_r[idx] = np.clip(candidate_r[idx], 0.001, 0.5)
            
            pen = validity_penalty(candidate_x, candidate_y, candidate_r)
            if pen == 0:
                val = -np.sum(candidate_r)
                if val < best_val:
                    best_x, best_y, best_r = candidate_x, candidate_y, candidate_r
                    best_val = val
        
        return best_x, best_y, best_r

    def maximize_radii(x, y, r, iterations=10):
        for _ in range(iterations):
            if time.time() > deadline:
                break
            for i in range(N):
                r_bound = min(x[i], 1 - x[i], y[i], 1 - y[i])
                r_neighbor = 0.5
                for j in range(N):
                    if i == j:
                        continue
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    r_neighbor = min(r_neighbor, dist - r[j])
                
                new_r = min(r_bound, r_neighbor)
                if new_r > r[i]:
                    r[i] = new_r
            if validity_penalty(x, y, r) != 0:
                r *= 0.95
        return x, y, r

    def try_scipy(x, y, r):
        """Use scipy to optimize the configuration"""
        def objective(params):
            x_ = params[:N]
            y_ = params[N:2*N]
            r_ = params[2*N:3*N]
            
            # Boundary violations
            bound_pen = 0.0
            for i in range(N):
                if r_[i] > x_[i]:
                    bound_pen += (r_[i] - x_[i])**2
                if r_[i] > 1 - x_[i]:
                    bound_pen += (r_[i] - (1 - x_[i]))**2
                if r_[i] > y_[i]:
                    bound_pen += (r_[i] - y_[i])**2
                if r_[i] > 1 - y_[i]:
                    bound_pen += (r_[i] - (1 - y_[i]))**2
            
            # Overlap violations
            overlap_pen = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    dx = x_[i] - x_[j]
                    dy = y_[i] - y_[j]
                    d = np.sqrt(dx*dx + dy*dy)
                    if d < r_[i] + r_[j]:
                        overlap_pen += (r_[i] + r_[j] - d)**2
            
            return -np.sum(r_) + 100 * (bound_pen + overlap_pen)
        
        params = np.concatenate([x, y, r])
        
        try:
            result = minimize(objective, params, method='L-BFGS-B', 
                             options={'maxiter': 500, 'ftol': 1e-10})
            if result.success or result.fun < -np.sum(r) + 1:
                x_new = result.x[:N]
                y_new = result.x[N:2*N]
                r_new = result.x[2*N:3*N]
                r_new = np.maximum(r_new, 0.001)
                x_new = np.clip(x_new, 0, 1)
                y_new = np.clip(y_new, 0, 1)
                return x_new, y_new, r_new
        except:
            pass
        return x, y, r

    # Multiple restarts with diverse initializations
    num_restarts = 30
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        np.random.seed(1000 + restart * 13)
        
        # Different initialization strategies
        strategy = restart % 3
        
        if strategy == 0:
            # Grid with jitter
            side = int(np.ceil(np.sqrt(N)))
            xs, ys, rs = [], [], []
            indices = list(range(side * side))
            np.random.shuffle(indices)
            for idx in indices:
                if len(xs) >= N:
                    break
                i = idx % side
                j = idx // side
                x = (i + 0.5) / side + np.random.uniform(-0.04, 0.04)
                y = (j + 0.5) / side + np.random.uniform(-0.04, 0.04)
                x = np.clip(x, 0.08, 0.92)
                y = np.clip(y, 0.08, 0.92)
                xs.append(x)
                ys.append(y)
                rs.append(0.06)
        elif strategy == 1:
            # Random uniform
            xs = np.random.uniform(0.1, 0.9, N)
            ys = np.random.uniform(0.1, 0.9, N)
            rs = np.full(N, 0.05)
        else:
            # Clustered: some larger, some smaller
            xs, ys, rs = [], [], []
            # Place 4 large circles in corners
            corners = [(0.2, 0.2), (0.8, 0.2), (0.2, 0.8), (0.8, 0.8)]
            for cx, cy in corners:
                xs.append(cx + np.random.uniform(-0.03, 0.03))
                ys.append(cy + np.random.uniform(-0.03, 0.03))
                rs.append(0.12)
            # Fill remaining randomly
            for _ in range(N - 4):
                xs.append(np.random.uniform(0.15, 0.85))
                ys.append(np.random.uniform(0.15, 0.85))
                rs.append(0.06)
        
        x = np.array(xs)
        y = np.array(ys)
        r = np.array(rs)
        
        # Optimize
        x, y, r = optimize_gradient(x, y, r, steps=4000, start_lr=0.012)
        x, y, r = repair_config(x, y, r)
        x, y, r = local_search(x, y, r, max_iter=500)
        x, y, r = repair_config(x, y, r)
        x, y, r = maximize_radii(x, y, r)
        
        # Try scipy refinement
        if time.time() < deadline - 20:
            x_sc, y_sc, r_sc = try_scipy(x, y, r)
            x_sc, y_sc, r_sc = repair_config(x_sc, y_sc, r_sc)
            x_sc, y_sc, r_sc = maximize_radii(x_sc, y_sc, r_sc)
            
            if validity_penalty(x_sc, y_sc, r_sc) == 0:
                score_sc = np.sum(r_sc)
                if score_sc > best_score:
                    best_score = score_sc
                    best_config = np.column_stack((x_sc, y_sc, r_sc))
        
        if validity_penalty(x, y, r) == 0:
            score = np.sum(r)
            if score > best_score:
                best_score = score
                best_config = np.column_stack((x, y, r))
        
        if time.time() > deadline - 25:
            break

    # Final polish on best config
    if best_config is not None and time.time() < deadline - 10:
        x, y, r = best_config[:, 0], best_config[:, 1], best_config[:, 2]
        x, y, r = local_search(x, y, r, max_iter=300)
        x, y, r = repair_config(x, y, r)
        x, y, r = maximize_radii(x, y, r)
        if validity_penalty(x, y, r) == 0:
            score = np.sum(r)
            if score > best_score:
                best_score = score
                best_config = np.column_stack((x, y, r))

    if best_config is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_config
