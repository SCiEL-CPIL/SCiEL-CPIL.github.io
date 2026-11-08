"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit
    
    best_sum_r = 0.0
    best_circles = None
    
    # Vectorized objective function with smooth penalty
    def objective(circles):
        """
        circles: (N, 3) array of (x, y, r)
        Returns: -sum(r) + penalty (minimize this)
        """
        x = circles[:, 0]
        y = circles[:, 1]
        r = circles[:, 2]
        
        # Wall constraints: r <= x, r <= 1-x, r <= y, r <= 1-y
        v1 = np.maximum(0, r - x)
        v2 = np.maximum(0, r - (1 - x))
        v3 = np.maximum(0, r - y)
        v4 = np.maximum(0, r - (1 - y))
        
        wall_penalty = np.sum(v1**2) + np.sum(v2**2) + np.sum(v3**2) + np.sum(v4**2)
        
        # Overlap constraints using smooth approximation
        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        dist_sq = dx**2 + dy**2
        
        r_sum = r[:, np.newaxis] + r[np.newaxis, :]
        r_sum_sq = r_sum**2
        
        i_idx, j_idx = np.triu_indices(N, k=1)
        dist_sq_pairs = dist_sq[i_idx, j_idx]
        r_sum_sq_pairs = r_sum_sq[i_idx, j_idx]
        
        # Smooth violation: use sqrt for better conditioning
        violations = np.maximum(0, r_sum_sq_pairs - dist_sq_pairs)
        overlap_penalty = np.sum(violations**2)
        
        penalty = 1000.0 * (wall_penalty + overlap_penalty)
        
        return -np.sum(r) + penalty
    
    # Vectorized gradient (numerical or analytical)
    # For now, we'll use a simple coordinate descent with local search
    
    def local_search(circles, max_iter=5000, lr=0.01):
        """
        Adaptive coordinate descent with Metropolis acceptance.
        """
        x = circles[:, 0].copy()
        y = circles[:, 1].copy()
        r = circles[:, 2].copy()
        
        current_obj = objective(np.column_stack([x, y, r]))
        best_obj = current_obj
        best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
        
        rng = np.random.RandomState(42)
        current_lr = lr
        no_improve = 0
        max_no_improve = 500
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Adaptive learning rate
            if no_improve > max_no_improve:
                current_lr *= 0.9
                no_improve = 0
            
            # Pick a random circle to update
            i = rng.randint(N)
            
            # Multiple perturbations per iteration for better exploration
            best_new_obj = current_obj
            best_new_x, best_new_y, best_new_r = x[i], y[i], r[i]
            
            for attempt in range(3):
                dx = rng.randn() * current_lr
                dy = rng.randn() * current_lr
                dr = rng.randn() * current_lr * 0.3
                
                new_x = np.clip(x[i] + dx, 0, 1)
                new_y = np.clip(y[i] + dy, 0, 1)
                new_r = np.clip(r[i] + dr, 0.001, 0.5)
                
                new_circles = np.column_stack([x, y, r]).copy()
                new_circles[i] = [new_x, new_y, new_r]
                
                new_obj = objective(new_circles)
                
                if new_obj < best_new_obj:
                    best_new_obj = new_obj
                    best_new_x, best_new_y, best_new_r = new_x, new_y, new_r
            
            # Metropolis acceptance
            delta = best_new_obj - current_obj
            t = max(0.001, 0.01 * (0.001 / 0.01) ** (it / max_iter))
            
            if delta < 0 or rng.rand() < np.exp(-delta / t):
                x[i] = best_new_x
                y[i] = best_new_y
                r[i] = best_new_r
                current_obj = best_new_obj
                
                if new_obj < best_obj:
                    best_obj = new_obj
                    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1
        
        return np.column_stack([best_x, best_y, best_r]), best_obj
    
    # Multiple initializations
    initializations = []
    
    # 1. Grid-based initialization
    side = 5
    for jitter_seed in range(3):
        rng_init = np.random.RandomState(jitter_seed)
        x = (np.arange(N) % side + 0.5 + rng_init.randn(N) * 0.1) / side
        y = (np.arange(N) // side + 0.5 + rng_init.randn(N) * 0.1) / side
        r = np.full(N, 0.5 / side * 0.8)
        x = np.clip(x, 0.02, 0.98)
        y = np.clip(y, 0.02, 0.98)
        initializations.append(np.column_stack([x, y, r]))
    
    # 2. Random initialization
    for rand_seed in range(5):
        rng_init = np.random.RandomState(100 + rand_seed)
        x = rng_init.rand(N) * 0.8 + 0.1
        y = rng_init.rand(N) * 0.8 + 0.1
        r = rng_init.rand(N) * 0.05 + 0.03
        initializations.append(np.column_stack([x, y, r]))
    
    # 3. Spiral initialization
    for spiral_seed in range(2):
        angles = 2 * np.pi * np.arange(N) / N * (1.618 + 0.1 * spiral_seed)
        x = 0.5 + 0.4 * np.cos(angles)
        y = 0.5 + 0.4 * np.sin(angles)
        r = np.full(N, 0.08)
        x = np.clip(x, 0.02, 0.98)
        y = np.clip(y, 0.02, 0.98)
        initializations.append(np.column_stack([x, y, r]))
    
    # Run local search on each initialization
    for init_circles in initializations:
        if time.time() > deadline:
            break
        
        # Use more iterations for earlier initializations
        remaining_time = deadline - time.time()
        max_iter = min(20000, int(remaining_time * 50))
        
        result_circles, result_obj = local_search(init_circles, max_iter=max_iter, lr=0.015)
        
        # Extract sum of radii
        sum_r = np.sum(result_circles[:, 2])
        
        if sum_r > best_sum_r:
            best_sum_r = sum_r
            best_circles = result_circles
    
    # Final validation and cleanup
    if best_circles is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        count = 0
        for i in range(side):
            for j in range(side):
                if count < N:
                    cells.append([(i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 0.01])
                    count += 1
        best_circles = np.array(cells)
    
    # Ensure validity: fix any remaining violations
    x, y, r = best_circles[:, 0].copy(), best_circles[:, 1].copy(), best_circles[:, 2].copy()
    
    # Fix wall violations
    for i in range(N):
        r[i] = min(r[i], x[i], 1 - x[i], y[i], 1 - y[i])
        r[i] = max(r[i], 1e-6)
    
    # Fix overlaps by shrinking radii
    for iteration in range(200):
        fixed = False
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    overlap = min_dist - dist
                    shrink = overlap / 2 + 0.0005
                    r[i] = max(1e-6, r[i] - shrink)
                    r[j] = max(1e-6, r[j] - shrink)
                    fixed = True
        if not fixed:
            break
    
    best_circles = np.column_stack([x, y, r])
    
    # High-precision polish using scipy L-BFGS-B if time permits
    if time.time() < deadline - 10:
        try:
            from scipy.optimize import minimize
            
            def smooth_obj(params):
                """Smooth objective for scipy optimizer."""
                x_p = params[:N]
                y_p = params[N:2*N]
                r_p = params[2*N:3*N]
                
                # Wall violations
                v1 = np.maximum(0, r_p - x_p)
                v2 = np.maximum(0, r_p - (1 - x_p))
                v3 = np.maximum(0, r_p - y_p)
                v4 = np.maximum(0, r_p - (1 - y_p))
                wall_pen = np.sum(v1**2) + np.sum(v2**2) + np.sum(v3**2) + np.sum(v4**2)
                
                # Overlap violations
                dx = x_p[:, np.newaxis] - x_p[np.newaxis, :]
                dy = y_p[:, np.newaxis] - y_p[np.newaxis, :]
                dist_sq = dx**2 + dy**2
                r_sum = r_p[:, np.newaxis] + r_p[np.newaxis, :]
                
                i_idx, j_idx = np.triu_indices(N, k=1)
                viol = np.maximum(0, r_sum[i_idx, j_idx]**2 - dist_sq[i_idx, j_idx])
                overlap_pen = np.sum(viol**2)
                
                return -np.sum(r_p) + 500.0 * (wall_pen + overlap_pen)
            
            # Initial parameters from best solution
            x0 = best_circles[:, 0].copy()
            y0 = best_circles[:, 1].copy()
            r0 = best_circles[:, 2].copy()
            params0 = np.concatenate([x0, y0, r0])
            
            # Bounds
            bounds = []
            for i in range(N):
                bounds.append((0.001, 1.0))  # x
            for i in range(N):
                bounds.append((0.001, 1.0))  # y
            for i in range(N):
                bounds.append((0.001, 0.5))  # r
            
            # Run L-BFGS-B
            result = minimize(
                smooth_obj,
                params0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.success and result.fun < objective(best_circles):
                polished_x = result.x[:N]
                polished_y = result.x[N:2*N]
                polished_r = result.x[2*N:3*N]
                
                # Validate and fix
                for i in range(N):
                    polished_r[i] = min(polished_r[i], polished_x[i], 1 - polished_x[i], 
                                        polished_y[i], 1 - polished_y[i])
                    polished_r[i] = max(polished_r[i], 1e-6)
                
                # Fix overlaps
                for iteration in range(50):
                    fixed = False
                    for i in range(N):
                        for j in range(i + 1, N):
                            dx = polished_x[i] - polished_x[j]
                            dy = polished_y[i] - polished_y[j]
                            dist = np.sqrt(dx*dx + dy*dy)
                            min_dist = polished_r[i] + polished_r[j]
                            if dist < min_dist:
                                overlap = min_dist - dist
                                shrink = overlap / 2 + 0.0001
                                polished_r[i] = max(1e-6, polished_r[i] - shrink)
                                polished_r[j] = max(1e-6, polished_r[j] - shrink)
                                fixed = True
                    if not fixed:
                        break
                
                polished_circles = np.column_stack([polished_x, polished_y, polished_r])
                polished_sum = np.sum(polished_r)
                
                if polished_sum > best_sum_r:
                    best_circles = polished_circles
                    best_sum_r = polished_sum
        
        except Exception:
            pass
    
    return best_circles
# EVOLVE-BLOCK-END
