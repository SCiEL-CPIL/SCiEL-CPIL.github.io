import numpy as np
from scipy.optimize import minimize
import time

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    best_score = 0.0
    best_config = None
    
    # Initial simple grid as a fallback/baseline
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    base_config = np.array(cells[:N])
    best_score = np.sum(base_config[:, 2])
    best_config = base_config.copy()
    
    # Helper to check validity and score
    def get_score(config):
        x, y, r = config[:, 0], config[:, 1], config[:, 2]
        # Check boundaries
        if np.any(x < r) or np.any(x > 1 - r) or np.any(y < r) or np.any(y > 1 - r):
            return -np.inf
        # Check overlaps
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < r[i] + r[j] - 1e-9:
                    return -np.inf
        return np.sum(r)

    def repair_config(config):
        """
        Ensure config is strictly valid by shrinking radii and clamping.
        """
        x = config[:, 0].copy()
        y = config[:, 1].copy()
        r = config[:, 2].copy()
        
        # Clamp to boundaries
        for i in range(N):
            r[i] = min(r[i], x[i], 1 - x[i], y[i], 1 - y[i])
            r[i] = max(r[i], 1e-6)
        
        # Iteratively resolve overlaps
        max_iter = 100
        for _ in range(max_iter):
            changed = False
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = r[i] + r[j]
                    if dist < min_dist:
                        # Shrink radii to fit
                        # We want r_i + r_j <= dist
                        # Let's shrink both proportionally to their current sizes
                        # Or simply set sum to dist * 0.999
                        target_sum = dist * 0.999
                        if target_sum <= 0:
                            target_sum = 1e-6
                        
                        # Distribute target_sum between i and j
                        # Keep ratio of radii? Or just split?
                        # Splitting equally might be bad if one is much larger.
                        # Let's shrink the larger one more?
                        # Simplest: shrink both by half the overlap
                        overlap = min_dist - dist
                        shrink_i = overlap / 2.0
                        shrink_j = overlap / 2.0
                        
                        r[i] -= shrink_i
                        r[j] -= shrink_j
                        
                        # Re-clamp to boundaries
                        r[i] = min(r[i], x[i], 1 - x[i], y[i], 1 - y[i])
                        r[j] = min(r[j], x[j], 1 - x[j], y[j], 1 - y[j])
                        
                        r[i] = max(r[i], 1e-6)
                        r[j] = max(r[j], 1e-6)
                        changed = True
            
            if not changed:
                break
                
        return np.column_stack((x, y, r))

    def run_sa(x0, y0, r0, steps=2000):
        """
        Simulated Annealing to optimize positions and radii.
        """
        x = x0.copy()
        y = y0.copy()
        r = r0.copy()
        
        T = 0.1
        alpha = 0.995
        step_size = 0.05
        
        def energy(x, y, r):
            # We want to maximize sum(r), so minimize -sum(r)
            # Plus penalties for overlaps and boundaries
            e = -np.sum(r)
            
            # Boundary penalties
            for i in range(N):
                if x[i] < r[i]:
                    e += 100 * (r[i] - x[i])**2
                if x[i] > 1 - r[i]:
                    e += 100 * (x[i] - (1 - r[i]))**2
                if y[i] < r[i]:
                    e += 100 * (r[i] - y[i])**2
                if y[i] > 1 - r[i]:
                    e += 100 * (y[i] - (1 - r[i]))**2
            
            # Overlap penalties
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = r[i] + r[j]
                    if dist < min_dist:
                        overlap = min_dist - dist
                        e += 100 * overlap * overlap
            
            return e

        current_energy = energy(x, y, r)
        
        for step in range(steps):
            if time.time() > deadline - 2.0:
                break
                
            # Pick a random circle to perturb
            i = np.random.randint(N)
            
            # Perturb position and radius
            dx = np.random.normal(0, step_size)
            dy = np.random.normal(0, step_size)
            dr = np.random.normal(0, step_size * 0.5)
            
            new_x = x[i] + dx
            new_y = y[i] + dy
            new_r = r[i] + dr
            
            # Basic bounds for new state to avoid extreme bad moves
            if new_r < 1e-4 or new_r > 0.5:
                continue
            if new_x < 0 or new_x > 1 or new_y < 0 or new_y > 1:
                continue

            # Calculate new energy
            # To speed up, we only need to recalculate energy terms involving i
            # But for simplicity and N=26, full recalc is fast enough (~325 pairs)
            old_x, old_y, old_r = x[i], y[i], r[i]
            x[i], y[i], r[i] = new_x, new_y, new_r
            new_energy = energy(x, y, r)
            
            delta_e = new_energy - current_energy
            
            if delta_e < 0 or np.random.random() < np.exp(-delta_e / T):
                current_energy = new_energy
            else:
                x[i], y[i], r[i] = old_x, old_y, old_r
            
            T *= alpha
            # Occasionally increase step size if stuck? No, standard SA cools.
            
        return np.column_stack((x, y, r))

    def polish_bfgs(config):
        """
        Use BFGS with penalty method to refine the configuration.
        Much faster than SLSQP for this problem size.
        """
        x0 = config[:, 0]
        y0 = config[:, 1]
        r0 = config[:, 2]
        
        # Variables: [x1, y1, r1, x2, y2, r2, ...]
        vars_ = np.concatenate([x0, y0, r0])
        
        def objective_and_grad(v):
            x = v[0::3]
            y = v[1::3]
            r = v[2::3]
            
            # Objective: -sum(r)
            obj = -np.sum(r)
            grad = np.zeros_like(v)
            
            # Gradient of -sum(r)
            grad[2::3] = -1.0
            
            # Penalty weight
            lam = 500.0
            
            # Boundary penalties
            for i in range(N):
                # x constraint: r_i - x_i > 0
                val_x = r[i] - x[i]
                if val_x > 0:
                    obj += lam * val_x * val_x
                    grad[i*3] += -2.0 * lam * val_x
                    grad[i*3+2] += 2.0 * lam * val_x
                
                # 1 - x_i >= r_i -> x_i + r_i - 1 > 0
                val_1x = x[i] + r[i] - 1.0
                if val_1x > 0:
                    obj += lam * val_1x * val_1x
                    grad[i*3] += 2.0 * lam * val_1x
                    grad[i*3+2] += 2.0 * lam * val_1x
                
                # y constraint: r_i - y_i > 0
                val_y = r[i] - y[i]
                if val_y > 0:
                    obj += lam * val_y * val_y
                    grad[i*3+1] += -2.0 * lam * val_y
                    grad[i*3+2] += 2.0 * lam * val_y
                
                # 1 - y_i >= r_i -> y_i + r_i - 1 > 0
                val_1y = y[i] + r[i] - 1.0
                if val_1y > 0:
                    obj += lam * val_1y * val_1y
                    grad[i*3+1] += 2.0 * lam * val_1y
                    grad[i*3+2] += 2.0 * lam * val_1y
            
            # Overlap penalties
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist_sq = dx*dx + dy*dy
                    dist = np.sqrt(dist_sq)
                    min_dist = r[i] + r[j]
                    
                    if dist > 1e-9:
                        overlap = min_dist - dist
                        if overlap > 0:
                            obj += lam * overlap * overlap
                            factor = 2.0 * lam * overlap / dist
                            grad[i*3] += -factor * dx
                            grad[i*3+1] += -factor * dy
                            grad[i*3+2] += 2.0 * lam * overlap
                            grad[j*3] += factor * dx
                            grad[j*3+1] += factor * dy
                            grad[j*3+2] += 2.0 * lam * overlap
                    else:
                        obj += lam * 1e-6
            
            return obj, grad
        
        bounds = [(1e-6, 1.0), (1e-6, 1.0), (1e-6, 0.5)] * N
        
        res = minimize(objective_and_grad, vars_, jac=True, method='BFGS', bounds=bounds, options={'maxiter': 100, 'gtol': 1e-5})
        
        if res.success:
            v_opt = res.x
            x_opt = v_opt[0::3]
            y_opt = v_opt[1::3]
            r_opt = v_opt[2::3]
            return np.column_stack((x_opt, y_opt, r_opt))
        else:
            return config

    # Main loop
    while time.time() < deadline - 5.0:
        # Random initialization
        x = np.random.uniform(0.1, 0.9, N)
        y = np.random.uniform(0.1, 0.9, N)
        r = np.random.uniform(0.02, 0.1, N)
        
        # Ensure initial config is somewhat valid to avoid huge penalties
        # Just run SA which will handle it
        
        config = np.column_stack((x, y, r))
        
        # Run SA
        sa_config = run_sa(x, y, r, steps=500)
        
        # Polish
        polished_config = polish_bfgs(sa_config)
        
        # Repair to ensure strict validity
        valid_config = repair_config(polished_config)
        
        score = get_score(valid_config)
        
        if score > best_score:
            best_score = score
            best_config = valid_config.copy()
            
        # Check time
        if time.time() > deadline - 10.0:
            break

    return best_config
