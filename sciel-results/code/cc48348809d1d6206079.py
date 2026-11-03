import numpy as np
import time
from scipy.optimize import minimize

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    def check_validity(x, y, r, tol=1e-9):
        """Check if circles are valid (non-overlapping, within bounds)."""
        if np.any(x < r - tol) or np.any(x > 1 - r + tol):
            return False
        if np.any(y < r - tol) or np.any(y > 1 - r + tol):
            return False
        diff_x = x[:, None] - x[None, :]
        diff_y = y[:, None] - y[None, :]
        dist = np.sqrt(diff_x**2 + diff_y**2)
        min_dist = r[:, None] + r[None, :]
        np.fill_diagonal(min_dist, 0)
        if np.any(dist < min_dist - tol):
            return False
        return True
    
    def make_initial_guess():
        """Create a grid-based initial guess with small radii."""
        side = int(np.ceil(np.sqrt(N)))
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        for i in range(N):
            row = i // side
            col = i % side
            x[i] = (col + 0.5) / side
            y[i] = (row + 0.5) / side
            r[i] = 0.5 / side * 0.85
        return x, y, r
    
    def sa_step(x, y, r, temp, rng):
        """One simulated annealing step: perturb and accept/reject."""
        x_new = x.copy()
        y_new = y.copy()
        r_new = r.copy()
        
        # Perturb random subset of circles
        n_perturb = max(1, N // 4)
        idx = rng.choice(N, size=n_perturb, replace=False)
        
        # Move positions
        step_size = 0.02 * temp
        x_new[idx] += rng.normal(0, step_size, size=len(idx))
        y_new[idx] += rng.normal(0, step_size, size=len(idx))
        
        # Adjust radii slightly
        r_change = rng.normal(0, 0.01 * temp, size=len(idx))
        r_new[idx] = np.maximum(r_new[idx] + r_change, 0.02)
        
        # Clamp to bounds
        x_new = np.clip(x_new, r_new, 1 - r_new)
        y_new = np.clip(y_new, r_new, 1 - r_new)
        
        # Evaluate energy: sum of overlaps + boundary violations - sum of radii
        def energy(xv, yv, rv):
            e = 0.0
            # Boundary penalty
            e += np.sum(np.maximum(0, rv - xv)**2)
            e += np.sum(np.maximum(0, xv - (1 - rv))**2)
            e += np.sum(np.maximum(0, rv - yv)**2)
            e += np.sum(np.maximum(0, yv - (1 - rv))**2)
            # Overlap penalty
            diff_x = xv[:, None] - xv[None, :]
            diff_y = yv[:, None] - yv[None, :]
            dist = np.sqrt(diff_x**2 + diff_y**2)
            min_dist = rv[:, None] + rv[None, :]
            np.fill_diagonal(min_dist, 0)
            viol = np.maximum(0, min_dist - dist)
            e += 10.0 * np.sum(viol**2)
            # Reward for larger radii
            e -= np.sum(rv)
            return e
        
        e_old = energy(x, y, r)
        e_new = energy(x_new, y_new, r_new)
        
        # Metropolis acceptance
        if e_new < e_old or rng.random() < np.exp(-(e_new - e_old) / max(temp, 1e-10)):
            return x_new, y_new, r_new
        return x, y, r
    
    def gradient_polish(x, y, r, max_iter=50):
        """Polish using gradient-based optimization."""
        def objective(params):
            xv = params[0::3]
            yv = params[1::3]
            rv = params[2::3]
            
            # Soft constraints with smooth penalty
            pen = 0.0
            # Boundary
            pen += np.sum(np.maximum(0, rv - xv)**2)
            pen += np.sum(np.maximum(0, xv - (1 - rv))**2)
            pen += np.sum(np.maximum(0, rv - yv)**2)
            pen += np.sum(np.maximum(0, yv - (1 - rv))**2)
            # Overlap
            diff_x = xv[:, None] - xv[None, :]
            diff_y = yv[:, None] - yv[None, :]
            dist = np.sqrt(diff_x**2 + diff_y**2)
            min_dist = rv[:, None] + rv[None, :]
            np.fill_diagonal(min_dist, 0)
            viol = np.maximum(0, min_dist - dist)
            pen += 50.0 * np.sum(viol**2)
            
            return -np.sum(rv) + pen
        
        params = np.column_stack((x, y, r)).flatten()
        res = minimize(objective, params, method='L-BFGS-B', options={'maxiter': max_iter, 'ftol': 1e-10})
        
        xv = res.x[0::3]
        yv = res.x[1::3]
        rv = res.x[2::3]
        
        # Clamp
        rv = np.maximum(rv, 0.02)
        xv = np.clip(xv, rv, 1 - rv)
        yv = np.clip(yv, rv, 1 - rv)
        
        return xv, yv, rv
    
    rng = np.random.RandomState(42)
    
    # Phase 1: Simulated Annealing
    x, y, r = make_initial_guess()
    temp = 1.0
    cooling_rate = 0.995
    
    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
    best_score = -np.inf
    
    sa_steps = 0
    max_sa_steps = 5000
    
    while time.time() < deadline - 30 and sa_steps < max_sa_steps:
        x, y, r = sa_step(x, y, r, temp, rng)
        sa_steps += 1
        
        # Track best valid solution
        if check_validity(x, y, r):
            score = np.sum(r)
            if score > best_score:
                best_score = score
                best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
        
        # Cool down
        temp *= cooling_rate
    
    # Phase 2: Gradient polish from best SA result
    if best_score > -np.inf:
        x, y, r = gradient_polish(best_x, best_y, best_r, max_iter=100)
        if check_validity(x, y, r):
            score = np.sum(r)
            if score > best_score:
                best_score = score
                best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
    
    # Phase 3: Multiple restarts with gradient polish
    remaining_time = deadline - time.time()
    if remaining_time > 20:
        n_restarts = int(min(5, remaining_time / 15))
        for restart in range(n_restarts):
            if time.time() > deadline - 15:
                break
            
            # New random initial guess
            x_init = rng.uniform(0.1, 0.9, N)
            y_init = rng.uniform(0.1, 0.9, N)
            r_init = rng.uniform(0.03, 0.08, N)
            
            # Quick SA
            temp = 0.5
            for _ in range(500):
                if time.time() > deadline - 15:
                    break
                x_s, y_s, r_s = sa_step(x_init, y_init, r_init, temp, rng)
                x_init, y_init, r_init = x_s, y_s, r_s
                temp *= 0.998
            
            # Gradient polish
            x_p, y_p, r_p = gradient_polish(x_init, y_init, r_init, max_iter=50)
            if check_validity(x_p, y_p, r_p):
                score = np.sum(r_p)
                if score > best_score:
                    best_score = score
                    best_x, best_y, best_r = x_p.copy(), y_p.copy(), r_p.copy()
    
    # If no valid result, return safe grid
    if best_score <= -np.inf:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i // side + 0.5) / side, (i % side + 0.5) / side, 0.5 / side - 1e-3) for i in range(N)]
        return np.array(cells)
    
    # Final repair: shrink radii if any overlap remains
    x, y, r = best_x, best_y, best_r
    for _ in range(20):
        diff_x = x[:, None] - x[None, :]
        diff_y = y[:, None] - y[None, :]
        dist = np.sqrt(diff_x**2 + diff_y**2)
        min_dist = r[:, None] + r[None, :]
        np.fill_diagonal(min_dist, 0)
        viol = np.maximum(0, min_dist - dist)
        np.fill_diagonal(viol, 0)
        if np.max(viol) < 1e-9:
            break
        idx_i, idx_j = np.unravel_index(np.argmax(viol), viol.shape)
        reduction = viol[idx_i, idx_j] / 2 + 1e-6
        r[idx_i] = max(r[idx_i] - reduction, 1e-4)
        r[idx_j] = max(r[idx_j] - reduction, 1e-4)
    
    # Final clamp
    r = np.maximum(r, 1e-4)
    x = np.clip(x, r, 1 - r)
    y = np.clip(y, r, 1 - r)
    
    return np.column_stack((x, y, r))
