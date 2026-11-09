import numpy as np
import time
from scipy.optimize import minimize

N = 26
_DEADLINE = 145.0  # Safety margin from 150s limit

def is_valid(x, y, r, tol=1e-6):
    """
    Checks if the configuration is valid:
    1. All radii are non-negative.
    2. All circles are inside the unit square.
    3. No circles overlap.
    """
    if np.any(r < -tol):
        return False
    if np.any(x < r - tol) or np.any(x > 1 - r + tol):
        return False
    if np.any(y < r - tol) or np.any(y > 1 - r + tol):
        return False
    
    # Check pairwise distances
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    min_dist = r[:, None] + r[None, :]
    
    # Check for overlaps with tolerance
    if np.any(dist < min_dist - tol):
        return False
    return True

def compute_max_radii(x, y):
    """
    Computes the maximum possible radius for each circle given the centers,
    such that no circles overlap and all are inside the unit square.
    r_i = min( min_j (d_ij / 2), dist_to_boundary_i )
    """
    n = len(x)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    
    # Distance to boundary for each circle
    b = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
    
    # Minimum distance to other circles / 2
    # Set diagonal to inf so it doesn't affect min
    np.fill_diagonal(dist, np.inf)
    min_dist_to_other = np.min(dist, axis=1) / 2.0
    
    r = np.minimum(b, min_dist_to_other)
    # Ensure radii are non-negative (should be due to b >= 0 if x,y in [0,1])
    r = np.maximum(r, 0)
    return r

def simulated_annealing_fast(x, y, start_time, deadline, T0=0.05, Tend=0.0001, n_steps=10000):
    """
    Fast Simulated Annealing using exact max-radius calculation.
    """
    best_score = 0.0
    best_config = None
    
    cur_x = x.copy()
    cur_y = y.copy()
    cur_r = compute_max_radii(cur_x, cur_y)
    cur_score = np.sum(cur_r)
    
    if cur_score > best_score:
        best_score = cur_score
        best_config = (cur_x.copy(), cur_y.copy(), cur_r.copy())
    
    cool_factor = (Tend / T0) ** (1.0 / n_steps)
    
    for step in range(n_steps):
        if time.time() > deadline - 10:
            break
            
        T = T0 * (cool_factor ** step)
        
        # Generate neighbor: move 1-3 random circles
        n_move = np.random.randint(1, 4)
        idxs = np.random.choice(N, n_move, replace=False)
        
        new_x = cur_x.copy()
        new_y = cur_y.copy()
        
        # Step size can be adaptive, but fixed small step is often robust
        step_size = 0.05 * (T / T0 + 0.1) # Larger steps at high T
        
        for i in idxs:
            new_x[i] += np.random.normal(0, step_size)
            new_y[i] += np.random.normal(0, step_size)
            new_x[i] = np.clip(new_x[i], 0.0, 1.0)
            new_y[i] = np.clip(new_y[i], 0.0, 1.0)
            
        new_r = compute_max_radii(new_x, new_y)
        new_score = np.sum(new_r)
        
        delta_E = cur_score - new_score # We maximize score, so delta E = E_new - E_old = -new_score - (-cur_score)
        
        if delta_E < 0 or np.random.rand() < np.exp(-delta_E / max(T, 1e-9)):
            cur_x = new_x
            cur_y = new_y
            cur_r = new_r
            cur_score = new_score
            
            if cur_score > best_score:
                best_score = cur_score
                best_config = (cur_x.copy(), cur_y.copy(), cur_r.copy())
                
    return best_score, best_config

def polish_l_bfgsb(x, y, r, deadline):
    """
    Polishes the configuration using L-BFGS-B with a penalty method.
    """
    if time.time() > deadline - 5:
        return np.sum(r), (x, y, r)
        
    best_score = np.sum(r)
    best_config = (x, y, r)
    
    z0 = np.concatenate([x, y, r])
    bounds = [(0, 1)] * N + [(0, 1)] * N + [(0, 0.5)] * N
    
    def objective_and_grad(z, penalty_weight):
        N_local = len(z) // 3
        x_c = z[:N_local]
        y_c = z[N_local:2*N_local]
        r_c = z[2*N_local:]
        
        obj = -np.sum(r_c)
        grad_x = np.zeros(N_local)
        grad_y = np.zeros(N_local)
        grad_r = -np.ones(N_local)
        
        # Overlap Penalty
        dx = x_c[:, None] - x_c[None, :]
        dy = y_c[:, None] - y_c[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        dist_safe = np.where(dist > 1e-6, dist, 1e-6)
        
        min_dist = r_c[:, None] + r_c[None, :]
        overlap = np.maximum(0, min_dist - dist)
        np.fill_diagonal(overlap, 0)
        
        obj += penalty_weight * np.sum(overlap**2)
        
        ux = dx / dist_safe
        uy = dy / dist_safe
        
        grad_x += -2.0 * penalty_weight * np.sum(overlap * ux, axis=1)
        grad_y += -2.0 * penalty_weight * np.sum(overlap * uy, axis=1)
        grad_r += 2.0 * penalty_weight * np.sum(overlap, axis=1)
        
        # Boundary Penalty
        vx1 = np.maximum(0, r_c - x_c)
        vx2 = np.maximum(0, r_c - (1 - x_c))
        vy1 = np.maximum(0, r_c - y_c)
        vy2 = np.maximum(0, r_c - (1 - y_c))
        
        bound_viol = vx1**2 + vx2**2 + vy1**2 + vy2**2
        obj += penalty_weight * np.sum(bound_viol)
        
        grad_x += penalty_weight * (-2.0 * vx1 + 2.0 * vx2)
        grad_y += penalty_weight * (-2.0 * vy1 + 2.0 * vy2)
        grad_r += penalty_weight * (2.0 * vx1 + 2.0 * vx2 + 2.0 * vy1 + 2.0 * vy2)
        
        grad = np.concatenate([grad_x, grad_y, grad_r])
        return obj, grad

    try:
        res = minimize(
            objective_and_grad,
            z0,
            args=(1000.0,), # High penalty weight for polishing
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
        )
        
        z_final = res.x
        x_f = z_final[:N]
        y_f = z_final[N:2*N]
        r_f = z_final[2*N:]
        
        if is_valid(x_f, y_f, r_f):
            score = np.sum(r_f)
            if score > best_score:
                best_score = score
                best_config = (x_f, y_f, r_f)
                
    except Exception:
        pass
        
    return best_score, best_config

def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    best_score = 0.0
    best_config = None
    
    # Multi-start Simulated Annealing
    n_starts = 10
    time_per_start = (_DEADLINE - 20) / n_starts # Leave 20s for polishing and overhead
    
    for i in range(n_starts):
        if time.time() > deadline - 25:
            break
            
        # Initial centers
        if i == 0:
            # Grid initialization
            side = int(np.ceil(np.sqrt(N)))
            xs = []
            ys = []
            for r in range(side):
                for c in range(side):
                    if len(xs) < N:
                        xs.append((c + 0.5) / side)
                        ys.append((r + 0.5) / side)
            x0 = np.array(xs[:N])
            y0 = np.array(ys[:N])
        else:
            # Random initialization with some structure
            x0 = np.random.uniform(0.1, 0.9, N)
            y0 = np.random.uniform(0.1, 0.9, N)
            
        sa_deadline = start_time + (i + 1) * time_per_start
        score, config = simulated_annealing_fast(x0, y0, start_time, sa_deadline)
        
        if config is not None:
            if score > best_score:
                best_score = score
                best_config = config
                
    # Polish the best result
    if best_config is not None:
        bx, by, br = best_config
        # Ensure br is the max possible radius for these centers to start polishing from a "tight" state
        br = compute_max_radii(bx, by)
        polish_score, polish_config = polish_l_bfgsb(bx, by, br, deadline)
        
        if polish_config is not None:
            if polish_score > best_score:
                best_score = polish_score
                best_config = polish_config
                
    # Final fallback
    if best_config is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    cells.append(((j + 0.5) / side, (i + 0.5) / side, 0.5 / side - 1e-3))
        return np.array(cells[:N])
        
    bx, by, br = best_config
    # Final validity check and correction
    if not is_valid(bx, by, br):
        # Shrink radii slightly if invalid
        br = br * 0.99
        if not is_valid(bx, by, br):
            br = br * 0.95
        if not is_valid(bx, by, br):
             # Fallback to safe grid
             side = int(np.ceil(np.sqrt(N)))
             cells = []
             for i in range(side):
                 for j in range(side):
                     if len(cells) < N:
                         cells.append(((j + 0.5) / side, (i + 0.5) / side, 0.5 / side - 1e-3))
             return np.array(cells[:N])
             
    return np.column_stack((bx, by, br))
