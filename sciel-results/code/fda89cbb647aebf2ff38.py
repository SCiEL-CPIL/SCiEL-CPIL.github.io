import numpy as np
import time
from scipy.optimize import minimize

N = 50

# Precompute upper triangle indices for pairwise distances
_iu, _ju = np.triu_indices(N, k=1)
_PAIR_IDX = np.column_stack([_iu, _ju])
_NUM_PAIRS = len(_iu)

def _compute_radius(c):
    """
    Compute the exact maximum radius for a given configuration c.
    """
    if len(c) == 0:
        return 0.0
    diff = c[_iu] - c[_ju]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    min_pair = np.min(dists)
    wall_dists = 0.5 - np.abs(c)
    min_wall = np.min(wall_dists)
    return min(min_pair / 2.0, min_wall)

def _softmin_radius_and_grad(c, t):
    """
    Compute the softmin of all pair distances (divided by 2) and wall distances,
    along with the gradient with respect to c.
    
    softmin(x) = -t * log(sum(exp(-x_i / t)))
    """
    diff = c[_iu] - c[_ju]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    pair_vals = dists / 2.0
    wall_vals = 0.5 - np.abs(c)
    
    # Combine pair and wall values for softmin
    # We want to minimize -softmin, which is equivalent to maximizing softmin.
    # The objective function for minimize will be -softmin_val.
    
    # To avoid overflow/underflow, we can use log-sum-exp trick if needed, 
    # but for t > 0.001 and values in [0, 0.5], exp is safe.
    
    # We treat pair distances and wall distances as a single set of constraints for the softmin.
    # However, the gradients are different.
    # Let S = sum(exp(-pair_vals/t)) + sum(exp(-wall_vals/t))
    # softmin = -t * log(S)
    
    pair_weights = np.exp(-pair_vals / t)
    wall_weights = np.exp(-wall_vals / t)
    
    total_sum = np.sum(pair_weights) + np.sum(wall_weights)
    
    if total_sum < 1e-300:
        return 0.0, np.zeros_like(c)
        
    softmin_val = -t * np.log(total_sum)
    
    grad_c = np.zeros_like(c)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        unit_vecs = diff / dists[:, np.newaxis]
    unit_vecs[np.isnan(unit_vecs)] = 0.0
    
    # Gradient of softmin: 
    # d/dx_i (-t * log(sum exp(-f_j/t))) = sum (exp(-f_j/t) / sum) * (1/t) * (-t) * df_j/dx_i
    # = sum (w_j) * (-df_j/dx_i)
    
    # Pair contribution to grad:
    # f_pair = dist/2. df/dx_i = 0.5 * (x_i - x_j)/dist
    # -df/dx_i = -0.5 * (x_i - x_j)/dist
    pair_grad = 0.5 * pair_weights[:, np.newaxis] * unit_vecs
    
    np.add.at(grad_c, _iu, -pair_grad)
    np.add.at(grad_c, _ju, pair_grad)
    
    # Wall contribution to grad:
    # f_wall = 0.5 - |x|. df/dx_i = -sign(x_i)
    # -df/dx_i = sign(x_i)
    signs = np.sign(c)
    grad_c += wall_weights * signs
    
    # Normalize by total_sum
    grad_c /= total_sum
    
    return softmin_val, grad_c

def _polish_softmin(c_init, deadline, t_start=0.1, t_end=0.001, iters_per_temp=50):
    """
    Polish a configuration using L-BFGS-B on the softmin objective.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    temps = np.logspace(np.log10(t_start), np.log10(t_end), 15)
    
    for t in temps:
        if time.time() > deadline:
            break
            
        def obj(x):
            r, _ = _softmin_radius_and_grad(x.reshape(N, 3), t)
            return -r
        
        def grad(x):
            _, g = _softmin_radius_and_grad(x.reshape(N, 3), t)
            return g.ravel()
        
        x0 = c.flatten()
        bounds = [(-0.5, 0.5)] * (N * 3)
        try:
            res = minimize(obj, x0, jac=grad, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            c_new = res.x.reshape(N, 3)
            c_new = np.clip(c_new, -0.499999, 0.499999)
            r_new = _compute_radius(c_new)
            if r_new > best_r:
                best_r = r_new
                best_c = c_new.copy()
                c = c_new.copy()
        except Exception:
            continue
            
    return best_c, best_r

def _pso_search(deadline, n_particles=20, max_iters=100):
    """
    Particle Swarm Optimization to find good configurations.
    """
    # Initialize particles
    positions = np.zeros((n_particles, N, 3))
    velocities = np.zeros((n_particles, N, 3))
    personal_bests = np.zeros((n_particles, N, 3))
    personal_best_scores = np.zeros(n_particles)
    
    # Initialize with diverse seeds
    for i in range(n_particles):
        if i == 0:
            positions[i] = _seed_fcc()
        elif i == 1:
            positions[i] = _seed_bcc()
        elif i == 2:
            positions[i] = _seed_grid()
        else:
            positions[i] = np.random.uniform(-0.45, 0.45, size=(N, 3))
        
        positions[i] = np.clip(positions[i], -0.499999, 0.499999)
        personal_bests[i] = positions[i].copy()
        personal_best_scores[i] = _compute_radius(positions[i])
        
    global_best_pos = personal_bests[np.argmax(personal_best_scores)].copy()
    global_best_score = np.max(personal_best_scores)
    
    w = 0.5  # Inertia weight
    c1 = 1.5 # Cognitive coefficient
    c2 = 1.5 # Social coefficient
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
            
        for i in range(n_particles):
            # Update velocity
            r1 = np.random.random((N, 3))
            r2 = np.random.random((N, 3))
            
            velocities[i] = w * velocities[i] + \
                           c1 * r1 * (personal_bests[i] - positions[i]) + \
                           c2 * r2 * (global_best_pos - positions[i])
            
            # Update position
            positions[i] += velocities[i]
            positions[i] = np.clip(positions[i], -0.499999, 0.499999)
            
            # Evaluate fitness
            score = _compute_radius(positions[i])
            
            if score > personal_best_scores[i]:
                personal_best_scores[i] = score
                personal_bests[i] = positions[i].copy()
                
                if score > global_best_score:
                    global_best_score = score
                    global_best_pos = personal_bests[i].copy()
                    
    return global_best_pos, global_best_score

def _seed_fcc():
    m = 4
    spacing = 1.0 / m
    points = []
    for i in range(m):
        for j in range(m):
            for k in range(m):
                x = (i + 0.5) * spacing - 0.5
                y = (j + 0.5) * spacing - 0.5
                z = (k + 0.5) * spacing - 0.5
                for dx, dy, dz in [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]:
                    px = x + dx * spacing
                    py = y + dy * spacing
                    pz = z + dz * spacing
                    if -0.5 <= px <= 0.5 and -0.5 <= py <= 0.5 and -0.5 <= pz <= 0.5:
                        points.append([px, py, pz])
    points = np.array(points)
    if len(points) < N:
        extra = np.random.uniform(-0.45, 0.45, size=(N - len(points), 3))
        points = np.vstack([points, extra])
    else:
        idx = np.random.choice(len(points), N, replace=False)
        points = points[idx]
    return points

def _seed_bcc():
    m = 3
    spacing = 1.0 / m
    points = []
    for i in range(m):
        for j in range(m):
            for k in range(m):
                x = (i + 0.5) * spacing - 0.5
                y = (j + 0.5) * spacing - 0.5
                z = (k + 0.5) * spacing - 0.5
                points.append([x, y, z])
                points.append([x + 0.5*spacing, y + 0.5*spacing, z + 0.5*spacing])
    points = np.array(points)
    if len(points) < N:
        extra = np.random.uniform(-0.45, 0.45, size=(N - len(points), 3))
        points = np.vstack([points, extra])
    else:
        idx = np.random.choice(len(points), N, replace=False)
        points = points[idx]
    return points

def _seed_grid():
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    idx = np.random.choice(len(grid), N, replace=False)
    return grid[idx]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # 1. Global Search with PSO
    pso_deadline = start_time + 60.0
    best_c, best_r = _pso_search(pso_deadline, n_particles=30, max_iters=100)
    
    # 2. Local Polish on the best PSO result
    polish_deadline = start_time + 120.0
    best_c, best_r = _polish_softmin(best_c, polish_deadline, t_start=0.05, t_end=0.0005)
    
    # 3. Try a few random restarts with polishing if time permits
    if time.time() < deadline - 20.0:
        for _ in range(3):
            if time.time() > deadline - 10.0:
                break
            c_rand = np.random.uniform(-0.45, 0.45, size=(N, 3))
            c_polished, r_polished = _polish_softmin(c_rand, deadline - 5.0, t_start=0.1, t_end=0.001)
            if r_polished > best_r:
                best_r = r_polished
                best_c = c_polished.copy()
                
    return best_c
