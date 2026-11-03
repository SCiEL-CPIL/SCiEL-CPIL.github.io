"""
Pack 50 equal circles in a unit square using Simulated Annealing and L-BFGS-B polishing.
Strategy:
1. Fast radius evaluation using spatial hashing (O(N) avg).
2. Simulated Annealing to explore the configuration space globally.
3. L-BFGS-B with a smooth log-sum-exp objective to polish local optima.
4. Multi-start with diverse initializations.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 135.0  # Safety margin before 150s kill

def _radius(centers):
    """
    Compute the max common radius for given centers.
    Uses spatial hashing for speed.
    """
    # Wall radius
    wall_r = np.min(0.5 - np.abs(centers))
    
    # Spatial hash for pairwise distances
    # Cell size should be roughly the current min distance or a fraction of the square
    # For safety and simplicity in the "hard" radius check, we can use a grid.
    # However, for 50 points, even O(N^2) is ~2500 ops, which is fast.
    # But we want speed for SA. Let's try O(N^2) first, it might be fast enough in numpy.
    # Actually, let's stick to vectorized O(N^2) for correctness in the "hard" check,
    # and use a faster approximation or the same for SA if needed.
    # 50 points -> 1225 pairs. Numpy handles this in microseconds.
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    
    return min(wall_r, min_pair / 2.0)

def _soft_min_radius(centers, temperature=0.01):
    """
    Differentiable approximation of the radius using log-sum-exp.
    r_soft = -1/beta * log( sum(exp(-beta * d_ij)) + exp(-beta * w_i) )
    Here we want to maximize the min distance.
    Let's define the objective as minimizing the "soft max" of the inverse distances?
    Or simpler: minimize -soft_min.
    
    Actually, a common trick is to minimize the sum of squared overlaps or use a soft-min.
    soft_min(x) = -1/beta * log(sum(exp(-beta * x_i)))
    As beta -> inf, soft_min -> min(x_i).
    
    We want to maximize min(d_ij, w_i).
    So we minimize -soft_min(d_ij, w_i).
    
    To avoid underflow/overflow, we shift by the max.
    """
    # Wall distances
    w = 0.5 - np.abs(centers)
    # Flatten wall distances for each coordinate? No, min over all coords and points.
    # w_i = min(0.5 - |x_i|, 0.5 - |y_i|)
    w_min = np.min(np.column_stack((0.5 - np.abs(centers[:, 0]), 0.5 - np.abs(centers[:, 1]))), axis=1)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    # Combine all distances
    # We want the min of all pairwise dists and all wall dists.
    # Note: radius is min(min_pair_dist / 2, min_wall_dist).
    # Let's define "effective distances" as 2*wall_dist and pair_dist.
    # Then radius = min(2*wall, pair) / 2.
    
    vals = np.concatenate([2.0 * w_min, dist[~np.eye(N, dtype=bool).ravel()]])
    
    # Soft min
    # To be stable:
    max_val = np.max(vals)
    exp_vals = np.exp(-(vals - max_val) / temperature)
    log_sum_exp = max_val + temperature * np.log(np.sum(exp_vals))
    soft_min_val = -log_sum_exp
    
    # We want to maximize this, so minimize -soft_min_val
    return -soft_min_val

def _objective_smooth(x, temperature=0.01):
    centers = x.reshape(N, 2)
    return _soft_min_radius(centers, temperature)

def _make_grid_rect(n_x, n_y):
    """Create a rectangular grid."""
    # Ensure n_x * n_y >= N
    # We will take the first N points
    ticks_x = (np.arange(n_x) + 0.5) / n_x - 0.5
    ticks_y = (np.arange(n_y) + 0.5) / n_y - 0.5
    grid = np.array([(x, y) for y in ticks_y for x in ticks_x])
    return grid[:N].copy()

def _make_hexagonal_grid(n_x, n_y, spacing, offset_x=0, offset_y=0):
    """Create a hexagonal lattice of points."""
    points = []
    dy = spacing * np.sqrt(3) / 2
    for j in range(n_y):
        for i in range(n_x):
            x = i * spacing + offset_x
            y = j * dy + offset_y
            if j % 2 != 0:
                x += spacing / 2
            points.append((x, y))
    points = np.array(points)
    return points[:N]

def _local_refine_lbfgsb(centers, time_limit=5.0, temp_start=0.05, temp_end=0.005):
    """
    Use L-BFGS-B to refine the configuration with a smooth objective.
    """
    x0 = centers.ravel()
    start_time = time.time()
    
    def callback(xk):
        if time.time() - start_time > time_limit:
            raise RuntimeError("Time limit exceeded")
        return False

    try:
        # Anneal the temperature
        best_x = x0
        best_f = _objective_smooth(x0, temp_start)
        
        # Simple schedule: linear or exponential decrease
        # We'll do a few passes with decreasing temperature
        temps = [0.05, 0.02, 0.01, 0.005, 0.002]
        
        for t in temps:
            if time.time() - start_time > time_limit:
                break
            
            def obj_func(x, t=t):
                return _objective_smooth(x, t)
            
            def grad_func(x, t=t):
                # Numerical gradient is slow, but L-BFGS-B can approximate it.
                # However, providing analytic gradient is better.
                # For now, let's rely on L-BFGS-B's finite difference or just use it.
                # Actually, implementing analytic gradient for soft-min is complex.
                # L-BFGS-B with finite difference on 100 vars is okay for short runs.
                # But it's slow.
                # Alternative: Use Nelder-Mead for the final polish?
                # Or just use L-BFGS-B and accept the FD cost.
                return None # Let scipy use FD
                
            res = minimize(
                obj_func, 
                x0, 
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * 2 * N,
                options={
                    'maxiter': 200,
                    'ftol': 1e-8,
                    'gtol': 1e-6,
                    'disp': False
                },
                callback=callback
            )
            
            if res.x is not None:
                x0 = res.x
                f_val = _objective_smooth(x0, t)
                if f_val < best_f:
                    best_f = f_val
                    best_x = x0.copy()
                    
    except RuntimeError:
        pass
    except Exception:
        pass
    
    return best_x.reshape(N, 2)

def _simulated_annealing(centers, time_limit=10.0, T_start=0.05, T_end=0.001, alpha=0.999):
    """
    Simulated Annealing for global exploration.
    """
    x = centers.ravel().copy()
    start_time = time.time()
    best_x = x.copy()
    best_r = _radius(x.reshape(N, 2))
    T = T_start
    
    step_size = 0.05
    
    while time.time() - start_time < time_limit:
        # Propose new position
        # Move a random subset of points
        num_move = np.random.randint(1, 5)
        indices = np.random.choice(N, num_move, replace=False)
        
        x_new = x.copy()
        for idx in indices:
            # Random jitter
            x_new[idx*2] += np.random.uniform(-step_size, step_size)
            x_new[idx*2+1] += np.random.uniform(-step_size, step_size)
        
        # Clip to bounds (softly or hard? Hard is better for feasibility)
        x_new[0::2] = np.clip(x_new[0::2], -0.5, 0.5)
        x_new[1::2] = np.clip(x_new[1::2], -0.5, 0.5)
        
        r_old = _radius(x.reshape(N, 2))
        r_new = _radius(x_new.reshape(N, 2))
        
        # Acceptance probability
        if r_new > r_old:
            x = x_new
            if r_new > best_r:
                best_r = r_new
                best_x = x_new.copy()
        else:
            delta = r_new - r_old
            if delta < 0:
                prob = np.exp(delta / T)
                if np.random.random() < prob:
                    x = x_new
        
        # Cool down
        T *= alpha
        if T < T_end:
            T = T_end
            # If we've cooled enough, maybe restart with a slightly higher T?
            # Or just stop.
            
        # Adapt step size?
        # If we've been stuck, reduce step size.
        # Simple heuristic: if best_r hasn't improved in last 100 steps, reduce step.
        # For simplicity, we'll just let it run.
        
    return best_x.reshape(N, 2)

def construct():
    global DEADLINE
    start_time = time.time()
    
    best_centers = None
    best_r = 0.0
    
    # Generate initial configurations
    inits = []
    
    # 1. Rectangular Grids
    c1 = _make_grid_rect(7, 8)
    inits.append(c1)
    
    c2 = _make_grid_rect(8, 7)
    inits.append(c2)
    
    # 2. Hexagonal Grids
    for sp in [0.14, 0.15, 0.16]:
        c_hex = _make_hexagonal_grid(10, 10, sp, offset_x=-0.5, offset_y=-0.5)
        c_hex = c_hex - np.mean(c_hex, axis=0)
        max_ext = np.max(np.abs(c_hex))
        if max_ext > 0.45:
            c_hex = c_hex * (0.45 / max_ext)
        inits.append(c_hex)
        
    # 3. Random
    np.random.seed(42)
    c_rand = np.random.uniform(-0.45, 0.45, size=(N, 2))
    inits.append(c_rand)
    
    # Phase 1: Simulated Annealing on diverse starts
    # SA is good for escaping local minima.
    time_per_sa = 15.0
    for i, init in enumerate(inits):
        if time.time() - start_time > DEADLINE - 30:
            break
        
        time_left = DEADLINE - 30 - (time.time() - start_time)
        alloc_time = min(time_per_sa, time_left / max(1, len(inits) - i))
        
        sa_refined = _simulated_annealing(init, time_limit=alloc_time)
        r = _radius(sa_refined)
        
        if r > best_r:
            best_r = r
            best_centers = sa_refined.copy()
            
    # Phase 2: L-BFGS-B Polishing on the best candidates from Phase 1
    # We can re-run SA to get more candidates, or just polish the best one found so far.
    # Better: Polish the top K candidates from SA.
    # Since we only kept the best, let's just polish the current best.
    # And also try a few new random restarts with SA+Polish.
    
    candidates = [best_centers] if best_centers is not None else []
    
    # Add a few more diverse candidates
    for _ in range(5):
        if time.time() - start_time > DEADLINE - 20:
            break
        np.random.seed(int(time.time() * 1000) % 100000)
        c_rand = np.random.uniform(-0.45, 0.45, size=(N, 2))
        # Quick SA
        sa_refined = _simulated_annealing(c_rand, time_limit=5.0)
        candidates.append(sa_refined)
        
    # Polish candidates
    time_per_polish = 10.0
    for i, cand in enumerate(candidates):
        if time.time() - start_time > DEADLINE - 5:
            break
        
        time_left = DEADLINE - 5 - (time.time() - start_time)
        alloc_time = min(time_per_polish, time_left / max(1, len(candidates) - i))
        
        polished = _local_refine_lbfgsb(cand, time_limit=alloc_time)
        r = _radius(polished)
        
        if r > best_r:
            best_r = r
            best_centers = polished.copy()

    return best_centers
