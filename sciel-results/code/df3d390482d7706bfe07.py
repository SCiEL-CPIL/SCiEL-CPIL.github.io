"""Pack 50 equal circles in a unit square, maximize common radius."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 148.0  # seconds, safety margin before 150s kill

def _radius(centers):
    """Compute the max common radius for given centers (exact, for scoring)."""
    # Wall radius = min of (0.5 - |coord|)
    wall_r = np.min(0.5 - np.abs(centers))
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    return min(wall_r, min_pair / 2.0)

def _soft_min_radius(centers, beta=100.0):
    """
    Compute a smooth approximation of the radius using log-sum-exp.
    r = min(wall_r, min_pair/2)
    """
    # Wall constraints
    c1 = 0.5 - centers[:, 0]
    c2 = 0.5 + centers[:, 0]
    c3 = 0.5 - centers[:, 1]
    c4 = 0.5 + centers[:, 1]
    
    # Pairwise constraints
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    # Take upper triangle
    iu = np.triu_indices(N, 1)
    pair_dists = dist[iu] / 2.0
    
    # Combine all constraints
    all_constraints = np.concatenate([c1, c2, c3, c4, pair_dists])
    
    y = -beta * all_constraints
    max_y = np.max(y)
    sum_exp = np.sum(np.exp(y - max_y))
    log_sum_exp = max_y + np.log(sum_exp)
    
    return -log_sum_exp / beta

def _objective_smooth(x, beta=100.0):
    """Objective to minimize: -soft_min_radius."""
    centers = x.reshape(N, 2)
    r = _soft_min_radius(centers, beta)
    return -r

def _make_grid_rect(n_x, n_y):
    """Create a rectangular grid."""
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

def _expand_from_start(centers, max_time=10.0):
    """
    Start with a configuration, calculate its current radius, then try to expand it
    by iteratively increasing r and resolving collisions.
    """
    start_time = time.time()
    centers = centers.copy()
    r = _radius(centers)
    
    # If r is already small, we can grow it.
    # We'll try to increase r in steps.
    # Step size: adaptive.
    
    while time.time() - start_time < max_time:
        # Try to increase r by a small amount
        # If we can resolve collisions at r + delta, we accept it.
        # Otherwise, reduce delta and try again.
        
        delta = 0.001
        success = False
        for _ in range(5):
            r_try = r + delta
            # Check if feasible
            # We need a solver to find feasible positions for r_try.
            # Let's use a simple projected gradient or just check if current positions work.
            # If current positions work for r_try, great.
            # If not, we need to move them.
            
            # Simple check:
            # If _radius(centers) >= r_try, then yes.
            if _radius(centers) >= r_try:
                r = r_try
                success = True
                break
            else:
                # We need to move centers to make r_try feasible.
                # This is the hard part. 
                # Let's use a few iterations of L-BFGS-B to minimize violation at r_try.
                # If it converges to near 0, we accept.
                
                def _obj_violation(x, r_fix):
                    c = x.reshape(N, 2)
                    v_walls = np.maximum(0, np.abs(c) + r_fix - 0.5)
                    v_wall_sum = np.sum(v_walls**2)
                    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
                    dist = np.sqrt(np.sum(diff**2, axis=2))
                    np.fill_diagonal(dist, np.inf)
                    v_pairs = np.maximum(0, 2*r_fix - dist)
                    v_pair_sum = np.sum(v_pairs**2)
                    return v_wall_sum + v_pair_sum
                
                try:
                    res = minimize(
                        _obj_violation, 
                        centers.ravel(), 
                        args=(r_try,), 
                        method='L-BFGS-B',
                        options={'maxiter': 50, 'disp': False}
                    )
                    if res.fun < 1e-6:
                        centers = res.x.reshape(N, 2)
                        r = r_try
                        success = True
                        break
                    else:
                        # Failed to find feasible config for r_try
                        # Maybe try a smaller delta
                        delta *= 0.5
                except Exception:
                    break
        
        if not success:
            # Can't increase r anymore with this delta strategy?
            # Or maybe we should try a larger delta? 
            # If we fail at small delta, we are stuck.
            # Let's break and return current r.
            # Actually, if we fail at 0.001, we might be in a local minimum of the feasible set.
            # But for "expansion", we usually just grow until stuck.
            break
            
    return centers, r

def _local_refine_smooth(centers, time_limit=5.0, beta=100.0):
    """
    Use L-BFGS-B to refine the configuration using the smooth objective.
    """
    x0 = centers.ravel()
    
    start_time = time.time()
    
    def callback(xk):
        if time.time() - start_time > time_limit:
            raise RuntimeError("Time limit exceeded")
        return False

    try:
        res = minimize(
            _objective_smooth, 
            x0, 
            args=(beta,),
            method='L-BFGS-B',
            options={
                'maxiter': 1000,
                'maxfun': 5000,
                'ftol': 1e-10,
                'gtol': 1e-8,
                'disp': False
            },
            callback=callback
        )
        if res.x is not None:
            return res.x.reshape(N, 2)
    except RuntimeError:
        pass
    except Exception:
        pass
    
    return centers

def _force_directed_relax(centers, r, max_iter=200, time_limit=5.0):
    """
    Force-directed relaxation: treat circles as particles with repulsive forces
    when overlapping, and attractive forces towards the center of the square.
    This quickly separates overlapping circles and pushes them into a reasonable spread.
    """
    start_time = time.time()
    centers = centers.copy()
    
    # Initial forces
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        forces = np.zeros_like(centers)
        
        # Pairwise repulsive forces
        for i in range(N):
            for j in range(i + 1, N):
                diff = centers[i] - centers[j]
                dist = np.linalg.norm(diff)
                if dist < 2 * r:
                    # Repulsive force proportional to overlap
                    overlap = 2 * r - dist
                    if dist > 1e-10:
                        force_dir = diff / dist
                    else:
                        force_dir = np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1)])
                        force_dir /= np.linalg.norm(force_dir)
                    force_mag = overlap * 10.0
                    forces[i] += force_mag * force_dir
                    forces[j] -= force_mag * force_dir
        
        # Wall repulsive forces
        for i in range(N):
            # Left wall
            if centers[i, 0] < -0.5 + r:
                overlap = (-0.5 + r) - centers[i, 0]
                forces[i, 0] += overlap * 10.0
            # Right wall
            if centers[i, 0] > 0.5 - r:
                overlap = centers[i, 0] - (0.5 - r)
                forces[i, 0] -= overlap * 10.0
            # Bottom wall
            if centers[i, 1] < -0.5 + r:
                overlap = (-0.5 + r) - centers[i, 1]
                forces[i, 1] += overlap * 10.0
            # Top wall
            if centers[i, 1] > 0.5 - r:
                overlap = centers[i, 1] - (0.5 - r)
                forces[i, 1] -= overlap * 10.0
        
        # Update positions
        learning_rate = 0.01 / (1 + it * 0.01)
        centers += learning_rate * forces
        
        # Clip to square
        centers = np.clip(centers, -0.5 + r, 0.5 - r)
    
    return centers

def _simulated_annealing(centers, r, max_iter=500, time_limit=10.0):
    """
    Simulated annealing to escape local minima.
    """
    start_time = time.time()
    centers = centers.copy()
    current_r = _radius(centers)
    best_r = current_r
    best_centers = centers.copy()
    
    T = 0.01  # Initial temperature
    T_min = 0.0001
    cooling_rate = 0.99
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Propose a random move
        idx = np.random.randint(N)
        delta = np.random.normal(0, T * 0.1, size=2)
        new_centers = centers.copy()
        new_centers[idx] += delta
        
        # Clip to square
        new_centers = np.clip(new_centers, -0.5 + r, 0.5 - r)
        
        new_r = _radius(new_centers)
        
        # Accept or reject
        if new_r >= current_r:
            centers = new_centers
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_centers = centers.copy()
        else:
            # Accept with probability exp((new_r - current_r) / T)
            prob = np.exp((new_r - current_r) / T)
            if np.random.rand() < prob:
                centers = new_centers
                current_r = new_r
        
        T *= cooling_rate
        if T < T_min:
            T = T_min
    
    return best_centers, best_r

def construct():
    global DEADLINE
    start_time = time.time()
    
    best_centers = None
    best_r = 0.0
    
    # Generate initial configurations
    inits = []
    
    # 1. Rectangular Grid 7x8
    c1 = _make_grid_rect(7, 8)
    inits.append(c1)
    
    # 2. Rectangular Grid 8x7
    c2 = _make_grid_rect(8, 7)
    inits.append(c2)
    
    # 3. Hexagonal Grids
    for sp in [0.14, 0.15, 0.16]:
        c_hex = _make_hexagonal_grid(10, 10, sp, offset_x=-0.5, offset_y=-0.5)
        c_hex = c_hex - np.mean(c_hex, axis=0)
        max_ext = np.max(np.abs(c_hex))
        if max_ext > 0.45:
            c_hex = c_hex * (0.45 / max_ext)
        inits.append(c_hex)
        
    # 4. Random
    np.random.seed(42)
    c_rand = np.random.uniform(-0.45, 0.45, size=(N, 2))
    inits.append(c_rand)
    
    # Strategy:
    # 1. For each init, perform force-directed relaxation to get a better spread.
    # 2. Then use simulated annealing to escape local minima.
    # 3. Finally, polish with L-BFGS-B soft-min to maximize r.
    
    for i, init in enumerate(inits):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        time_left = DEADLINE - 10 - (time.time() - start_time)
        
        # Phase 1: Force-directed relaxation
        # Estimate initial radius
        r_est = _radius(init)
        
        # Relax with a slightly larger target radius
        r_target = min(r_est * 1.1, 0.08)
        
        alloc_fd = min(3.0, time_left * 0.2)
        if alloc_fd > 0.5:
            relaxed_centers = _force_directed_relax(init, r_target, time_limit=alloc_fd)
            r_relaxed = _radius(relaxed_centers)
            
            if r_relaxed > best_r:
                best_r = r_relaxed
                best_centers = relaxed_centers.copy()
        
        # Phase 2: Simulated annealing
        if time.time() - start_time < DEADLINE - 5:
            alloc_sa = min(5.0, (DEADLINE - 5) - (time.time() - start_time))
            if alloc_sa > 1.0:
                sa_centers, r_sa = _simulated_annealing(relaxed_centers, r_target, time_limit=alloc_sa)
                r_sa = _radius(sa_centers)
                
                if r_sa > best_r:
                    best_r = r_sa
                    best_centers = sa_centers.copy()
        
        # Phase 3: Polish with L-BFGS-B
        if time.time() - start_time < DEADLINE - 2:
            alloc_polish = min(5.0, (DEADLINE - 2) - (time.time() - start_time))
            if alloc_polish > 1.0:
                polished = _local_refine_smooth(best_centers, time_limit=alloc_polish, beta=100.0)
                r_pol = _radius(polished)
                
                if r_pol > best_r:
                    best_r = r_pol
                    best_centers = polished.copy()
        
        # If time remains, do more random restarts with the full pipeline
        while time.time() - start_time < DEADLINE - 5:
            seed = int(time.time() * 1000) % 100000
            np.random.seed(seed)
            c_rand = np.random.uniform(-0.45, 0.45, size=(N, 2))
            
            time_left = DEADLINE - 5 - (time.time() - start_time)
            if time_left < 1: break
            
            # Short force-directed relaxation
            r_est = _radius(c_rand)
            r_target = min(r_est * 1.1, 0.08)
            alloc_fd = min(2.0, time_left * 0.2)
            if alloc_fd > 0.5:
                relaxed_centers = _force_directed_relax(c_rand, r_target, time_limit=alloc_fd)
                r_relaxed = _radius(relaxed_centers)
                
                if r_relaxed > best_r:
                    best_r = r_relaxed
                    best_centers = relaxed_centers.copy()
            
            # Short simulated annealing
            if time.time() - start_time < DEADLINE - 3:
                alloc_sa = min(3.0, (DEADLINE - 3) - (time.time() - start_time))
                if alloc_sa > 0.5:
                    sa_centers, r_sa = _simulated_annealing(relaxed_centers, r_target, time_limit=alloc_sa)
                    r_sa = _radius(sa_centers)
                    
                    if r_sa > best_r:
                        best_r = r_sa
                        best_centers = sa_centers.copy()
            
            # Short polish
            if time.time() - start_time < DEADLINE - 1:
                alloc_polish = min(3.0, (DEADLINE - 1) - (time.time() - start_time))
                if alloc_polish > 0.5:
                    polished = _local_refine_smooth(best_centers, time_limit=alloc_polish, beta=100.0)
                    r_pol = _radius(polished)
                    
                    if r_pol > best_r:
                        best_r = r_pol
                        best_centers = polished.copy()

    return best_centers
