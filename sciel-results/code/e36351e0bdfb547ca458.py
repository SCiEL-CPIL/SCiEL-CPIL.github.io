import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, cdist

N = 100

def _compute_radius(centers):
    """Compute the largest common radius for given centers."""
    if len(centers) < 2:
        return float(np.min(0.5 - np.abs(centers)))
    dists = pdist(centers)
    min_pair_dist = np.min(dists)
    dist_to_walls = 0.5 - np.abs(centers)
    min_wall_dist = np.min(dist_to_walls)
    return float(min(min_pair_dist / 2.0, min_wall_dist))

def _riesz_energy_and_grad(centers, s=6.0, wall_weight=1.0):
    """
    Compute Riesz s-energy and its gradient.
    Energy = sum_{i<j} |c_i - c_j|^(-s) + wall_weight * sum_i sum_k |0.5 - |c_ik||^(-s)
    
    For s > 3, this strongly penalizes close pairs and wall proximity.
    Minimizing this energy maximizes the minimum distance.
    """
    n = len(centers)
    energy = 0.0
    grad = np.zeros_like(centers)
    
    # Pairwise repulsion
    for i in range(n):
        for j in range(i + 1, n):
            d_vec = centers[i] - centers[j]
            d2 = np.dot(d_vec, d_vec)
            if d2 < 1e-20:
                d2 = 1e-20
            d = np.sqrt(d2)
            # E_ij = d^(-s)
            energy += d ** (-s)
            # dE_ij/d(centers[i]) = -s * d^(-s-2) * d_vec
            coeff = -s * d ** (-s - 2)
            grad[i] += coeff * d_vec
            grad[j] -= coeff * d_vec
    
    # Wall repulsion: for each coordinate, penalize proximity to walls
    for i in range(n):
        for k in range(3):
            c = centers[i, k]
            # Distance to nearest wall in this coordinate
            d_wall = 0.5 - abs(c)
            if d_wall < 1e-10:
                d_wall = 1e-10
            # E_wall = d_wall^(-s)
            energy += wall_weight * d_wall ** (-s)
            # dE_wall/d(centers[i,k]) = -s * d_wall^(-s-1) * sign(c)
            sign_c = np.sign(c)
            if sign_c == 0:
                sign_c = 1.0
            grad[i, k] += wall_weight * (-s) * d_wall ** (-s - 1) * sign_c
    
    return energy, grad

def _minimize_energy(centers_init, s=6.0, time_limit=10.0, wall_weight=1.0):
    """Minimize Riesz energy using L-BFGS-B."""
    centers_init = np.clip(centers_init, -0.499, 0.499)
    x0 = centers_init.flatten()
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    def obj(x):
        c = x.reshape(N, 3)
        e, _ = _riesz_energy_and_grad(c, s=s, wall_weight=wall_weight)
        return e
    
    def grad(x):
        c = x.reshape(N, 3)
        _, g = _riesz_energy_and_grad(c, s=s, wall_weight=wall_weight)
        return g.flatten()
    
    start_t = time.time()
    stop_flag = [False]
    
    def callback(xk):
        if time.time() > start_t + time_limit:
            stop_flag[0] = True
        return stop_flag[0]
    
    try:
        res = minimize(
            obj, x0, jac=grad, method='L-BFGS-B',
            bounds=bounds, options={
                'maxiter': 2000,
                'ftol': 1e-14,
                'gtol': 1e-10,
                'maxls': 100,
                'callback': callback,
                'disp': False
            }
        )
        best_x = res.x
    except Exception:
        best_x = x0
    
    best_centers = best_x.reshape(N, 3)
    best_centers = np.clip(best_centers, -0.5, 0.5)
    return best_centers

def _simulated_annealing(centers_init, time_limit=20.0, s=6.0, wall_weight=1.0):
    """
    Simulated annealing to escape local minima.
    Uses Riesz energy as the objective.
    """
    centers = np.clip(centers_init, -0.499, 0.499).copy()
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    current_energy, _ = _riesz_energy_and_grad(centers, s=s, wall_weight=wall_weight)
    best_energy = current_energy
    
    start_t = time.time()
    T = 0.05  # Initial temperature
    T_min = 0.001
    cooling_rate = 0.999
    step_scale = 0.02
    
    rng = np.random.RandomState(42)
    
    while time.time() < start_t + time_limit:
        if T < T_min:
            break
        
        # Generate random perturbation
        n_moves = rng.randint(1, 5)
        new_centers = centers.copy()
        for _ in range(n_moves):
            idx = rng.randint(N)
            new_centers[idx] += rng.normal(0, step_scale, 3)
        new_centers = np.clip(new_centers, -0.499, 0.499)
        
        new_energy, _ = _riesz_energy_and_grad(new_centers, s=s, wall_weight=wall_weight)
        delta_e = new_energy - current_energy
        
        # Metropolis criterion
        if delta_e < 0 or rng.random() < np.exp(-delta_e / T):
            centers = new_centers
            current_energy = new_energy
            
            r = _compute_radius(centers)
            if r > best_radius:
                best_radius = r
                best_centers = centers.copy()
                best_energy = current_energy
        
        T *= cooling_rate
    
    return best_centers, best_radius

def _generate_initial_positions(seed=42):
    """Generate diverse initial configurations."""
    rng = np.random.RandomState(seed)
    inits = []
    
    # 5x5x4 grid
    m1, m2, m3 = 5, 5, 4
    t1 = (np.arange(m1) + 0.5) / m1 - 0.5
    t2 = (np.arange(m2) + 0.5) / m2 - 0.5
    t3 = (np.arange(m3) + 0.5) / m3 - 0.5
    grid = np.array([(x, y, z) for z in t3 for y in t2 for x in t1])
    inits.append(grid[:N].copy())
    
    # 4x5x5 grid
    m1, m2, m3 = 4, 5, 5
    t1 = (np.arange(m1) + 0.5) / m1 - 0.5
    t2 = (np.arange(m2) + 0.5) / m2 - 0.5
    t3 = (np.arange(m3) + 0.5) / m3 - 0.5
    grid = np.array([(x, y, z) for z in t3 for y in t2 for x in t1])
    inits.append(grid[:N].copy())
    
    # Random uniform
    inits.append(rng.uniform(-0.45, 0.45, (N, 3)))
    
    # Jittered grid
    base = inits[0].copy()
    jitter = rng.normal(0, 0.015, (N, 3))
    inits.append(np.clip(base + jitter, -0.499, 0.499))
    
    # Another random with different seed
    rng2 = np.random.RandomState(123)
    inits.append(rng2.uniform(-0.4, 0.4, (N, 3)))
    
    return inits

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin for 150s limit
    
    best_radius = 0.0
    best_centers = None
    
    # Generate initial candidates
    inits = _generate_initial_positions()
    
    # Phase 1: Optimize each initial configuration with Riesz energy
    for i, init_centers in enumerate(inits):
        if time.time() > deadline - 50.0:
            break
        
        time_left = deadline - time.time() - 30.0
        if time_left < 5.0:
            break
        
        opt_centers = _minimize_energy(init_centers, s=6.0, time_limit=time_left * 0.7, wall_weight=1.0)
        r = _compute_radius(opt_centers)
        
        if r > best_radius:
            best_radius = r
            best_centers = opt_centers.copy()
    
    if best_centers is None:
        best_centers = inits[0]
        best_radius = _compute_radius(best_centers)
    
    # Phase 2: Simulated annealing from best configuration
    if time.time() < deadline - 15.0:
        sa_time = min(25.0, deadline - time.time() - 5.0)
        sa_centers, sa_radius = _simulated_annealing(
            best_centers, time_limit=sa_time, s=6.0, wall_weight=1.0
        )
        if sa_radius > best_radius:
            best_radius = sa_radius
            best_centers = sa_centers
    
    # Phase 3: Final polish with L-BFGS-B
    if time.time() < deadline - 3.0 and best_centers is not None:
        polish_time = min(10.0, deadline - time.time() - 2.0)
        polished = _minimize_energy(best_centers, s=8.0, time_limit=polish_time, wall_weight=2.0)
        r_final = _compute_radius(polished)
        if r_final > best_radius:
            best_radius = r_final
            best_centers = polished
    
    if best_centers is None:
        # Fallback: simple grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centers = grid[:N]
    
    return best_centers
