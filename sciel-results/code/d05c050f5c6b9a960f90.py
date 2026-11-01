import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 100

def _compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(centers > 0.5) or np.any(centers < -0.5):
        return 0.0
    wall_dists = 0.5 - np.abs(centers)
    r_wall = np.min(wall_dists)
    tree = cKDTree(centers)
    dists, _ = tree.query(centers, k=2)
    r_pair = np.min(dists[:, 1]) / 2.0
    return min(r_wall, r_pair)

def _soft_min_objective(x, alpha=100.0):
    """
    Objective: minimize -softmin(2*r_wall, 2*r_pair)
    where softmin uses a smooth approximation.
    x is flattened centers (N*3,), reshaped to (N,3)
    """
    centers = x.reshape(N, 3)
    if not np.all(np.isfinite(centers)):
        return 1e10
    if np.any(centers > 0.5) or np.any(centers < -0.5):
        return 1e10
    
    # Wall distances
    wall_dists = 0.5 - np.abs(centers)
    r_wall = np.min(wall_dists)
    
    # Pairwise distances using KDTree
    tree = cKDTree(centers)
    dists, _ = tree.query(centers, k=2)
    r_pair = np.min(dists[:, 1]) / 2.0
    
    # Soft min: -softmin(a, b) = log(exp(-alpha*a) + exp(-alpha*b)) / alpha
    # This is a smooth approximation to min(a, b)
    a = 2.0 * r_wall
    b = 2.0 * r_pair
    
    # Numerically stable softmin
    m = np.minimum(a, b)
    exp_a = np.exp(-alpha * (a - m))
    exp_b = np.exp(-alpha * (b - m))
    soft_min_val = m + np.log(exp_a + exp_b) / alpha
    
    return -soft_min_val

def _soft_min_gradient(x, alpha=100.0):
    """
    Gradient of the soft-min objective via finite differences for reliability.
    """
    centers = x.reshape(N, 3)
    if not np.all(np.isfinite(centers)):
        return np.zeros_like(x)
    
    eps = 1e-7
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += eps
        x_minus[i] -= eps
        f_plus = _soft_min_objective(x_plus, alpha)
        f_minus = _soft_min_objective(x_minus, alpha)
        grad[i] = (f_plus - f_minus) / (2 * eps)
    
    return grad

def _build_initial_grid():
    """Build a regular grid initialization."""
    nx, ny, nz = 5, 5, 4
    xs = np.linspace(-0.42, 0.42, nx)
    ys = np.linspace(-0.42, 0.42, ny)
    zs = np.linspace(-0.42, 0.42, nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    centers = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    return centers

def _build_bcc_init():
    """Build a body-centered cubic initialization."""
    a = 0.65
    points = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
                points.append([i*a, j*a, k*a])
                points.append([i*a + a/2, j*a + a/2, k*a + a/2])
    
    arr = np.array(points)
    mask = np.all(np.abs(arr) <= 0.45, axis=1)
    arr = arr[mask]
    if len(arr) >= N:
        idx = np.random.choice(len(arr), N, replace=False)
        return arr[idx]
    else:
        return _build_initial_grid()

def _build_halton_init():
    """Build a Halton sequence initialization."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r
    
    centers = []
    for i in range(1, N + 1):
        x = halton(i, 2)
        y = halton(i, 3)
        z = halton(i, 5)
        # Map to [-0.45, 0.45]
        centers.append([x * 0.9 - 0.45, y * 0.9 - 0.45, z * 0.9 - 0.45])
    
    return np.array(centers)

def _polish_l_bfgs(centers, time_limit, alpha=100.0):
    """
    Polish using L-BFGS-B optimization with soft-min objective.
    """
    x0 = centers.flatten().copy()
    
    bounds = [(-0.499, 0.499)] * (N * 3)
    
    def objective(x):
        return _soft_min_objective(x, alpha)
    
    start_time = time.time()
    
    try:
        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={
                'maxiter': 500,
                'maxfun': 20000,
                'ftol': 1e-12,
                'gtol': 1e-8,
                'maxls': 50,
                'maxcor': 10
            }
        )
        
        if result.success or result.fun < objective(x0):
            new_centers = result.x.reshape(N, 3)
            new_centers = np.clip(new_centers, -0.499, 0.499)
            return new_centers
        else:
            return centers
    
    except Exception:
        return centers

def _anneal(centers, time_limit, initial_temp=0.02, cooling=0.9998):
    """
    Simulated annealing to escape local minima.
    """
    centers = centers.copy()
    current_r = _compute_radius(centers)
    best_centers = centers.copy()
    best_r = current_r
    temp = initial_temp
    
    while time.time() < time_limit and temp > 1e-6:
        n_perturb = np.random.choice([1, 2, 3], p=[0.7, 0.2, 0.1])
        indices = np.random.choice(N, n_perturb, replace=False)
        
        step_size = temp * 0.5
        new_centers = centers.copy()
        for idx in indices:
            new_centers[idx] += np.random.normal(0, step_size, 3)
        
        new_centers = np.clip(new_centers, -0.499, 0.499)
        
        new_r = _compute_radius(new_centers)
        
        delta = new_r - current_r
        if delta > 0 or np.random.random() < np.exp(delta / (temp + 1e-10)):
            current_r = new_r
            centers = new_centers
            if current_r > best_r:
                best_r = current_r
                best_centers = centers.copy()
        
        temp *= cooling
    
    return best_centers, best_r

def _optimize_from_initial(centers, time_limit):
    """
    Full optimization pipeline from an initial configuration:
    1. Anneal to find a good basin
    2. Polish with L-BFGS-B
    3. Repeat if time allows
    """
    best_centers = centers.copy()
    best_r = _compute_radius(centers)
    
    # Phase 1: Annealing
    remaining = time_limit - time.time()
    if remaining > 5:
        anneal_limit = time.time() + min(remaining * 0.5, 20)
        annealed_centers, annealed_r = _anneal(centers, anneal_limit, initial_temp=0.03, cooling=0.9998)
        
        if annealed_r > best_r:
            best_r = annealed_r
            best_centers = annealed_centers
    
    # Phase 2: L-BFGS-B polish
    remaining = time_limit - time.time()
    if remaining > 3:
        polish_limit = time.time() + min(remaining * 0.4, 15)
        polished_centers = _polish_l_bfgs(best_centers, polish_limit, alpha=100.0)
        
        polished_r = _compute_radius(polished_centers)
        if polished_r > best_r:
            best_r = polished_r
            best_centers = polished_centers
    
    # Phase 3: Another anneal + polish cycle if time allows
    remaining = time_limit - time.time()
    if remaining > 10:
        anneal_limit = time.time() + min(remaining * 0.3, 10)
        annealed_centers, annealed_r = _anneal(best_centers, anneal_limit, initial_temp=0.01, cooling=0.9998)
        
        if annealed_r > best_r:
            best_r = annealed_r
            best_centers = annealed_centers
        
        remaining = time_limit - time.time()
        if remaining > 5:
            polish_limit = time.time() + min(remaining * 0.5, 10)
            polished_centers = _polish_l_bfgs(best_centers, polish_limit, alpha=150.0)
            
            polished_r = _compute_radius(polished_centers)
            if polished_r > best_r:
                best_r = polished_r
                best_centers = polished_centers
    
    return best_centers, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s limit
    
    np.random.seed(42)
    
    # Generate multiple initializations
    initializations = []
    
    # Grid
    initializations.append(_build_initial_grid())
    
    # BCC
    initializations.append(_build_bcc_init())
    
    # Halton
    initializations.append(_build_halton_init())
    
    # Grid with jitter
    for jitter in [0.02, 0.04, 0.06]:
        centers = _build_initial_grid() + np.random.normal(0, jitter, (N, 3))
        centers = np.clip(centers, -0.49, 0.49)
        initializations.append(centers)
    
    # BCC with jitter
    for jitter in [0.02, 0.04]:
        centers = _build_bcc_init() + np.random.normal(0, jitter, (N, 3))
        centers = np.clip(centers, -0.49, 0.49)
        initializations.append(centers)
    
    # Random
    for _ in range(3):
        centers = np.random.uniform(-0.45, 0.45, (N, 3))
        initializations.append(centers)
    
    best_centers = None
    best_radius = 0.0
    
    # Sort initializations by initial radius (best first)
    init_radii = []
    for init in initializations:
        r = _compute_radius(init)
        init_radii.append((r, init))
    
    init_radii.sort(key=lambda x: -x[0])
    
    # Optimize each initialization
    for idx, (r_init, init_centers) in enumerate(init_radii):
        if time.time() > deadline - 15:
            break
        
        remaining = deadline - 15 - time.time()
        if remaining < 10:
            break
        
        # Allocate time based on initial quality
        if r_init > 0.08:
            time_budget = min(remaining * 0.35, 25)
        elif r_init > 0.06:
            time_budget = min(remaining * 0.25, 20)
        else:
            time_budget = min(remaining * 0.15, 15)
        
        opt_limit = time.time() + time_budget
        
        opt_centers, opt_r = _optimize_from_initial(init_centers, opt_limit)
        
        if opt_r > best_radius:
            best_radius = opt_r
            best_centers = opt_centers
    
    # Final polish on the best solution
    if best_centers is not None and time.time() < deadline - 5:
        remaining = deadline - 5 - time.time()
        if remaining > 3:
            polished = _polish_l_bfgs(best_centers, time.time() + min(remaining, 10), alpha=200.0)
            polished_r = _compute_radius(polished)
            if polished_r > best_radius:
                best_radius = polished_r
                best_centers = polished
    
    # Ensure valid output
    if best_centers is None or not np.all(np.isfinite(best_centers)):
        best_centers = _build_initial_grid()
    
    # Final clip
    best_centers = np.clip(best_centers, -0.5, 0.5)
    
    return best_centers
