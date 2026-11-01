"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

def _compute_radius(centers):
    """Compute the maximum valid radius for given centers."""
    wall_dists = np.min(0.5 - np.abs(centers), axis=1)
    min_wall = np.min(wall_dists)
    dists = cdist(centers, centers)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    return min(min_wall, min_pair / 2.0)

def _soft_min(values, k=50.0):
    """Smooth approximation of min using log-sum-exp."""
    m = np.min(values)
    if not np.isfinite(m):
        return m
    return m - np.log(np.sum(np.exp(k * (values - m)))) / k

def _neg_soft_radius(x, k=50.0):
    """Negative soft radius for optimization."""
    centers = x.reshape(-1, 3)
    wall_dists = 0.5 - np.abs(centers)
    min_wall = np.min(wall_dists, axis=1)
    
    dists = cdist(centers, centers)
    np.fill_diagonal(dists, np.inf)
    pair_dists = dists / 2.0
    
    # Soft min of all wall distances and pair distances
    all_dists = np.concatenate([min_wall, np.min(pair_dists, axis=1)])
    soft_r = _soft_min(all_dists, k)
    return -soft_r

def _exact_objective(x):
    """Exact negative radius."""
    centers = x.reshape(-1, 3)
    return -_compute_radius(centers)

def _init_grid():
    """5x5x2 grid initialization."""
    xs = np.linspace(-0.4, 0.4, 5)
    ys = np.linspace(-0.4, 0.4, 5)
    zs = np.array([-0.25, 0.25])
    centers = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    return centers[:N]

def _init_fcc():
    """FCC lattice initialization with greedy selection."""
    pts = []
    a = 0.28
    for i in range(-3, 4):
        for j in range(-3, 4):
            for k in range(-3, 4):
                for dx, dy, dz in [(0,0,0), (0,a/2,a/2), (a/2,0,a/2), (a/2,a/2,0)]:
                    x = i*a + dx
                    y = j*a + dy
                    z = k*a + dz
                    if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5 and -0.5 <= z <= 0.5:
                        pts.append((x, y, z))
    pts = np.array(pts)
    if len(pts) < N:
        return _init_grid()
    
    selected = []
    # Start with corner
    selected.append(pts[0])
    for _ in range(N - 1):
        dists = cdist(pts, np.array(selected))
        min_dists = np.min(dists, axis=1)
        for idx in range(len(selected)):
            min_dists[np.where(np.all(pts == selected[idx], axis=1))[0]] = -1
        best_idx = np.argmax(min_dists)
        selected.append(pts[best_idx])
    return np.array(selected)

def _init_bcc():
    """BCC lattice initialization."""
    pts = []
    a = 0.3
    for i in range(-3, 4):
        for j in range(-3, 4):
            for k in range(-3, 4):
                # Corner
                pts.append((i*a, j*a, k*a))
                # Body center
                pts.append((i*a + a/2, j*a + a/2, k*a + a/2))
    pts = np.array(pts)
    pts = pts[(pts >= -0.5).all(axis=1) & (pts <= 0.5).all(axis=1)]
    if len(pts) < N:
        return _init_grid()
    
    selected = []
    selected.append(pts[0])
    for _ in range(N - 1):
        dists = cdist(pts, np.array(selected))
        min_dists = np.min(dists, axis=1)
        for idx in range(len(selected)):
            min_dists[np.where(np.all(pts == selected[idx], axis=1))[0]] = -1
        best_idx = np.argmax(min_dists)
        selected.append(pts[best_idx])
    return np.array(selected)

def _init_random(seed=42):
    """Random initialization with rejection sampling for minimum distance."""
    rng = np.random.RandomState(seed)
    centers = []
    for _ in range(N):
        for attempt in range(1000):
            p = rng.uniform(-0.45, 0.45, 3)
            if not centers or np.min(cdist(np.array([p]), np.array(centers))) > 0.15:
                centers.append(p)
                break
        else:
            centers.append(p)
    return np.array(centers)

def _repulsion_anneal(x0, time_limit=30.0):
    """Repulsion-based annealing to spread points."""
    centers = x0.copy()
    start = time.time()
    
    # Parameters
    n_iter = 500
    k_repel = 2.0  # Repulsion strength
    k_wall = 5.0   # Wall repulsion strength
    cooling = 0.98
    
    for it in range(n_iter):
        if time.time() - start > time_limit:
            break
        
        forces = np.zeros_like(centers)
        
        # Pairwise repulsion
        dists = cdist(centers, centers)
        np.fill_diagonal(dists, np.inf)
        
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                d = dists[i, j]
                if d < 1e-10:
                    d = 1e-10
                # Repulsive force ~ 1/d^2
                force_mag = k_repel / (d * d)
                direction = (centers[j] - centers[i]) / d
                forces[i] -= force_mag * direction
        
        # Wall repulsion
        for i in range(N):
            for dim in range(3):
                if centers[i, dim] > 0:
                    d = 0.5 - centers[i, dim]
                    if d < 0.3:
                        force_mag = k_wall / max(d, 0.01)
                        forces[i, dim] -= force_mag
                else:
                    d = 0.5 + centers[i, dim]
                    if d < 0.3:
                        force_mag = k_wall / max(d, 0.01)
                        forces[i, dim] += force_mag
        
        # Move with cooling
        step_size = 0.01 * cooling ** (it / 50.0)
        centers += step_size * forces
        
        # Clamp to bounds
        centers = np.clip(centers, -0.499, 0.499)
    
    return centers

def _refine_smooth(x, k=50.0, maxiter=300):
    """Refine using smooth surrogate."""
    bounds = [(-0.5, 0.5)] * (N * 3)
    try:
        result = minimize(_neg_soft_radius, x, method='L-BFGS-B', 
                         bounds=bounds, 
                         options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10, 'k': k})
        if result.fun < -_compute_radius(result.x.reshape(-1, 3)):
            return result.x
    except:
        pass
    return x

def _refine_exact(x, maxiter=200):
    """Refine using exact objective."""
    bounds = [(-0.5, 0.5)] * (N * 3)
    try:
        result = minimize(_exact_objective, x, method='L-BFGS-B',
                         bounds=bounds,
                         options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10})
        if result.fun < -_compute_radius(result.x.reshape(-1, 3)):
            return result.x
    except:
        pass
    return x

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    best_x = None
    best_score = 0.0
    
    # Multiple initializations
    initializations = [
        ("grid", _init_grid),
        ("fcc", _init_fcc),
        ("bcc", _init_bcc),
        ("random1", lambda: _init_random(42)),
        ("random2", lambda: _init_random(123)),
        ("random3", lambda: _init_random(456)),
    ]
    
    for name, init_func in initializations:
        if time.time() > deadline - 20:
            break
        
        try:
            centers = init_func()
        except:
            continue
        
        if centers is None or len(centers) != N:
            continue
        
        x0 = centers.flatten()
        current_score = _compute_radius(centers)
        
        if current_score > best_score:
            best_score = current_score
            best_x = x0.copy()
        
        # Repulsion annealing
        if time.time() < deadline - 15:
            try:
                annealed = _repulsion_anneal(centers, time_limit=10.0)
                anneal_score = _compute_radius(annealed)
                if anneal_score > best_score:
                    best_score = anneal_score
                    best_x = annealed.flatten().copy()
                
                # Smooth refinement
                if time.time() < deadline - 10:
                    refined = _refine_smooth(annealed.flatten(), k=30.0, maxiter=200)
                    ref_score = _compute_radius(refined.reshape(-1, 3))
                    if ref_score > best_score:
                        best_score = ref_score
                        best_x = refined.copy()
            except:
                pass
        
        # Direct smooth refinement from initial
        if time.time() < deadline - 10:
            try:
                refined = _refine_smooth(x0, k=30.0, maxiter=200)
                ref_score = _compute_radius(refined.reshape(-1, 3))
                if ref_score > best_score:
                    best_score = ref_score
                    best_x = refined.copy()
            except:
                pass
    
    # Final exact polish on best solution
    if best_x is not None and time.time() < deadline - 5:
        try:
            polished = _refine_exact(best_x, maxiter=300)
            pol_score = _compute_radius(polished.reshape(-1, 3))
            if pol_score > best_score:
                best_score = pol_score
                best_x = polished.copy()
        except:
            pass
    
    # Additional smooth refinement with different k values
    if best_x is not None and time.time() < deadline - 3:
        for k_val in [20.0, 80.0]:
            if time.time() > deadline - 3:
                break
            try:
                refined = _refine_smooth(best_x, k=k_val, maxiter=150)
                ref_score = _compute_radius(refined.reshape(-1, 3))
                if ref_score > best_score:
                    best_score = ref_score
                    best_x = refined.copy()
            except:
                pass
    
    if best_x is None:
        # Fallback
        centers = _init_grid()
        return centers
    
    centers = best_x.reshape(-1, 3)
    # Final clamp
    centers = np.clip(centers, -0.5, 0.5)
    return centers

# EVOLVE-BLOCK-END
