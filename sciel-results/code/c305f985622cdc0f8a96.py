import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 100

# EVOLVE-BLOCK-START

def _compute_radius(centers):
    """
    Compute the largest common radius for the given centers.
    Centers are in [-0.5, 0.5]^3.
    """
    dists_to_walls = 0.5 - np.abs(centers)
    min_wall_dist = np.min(dists_to_walls)
    
    dists = cdist(centers, centers, metric='euclidean')
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    return min(min_wall_dist, min_pair_dist / 2.0)

def _soft_min_objective(centers_flat, beta):
    """
    Objective function for L-BFGS-B.
    Minimizes the soft-min approximation of the radius.
    """
    centers = centers_flat.reshape(-1, 3)
    
    dists_to_walls = 0.5 - np.abs(centers)
    wall_vals = dists_to_walls.flatten()
    
    dists = cdist(centers, centers, metric='euclidean')
    np.fill_diagonal(dists, np.inf)
    pair_vals = dists / 2.0
    iu = np.triu_indices(N, k=1)
    pair_vals_unique = pair_vals[iu]
    
    all_vals = np.concatenate([wall_vals, pair_vals_unique])
    
    neg_beta_x = -beta * all_vals
    max_val = np.max(neg_beta_x)
    sum_exp = np.sum(np.exp(neg_beta_x - max_val))
    obj = (max_val + np.log(sum_exp)) / beta
    
    return obj

def _build_bcc_init():
    """
    Build a BCC (body-centered cubic) lattice initialization.
    """
    nx, ny, nz = 5, 5, 2
    
    xs = np.linspace(-0.45, 0.45, nx)
    ys = np.linspace(-0.45, 0.45, ny)
    zs = np.linspace(-0.45, 0.45, nz)
    
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    corners = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    
    dx = xs[1] - xs[0] if nx > 1 else 1.0
    dy = ys[1] - ys[0] if ny > 1 else 1.0
    dz = zs[1] - zs[0] if nz > 1 else 1.0
    
    body_xs = xs + dx / 2.0
    body_ys = ys + dy / 2.0
    body_zs = zs + dz / 2.0
    
    BX, BY, BZ = np.meshgrid(body_xs, body_ys, body_zs, indexing='ij')
    bodies = np.column_stack((BX.ravel(), BY.ravel(), BZ.ravel()))
    
    all_centers = np.vstack([corners, bodies])
    all_centers = np.clip(all_centers, -0.49, 0.49)
    all_centers += np.random.normal(0, 0.005, all_centers.shape)
    all_centers = np.clip(all_centers, -0.49, 0.49)
    
    return all_centers

def _build_grid_init():
    """
    Build a simple cubic grid initialization.
    """
    nx, ny, nz = 5, 5, 4
    
    xs = np.linspace(-0.45, 0.45, nx)
    ys = np.linspace(-0.45, 0.45, ny)
    zs = np.linspace(-0.45, 0.45, nz)
    
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    centers = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    
    centers += np.random.normal(0, 0.01, centers.shape)
    centers = np.clip(centers, -0.49, 0.49)
    
    return centers

def _build_halton_init():
    """
    Build a Halton sequence initialization for better space-filling.
    """
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    points = np.zeros((N, 3))
    for i in range(N):
        points[i, 0] = halton(i + 1, 2) - 0.5
        points[i, 1] = halton(i + 1, 3) - 0.5
        points[i, 2] = halton(i + 1, 5) - 0.5
    
    points += np.random.normal(0, 0.02, points.shape)
    points = np.clip(points, -0.49, 0.49)
    
    return points

def _build_sobol_init():
    """
    Build a Sobol sequence initialization for better space-filling.
    """
    from scipy.stats import qmc
    sampler = qmc.Sobol(d=3, scramble=True, seed=42)
    points = sampler.random(N)
    centers = points * 0.98 - 0.49
    centers = np.clip(centers, -0.49, 0.49)
    return centers

def _optimize_from_initial(centers, deadline, beta_start=10.0, beta_end=500.0):
    """
    Optimize from a given initial configuration using soft-min with increasing beta.
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    beta = beta_start
    while time.time() < deadline - 15.0 and beta < beta_end:
        x0 = best_centers.flatten()
        
        try:
            res = minimize(
                _soft_min_objective,
                x0,
                args=(beta,),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            
            new_centers = res.x.reshape(-1, 3)
            new_centers = np.clip(new_centers, -0.5, 0.5)
            
            r = _compute_radius(new_centers)
            if r > best_radius:
                best_radius = r
                best_centers = new_centers
        except Exception:
            pass
        
        beta *= 1.8
        
        if time.time() > deadline - 15.0:
            break
    
    return best_centers, best_radius

def _polish(centers, deadline):
    """
    Final polish pass with tighter tolerances and higher beta.
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    for beta in [500.0, 1000.0, 2000.0, 5000.0]:
        if time.time() > deadline - 5.0:
            break
            
        x0 = best_centers.flatten()
        
        try:
            res = minimize(
                _soft_min_objective,
                x0,
                args=(beta,),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 200}
            )
            
            new_centers = res.x.reshape(-1, 3)
            new_centers = np.clip(new_centers, -0.5, 0.5)
            
            r = _compute_radius(new_centers)
            if r > best_radius:
                best_radius = r
                best_centers = new_centers
        except Exception:
            pass
    
    return best_centers, best_radius

def _local_search_refine(centers, deadline):
    """
    Additional local search refinement using multiple random perturbations.
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    num_attempts = 10
    for _ in range(num_attempts):
        if time.time() > deadline - 10.0:
            break
        
        # Random perturbation
        perturbation = np.random.normal(0, 0.001, centers.shape)
        perturbed = centers + perturbation
        perturbed = np.clip(perturbed, -0.49, 0.49)
        
        # Quick optimization
        try:
            res = minimize(
                _soft_min_objective,
                perturbed.flatten(),
                args=(1000.0,),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            
            new_centers = res.x.reshape(-1, 3)
            new_centers = np.clip(new_centers, -0.5, 0.5)
            
            r = _compute_radius(new_centers)
            if r > best_radius:
                best_radius = r
                best_centers = new_centers
        except Exception:
            pass
    
    return best_centers, best_radius

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # Try multiple initializations
    initializations = [
        ("grid", _build_grid_init),
        ("bcc", _build_bcc_init),
        ("halton", _build_halton_init),
        ("sobol", _build_sobol_init),
    ]
    
    best_centers = None
    best_radius = 0.0
    
    for name, init_func in initializations:
        if time.time() > deadline - 60.0:
            break
        
        centers = init_func()
        r = _compute_radius(centers)
        
        if r > best_radius:
            best_radius = r
            best_centers = centers.copy()
    
    # Optimization loop with increasing beta
    best_centers, best_radius = _optimize_from_initial(best_centers, deadline, beta_start=10.0, beta_end=500.0)
    
    # Local search refinement
    best_centers, best_radius = _local_search_refine(best_centers, deadline)
    
    # Final polish
    best_centers, best_radius = _polish(best_centers, deadline)
    
    return best_centers

# EVOLVE-BLOCK-END
