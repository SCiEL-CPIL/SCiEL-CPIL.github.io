"""Physics-based optimization for packing 200 equal circles in a unit square."""
import numpy as np
import time
from scipy.optimize import minimize

N = 200
SIDE = 1.0
HALF = 0.5

def _compute_radius(pts):
    """Compute the maximum common radius for given centers."""
    # Wall constraint
    wall_dist = np.minimum.reduce([
        HALF - pts[:, 0],
        HALF + pts[:, 0],
        HALF - pts[:, 1],
        HALF + pts[:, 1]
    ])
    min_wall = np.min(wall_dist)
    
    if min_wall < 0:
        return 0.0
    
    # Pairwise distances
    from scipy.spatial.distance import pdist
    dists = pdist(pts)
    if len(dists) == 0:
        return min_wall
    min_pair = np.min(dists) / 2.0
    
    return min(min_wall, min_pair)


def _energy(pts, r_target=0.035):
    """
    Energy function: soft repulsion between particles + wall penalty.
    We want to maximize the minimum distance, so we minimize a function
    that penalizes close pairs and wall violations.
    """
    pts = np.clip(pts, -HALF + 1e-10, HALF - 1e-10)
    
    # Wall penalty: quadratic penalty for being too close to walls
    wall_violation = np.maximum(0, np.abs(pts) - (HALF - r_target))
    wall_energy = np.sum(wall_violation ** 2) * 1000.0
    
    # Pairwise repulsion: soft potential
    from scipy.spatial.distance import pdist
    dists = pdist(pts)
    
    # Only consider pairs closer than some threshold for efficiency
    # Use a power-law repulsion
    close_mask = dists < 0.15
    if np.any(close_mask):
        close_dists = dists[close_mask]
        # Repulsive energy: 1/d^2 style but softened
        rep_energy = np.sum(1.0 / (close_dists ** 2 + 1e-6))
    else:
        rep_energy = 0.0
    
    # Also add a term to push apart particles that are too close
    # This helps break symmetry
    if np.any(close_mask):
        close_dists = dists[close_mask]
        push_energy = np.sum(np.maximum(0, (2 * r_target - close_dists)) ** 2) * 50.0
    else:
        push_energy = 0.0
    
    return wall_energy + rep_energy + push_energy


def _energy_and_grad(pts, r_target=0.035):
    """Energy and gradient for L-BFGS-B."""
    pts = np.clip(pts, -HALF + 1e-10, HALF - 1e-10)
    
    # Wall penalty gradient
    wall_violation = np.maximum(0, np.abs(pts) - (HALF - r_target))
    wall_grad = np.zeros_like(pts)
    for i in range(N):
        for j in range(2):
            val = pts[i, j]
            limit = HALF - r_target
            if val > limit:
                wall_grad[i, j] += 2 * 1000.0 * (val - limit)
            elif val < -limit:
                wall_grad[i, j] += 2 * 1000.0 * (val + limit)
    
    # Pairwise repulsion gradient
    grad = np.zeros_like(pts)
    from scipy.spatial.distance import pdist, squareform
    dists = squareform(pdist(pts))
    np.fill_diagonal(dists, np.inf)
    
    for i in range(N):
        for j in range(i + 1, N):
            d = dists[i, j]
            if d < 0.15:
                diff = pts[i] - pts[j]
                # dE/dpts_i for 1/d^2: -2 * diff / d^4
                factor = 2.0 / (d ** 4 + 1e-6)
                grad[i] += factor * diff
                grad[j] -= factor * diff
                # Push energy gradient
                if d < 2 * r_target:
                    push_factor = 2 * 50.0 * (2 * r_target - d)
                    grad[i] += push_factor * diff / d
                    grad[j] -= push_factor * diff / d
    
    energy = _energy(pts, r_target)
    return energy, wall_grad + grad


def _random_init(rng):
    """Random initial configuration inside the square."""
    return rng.uniform(-HALF + 0.02, HALF - 0.02, size=(N, 2))


def _grid_init():
    """Grid initialization."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10 second safety margin
    
    best_pts = None
    best_radius = 0.0
    
    rng = np.random.default_rng(42)
    
    # Try multiple random restarts with L-BFGS-B
    n_restarts = 10
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        if restart == 0:
            pts = _grid_init()
        else:
            pts = _random_init(rng)
        
        # Optimize
        try:
            result = minimize(
                _energy_and_grad,
                pts.flatten(),
                args=(0.035,),
                method='L-BFGS-B',
                jac=True,
                bounds=[(-HALF + 1e-10, HALF - 1e-10)] * (2 * N),
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            pts_opt = result.x.reshape(N, 2)
        except Exception:
            pts_opt = pts
        
        # Clip to valid region
        pts_opt = np.clip(pts_opt, -HALF + 1e-10, HALF - 1e-10)
        
        # Compute radius
        r = _compute_radius(pts_opt)
        
        if r > best_radius:
            best_radius = r
            best_pts = pts_opt.copy()
        
        # If we have a good radius, try to refine with a higher target
        if best_radius > 0.035 and time.time() < deadline - 20:
            # Try local refinement with higher r_target
            try:
                result2 = minimize(
                    _energy_and_grad,
                    best_pts.flatten(),
                    args=(best_radius,),
                    method='L-BFGS-B',
                    jac=True,
                    bounds=[(-HALF + 1e-10, HALF - 1e-10)] * (2 * N),
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                pts_refined = result2.x.reshape(N, 2)
                pts_refined = np.clip(pts_refined, -HALF + 1e-10, HALF - 1e-10)
                r2 = _compute_radius(pts_refined)
                if r2 > best_radius:
                    best_radius = r2
                    best_pts = pts_refined.copy()
            except Exception:
                pass
    
    # Final check
    if best_pts is None:
        best_pts = _grid_init()
    
    return best_pts
