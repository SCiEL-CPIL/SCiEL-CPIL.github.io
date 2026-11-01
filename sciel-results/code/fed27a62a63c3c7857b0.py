import numpy as np
import time
from scipy.optimize import minimize

N = 100

def _compute_radius(centers):
    """Compute the maximum radius for given centers."""
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    wall_dists = 0.5 - np.abs(centers)
    min_wall_dist = np.min(wall_dists)
    
    return min(min_pair_dist / 2, min_wall_dist)

def _soft_min_objective_and_grad(x, centers_flat, r, alpha):
    """
    Objective: maximize r subject to constraints.
    We minimize -r + penalty for violations.
    
    x = [r, centers_flat] where centers_flat is length 300
    """
    r = x[0]
    c = x[1:].reshape(N, 3)
    
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    
    # Wall distances
    wall_dists = 0.5 - np.abs(c)
    min_wall = np.min(wall_dists)
    
    # Soft min of all constraints: pair distances and wall distances
    # Constraint: dist_ij >= 2r for all i<j
    # Constraint: wall_dist_i >= r for all i,k
    
    # Build vector of all constraint violations (should be >= 0)
    # Pair constraints: dist_ij - 2r
    # Wall constraints: wall_dist_ik - r
    
    # For efficiency, compute soft min using log-sum-exp
    # violations = [dist_ij - 2r for i<j, wall_dist_ik - r for all i,k]
    
    # Get upper triangle indices
    iu = np.triu_indices(N, k=1)
    pair_violations = dists[iu] - 2*r
    wall_violations = wall_dists.ravel() - r
    
    all_violations = np.concatenate([pair_violations, wall_violations])
    
    # Soft min: -logsumexp(-alpha * violations) / alpha
    # For large alpha, this approximates min(violations)
    max_neg = np.max(-alpha * all_violations)
    if max_neg > 500:
        max_neg = 500
    soft_min_val = -(np.log(np.sum(np.exp(-alpha * all_violations - max_neg))) + max_neg) / alpha
    
    # Objective: -r + penalty * max(0, -soft_min_val)
    # If soft_min_val < 0, there's a violation
    penalty = 100.0
    violation = max(0, -soft_min_val)
    obj = -r + penalty * violation
    
    # Gradient
    grad_r = -1.0
    grad_c = np.zeros_like(c)
    
    if violation > 0:
        # d(obj)/d(r) = -1 + penalty * d(violation)/d(r)
        # d(soft_min)/d(r) = -2 * (weighted avg of pair weights) - (weighted avg of wall weights)
        # weights = exp(-alpha * violations - max_neg) / sum(exp(...))
        weights = np.exp(-alpha * all_violations - max_neg)
        weights /= np.sum(weights)
        
        # d(pair_violations)/dr = -2 for each pair
        # d(wall_violations)/dr = -1 for each wall
        d_soft_min_dr = -2 * np.sum(weights[:len(pair_violations)]) - np.sum(weights[len(pair_violations):])
        
        grad_r += penalty * (-d_soft_min_dr)
        
        # d(pair_violations)/dc_i = (c_i - c_j)/dist_ij for j, and (c_i - c_k)/dist_ik for k
        # This is complex; approximate with finite difference or skip for now
        # For simplicity, use a simpler penalty approach
    
    return obj, np.concatenate([[grad_r], grad_c.ravel()])

def _optimize_from_initial(centers, time_budget=30):
    """Optimize from given initial centers using gradient-based method."""
    start = time.time()
    deadline = start + time_budget
    
    best_r = _compute_radius(centers)
    best_centers = centers.copy()
    
    # Try different alpha values for soft min
    alphas = [50, 100, 200]
    
    for alpha in alphas:
        if time.time() > deadline - 5:
            break
            
        # Start with current best radius estimate
        r0 = best_r
        x0 = np.concatenate([[r0], best_centers.ravel()])
        
        def obj_and_grad(x):
            return _soft_min_objective_and_grad(x, None, None, alpha)
        
        try:
            res = minimize(
                lambda x: _soft_min_objective_and_grad(x, None, None, alpha)[0],
                x0,
                jac=lambda x: _soft_min_objective_and_grad(x, None, None, alpha)[1],
                method='L-BFGS-B',
                options={'maxiter': 100, 'maxfun': 500}
            )
            
            if res.success:
                r_new = res.x[0]
                c_new = res.x[1:].reshape(N, 3)
                r_actual = _compute_radius(c_new)
                
                if r_actual > best_r:
                    best_r = r_actual
                    best_centers = c_new.copy()
        except Exception:
            pass
    
    return best_centers, best_r

def _init_grid():
    x = np.linspace(-0.4, 0.4, 5)
    y = np.linspace(-0.4, 0.4, 5)
    z = np.linspace(-0.375, 0.375, 4)
    return np.array([(x[i], y[j], z[k]) for i in range(5) for j in range(5) for k in range(4)])

def _init_fcc_like():
    """FCC-like initialization with 100 points."""
    # FCC has 4 atoms per unit cell
    # 100 = 25 * 4, so 25 unit cells
    # Use 3x3x3 = 27 cells, take 25 of them
    a = 1.0 / 25 ** (1/3)  # ~0.342
    # Actually, let's just do a perturbed grid
    centers = _init_grid()
    np.random.seed(42)
    centers += np.random.normal(0, 0.02, centers.shape)
    return centers

def _init_random(seed=0):
    np.random.seed(seed)
    return np.random.uniform(-0.45, 0.45, (N, 3))

def _init_halton():
    """Halton sequence initialization."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    centers = np.zeros((N, 3))
    for i in range(N):
        for j in range(3):
            centers[i, j] = 2 * halton(i + 1, [2, 3, 5][j]) - 1
    return centers

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 40s margin
    
    # Multiple initializations
    inits = [
        _init_grid(),
        _init_fcc_like(),
        _init_random(0),
        _init_halton(),
        _init_random(1),
    ]
    
    best_r = -1
    best_centers = None
    
    for i, init_centers in enumerate(inits):
        if time.time() > deadline - 15:
            break
        
        # Allocate remaining time divided among remaining inits
        remaining_inits = len(inits) - i
        time_per_init = min(20, (deadline - 15 - time.time()) / max(1, remaining_inits))
        
        centers, r = _optimize_from_initial(init_centers, time_budget=time_per_init)
        
        if r > best_r:
            best_r = r
            best_centers = centers.copy()
    
    if best_centers is None:
        best_centers = _init_grid()
    
    return best_centers
