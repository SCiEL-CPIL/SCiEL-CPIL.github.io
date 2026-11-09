"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _compute_radius(centers):
    """Compute the largest common radius for the given centers."""
    # Distance from each center to origin
    norms = np.linalg.norm(centers, axis=1)
    # Minimum distance to boundary
    min_boundary = 1.0 - norms.min()
    
    # Pairwise distances
    n = len(centers)
    if n < 2:
        return min_boundary
    
    # Compute all pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # Get minimum pairwise distance (excluding self)
    np.fill_diagonal(dists, np.inf)
    min_pair = dists.min() / 2.0
    
    return min(min_boundary, min_pair)


def _energy_and_grad(centers):
    """
    Soft-min surrogate for the radius objective.
    We want to maximize r = min(min_pair/2, min_boundary).
    Equivalent to minimizing -r.
    
    Using a log-sum-exp based soft-min for smoothness:
    soft_min(x) = -logsumexp(-x) / scale... 
    Actually, let's use a simpler approach: 
    r_soft = -logsumexp(-[all_pair_dists/2, all_boundary_dists]) / n_terms
    But this is complex. Let's use a different smooth surrogate.
    
    Alternative: minimize the sum of squares of violations if we fix a target r,
    or use the fact that maximizing min is equivalent to pushing up the smallest.
    
    Let's use: objective = -soft_min where soft_min uses log-sum-exp.
    For a set of values v_i, soft_min = -1/beta * log(sum(exp(-beta*v_i)))
    As beta -> inf, this approaches min(v_i).
    """
    centers = centers.reshape(-1, 2)
    n = len(centers)
    
    # Boundary distances
    norms = np.linalg.norm(centers, axis=1)
    boundary_dists = 1.0 - norms  # shape (n,)
    
    # Pairwise distances / 2
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    pair_dists = dists / 2.0  # shape (n, n)
    
    # Collect all constraint values
    all_vals = np.concatenate([boundary_dists, pair_dists.ravel()])
    
    # Soft-min with log-sum-exp
    beta = 50.0  # sharpness parameter
    # soft_min = -1/beta * log(sum(exp(-beta * v)))
    # For numerical stability:
    max_val = np.max(all_vals)
    shifted = -beta * (all_vals - max_val)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min_val = max_val - log_sum / beta
    
    # Gradient of soft_min with respect to centers
    # d(soft_min)/d(v_i) = -exp(-beta*v_i) / sum(exp(-beta*v_j))
    # But we need d(soft_min)/d(centers)
    
    # For simplicity, let's compute gradient numerically or use a simpler objective
    # Actually, let's use a different approach: minimize the negative of the soft-min
    # and compute gradient analytically.
    
    # Let's compute the gradient properly
    # soft_min = max_val - (1/beta) * log(sum(exp(-beta*(v_i - max_val))))
    # d(soft_min)/d(v_k) = -1/beta * (-beta * exp(-beta*(v_k - max_val))) / sum(exp(-beta*(v_i - max_val)))
    #                     = exp(-beta*(v_k - max_val)) / sum(exp(-beta*(v_i - max_val)))
    
    exp_vals = np.exp(shifted)
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp  # shape (n + n*(n-1),)
    
    # Now we need to map weights back to centers
    # First n weights are for boundary_dists
    # Next n*(n-1) weights are for pair_dists
    
    grad = np.zeros_like(centers)
    
    # Boundary contribution
    # d(boundary_dists_i)/d(centers_i) = -centers_i / norms_i
    for i in range(n):
        if norms[i] > 1e-10:
            grad[i] += -weights[i] * (-centers[i] / norms[i])
        # else centers[i] is near origin, gradient of (1-|c|) is -c/|c| which is undefined,
        # but numerically it's fine
    
    # Pairwise contribution
    # pair_dists[i,j] = |c_i - c_j| / 2
    # d(pair_dists[i,j])/d(c_i) = (c_i - c_j) / (2 * |c_i - c_j|) = (c_i - c_j) / (4 * pair_dists[i,j])
    # d(pair_dists[i,j])/d(c_j) = (c_j - c_i) / (4 * pair_dists[i,j])
    
    # We need to iterate over pairs
    pair_idx = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            w = weights[n + pair_idx]
            pair_idx += 1
            d = dists[i, j]
            if d > 1e-10:
                direction = (centers[i] - centers[j]) / (2.0 * d)
                grad[i] += w * direction
                grad[j] += w * (-direction)
    
    # We want to maximize soft_min, so minimize -soft_min
    # Gradient of -soft_min is -grad
    objective = -soft_min_val
    grad_obj = -grad
    
    return objective, grad_obj


def _hex_lattice_init(n_points):
    """Generate a hexagonal lattice initialization."""
    spacing = 2.0 / (np.sqrt(n_points) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        mask = radius <= 1.0 - spacing / 2.0
        points = points[mask]
        if len(points) >= n_points:
            # Sort by radius to get the innermost points
            idx = np.argsort(radius[mask], kind="stable")[:n_points]
            return points[idx]
        spacing *= 0.98


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    # Initial guess: hexagonal lattice
    centers = _hex_lattice_init(N)
    
    # Flatten for scipy
    x0 = centers.flatten()
    
    # Objective and gradient function for scipy
    def obj_func(x):
        c = x.reshape(-1, 2)
        val, grad = _energy_and_grad(c)
        return val, grad.flatten()
    
    # Use L-BFGS-B for optimization
    # We can also try SLSQP with constraints, but L-BFGS-B is faster
    best_centers = centers.copy()
    best_radius = _compute_radius(centers)
    
    # Try multiple restarts with different initializations
    rng = np.random.RandomState(42)
    
    for restart in range(20):
        if time.time() > deadline - 30.0:
            break
        
        if restart == 0:
            x0 = centers.flatten()
        else:
            # Perturb the best solution found so far
            if restart % 2 == 0:
                # Small perturbation
                x0 = best_centers.flatten() + rng.normal(0, 0.001, size=(N, 2)).flatten()
            else:
                # Larger perturbation
                x0 = best_centers.flatten() + rng.normal(0, 0.01, size=(N, 2)).flatten()
        
        # Clip to stay within reasonable bounds
        x0 = np.clip(x0, -1.0, 1.0)
        
        # Optimize
        try:
            result = minimize(
                obj_func,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < obj_func(x0)[0]:
                new_centers = result.x.reshape(-1, 2)
                new_radius = _compute_radius(new_centers)
                
                if new_radius > best_radius:
                    best_radius = new_radius
                    best_centers = new_centers.copy()
        
        except Exception as e:
            pass
        
        # Check time
        if time.time() > deadline - 10.0:
            break
    
    # Final polish with SLSQP if time permits
    if time.time() < deadline - 15.0:
        try:
            # Use SLSQP for a final polish
            def slsqp_obj(x):
                c = x.reshape(-1, 2)
                val, _ = _energy_and_grad(c)
                return val
            
            def slsqp_grad(x):
                c = x.reshape(-1, 2)
                _, grad = _energy_and_grad(c)
                return grad.flatten()
            
            result = minimize(
                slsqp_obj,
                best_centers.flatten(),
                method='SLSQP',
                jac=slsqp_grad,
                options={'maxiter': 50, 'ftol': 1e-12}
            )
            
            if result.fun < slsqp_obj(best_centers.flatten()):
                new_centers = result.x.reshape(-1, 2)
                new_radius = _compute_radius(new_centers)
                if new_radius > best_radius:
                    best_radius = new_radius
                    best_centers = new_centers.copy()
        
        except Exception as e:
            pass
    
    return best_centers
# EVOLVE-BLOCK-END
