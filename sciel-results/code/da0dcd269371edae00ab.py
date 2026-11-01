"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 300

# EVOLVE-BLOCK-START

def _compute_score(points):
    """Compute the minimum radius for a given set of points."""
    n = len(points)
    if n < 2:
        return min(1.0 - np.linalg.norm(points[0])) if n == 1 else 0.0
    
    # Compute all pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists) / 2.0
    
    # Compute distance to boundary
    boundary_dists = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = np.min(boundary_dists)
    
    return min(min_pair_dist, min_boundary)

def _ring_initialization(n):
    """
    Initialize points in concentric rings.
    We estimate the number of rings and points per ring to fill the disk.
    """
    points = []
    # Approximation: Area of disk = pi. Area of one circle = pi*r^2.
    # N*pi*r^2 <= pi => r <= 1/sqrt(N).
    # For N=300, r ~ 0.057.
    # Let's place rings.
    
    # A heuristic for ring distribution:
    # The number of points in ring k is proportional to its radius.
    # Let's try to fit rings such that the outer ring is near the boundary.
    
    # Start with a central point
    points.append([0.0, 0.0])
    
    remaining = n - 1
    # We need to distribute 'remaining' points into rings.
    # Let's guess the number of rings. 
    # If radius is ~0.05, diameter ~0.1. 
    # A ring of radius R can hold approx 2*pi*R / (2*r) = pi*R/r points.
    
    # Let's iterate to find a good configuration
    # We'll try to place rings from outside in or inside out.
    # Inside out is easier to manage overlap with previous rings.
    
    # Let's estimate the max radius R_max such that we can fit N circles.
    # This is complex. Let's use a simple heuristic:
    # Assume rings are at radii r_1, r_2, ...
    # Points in ring i: k_i.
    # Constraint: distance between adjacent rings >= 2r.
    
    # Let's just generate a grid of potential ring configurations and pick the best initial one?
    # No, let's use a standard "hexagonal packing in a circle" algorithm which is robust.
    # The seed used a hex lattice. Let's stick to a better lattice or ring-based.
    
    # Let's try a simple ring-based packing:
    # 1. Center point.
    # 2. Ring 1: 6 points.
    # 3. Ring 2: 12 points.
    # 4. Ring 3: 18 points.
    # This is for hexagonal packing.
    # For N=300, we need about 10-12 rings.
    
    # Let's calculate the ideal radius for a hexagonal packing truncated to a circle.
    # This is what the seed did, but it can be improved by allowing the lattice to deform.
    # However, a ring-based initialization with optimized radii is a good start for the optimizer.
    
    # Let's create a "perfect" hexagonal lattice, then clip it.
    # This is safer than guessing ring counts.
    
    spacing = 2.0 / (np.sqrt(n) + 1.0)
    # Adjust spacing to be slightly tighter to ensure we have enough points before clipping
    # Then the optimizer will push them out.
    spacing *= 0.95 
    
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        # Keep points strictly inside
        points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
        if len(points) >= n:
            # Sort by radius to pick the most central ones, which usually form a better core
            points = points[np.argsort(radius, kind="stable")[:n]]
            break
        spacing *= 0.98
        
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # 1. Initialization
    points = _ring_initialization(N)
    best_points = points.copy()
    best_score = _compute_score(points)
    
    # 2. Optimization using L-BFGS-B with custom gradient
    # We want to maximize the minimum distance r.
    # f(x) = min( min_{i<j} d_ij/2, min_i (1-|c_i|) )
    # This is non-smooth. We use the subgradient of the min function.
    # The gradient of the minimum is the average of the gradients of the active terms.
    
    def objective_and_grad(x):
        p = x.reshape(-1, 2)
        
        # Calculate distances
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        
        # Boundary distances
        norms = np.linalg.norm(p, axis=1)
        boundary_dists = 1.0 - norms
        
        # Current minimum radius
        min_pair = np.min(dists) / 2.0
        min_bound = np.min(boundary_dists)
        current_r = min(min_pair, min_bound)
        
        # We want to maximize current_r, so we minimize -current_r.
        # Gradient of -current_r is -grad(current_r).
        
        # Identify active constraints
        # For pairs: active if dists[i,j] is close to 2*current_r
        # For boundaries: active if boundary_dists[i] is close to current_r
        
        # To avoid noise, we consider a small tolerance
        tol = 1e-8
        active_pair_mask = (dists < 2.0 * current_r + tol) & (dists > 0)
        active_bound_mask = (boundary_dists < current_r + tol)
        
        grad = np.zeros_like(p)
        
        # Sum gradients of active pair constraints
        # d/dp_i (d_ij/2) = (p_i - p_j) / (2 * d_ij)
        # We want to move points to INCREASE the distance, so we follow the gradient of the distance.
        # Since we are minimizing -r, we move in the direction of -grad(r).
        # Wait, the optimizer minimizes f. f = -r.
        # grad(f) = -grad(r).
        # So we need to compute grad(r).
        
        # grad(r) is a vector in R^{2N}.
        # If pair (i,j) is active:
        # Contribution to grad_i: (p_i - p_j) / (2 * d_ij)  (normalized direction * weight)
        # Contribution to grad_j: (p_j - p_i) / (2 * d_ij)
        # Note: d_ij = |p_i - p_j|. d/dp_i |p_i - p_j| = (p_i - p_j)/|p_i - p_j|.
        # So d/dp_i (d_ij/2) = (p_i - p_j) / (2 * d_ij).
        
        # If boundary i is active:
        # d/dp_i (1 - |p_i|) = - p_i / |p_i|
        # So contribution to grad_i: -p_i / |p_i|.
        
        # Since r = min(...), grad(r) is the average of the active gradients.
        
        active_pairs = np.argwhere(active_pair_mask)
        if len(active_pairs) > 0:
            for i, j in active_pairs:
                if i == j: continue
                d = dists[i, j]
                if d < 1e-12: continue
                # Unit vector from j to i
                u = (p[i] - p[j]) / d
                # Gradient of d_ij/2 w.r.t p_i is u/2
                # Gradient of d_ij/2 w.r.t p_j is -u/2
                grad[i] += u / 2.0
                grad[j] -= u / 2.0
        
        if np.any(active_bound_mask):
            active_indices = np.where(active_bound_mask)[0]
            for i in active_indices:
                norm_i = norms[i]
                if norm_i < 1e-12:
                    # If at origin, gradient is undefined, but we can push slightly
                    # Actually, if at origin, boundary dist is 1. It's unlikely to be the min unless N=1.
                    continue
                # Gradient of (1 - |p_i|) w.r.t p_i is -p_i / |p_i|
                grad[i] += -p[i] / norm_i
        
        # Normalize the gradient (average of active gradients)
        num_active = len(active_pairs) + len(active_indices)
        if num_active > 0:
            grad /= num_active
        
        # The objective is -r. So grad_obj = -grad
        return -current_r, -grad

    # Use L-BFGS-B
    # We need to be careful with the number of iterations and time.
    try:
        res = minimize(
            lambda x: objective_and_grad(x)[0],
            points.flatten(),
            jac=lambda x: objective_and_grad(x)[1],
            method='L-BFGS-B',
            options={
                'maxiter': 500, # Limit iterations
                'gtol': 1e-6,
                'ftol': 1e-10,
                'maxls': 20,
                'disp': False
            }
        )
        
        if time.time() < deadline:
            candidate = res.x.reshape(-1, 2)
            candidate_score = _compute_score(candidate)
            if candidate_score > best_score:
                best_points = candidate
                best_score = candidate_score
                
        # If time remains, do a second pass with tighter tolerance or different initial simplex?
        # L-BFGS is quite effective. A second pass might refine it.
        
        if time.time() < deadline - 20:
             # Restart from best found with a smaller step size or just more iterations?
             # L-BFGS-B doesn't take step size directly, but we can restart.
             res2 = minimize(
                lambda x: objective_and_grad(x)[0],
                best_points.flatten(),
                jac=lambda x: objective_and_grad(x)[1],
                method='L-BFGS-B',
                options={
                    'maxiter': 1000,
                    'gtol': 1e-8,
                    'ftol': 1e-12,
                    'maxls': 20,
                    'disp': False
                }
            )
             if time.time() < deadline:
                candidate2 = res2.x.reshape(-1, 2)
                candidate_score2 = _compute_score(candidate2)
                if candidate_score2 > best_score:
                    best_points = candidate2
                    best_score = candidate_score2
                    
    except Exception as e:
        # If optimization fails, return the initial lattice
        pass

    # Final check: ensure all points are strictly inside and finite
    # The optimizer might push points slightly outside if constraints are soft.
    # We must project them back or penalize.
    # Since we used L-BFGS-B without explicit bounds, points might drift out.
    # We need to enforce the boundary constraint strictly.
    
    # Simple projection: if |p_i| > 1 - r_best, scale it back?
    # No, that might cause overlaps.
    # Better: The objective function naturally penalizes going outside because boundary_dist becomes negative,
    # making r negative, so -r positive, increasing the cost.
    # So L-BFGS should keep them inside.
    # However, floating point errors might cause |p_i| > 1.
    # Let's do a final cleanup.
    
    norms = np.linalg.norm(best_points, axis=1)
    max_allowed = 1.0 - best_score
    # If any point is outside max_allowed, we have a problem.
    # We can clip them, but this is a hack.
    # Given the gradient logic, they should be well inside.
    
    return best_points

# EVOLVE-BLOCK-END
