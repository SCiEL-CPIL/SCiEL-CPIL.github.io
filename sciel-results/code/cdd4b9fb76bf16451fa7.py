"""Optimize packing of 100 equal spheres in a unit cube via L-BFGS-B on an energy
function that penalizes violations of non-overlap and wall constraints."""
import time
import numpy as np

N = 100
DEADLINE_FRACTION = 0.85  # return by 150 * 0.85 = 127.5 s

def _pack_energy(x, K=400.0):
    """
    Energy = sum of overlap penalties + wall penalties.
    Returns scalar energy and gradient (flattened 3N vector).
    """
    c = x.reshape(N, 3)
    # Pairwise distances
    diff = c[:, None, :] - c[None, :, :]  # (N, N, 3)
    dist = np.linalg.norm(diff, axis=2)   # (N, N)
    np.fill_diagonal(dist, 1e10)
    
    # Overlap penalty: if dist < 2r, penalty = (2r - dist)^2
    # We use r as a parameter? Actually we maximize r such that all constraints hold.
    # Alternative: minimize sum of (min dist to wall, min pair dist) violations.
    # Better: define radius r = min(min_pair_dist/2, min_wall_dist)
    # We want to maximize r. Equivalent to minimize -r, but r is non-smooth.
    # Use soft constraint: minimize sum of max(0, 2r - dist_ij)^2 + sum of max(0, r - wall_dist_i)^2
    # But r is what we're optimizing... 
    #
    # Cleaner approach: parameterize centers, define objective as -min_dist where
    # min_dist = min over all pair distances/2 and wall distances.
    # This is non-smooth. Use smoothed min.
    #
    # Simplest: minimize sum of violations for a FIXED target radius.
    # But we don't know target radius. 
    #
    # Best approach: minimize sum of squared violations where violation for pair (i,j)
    # is max(0, 2r - dist_ij) and for wall is max(0, r - wall_dist_i),
    # where r is chosen as the current minimum feasible radius.
    # This is still coupled.
    #
    # Practical approach used in packing: minimize sum of (dist_ij - 2r)^2 for dist_ij < 2r
    # and (wall_dist_i - r)^2 for wall_dist_i < r, where r is a parameter we also optimize.
    # Make r a separate variable.
    pass

def construct():
    t0 = time.time()
    deadline = t0 + 150 * DEADLINE_FRACTION
    
    rng = np.random.default_rng(42)
    
    # Initialize with simple grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    best_x = grid[:N].copy()
    best_score = _compute_score(best_x)
    
    from scipy.optimize import minimize
    
    def objective_and_grad(x):
        """
        x: 3N vector of centers.
        Objective: minimize sum of squared constraint violations for the current best radius.
        Actually, let's use a simpler energy: for each pair, if dist < d_target, penalty.
        We'll use the approach of minimizing -min_dist via soft min.
        """
        c = x.reshape(N, 3)
        # Compute all pair distances
        diff = c[:, None, :] - c[None, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        
        # Wall distances: distance to each of 6 faces
        # cube is [-0.5, 0.5]^3
        wall_dists = np.min(np.abs(np.column_stack([
            0.5 - c[:, 0],  # right face
            c[:, 0] + 0.5,   # left face  
            0.5 - c[:, 1],   # top
            c[:, 1] + 0.5,   # bottom
            0.5 - c[:, 2],   # front
            c[:, 2] + 0.5    # back
        ])), axis=1)
        
        # Current radius estimate
        min_pair = np.min(dist)
        min_wall = np.min(wall_dists)
        r = min(min_pair / 2, min_wall)
        
        # Energy: sum of squared violations
        # Pair violations: (2r - dist_ij)^2 where dist_ij < 2r
        mask_pair = dist < 2 * r
        pair_viol = np.zeros_like(dist)
        pair_viol[mask_pair] = (2 * r - dist[mask_pair]) ** 2
        # Only count upper triangle
        pair_energy = 0.5 * np.sum(pair_viol)  # 0.5 to avoid double counting
        
        # Wall violations: (r - wall_dist_i)^2 where wall_dist_i < r
        mask_wall = wall_dists < r
        wall_viol = np.zeros_like(wall_dists)
        wall_viol[mask_wall] = (r - wall_dists[mask_wall]) ** 2
        wall_energy = np.sum(wall_viol)
        
        energy = pair_energy + wall_energy
        
        # Gradient
        grad = np.zeros_like(x)
        c_grad = grad.reshape(N, 3)
        
        if energy > 1e-15:
            # Gradient of pair energy
            # d(pair_energy)/d(c_i) = sum_j where i<j and dist_ij < 2r of -2(2r - dist_ij) * (c_i - c_j)/dist_ij
            # This is complex. Let's use numerical gradient for simplicity if time permits,
            # or use a simpler energy function.
            
            # Actually, let's use a simpler approach: minimize sum of 1/dist for close pairs
            # and sum of 1/wall_dist for close walls, which is smooth.
            pass
        
        return energy, grad
    
    # Let's use a cleaner energy function that is smooth
    def smooth_energy(x, beta=50.0):
        c = x.reshape(N, 3)
        diff = c[:, None, :] - c[None, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, 1e10)
        
        wall_dists = np.min(np.abs(np.column_stack([
            0.5 - c[:, 0], c[:, 0] + 0.5,
            0.5 - c[:, 1], c[:, 1] + 0.5,
            0.5 - c[:, 2], c[:, 2] + 0.5
        ])), axis=1)
        
        # Smooth min approximation using log-sum-exp style
        # For pairs: E_pair = -log(sum(exp(-beta * dist_ij))) for i<j, normalized
        # This gives approx min dist / beta
        upper_idx = np.triu_indices(N, k=1)
        pair_dists = dist[upper_idx]
        # Clamp to avoid underflow
        pair_dists = np.clip(pair_dists, 1e-8, 1.0)
        E_pair = -np.log(np.sum(np.exp(-beta * pair_dists))) / beta
        
        # For walls: E_wall = -log(sum(exp(-beta * wall_dist_i))) / beta
        wall_dists = np.clip(wall_dists, 1e-8, 1.0)
        E_wall = -np.log(np.sum(np.exp(-beta * wall_dists))) / beta
        
        # Total energy to minimize (we want to maximize min distance, so minimize this)
        energy = E_pair + E_wall
        
        return energy
    
    # Use L-BFGS-B with numerical gradient
    best_x = best_x.copy()
    
    for restart in range(20):
        if time.time() > deadline:
            break
        
        if restart == 0:
            x0 = best_x.copy()
        else:
            # Perturb best solution
            x0 = best_x + rng.normal(0, 0.01, best_x.shape)
            x0 = np.clip(x0, -0.49, 0.49)
        
        def obj(x):
            return smooth_energy(x)
        
        try:
            res = minimize(
                obj, x0, method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (3 * N),
                options={'maxiter': 2000, 'gtol': 1e-7, 'ftol': 1e-12}
            )
            if res.success or res.fun < _compute_score(best_x) * 0.999:
                # Validate and update
                candidate = np.clip(res.x.reshape(N, 3), -0.5, 0.5)
                score = _compute_score(candidate)
                if score > best_score:
                    best_score = score
                    best_x = candidate.copy()
        except Exception:
            pass
        
        if time.time() > deadline:
            break
    
    return best_x

def _compute_score(centers):
    """Compute the packing radius for given centers."""
    c = centers
    if c.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(c)):
        return 0.0
    
    # Check bounds
    if np.any(c < -0.5) or np.any(c > 0.5):
        return 0.0
    
    # Wall distances
    wall_dists = np.min(np.abs(np.column_stack([
        0.5 - c[:, 0], c[:, 0] + 0.5,
        0.5 - c[:, 1], c[:, 1] + 0.5,
        0.5 - c[:, 2], c[:, 2] + 0.5
    ])), axis=1)
    min_wall = np.min(wall_dists)
    
    # Pair distances
    diff = c[:, None, :] - c[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    
    r = min(min_pair / 2, min_wall)
    return r
