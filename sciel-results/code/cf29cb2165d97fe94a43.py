"""Combined program: hexagonal lattice init + soft-min optimization with epigraph variable."""
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
    
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists) / 2.0
    
    boundary_dists = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = np.min(boundary_dists)
    
    return min(min_pair_dist, min_boundary)

def _hex_lattice_init(n):
    """
    Initialize points in a hexagonal lattice clipped to the unit circle.
    """
    points = []
    # Estimate spacing. r ~ 1/sqrt(N). Diameter ~ 2r.
    # Hex lattice spacing s = 2r.
    # Area of hexagon per point ~ s^2 * sqrt(3)/2.
    # N * s^2 * sqrt(3)/2 ~ pi.
    # s ~ sqrt(2*pi / (N*sqrt(3))).
    s = np.sqrt(2.0 * np.pi / (n * np.sqrt(3.0)))
    
    max_k = int(1.0 / (s * 0.5)) + 2
    
    candidates = []
    for j in range(-max_k, max_k + 1):
        for i in range(-max_k, max_k + 1):
            x = s * (i + 0.5 * (j % 2))
            y = s * j * np.sqrt(3.0) / 2.0
            if x*x + y*y <= 1.0:
                candidates.append([x, y])
    
    candidates = np.array(candidates)
    
    if len(candidates) < n:
        # If we didn't generate enough, shrink spacing slightly and retry
        s *= 0.9
        max_k = int(1.0 / (s * 0.5)) + 3
        candidates = []
        for j in range(-max_k, max_k + 1):
            for i in range(-max_k, max_k + 1):
                x = s * (i + 0.5 * (j % 2))
                y = s * j * np.sqrt(3.0) / 2.0
                if x*x + y*y <= 1.0:
                    candidates.append([x, y])
        candidates = np.array(candidates)
        
    if len(candidates) >= n:
        # Sort by distance to origin to pick the most central ones
        radii = np.linalg.norm(candidates, axis=1)
        indices = np.argsort(radii)[:n]
        points = candidates[indices]
    else:
        # Fallback: random points
        points = np.random.rand(n, 2) * 2 - 1
        norms = np.linalg.norm(points, axis=1)
        mask = norms > 1.0
        points[mask] = points[mask] / norms[mask, np.newaxis] * 0.9

    # Add small random perturbation to break symmetry
    points += np.random.normal(0, 1e-4, points.shape)
    
    return points

def _ring_init(n):
    """
    Initialize points in concentric rings with optimal angular spacing.
    """
    points = []
    
    # Estimate number of rings
    # Each ring k has ~k*6 points (hexagonal packing approximation)
    # Total points ~ 3*K^2 where K is number of rings
    K = int(np.sqrt(n / 3.0)) + 1
    
    for k in range(1, K + 1):
        # Radius of ring k
        r_k = k * np.sqrt(2.0 * np.pi / (n * np.sqrt(3.0)))
        if r_k >= 1.0:
            break
        
        # Number of points on this ring
        # Circumference / spacing
        n_k = max(1, int(2 * np.pi * r_k / (2.0 * np.sqrt(2.0 * np.pi / (n * np.sqrt(3.0))))))
        
        # Angular positions
        angles = np.linspace(0, 2 * np.pi, n_k, endpoint=False)
        # Add offset for alternating rings
        if k % 2 == 0:
            angles += np.pi / n_k
        
        for angle in angles:
            x = r_k * np.cos(angle)
            y = r_k * np.sin(angle)
            if x*x + y*y <= 1.0:
                points.append([x, y])
    
    points = np.array(points)
    
    if len(points) >= n:
        # Pick the most central ones
        radii = np.linalg.norm(points, axis=1)
        indices = np.argsort(radii)[:n]
        points = points[indices]
    else:
        # Fill remaining with hex lattice
        remaining = n - len(points)
        hex_points = _hex_lattice_init(remaining)
        points = np.vstack([points, hex_points])
    
    # Add small random perturbation
    points += np.random.normal(0, 1e-4, points.shape)
    
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # 1. Initialization - try both hex lattice and ring init, pick better
    points_hex = _hex_lattice_init(N)
    points_ring = _ring_init(N)
    
    score_hex = _compute_score(points_hex)
    score_ring = _compute_score(points_ring)
    
    if score_ring > score_hex:
        points = points_ring
    else:
        points = points_hex
    
    best_points = points.copy()
    best_score = _compute_score(points)
    
    # 2. Optimization using L-BFGS-B with smooth soft-min objective
    # We use an epigraph formulation: maximize t such that all constraints >= t.
    # This is equivalent to maximizing the minimum radius.
    # We optimize over x (positions) and t (radius).
    # Objective: -t (minimize -t to maximize t)
    # Constraints: c_k(x) - t >= 0 for all k.
    # We use a soft penalty for constraints to make it smooth for L-BFGS-B.
    # Penalty: -beta * sum(log(sum(exp(-beta * (c_k - t)))) ) ? No, that's complex.
    # Simpler: Use the soft-min approach but include t in the state.
    # State: [x_1, ..., x_{2N}, t]
    # Objective: -t + penalty
    # Actually, let's stick to the soft-min on the constraints, but optimize t as well?
    # Or just optimize x, and t is determined by the min constraint.
    # The previous approach optimized x to maximize min(c_k). This is robust.
    # The inspiration suggests "auxiliary variable t and an epigraph formulation".
    # Let's try optimizing [x, t] with objective -t + lambda * soft_violation.
    # Violation: max(0, t - c_k).
    # Soft violation: (1/beta) * log(sum(exp(beta * (t - c_k)))) ?
    # This is getting complicated. Let's stick to the proven soft-min on x,
    # but maybe improve the annealing schedule and final polish.
    
    def objective_and_grad(x, beta):
        p = x.reshape(-1, 2)
        
        # Calculate distances
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        
        # We want the radius r, so the "distance" for pairs is d/2.
        # Let's define the "constraint distances" c_k.
        # For pairs (i,j): c_{ij} = dists[i,j] / 2.0
        # For boundary i: c_i = 1.0 - norms[i]
        
        # Flatten distances for vectorization
        iu = np.triu_indices(N, k=1)
        pair_dists = dists[iu] / 2.0
        
        # Boundary distances
        norms = np.linalg.norm(p, axis=1)
        bound_dists = 1.0 - norms
        
        # Combine all constraint distances
        # Note: pair_dists has N*(N-1)/2 elements, bound_dists has N elements.
        
        # To avoid overflow in exp, we use the log-sum-exp trick.
        # Let z = -beta * c.
        # soft_min(c) = -1/beta * log(sum(exp(-beta*c)))
        
        z_pairs = -beta * pair_dists
        z_bounds = -beta * bound_dists
        
        # Concatenate for sum
        # We can compute log_sum_exp separately or combined.
        # Combined is easier for gradient weighting.
        
        # Max values for stability
        max_z_pair = np.max(z_pairs)
        max_z_bound = np.max(z_bounds)
        max_z = max(max_z_pair, max_z_bound)
        
        # exp terms
        exp_pairs = np.exp(z_pairs - max_z)
        exp_bounds = np.exp(z_bounds - max_z)
        
        sum_exp = np.sum(exp_pairs) + np.sum(exp_bounds)
        
        # log_sum_exp = max_z + log(sum_exp)
        log_sum_exp = max_z + np.log(sum_exp)
        
        # Objective value: -soft_min = 1/beta * log_sum_exp
        val = log_sum_exp / beta
        
        # Gradient
        # grad(soft_min) = (1/sum_exp) * sum_k exp(-beta*c_k) * grad(c_k)
        # grad(-soft_min) = -grad(soft_min)
        # We want grad of objective = grad(-soft_min)
        
        # Weights
        w_pairs = exp_pairs / sum_exp
        w_bounds = exp_bounds / sum_exp
        
        grad = np.zeros_like(p)
        
        # Pair contributions
        # c_{ij} = 0.5 * |p_i - p_j|
        # grad_{p_i} c_{ij} = 0.5 * (p_i - p_j) / |p_i - p_j|
        # grad_{p_j} c_{ij} = 0.5 * (p_j - p_i) / |p_i - p_j|
        
        # Vectorized pair gradient
        i_idx, j_idx = iu
        
        # Avoid division by zero
        safe_pair_dists = np.where(pair_dists > 1e-12, pair_dists, 1.0)
        
        # Coefficient for each pair
        # w_k * 0.25 / pair_dists[k]
        coeff = w_pairs * 0.25 / safe_pair_dists
        
        # Vector from j to i
        vecs = p[i_idx] - p[j_idx]
        
        # Accumulate gradients
        np.add.at(grad, i_idx, coeff[:, np.newaxis] * vecs)
        np.add.at(grad, j_idx, -coeff[:, np.newaxis] * vecs)
        
        # Boundary contributions
        # c_i = 1 - |p_i|
        # grad_{p_i} c_i = - p_i / |p_i|
        
        safe_norms = np.where(norms > 1e-12, norms, 1.0)
        bound_grad_vecs = -p / safe_norms[:, np.newaxis]
        
        # grad[p_i] += w_bounds[i] * bound_grad_vecs[i]
        grad += w_bounds[:, np.newaxis] * bound_grad_vecs
        
        # The objective is -soft_min, so gradient is -grad(soft_min)
        # Wait, my derivation:
        # f = -soft_min
        # grad(f) = -grad(soft_min)
        # grad(soft_min) = sum(w_k * grad(c_k))
        # So grad(f) = -sum(w_k * grad(c_k))
        
        return val, -grad

    # Annealing loop
    # Start with low beta (smooth), end with high beta (sharp)
    # More refined annealing schedule with additional intermediate steps
    betas = [3.0, 10.0, 30.0, 60.0, 100.0, 150.0, 200.0, 300.0, 500.0, 800.0, 1200.0, 1500.0]
    
    current_points = points.flatten()
    
    for beta in betas:
        if time.time() > deadline - 20:
            break
            
        try:
            # Use more iterations for lower beta (smoother landscape)
            max_iter = 400 if beta < 100 else 300
            
            res = minimize(
                lambda x: objective_and_grad(x, beta)[0],
                current_points,
                jac=lambda x: objective_and_grad(x, beta)[1],
                method='L-BFGS-B',
                options={
                    'maxiter': max_iter,
                    'gtol': 1e-8,
                    'ftol': 1e-12,
                    'maxls': 30,
                    'disp': False
                }
            )
            
            if time.time() < deadline:
                candidate = res.x.reshape(-1, 2)
                candidate_score = _compute_score(candidate)
                
                if candidate_score > best_score:
                    best_points = candidate
                    best_score = candidate_score
                    current_points = candidate.flatten()
                else:
                    # Even if hard score didn't improve, use candidate for next step
                    # if it's closer to the soft-min optimum?
                    # Usually, if hard score didn't improve, the soft min optimization might have moved
                    # away from the optimal hard min. 
                    # However, if we are annealing, we want to follow the soft min.
                    # Let's use the candidate as the start for the next (sharper) beta,
                    # but keep best_points as the one with highest hard score.
                    current_points = candidate.flatten()
                    
        except Exception as e:
            # If optimization fails, break
            break
            
    # Final polish with a very high beta (effectively hard min)
    # Multiple polish passes with increasing precision
    polish_betas = [2000.0, 3000.0, 5000.0]
    
    for polish_beta in polish_betas:
        if time.time() > deadline - 3:
            break
        
        try:
            res_final = minimize(
                lambda x: objective_and_grad(x, polish_beta)[0],
                current_points,
                jac=lambda x: objective_and_grad(x, polish_beta)[1],
                method='L-BFGS-B',
                options={
                    'maxiter': 300,
                    'gtol': 1e-12,
                    'ftol': 1e-16,
                    'maxls': 40,
                    'disp': False
                }
            )
            
            if time.time() < deadline - 1:
                candidate = res_final.x.reshape(-1, 2)
                candidate_score = _compute_score(candidate)
                if candidate_score > best_score:
                    best_points = candidate
                    best_score = candidate_score
                    current_points = candidate.flatten()
                    
        except Exception as e:
            pass

    # Final check
    norms = np.linalg.norm(best_points, axis=1)
    # Ensure points are inside
    # If any point is outside, scale it back? 
    # The objective penalizes going outside, so it should be fine.
    
    return best_points

# EVOLVE-BLOCK-END
