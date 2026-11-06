import numpy as np
import time
from scipy.optimize import minimize

N = 50

def compute_radius(points):
    """
    Computes the maximum radius for the given points.
    r = min(min_pairwise_dist / 2, min_dist_to_boundary)
    """
    norms = np.linalg.norm(points, axis=1)
    min_boundary = np.min(1.0 - norms)
    
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    return min(min_boundary, min_pair)

def objective_and_grad(x):
    """
    Vectorized objective and gradient for circle packing.
    Minimizes sum of inverse squared distances (pairwise repulsion)
    plus boundary penalty.
    """
    points = x.reshape(N, 2)
    
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    
    # Safe distance squared (avoid division by zero)
    dists_sq_safe = np.maximum(dists_sq, 1e-10)
    
    # Upper triangle indices
    iu = np.triu_indices(N, k=1)
    
    # Objective: sum 1/d^2 for pairs
    e_pair = np.sum(1.0 / dists_sq_safe[iu])
    
    # Boundary penalty: penalize points outside unit circle
    norms = np.linalg.norm(points, axis=1)
    outside = np.maximum(0, norms - 1.0)
    e_bound = 1000.0 * np.sum(outside**2)
    
    total_obj = e_pair + e_bound
    
    # Gradient computation (vectorized)
    grad = np.zeros_like(points)
    
    # Pairwise gradient: d/dx_i sum_{j!=i} 1/||x_i - x_j||^2
    # = sum_{j!=i} -2*(x_i - x_j)/||x_i - x_j||^4
    # F[i,j] = -2 * diff[i,j] / dists_sq[i,j]^2
    inv_d4 = 1.0 / (dists_sq_safe**2)
    F = -2.0 * diff * inv_d4[:, :, np.newaxis]
    
    # Mask diagonal (no self-interaction)
    mask = ~np.eye(N, dtype=bool)
    F = F * mask[:, :, np.newaxis]
    
    # Sum over j for each i
    grad_pair = np.sum(F, axis=1)
    grad += grad_pair
    
    # Boundary gradient: d/dx_i 1000 * max(0, r_i - 1)^2
    # = 1000 * 2 * (r_i - 1) * x_i / r_i for r_i > 1
    norms_safe = np.maximum(norms, 1e-8)
    bound_grad_factor = 2000.0 * np.where(norms > 1.0, (norms - 1.0) / norms_safe, 0.0)
    grad += bound_grad_factor[:, np.newaxis] * points
    
    return total_obj, grad.flatten()

def seed_hexagonal():
    """Generate hexagonal lattice seed."""
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    for _ in range(20):
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        mask = radius <= 1.0 - spacing / 2.0
        points, radius = points[mask], radius[mask]
        if len(points) >= N:
            return points[np.argsort(radius, kind="stable")[:N]]
        spacing *= 0.95
    # Fallback: random
    angles = np.random.uniform(0, 2 * np.pi, N)
    radii = np.sqrt(np.random.uniform(0, 0.9, N))
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

def seed_rings():
    """Generate concentric ring seed."""
    ring_radii = [0, 0.3, 0.55, 0.8, 0.95]
    ring_counts = [1, 6, 12, 18, 13]
    pts = []
    for r, count in zip(ring_radii, ring_counts):
        if count == 0:
            continue
        angles = np.linspace(0, 2*np.pi, count, endpoint=False)
        for a in angles:
            pts.append([r * np.cos(a), r * np.sin(a)])
    return np.array(pts[:N])

def seed_random(seed_val):
    """Generate random seed."""
    np.random.seed(seed_val)
    angles = np.random.uniform(0, 2 * np.pi, N)
    radii = np.sqrt(np.random.uniform(0, 0.95, N))
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

def optimize_from_seed(seed_points, deadline):
    """Run L-BFGS-B optimization from a given seed."""
    x0 = seed_points.flatten()
    
    result = minimize(
        objective_and_grad,
        x0,
        method='L-BFGS-B',
        jac=True,
        options={'maxiter': 500, 'gtol': 1e-8, 'ftol': 1e-10}
    )
    
    if result.success or result.fun < 1e6:
        opt_points = result.x.reshape(N, 2)
        # Ensure points are inside the circle
        norms = np.linalg.norm(opt_points, axis=1)
        mask_outside = norms > 1.0
        if np.any(mask_outside):
            opt_points[mask_outside] *= 0.99 / norms[mask_outside, np.newaxis]
        return opt_points
    return seed_points

def local_refinement(points, deadline):
    """Fine-tune with small perturbations and re-optimization."""
    best = points.copy()
    best_score = compute_radius(best)
    
    remaining = deadline - time.time()
    if remaining < 5:
        return best
    
    # Try several perturbation strengths
    for perturb_scale in [0.01, 0.005, 0.002]:
        if time.time() > deadline - 5:
            break
        
        np.random.seed(int(time.time() * 1000) % 100000)
        perturbed = best + np.random.normal(0, perturb_scale, (N, 2))
        # Keep inside circle
        norms = np.linalg.norm(perturbed, axis=1)
        mask = norms > 1.0
        perturbed[mask] *= 0.99 / norms[mask, np.newaxis]
        
        refined = optimize_from_seed(perturbed, deadline)
        score = compute_radius(refined)
        
        if score > best_score:
            best = refined
            best_score = score
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s kill
    
    np.random.seed(42)
    
    # Generate diverse seeds
    seeds = [
        seed_hexagonal(),
        seed_rings(),
        seed_random(1),
        seed_random(2),
        seed_random(3),
    ]
    
    best_points = None
    best_score = -1.0
    
    # Optimize from each seed
    for i, seed in enumerate(seeds):
        if time.time() > deadline - 10:
            break
        
        opt_points = optimize_from_seed(seed, deadline)
        score = compute_radius(opt_points)
        
        if score > best_score:
            best_score = score
            best_points = opt_points
    
    # Local refinement on best solution
    if best_points is not None and time.time() < deadline - 15:
        best_points = local_refinement(best_points, deadline)
        best_score = compute_radius(best_points)
    
    # If we have time, try more random seeds
    if time.time() < deadline - 20:
        for seed_val in range(4, 10):
            if time.time() > deadline - 5:
                break
            seed = seed_random(seed_val)
            opt_points = optimize_from_seed(seed, deadline)
            score = compute_radius(opt_points)
            if score > best_score:
                best_score = score
                best_points = opt_points
    
    # Final refinement if time permits
    if best_points is not None and time.time() < deadline - 5:
        best_points = local_refinement(best_points, deadline)
    
    return best_points
