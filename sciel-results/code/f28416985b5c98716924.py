"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _score(c):
    """Compute the packing score: min(min_pair_dist/2, 1 - max_radius)."""
    c = c.reshape(N, 2)
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    radii = np.hypot(c[:, 0], c[:, 1])
    min_r = min(min_dist / 2.0, 1.0 - np.max(radii))
    return min_r

def _physics_energy(c, k_rep=10.0, k_wall=50.0, wall_margin=0.05):
    """
    Physics-based energy: repulsive inverse-power potential + soft wall.
    c: (N, 2) array of centers.
    Returns (energy, gradient).
    """
    c = c.reshape(N, 2)
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=-1)
    np.fill_diagonal(dists_sq, np.inf)
    
    # Repulsive force: F ~ 1/d^3 (potential ~ 1/d^2)
    # Use softened distance to avoid singularity
    eps = 1e-6
    dists = np.sqrt(dists_sq + eps)
    
    # Pairwise repulsion energy: sum_{i<j} k_rep / dist^2
    # Only compute upper triangle
    triu_idx = np.triu_indices(N, k=1)
    pair_dists_sq = dists_sq[triu_idx]
    pair_dists = np.sqrt(pair_dists_sq + eps)
    
    energy_rep = np.sum(k_rep / (pair_dists**2))
    
    # Wall potential: for each point, if r > (1 - wall_margin), penalty
    radii = np.hypot(c[:, 0], c[:, 1])
    wall_violation = np.maximum(0, radii - (1.0 - wall_margin))
    energy_wall = np.sum(k_wall * wall_violation**2)
    
    total_energy = energy_rep + energy_wall
    
    # Gradient computation
    grad = np.zeros_like(c)
    
    # Repulsive gradient: d/dx_i (k_rep / |x_i - x_j|^2) = -2*k_rep*(x_i - x_j)/|x_i - x_j|^4
    # For each pair (i,j), force on i is -2*k_rep*(x_i-x_j)/dist^4
    pair_diff = c[triu_idx[0]] - c[triu_idx[1]]
    pair_dists4 = (pair_dists**4)
    forces = -2 * k_rep * pair_diff / pair_dists4[:, np.newaxis]
    
    # Accumulate forces
    for idx in range(len(triu_idx[0])):
        i = triu_idx[0][idx]
        j = triu_idx[1][idx]
        grad[i] += forces[idx]
        grad[j] -= forces[idx]
    
    # Wall gradient: d/dx (k_wall * max(0, r - (1-margin))^2)
    # = 2 * k_wall * max(0, r - (1-margin)) * (x/r) for r > (1-margin)
    wall_active = wall_violation > 0
    if np.any(wall_active):
        r_active = radii[wall_active]
        c_active = c[wall_active]
        grad_wall = 2 * k_wall * wall_violation[wall_active, np.newaxis] * (c_active / r_active[:, np.newaxis])
        grad[wall_active] += grad_wall
    
    return total_energy, grad

def _soft_min_objective(c, beta=0.01):
    """Smooth approximation of negative minimum distance for L-BFGS polish."""
    c = c.reshape(N, 2)
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    D = np.sum(diff**2, axis=-1)
    np.fill_diagonal(D, np.inf)
    
    D_valid = D[D < np.inf].ravel()
    D_max = np.max(D_valid)
    exp_terms = np.exp(-(D - D_max) / beta)
    np.fill_diagonal(exp_terms, 0)
    sum_exp = np.sum(exp_terms)
    smooth_min_D = D_max - beta * np.log(sum_exp + 1e-300)
    
    # Also add wall constraint softly
    radii = np.hypot(c[:, 0], c[:, 1])
    wall_pen = np.sum(np.maximum(0, radii - 1.0)**2)
    
    f_val = -smooth_min_D + 100 * wall_pen
    
    # Gradient
    grad = np.zeros_like(c)
    w = exp_terms / sum_exp
    for k in range(N):
        for j in range(N):
            if k != j:
                grad[k] += -2 * w[k, j] * (c[k] - c[j])
    
    # Wall gradient
    wall_active = radii > 1.0
    if np.any(wall_active):
        grad[wall_active] += 200 * (radii[wall_active] - 1.0)[:, np.newaxis] * c[wall_active] / radii[wall_active, np.newaxis]
    
    return f_val, grad

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_score = -np.inf
    best_centers = None
    
    # Initial guess: concentric rings
    def _concentric_rings(n_circles):
        """Generate initial positions on concentric rings."""
        points = [np.array([0.0, 0.0])]
        remaining = n_circles - 1
        ring_radius = 0.3
        while remaining > 0 and ring_radius < 0.95:
            # How many fit on this ring?
            circumference = 2 * np.pi * ring_radius
            spacing = 2 * 0.12  # approximate diameter
            n_on_ring = max(1, int(circumference / spacing))
            n_on_ring = min(n_on_ring, remaining)
            
            angles = np.linspace(0, 2 * np.pi, n_on_ring, endpoint=False)
            ring_pts = np.column_stack([ring_radius * np.cos(angles), ring_radius * np.sin(angles)])
            points.extend(ring_pts)
            remaining -= n_on_ring
            ring_radius += 0.15
        
        # Pad if needed
        while len(points) < n_circles:
            points.append(np.array([0.0, 0.0]))
        
        return np.array(points[:n_circles])
    
    # Physics-based relaxation
    c = _concentric_rings(N)
    
    # Adaptive learning rate physics relaxation
    lr = 0.01
    for step in range(5000):
        if time.time() > deadline - 30:
            break
        
        energy, grad = _physics_energy(c, k_rep=5.0, k_wall=30.0, wall_margin=0.03)
        
        # Adaptive step size
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-8:
            c = c - lr * grad / grad_norm * 0.01
        
        # Keep inside unit circle
        radii = np.hypot(c[:, 0], c[:, 1])
        mask = radii > 0.98
        if np.any(mask):
            c[mask] *= 0.98 / radii[mask, np.newaxis]
        
        # Check for improvement
        current_score = _score(c)
        if current_score > best_score:
            best_score = current_score
            best_centers = c.copy()
        
        # Decay learning rate
        lr *= 0.999
    
    # Polish with L-BFGS-B on soft-min objective
    if best_centers is not None:
        c0 = best_centers.copy()
        try:
            result = minimize(
                _soft_min_objective,
                c0.ravel(),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            c_opt = result.x.reshape(N, 2)
            
            # Ensure inside unit circle
            radii = np.hypot(c_opt[:, 0], c_opt[:, 1])
            mask = radii > 0.999
            if np.any(mask):
                c_opt[mask] *= 0.999 / radii[mask, np.newaxis]
            
            opt_score = _score(c_opt)
            if opt_score > best_score:
                best_score = opt_score
                best_centers = c_opt.copy()
        except Exception:
            pass
    
    # Multi-start with random perturbations if time remains
    if time.time() < deadline - 20:
        for restart in range(5):
            if time.time() > deadline - 5:
                break
            
            c0 = best_centers + np.random.normal(0, 0.05, (N, 2))
            radii = np.hypot(c0[:, 0], c0[:, 1])
            mask = radii > 0.95
            c0[mask] *= 0.95 / radii[mask, np.newaxis]
            
            # Short physics relaxation
            for step in range(2000):
                if time.time() > deadline - 5:
                    break
                energy, grad = _physics_energy(c0, k_rep=5.0, k_wall=30.0, wall_margin=0.03)
                grad_norm = np.linalg.norm(grad)
                if grad_norm > 1e-8:
                    c0 = c0 - 0.01 * grad / grad_norm
                radii = np.hypot(c0[:, 0], c0[:, 1])
                mask = radii > 0.98
                if np.any(mask):
                    c0[mask] *= 0.98 / radii[mask, np.newaxis]
            
            current_score = _score(c0)
            if current_score > best_score:
                best_score = current_score
                best_centers = c0.copy()
    
    return best_centers
# EVOLVE-BLOCK-END
