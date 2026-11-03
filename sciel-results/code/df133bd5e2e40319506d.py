"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_score(P):
    """Calculate the radius score for a given set of points."""
    if P.shape != (N, 2) or not np.all(np.isfinite(P)):
        return -np.inf
    
    d_bound = 1.0 - np.linalg.norm(P, axis=1)
    r_bound = np.min(d_bound)
    
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair_dist = np.min(dist)
    r_pair = min_pair_dist / 2.0
    
    return min(r_pair, r_bound)

def _get_ring_initialization(n_rings=1):
    """Generate ring-based initialization matching known optimal topology."""
    # Known best packing has 50 circles in a single dense outer ring
    # with some inner structure
    if n_rings == 1:
        # Single ring of 50 circles
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        # Optimal radius for single ring of 50 circles
        # r_ring ≈ sin(π/N) / (1 + sin(π/N)) ≈ 0.0631 for N=50
        # But we need to account for circle radius
        # For N circles in a ring: ring_radius = r / sin(π/N)
        # Constraint: ring_radius + r <= 1
        # So r = sin(π/N) / (1 + sin(π/N))
        r_est = np.sin(np.pi / N) / (1 + np.sin(np.pi / N))
        ring_radius = r_est / np.sin(np.pi / N)
        
        P = np.column_stack([ring_radius * np.cos(angles), ring_radius * np.sin(angles)])
        return P
    
    # Multi-ring initialization (fallback)
    # Ring 1: outer ring with ~30 circles
    # Ring 2: inner ring with ~20 circles
    n_outer = 30
    n_inner = N - n_outer
    
    angles_outer = np.linspace(0, 2*np.pi, n_outer, endpoint=False)
    angles_inner = np.linspace(0, 2*np.pi, n_inner, endpoint=False) + np.pi/n_inner
    
    r_est = np.sin(np.pi / n_outer) / (1 + np.sin(np.pi / n_outer))
    ring_radius_outer = r_est / np.sin(np.pi / n_outer)
    ring_radius_inner = ring_radius_outer * 0.5
    
    P_outer = np.column_stack([ring_radius_outer * np.cos(angles_outer), 
                               ring_radius_outer * np.sin(angles_outer)])
    P_inner = np.column_stack([ring_radius_inner * np.cos(angles_inner), 
                               ring_radius_inner * np.sin(angles_inner)])
    
    return np.vstack([P_outer, P_inner])

def _energy_smooth(P, r, beta=10.0):
    """Smooth energy using soft-min for better gradient behavior."""
    # Circle-circle: soft-min of distances
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    dist = np.sqrt(dist_sq)
    
    # Soft-min of (2r - dist) to handle overlaps smoothly
    # Use log-sum-exp trick for soft-min
    x = 2.0 * r - dist
    x_clipped = np.clip(x, -10, 10)
    # Soft-min approximation: -log(sum(exp(-x/beta))) * beta
    # But we want sum of positive parts, so use soft-plus
    soft_overlaps_cc = np.sum(np.maximum(x, 0)**2 + 1e-10)
    
    # Circle-boundary
    dist_origin = np.linalg.norm(P, axis=1)
    overlaps_cb = np.maximum(dist_origin + r - 1.0, 0.0)
    soft_overlaps_cb = np.sum(overlaps_cb**2 + 1e-10)
    
    return soft_overlaps_cc + soft_overlaps_cb

def _optimize_positions(P_init, r_init, deadline):
    """Optimize positions using scipy L-BFGS-B."""
    P_flat = P_init.flatten()
    
    def objective(x):
        if time.time() > deadline:
            return 1e10
        P = x.reshape(N, 2)
        # Penalize invalid configurations
        if not np.all(np.isfinite(P)):
            return 1e10
        score = _compute_score(P)
        if score < 0:
            return 1e10
        # We want to maximize score, so minimize negative
        # Add penalty for boundary violations
        dist_origin = np.linalg.norm(P, axis=1)
        boundary_penalty = np.sum(np.maximum(dist_origin - (1.0 - 0.01), 0)**2)
        return -score + 0.1 * boundary_penalty
    
    # Bounds: keep points within unit circle with margin
    bounds = [(-1.0, 1.0)] * (2 * N)
    
    try:
        result = minimize(objective, P_flat, method='L-BFGS-B', 
                         bounds=bounds, 
                         options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
        P_opt = result.x.reshape(N, 2)
        score_opt = _compute_score(P_opt)
        return P_opt, score_opt
    except Exception:
        return P_init, r_init

def _local_refine(P, r, max_iter=500, tol=1e-12):
    """High-precision local refinement."""
    P_curr = P.copy()
    r_curr = r
    
    for it in range(max_iter):
        # Compute forces from overlaps
        diff = P_curr[:, np.newaxis, :] - P_curr[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        dist = np.sqrt(dist_sq)
        
        forces = np.zeros_like(P_curr)
        
        # Circle-circle repulsion
        overlap_cc = np.maximum(2.0 * r_curr - dist, 0.0)
        mask_cc = overlap_cc > 0
        if np.any(mask_cc):
            inv_dist = np.zeros_like(dist)
            np.divide(1.0, dist, out=inv_dist, where=dist > 1e-10)
            weight = np.zeros((N, N))
            weight[mask_cc] = overlap_cc[mask_cc] * inv_dist[mask_cc]
            sum_w = np.sum(weight, axis=1)
            term1 = P_curr * sum_w[:, np.newaxis]
            term2 = (P_curr.T @ weight).T
            forces += 2.0 * (term1 - term2)
        
        # Boundary attraction
        dist_origin = np.linalg.norm(P_curr, axis=1)
        overlap_cb = np.maximum(dist_origin + r_curr - 1.0, 0.0)
        mask_cb = overlap_cb > 0
        if np.any(mask_cb):
            unit_vec = P_curr[mask_cb] / dist_origin[mask_cb, np.newaxis]
            forces[mask_cb] -= 2.0 * unit_vec * overlap_cb[mask_cb]
        
        # Adaptive step size
        force_norm = np.linalg.norm(forces)
        if force_norm < 1e-15:
            break
        step = 0.005 * force_norm / (N * N + 1)
        P_new = P_curr + step * forces
        
        # Check if we can increase r
        new_score = _compute_score(P_new)
        if new_score > r_curr + tol:
            P_curr = P_new
            r_curr = new_score
        else:
            # Try smaller step
            step *= 0.3
            P_new = P_curr + step * forces
            new_score = _compute_score(P_new)
            if new_score > r_curr + tol:
                P_curr = P_new
                r_curr = new_score
            else:
                break
    
    return P_curr, r_curr

def _simulated_annealing(P_init, r_init, start_time, deadline):
    """Simulated annealing to find a good packing."""
    P_curr = P_init.copy()
    r_curr = r_init
    score_best = r_curr
    P_best = P_init.copy()
    
    T_start = 0.03
    T_end = 0.0005
    n_iter = 8000
    
    for it in range(n_iter):
        if time.time() > deadline - 10.0:
            break
        
        # Temperature schedule
        T = T_start * (T_end / T_start) ** (it / n_iter)
        
        # Generate a random move: perturb a random subset of points
        n_perturb = np.random.randint(1, 4)
        idx = np.random.choice(N, n_perturb, replace=False)
        P_trial = P_curr.copy()
        P_trial[idx] += np.random.normal(0, 0.015, (n_perturb, 2))
        
        # Compute energy change
        E_curr = _energy_smooth(P_curr, r_curr)
        E_trial = _energy_smooth(P_trial, r_curr)
        dE = E_trial - E_curr
        
        # Accept if better or with probability exp(-dE/T)
        if dE < 0 or (dE < 5.0 * T and np.random.random() < np.exp(-dE / T)):
            P_curr = P_trial
            # Try to increase r
            new_score = _compute_score(P_curr)
            if new_score > r_curr:
                r_curr = new_score
                if new_score > score_best:
                    score_best = new_score
                    P_best = P_curr.copy()
    
    return P_best, score_best

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    # 1. Ring-based initialization
    P_init = _get_ring_initialization(n_rings=1)
    P_best = P_init.copy()
    score_best = _compute_score(P_best)
    
    # 2. Initial optimization with scipy
    if time.time() < deadline - 60.0:
        P_opt, score_opt = _optimize_positions(P_best, score_best, deadline)
        if score_opt > score_best:
            P_best = P_opt
            score_best = score_opt
    
    # 3. Simulated annealing
    P_sa, score_sa = _simulated_annealing(P_best, score_best, start_time, deadline)
    if score_sa > score_best:
        P_best = P_sa
        score_best = score_sa
    
    # 4. Local refinement
    if time.time() < deadline - 15.0:
        P_ref, score_ref = _local_refine(P_best, score_best, max_iter=300, tol=1e-11)
        if score_ref > score_best:
            P_best = P_ref
            score_best = score_ref
    
    # 5. Final high-precision polish
    if time.time() < deadline - 5.0:
        P_polish, score_polish = _local_refine(P_best, score_best, max_iter=1000, tol=1e-14)
        if score_polish > score_best:
            P_best = P_polish
            score_best = score_polish
    
    return P_best
# EVOLVE-BLOCK-END
