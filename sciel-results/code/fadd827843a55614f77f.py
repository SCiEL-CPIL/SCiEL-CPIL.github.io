"""Improved packing: Physical repulsion search with multi-start and radius expansion."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START

def _score_raw(centers):
    """Compute the actual score (minimum radius) for a set of centers."""
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    
    # Distance from origin
    norms = np.sqrt(np.sum(centers**2, axis=1))
    # Check all inside unit circle
    if np.any(norms >= 1.0):
        return 0.0
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    min_pair_dist = np.min(dists)
    min_edge_dist = np.min(1.0 - norms)
    
    r = min(min_pair_dist / 2.0, min_edge_dist)
    return max(r, 0.0)

def _is_feasible(centers, r, eps=1e-9):
    """Check if centers are valid for radius r."""
    if not np.all(np.isfinite(centers)):
        return False
    norms = np.sqrt(np.sum(centers**2, axis=1))
    if np.any(norms + r > 1.0 + eps):
        return False
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    if np.any(dists < 2.0 * r - eps):
        return False
    return True

def _relax(centers, r, max_iter=500, dt=0.05):
    """
    Relax the configuration for a given radius r using repulsive forces.
    Returns the relaxed centers and a measure of the minimum distance achieved.
    """
    centers = centers.copy()
    # We want to maximize the minimum distance. 
    # We simulate forces that push circles apart if they are closer than 2*r.
    
    for _ in range(max_iter):
        forces = np.zeros_like(centers)
        
        # Boundary forces: push inwards if too close to edge
        norms = np.sqrt(np.sum(centers**2, axis=1))
        # Ideal distance from center is 1 - r. If norm > 1 - r, push in.
        # Use a soft constraint: force proportional to violation
        violations_boundary = np.maximum(0, norms - (1.0 - r))
        # Direction is towards origin: -centers / norms
        dir_boundary = -centers / np.where(norms > 0, norms, 1.0)[:, np.newaxis]
        forces += violations_boundary[:, np.newaxis] * dir_boundary * 10.0
        
        # Pairwise forces: push apart if distance < 2*r
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Overlap violation: max(0, 2*r - dist)
        violations_pair = np.maximum(0, 2.0 * r - dists)
        # Direction: from j to i is (c_i - c_j)/dist. We want to push i away from j.
        # Force on i from j: (c_i - c_j)/dist * violation
        # Sum over j
        # To avoid division by zero, use dist where dist > 0
        safe_dists = np.where(dists > 0, dists, 1.0)
        unit_vecs = diff / safe_dists[:, :, np.newaxis]
        
        # Force on i is sum over j of unit_vec_ij * violation_ij
        # Note: unit_vec_ij points from j to i. We want to push i further in that direction.
        # So force_i += sum_j (unit_vec_ij * violations_pair_ij)
        
        # Efficient computation:
        # F_i = sum_j ( (c_i - c_j) / dist_ij * violations_ij )
        # We can compute this by broadcasting
        weighted_vecs = unit_vecs * violations_pair[:, :, np.newaxis]
        forces += np.sum(weighted_vecs, axis=1) * 10.0
        
        # Update positions
        centers += forces * dt
        
        # Hard constraint: keep inside unit circle (project back if necessary to prevent explosion)
        norms = np.sqrt(np.sum(centers**2, axis=1))
        max_allowed = 1.0 - r
        mask = norms > max_allowed
        if np.any(mask):
            centers[mask] *= max_allowed / norms[mask, np.newaxis]
            
        # Check convergence: if max force is small, we are at a local minimum for this r
        if np.max(np.abs(forces)) < 1e-4:
            break
            
    return centers

def _try_expand_radius(centers, r_start, r_end, max_iter_per_step=200, steps=20):
    """
    Try to increase radius from r_start to r_end by relaxing at intermediate steps.
    Returns the best feasible centers and the max radius achieved.
    """
    current_centers = centers.copy()
    current_r = r_start
    
    # Binary search or linear expansion? Linear is often more stable for relaxation.
    # Let's do a coarse linear expansion first, then binary refine.
    
    # Coarse expansion
    r_steps = np.linspace(r_start, r_end, steps)
    for r in r_steps:
        current_centers = _relax(current_centers, r, max_iter=max_iter_per_step, dt=0.05)
        # Check feasibility
        if not _is_feasible(current_centers, r, eps=1e-6):
            # If it fails, we might have overshot or fallen into a bad local minimum.
            # Try to recover by relaxing at a slightly smaller r?
            # For now, just break and try to refine around the last good r.
            # Actually, if it's infeasible, the score will be lower. 
            # We want to find the largest r where it IS feasible.
            pass
    
    # Refine around the end
    # Let's try to push a bit further
    final_centers = _relax(current_centers, r_end, max_iter=500, dt=0.01)
    
    return final_centers

def _generate_hexagonal_initial(r_init):
    """Generate a hexagonal lattice packing with given circle radius."""
    points = []
    spacing = 2 * r_init
    # Generate hexagonal lattice points
    for j in range(-15, 16):
        for i in range(-15, 16):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            r = np.sqrt(x*x + y*y)
            if r <= 1.0 - r_init - 1e-9:
                points.append([x, y])
    
    points = np.array(points)
    if len(points) < N:
        return None
    
    # Sort by distance from origin and take N
    norms = np.sqrt(np.sum(points**2, axis=1))
    idx = np.argsort(norms, kind='stable')[:N]
    return points[idx]

def _generate_ring_initial():
    """Generate a ring-based initial configuration."""
    centers = []
    centers.append([0.0, 0.0])
    
    # Rings
    ring_radii = [0.25, 0.5, 0.75, 0.95]
    counts = [6, 12, 18, 13]
    
    for r, count in zip(ring_radii, counts):
        for k in range(count):
            theta = 2 * np.pi * k / count
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            centers.append([x, y])
    
    centers = np.array(centers[:N])
    return centers

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    best_score = 0.0
    best_centers = None
    
    # Initial configurations
    initial_configs = []
    
    # 1. Hexagonal
    for r_init in [0.12, 0.13, 0.14]:
        cfg = _generate_hexagonal_initial(r_init)
        if cfg is not None:
            initial_configs.append(cfg)
            
    # 2. Rings
    ring_cfg = _generate_ring_initial()
    if ring_cfg is not None:
        initial_configs.append(ring_cfg)
        
    # 3. Random
    np.random.seed(42)
    for _ in range(5):
        cfg = np.random.uniform(-0.9, 0.9, size=(N, 2))
        norms = np.sqrt(np.sum(cfg**2, axis=1))
        mask = norms > 0.9
        cfg[mask] *= 0.9 / norms[mask, np.newaxis]
        initial_configs.append(cfg)
        
    # Optimize each configuration
    for idx, x0 in enumerate(initial_configs):
        if time.time() > deadline - 10.0:
            break
            
        # Try to expand radius from a small value to a large value
        # Start with a small radius to ensure feasibility, then grow
        r_start = 0.05
        r_end = 0.16 # Known best is ~0.1258, try to go higher
        
        # Step 1: Relax at a small radius to untangle
        relaxed = _relax(x0, r_start, max_iter=100, dt=0.1)
        
        # Step 2: Gradually increase radius
        # Use a few steps
        r_steps = np.linspace(r_start, r_end, 10)
        current_centers = relaxed
        for r in r_steps:
            if time.time() > deadline - 5.0:
                break
            current_centers = _relax(current_centers, r, max_iter=100, dt=0.05)
            
        # Step 3: Final fine-tuning with L-BFGS-B on the soft-min objective
        # This helps polish the local minimum found by the physical relaxation
        if time.time() < deadline - 2.0:
            def obj(centers_flat):
                centers = centers_flat.reshape(N, 2)
                norms = np.sqrt(np.sum(centers**2, axis=1))
                edge_constraints = 1.0 - norms
                diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
                dists = np.sqrt(np.sum(diff**2, axis=2))
                iu = np.triu_indices(N, 1)
                pair_constraints = dists[iu] / 2.0
                all_constraints = np.concatenate([edge_constraints, pair_constraints])
                # Soft min with high beta
                beta = 500.0
                neg_beta_x = -beta * all_constraints
                max_val = np.max(neg_beta_x)
                log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_x - max_val)))
                soft_min_val = -log_sum_exp / beta
                return -soft_min_val
                
            try:
                result = minimize(
                    obj,
                    current_centers.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                if result.success or result.status == 0:
                    centers = result.x.reshape(N, 2)
                    score = _score_raw(centers)
                    if score > best_score:
                        best_score = score
                        best_centers = centers
            except Exception:
                pass
        
        # Also check the relaxed version directly in case L-BFGS-B fails or hurts
        score_relaxed = _score_raw(current_centers)
        if score_relaxed > best_score:
            best_score = score_relaxed
            best_centers = current_centers

    # Final polish on the best solution found
    if best_centers is not None and time.time() < deadline - 2.0:
        # Try a few random perturbations and re-optimize
        for _ in range(3):
            if time.time() > deadline - 2.0:
                break
            perturbed = best_centers + np.random.normal(0, 0.01, size=(N, 2))
            # Project inside
            norms = np.sqrt(np.sum(perturbed**2, axis=1))
            mask = norms > 0.98
            perturbed[mask] *= 0.98 / norms[mask, np.newaxis]
            
            # Quick relaxation
            relaxed_p = _relax(perturbed, best_score, max_iter=50, dt=0.05)
            
            # L-BFGS polish
            def obj(centers_flat):
                centers = centers_flat.reshape(N, 2)
                norms = np.sqrt(np.sum(centers**2, axis=1))
                edge_constraints = 1.0 - norms
                diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
                dists = np.sqrt(np.sum(diff**2, axis=2))
                iu = np.triu_indices(N, 1)
                pair_constraints = dists[iu] / 2.0
                all_constraints = np.concatenate([edge_constraints, pair_constraints])
                beta = 500.0
                neg_beta_x = -beta * all_constraints
                max_val = np.max(neg_beta_x)
                log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_x - max_val)))
                soft_min_val = -log_sum_exp / beta
                return -soft_min_val

            try:
                result = minimize(
                    obj,
                    relaxed_p.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                if result.success or result.status == 0:
                    centers = result.x.reshape(N, 2)
                    score = _score_raw(centers)
                    if score > best_score:
                        best_score = score
                        best_centers = centers
            except Exception:
                pass

    # Final validation
    if best_centers is not None:
        final_score = _score_raw(best_centers)
        if final_score > 0.0 and np.all(np.isfinite(best_centers)):
            return best_centers
    
    # Fallback
    fallback = _generate_hexagonal_initial(0.12)
    if fallback is not None:
        return fallback
    
    return np.random.uniform(-0.5, 0.5, size=(N, 2))

# EVOLVE-BLOCK-END
