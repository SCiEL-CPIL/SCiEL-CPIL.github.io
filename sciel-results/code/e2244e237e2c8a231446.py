"""Improved program for packing 50 equal circles in a unit circle."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

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

def _energy(P, r):
    """
    Smooth energy function: sum of squared overlaps.
    """
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    overlaps_cc = np.maximum(2.0 * r - dist, 0.0)
    
    dist_origin = np.linalg.norm(P, axis=1)
    overlaps_cb = np.maximum(dist_origin + r - 1.0, 0.0)
    
    return 0.5 * (np.sum(overlaps_cc**2) + np.sum(overlaps_cb**2))

def _energy_grad(P, r):
    """
    Gradient of the energy function.
    """
    grad = np.zeros_like(P)
    
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    V_cc = np.maximum(2.0 * r - dist, 0.0)
    inv_dist = np.zeros_like(dist)
    np.divide(1.0, dist, out=inv_dist, where=dist > 0)
    
    W = V_cc * inv_dist
    
    sum_W = np.sum(W, axis=1) 
    term1 = (P.T @ W).T 
    term2 = P * sum_W[:, np.newaxis] 
    
    grad += term1 - term2
    
    dist_origin = np.linalg.norm(P, axis=1)
    V_cb = np.maximum(r + dist_origin - 1.0, 0.0)
    
    inv_dist_origin = np.zeros_like(dist_origin)
    np.divide(1.0, dist_origin, out=inv_dist_origin, where=dist_origin > 0)
    
    grad += (V_cb * inv_dist_origin)[:, np.newaxis] * P
    
    return grad

def _get_ring_initialization(params, ring_counts):
    """
    Generate points based on concentric rings.
    params: [R1, R2, ..., Rk, offset1, offset2, ..., offsetk]
    ring_counts: list of number of points in each ring.
    """
    P = np.zeros((N, 2))
    k = len(ring_counts)
    
    radii = params[:k]
    offsets = params[k:2*k]
    
    count = 0
    for i, n_pts in enumerate(ring_counts):
        if n_pts == 1:
            P[count] = [0.0, 0.0]
            count += 1
        else:
            R = radii[i]
            offset = offsets[i]
            angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False) + offset
            P[count:count+n_pts, 0] = R * np.cos(angles)
            P[count:count+n_pts, 1] = R * np.sin(angles)
            count += n_pts
            
    return P

def _optimize_positions(P_init, r_target, deadline):
    """
    Optimize positions P to minimize energy for a fixed r_target.
    Returns (P_opt, min_energy).
    """
    def objective(P_flat):
        P = P_flat.reshape(N, 2)
        return _energy(P, r_target)
    
    def grad(P_flat):
        P = P_flat.reshape(N, 2)
        return _energy_grad(P, r_target).flatten()
        
    try:
        res = minimize(objective, P_init.flatten(), jac=grad, method='L-BFGS-B', 
                       options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
        P_opt = res.x.reshape(N, 2)
        min_energy = objective(P_opt)
        return P_opt, min_energy
    except Exception:
        return P_init, objective(P_init.flatten())

def _global_search_ring_params(ring_counts, deadline):
    """
    Use differential evolution to find good ring radii and offsets.
    """
    k = len(ring_counts)
    bounds = [(0.0, 1.0)] * k + [(0.0, 2 * np.pi)] * k
    
    def neg_score(params):
        if time.time() > deadline - 5.0:
            return 0.0
        P = _get_ring_initialization(params, ring_counts)
        score = _compute_score(P)
        return -score
        
    try:
        res = differential_evolution(neg_score, bounds, maxiter=50, popsize=15, 
                                     tol=1e-5, seed=42, polish=True, 
                                     workers=1,
                                     init='latinhypercube')
        
        best_params = res.x
        best_P = _get_ring_initialization(best_params, ring_counts)
        
    except Exception:
        best_P = _get_hex_initial_points()
        
    return best_P

def _get_hex_initial_points():
    """Fallback initialization using hexagonal lattice."""
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        margin = spacing / 2.0 
        points, radius = points[radius <= 1.0 - margin], radius[radius <= 1.0 - margin]
        if len(points) >= N:
            indices = np.argsort(radius, kind="stable")
            return points[indices[:N]]
        spacing *= 0.98

def _local_refine(P_init, r_init, deadline):
    """
    Local refinement using gradient-based optimization with adaptive steps.
    """
    P_curr = P_init.copy()
    r_curr = r_init
    
    # Use scipy's L-BFGS-B for more effective local optimization
    def objective(P_flat):
        P_test = P_flat.reshape(N, 2)
        score = _compute_score(P_test)
        norms = np.linalg.norm(P_test, axis=1)
        outside_penalty = np.sum(np.maximum(norms - 1.0, 0.0)**2)
        return -score + outside_penalty * 100
    
    P_flat = P_curr.flatten()
    
    try:
        result = minimize(objective, P_flat, method='L-BFGS-B', 
                         options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12})
        if result.fun < -r_curr + 1e-10:
            P_opt = result.x.reshape(N, 2)
            score_opt = _compute_score(P_opt)
            if score_opt > r_curr:
                P_curr = P_opt
                r_curr = score_opt
    except:
        pass
    
    # Additional manual refinement pass
    max_iter = 100
    tol = 1e-12
    for it in range(max_iter):
        if time.time() > deadline - 5.0:
            break
            
        diff = P_curr[:, np.newaxis, :] - P_curr[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        forces = np.zeros_like(P_curr)
        
        overlap_cc = np.maximum(2.0 * r_curr - dist, 0.0)
        mask_cc = overlap_cc > 0
        if np.any(mask_cc):
            inv_dist = np.zeros_like(dist)
            np.divide(1.0, dist, out=inv_dist, where=dist > 0)
            weight = np.zeros((N, N))
            weight[mask_cc] = overlap_cc[mask_cc] * inv_dist[mask_cc]
            sum_w = np.sum(weight, axis=1)
            term1 = P_curr * sum_w[:, np.newaxis]
            term2 = (P_curr.T @ weight).T
            forces += 2.0 * (term1 - term2)
        
        dist_origin = np.linalg.norm(P_curr, axis=1)
        overlap_cb = np.maximum(dist_origin + r_curr - 1.0, 0.0)
        mask_cb = overlap_cb > 0
        if np.any(mask_cb):
            unit_vec = P_curr[mask_cb] / dist_origin[mask_cb, np.newaxis]
            forces[mask_cb] -= 2.0 * unit_vec * overlap_cb[mask_cb, np.newaxis]
        
        norm_forces = np.linalg.norm(forces)
        if norm_forces < 1e-15:
            break
        
        best_step = 0
        best_score = r_curr
        for factor in [0.1, 0.05, 0.02, 0.01, 0.005]:
            step = factor * norm_forces / (N * N + 1)
            P_new = P_curr + step * forces
            new_score = _compute_score(P_new)
            if new_score > best_score:
                best_score = new_score
                best_step = step
        
        if best_step > 0:
            P_curr = P_curr + best_step * forces
            r_curr = best_score
        else:
            break
            
    return P_curr, r_curr

def _binary_search_radius(P, r_lo, r_hi, deadline, max_iter=60):
    """
    High-precision polish: binary search on radius with constraint projection.
    """
    best_P = P.copy()
    best_r = r_lo
    
    for it in range(max_iter):
        if time.time() > deadline - 3.0:
            break
        
        r_mid = (r_lo + r_hi) / 2.0
        P_test = P.copy()
        
        # Check boundary constraint
        norms = np.linalg.norm(P_test, axis=1)
        boundary_ok = np.all(norms + r_mid <= 1.0 + 1e-12)
        
        # Check pairwise constraint
        diff = P_test[:, np.newaxis, :] - P_test[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        pairwise_ok = np.all(dist >= 2.0 * r_mid - 1e-12)
        
        if boundary_ok and pairwise_ok:
            r_lo = r_mid
            best_P = P_test.copy()
            best_r = r_mid
        else:
            r_hi = r_mid
            
            # Try to fix by moving violating points
            P_fixed = P_test.copy()
            
            # Fix boundary violations
            norms = np.linalg.norm(P_fixed, axis=1)
            viol_b = norms + r_mid > 1.0
            if np.any(viol_b):
                target_norm = 1.0 - r_mid
                P_fixed[viol_b] *= target_norm / norms[viol_b, np.newaxis]
            
            # Fix pairwise violations iteratively
            for _ in range(10):
                diff = P_fixed[:, np.newaxis, :] - P_fixed[np.newaxis, :, :]
                dist = np.sqrt(np.sum(diff**2, axis=2))
                np.fill_diagonal(dist, np.inf)
                
                violating = dist < 2.0 * r_mid
                if not np.any(violating):
                    break
                
                for i in range(N):
                    for j in range(i+1, N):
                        if violating[i, j]:
                            d = P_fixed[i] - P_fixed[j]
                            norm_d = np.linalg.norm(d)
                            if norm_d < 1e-10:
                                d = np.array([1.0, 0.0])
                                norm_d = 1.0
                            unit_d = d / norm_d
                            gap = (2.0 * r_mid - norm_d) / 2.0
                            P_fixed[i] += unit_d * gap
                            P_fixed[j] -= unit_d * gap
            
            # Re-check if fixed configuration works
            norms = np.linalg.norm(P_fixed, axis=1)
            boundary_ok2 = np.all(norms + r_mid <= 1.0 + 1e-12)
            if boundary_ok2:
                diff2 = P_fixed[:, np.newaxis, :] - P_fixed[np.newaxis, :, :]
                dist2 = np.sqrt(np.sum(diff2**2, axis=2))
                np.fill_diagonal(dist2, np.inf)
                if np.all(dist2 >= 2.0 * r_mid - 1e-12):
                    r_lo = r_mid
                    best_P = P_fixed.copy()
                    best_r = r_mid
                    P = P_fixed
    
    return best_P, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    np.random.seed(42)
    
    # Try multiple ring structures
    ring_structures = [
        [1, 6, 12, 18, 13],
        [1, 5, 10, 15, 19],
        [1, 7, 14, 21, 7],
        [1, 6, 11, 17, 15],
        [1, 8, 16, 24, 1],
        [1, 6, 12, 16, 15],
    ]
    
    best_P = None
    best_score = -np.inf
    
    # Phase 1: Global search over ring parameters for each structure
    for ring_counts in ring_structures:
        if time.time() > deadline - 60.0:
            break
        
        P_ring = _global_search_ring_params(ring_counts, deadline)
        score_ring = _compute_score(P_ring)
        
        # Local refine from ring structure
        P_ref, score_ref = _local_refine(P_ring, score_ring, deadline)
        
        if score_ref > best_score:
            best_score = score_ref
            best_P = P_ref
    
    # Also try hexagonal lattice
    P_hex = _get_hex_initial_points()
    score_hex = _compute_score(P_hex)
    P_hex_ref, score_hex_ref = _local_refine(P_hex, score_hex, deadline)
    
    if score_hex_ref > best_score:
        best_score = score_hex_ref
        best_P = P_hex_ref
    
    # Phase 2: Binary search on radius for high-precision final answer
    if time.time() < deadline - 10.0:
        r_lo = best_score * 0.99
        r_hi = best_score * 1.01
        P_final, r_final = _binary_search_radius(best_P, r_lo, r_hi, deadline, max_iter=60)
        if r_final > best_score:
            best_P = P_final
            best_score = r_final
    
    # Ensure all points are inside the unit circle
    norms = np.linalg.norm(best_P, axis=1)
    mask = norms > 1.0
    if np.any(mask):
        best_P[mask] *= (1.0 - 1e-9) / norms[mask, np.newaxis]
    
    # Final safety check
    final_score = _compute_score(best_P)
    if not np.isfinite(final_score) or final_score <= 0:
        best_P = P_hex
        final_score = score_hex
    
    return best_P
# EVOLVE-BLOCK-END
