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

def _energy(P, r, softness=10.0):
    """Energy: smooth approximation using soft-min for better gradient flow."""
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    
    # Use soft-min approximation: -softness * log(sum(exp(-softness * x)))
    # For circle-circle: want to penalize when dist < 2r
    # Smooth penalty: -softness * log(sum(exp(-softness * (dist - 2r)))) for dist < 2r
    # Simpler: use smooth max(0, 2r - dist) approximation
    
    # Circle-circle: smooth penalty for violations
    # Use tanh-based smooth max: max(0, x) ≈ 0.5*(x + sqrt(x^2 + eps))
    eps = 1e-6
    cc_violation = 2.0 * r - np.sqrt(dist_sq + eps)
    cc_smooth = 0.5 * (cc_violation + np.sqrt(cc_violation**2 + eps))
    cc_smooth = np.where(cc_violation > -0.01, cc_smooth, 0.0)
    
    # Circle-boundary
    dist_origin = np.linalg.norm(P, axis=1)
    cb_violation = dist_origin + r - 1.0
    cb_smooth = 0.5 * (cb_violation + np.sqrt(cb_violation**2 + eps))
    cb_smooth = np.where(cb_violation > -0.01, cb_smooth, 0.0)
    
    return np.sum(cc_smooth**2) + np.sum(cb_smooth**2)

def _get_ring_initialization():
    """Generate a ring-structured initialization."""
    # Use concentric rings: 1 center + 6 + 12 + 18 + 13 = 50
    # Actually let's use: 1 + 6 + 12 + 18 + 13 = 50
    # Or: 1 + 6 + 12 + 18 + 13
    # Better: 1 + 6 + 12 + 18 + 13 = 50
    # Let's use: center(1), ring1(6), ring2(12), ring3(18), ring4(13)
    
    points = []
    
    # Center
    points.append([0.0, 0.0])
    
    # Ring 1: 6 points at radius ~0.3
    r1 = 0.3
    for i in range(6):
        angle = 2 * np.pi * i / 6
        points.append([r1 * np.cos(angle), r1 * np.sin(angle)])
    
    # Ring 2: 12 points at radius ~0.55
    r2 = 0.55
    for i in range(12):
        angle = 2 * np.pi * i / 12 + np.pi/12  # offset
        points.append([r2 * np.cos(angle), r2 * np.sin(angle)])
    
    # Ring 3: 18 points at radius ~0.78
    r3 = 0.78
    for i in range(18):
        angle = 2 * np.pi * i / 18
        points.append([r3 * np.cos(angle), r3 * np.sin(angle)])
    
    # Ring 4: 13 points at radius ~0.92
    r4 = 0.92
    for i in range(13):
        angle = 2 * np.pi * i / 13 + np.pi/13
        points.append([r4 * np.cos(angle), r4 * np.sin(angle)])
    
    return np.array(points)

def _get_hex_initial_points():
    """Generate a distorted hexagonal lattice and select the best N points."""
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

def _local_refine(P, r, max_iter=300, tol=1e-12):
    """Local refinement using gradient-based optimization with adaptive steps."""
    P_curr = P.copy()
    r_curr = r
    
    # Use scipy's L-BFGS-B for more effective local optimization
    def objective(P_flat):
        P_test = P_flat.reshape(N, 2)
        # Maximize the minimum radius, so minimize negative radius
        score = _compute_score(P_test)
        # Add penalty for points outside unit circle
        norms = np.linalg.norm(P_test, axis=1)
        outside_penalty = np.sum(np.maximum(norms - 1.0, 0.0)**2)
        return -score + outside_penalty * 100
    
    P_flat = P_curr.flatten()
    
    # Try L-BFGS-B optimization
    try:
        result = minimize(objective, P_flat, method='L-BFGS-B', 
                         options={'maxiter': max_iter, 'ftol': tol, 'gtol': tol})
        if result.fun < -r_curr + 1e-10:
            P_opt = result.x.reshape(N, 2)
            score_opt = _compute_score(P_opt)
            if score_opt > r_curr:
                P_curr = P_opt
                r_curr = score_opt
    except:
        pass
    
    # Additional manual refinement pass
    for it in range(max_iter // 2):
        diff = P_curr[:, np.newaxis, :] - P_curr[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        forces = np.zeros_like(P_curr)
        
        # Circle-circle repulsion
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
        
        # Boundary attraction
        dist_origin = np.linalg.norm(P_curr, axis=1)
        overlap_cb = np.maximum(dist_origin + r_curr - 1.0, 0.0)
        mask_cb = overlap_cb > 0
        if np.any(mask_cb):
            # Ensure dist_origin is reshaped for broadcasting with (k, 2) unit_vec
            # dist_origin[mask_cb] is (k,), we need (k, 1) to multiply with (k, 2)
            unit_vec = P_curr[mask_cb] / dist_origin[mask_cb, np.newaxis]
            # overlap_cb[mask_cb] is (k,), reshape to (k, 1) for broadcasting
            forces[mask_cb] -= 2.0 * unit_vec * overlap_cb[mask_cb, np.newaxis]
        
        # Adaptive step size with line search
        norm_forces = np.linalg.norm(forces)
        if norm_forces < 1e-15:
            break
        
        # Try multiple step sizes
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
        
        if r_curr - best_score < tol:
            break
    
    return P_curr, r_curr

def _simulated_annealing(P_init, r_init, start_time, deadline):
    """Simulated annealing with adaptive perturbation and local refinement."""
    P_curr = P_init.copy()
    r_curr = r_init
    score_best = r_curr
    P_best = P_init.copy()
    
    T_start = 0.05
    T_end = 0.001
    max_iter = 5000
    no_improve_count = 0
    max_no_improve = 500
    
    for it in range(max_iter):
        if time.time() > deadline - 10.0:
            break
        
        T = T_start * (T_end / T_start) ** (it / max_iter)
        
        # Adaptive perturbation: larger when stuck
        perturb_scale = 0.02 + 0.03 * (1 + no_improve_count / max_no_improve)
        n_perturb = np.random.randint(1, 8)
        idx = np.random.choice(N, n_perturb, replace=False)
        P_trial = P_curr.copy()
        P_trial[idx] += np.random.normal(0, perturb_scale, (n_perturb, 2))
        
        # Keep points inside container
        norms = np.linalg.norm(P_trial, axis=1)
        mask = norms > 0.98
        if np.any(mask):
            P_trial[mask] *= 0.98 / norms[mask, np.newaxis]
        
        E_curr = _energy(P_curr, r_curr)
        E_trial = _energy(P_trial, r_curr)
        dE = E_trial - E_curr
        
        if dE < 0 or np.random.random() < np.exp(-dE / T):
            P_curr = P_trial
            new_score = _compute_score(P_curr)
            if new_score > score_best:
                r_curr = new_score
                score_best = new_score
                P_best = P_curr.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
        
        # Periodic local refinement
        if it % 200 == 0 and it > 0:
            P_ref, score_ref = _local_refine(P_curr, r_curr, max_iter=50)
            if score_ref > score_best:
                P_curr = P_ref
                r_curr = score_ref
                score_best = score_ref
                P_best = P_ref.copy()
                no_improve_count = 0
    
    return P_best, score_best

def _optuna_like_search(P_init, r_init, start_time, deadline):
    """A more systematic search combining multiple strategies."""
    P_best = P_init.copy()
    score_best = r_init
    
    # Try multiple random restarts with local refinement
    n_restarts = 10
    for restart in range(n_restarts):
        if time.time() > deadline - 15.0:
            break
        
        # Perturb the current best
        P_trial = P_best + np.random.normal(0, 0.05, (N, 2))
        # Keep inside container
        norms = np.linalg.norm(P_trial, axis=1)
        mask = norms > 0.95
        P_trial[mask] *= 0.95 / norms[mask, np.newaxis]
        
        score_trial = _compute_score(P_trial)
        if score_trial > score_best:
            P_best = P_trial
            score_best = score_trial
        
        # Local refine
        P_ref, score_ref = _local_refine(P_trial, score_trial, max_iter=100)
        if score_ref > score_best:
            P_best = P_ref
            score_best = score_ref
    
    return P_best, score_best

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    np.random.seed(42)
    
    # Try multiple initializations and pick the best
    initializations = []
    
    # Hex lattice
    P_hex = _get_hex_initial_points()
    initializations.append(P_hex)
    
    # Ring structure
    P_ring = _get_ring_initialization()
    initializations.append(P_ring)
    
    # Random points
    for _ in range(3):
        angles = np.random.uniform(0, 2*np.pi, N)
        radii = np.random.uniform(0, 0.9, N)
        P_rand = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        initializations.append(P_rand)
    
    # Pick the best initialization
    P_best = initializations[0]
    score_best = _compute_score(P_best)
    for P_init in initializations[1:]:
        s = _compute_score(P_init)
        if s > score_best:
            P_best = P_init
            score_best = s
    
    # Phase 1: Simulated annealing with periodic local refinement
    if time.time() < deadline - 40.0:
        P_sa, score_sa = _simulated_annealing(P_best, score_best, start_time, deadline)
        if score_sa > score_best:
            P_best = P_sa
            score_best = score_sa
    
    # Phase 2: Multiple local refinement passes from different perturbations
    if time.time() < deadline - 20.0:
        for restart in range(5):
            if time.time() > deadline - 15.0:
                break
            P_trial = P_best + np.random.normal(0, 0.03, (N, 2))
            norms = np.linalg.norm(P_trial, axis=1)
            mask = norms > 0.95
            if np.any(mask):
                P_trial[mask] *= 0.95 / norms[mask, np.newaxis]
            
            P_ref, score_ref = _local_refine(P_trial, _compute_score(P_trial), max_iter=150)
            if score_ref > score_best:
                P_best = P_ref
                score_best = score_ref
    
    # Phase 3: Final high-precision polish
    if time.time() < deadline - 5.0:
        P_polish, score_polish = _local_refine(P_best, score_best, max_iter=500, tol=1e-14)
        if score_polish > score_best:
            P_best = P_polish
            score_best = score_polish
    
    # Ensure all points are inside the unit circle
    norms = np.linalg.norm(P_best, axis=1)
    mask = norms > 1.0
    if np.any(mask):
        P_best[mask] *= (1.0 - 1e-9) / norms[mask, np.newaxis]
    
    return P_best
# EVOLVE-BLOCK-END
