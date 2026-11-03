"""Pack 50 equal circles in the unit disk via Riesz-kernel gradient descent + L-BFGS-B polish."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
DEADLINE = 135.0  # safety margin before 150s limit


def _compute_score(pts):
    """Compute the packing radius: min of half-min-pair-dist and min-container-margin."""
    n = len(pts)
    container_margin = 1.0 - np.sqrt(np.sum(pts ** 2, axis=1))
    min_container = np.min(container_margin)
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist) / 2.0
    
    return min(min_container, min_pair)


def _riesz_energy(pts, s=20.0, eps=1e-8):
    """
    Riesz kernel repulsive energy: E = sum_{i<j} |c_i - c_j|^{-s} + container penalty.
    Large s makes the energy very sensitive to close pairs, driving points apart.
    """
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    # Repulsive pair energy
    pair_energy = np.sum(dist[dist < np.inf] ** (-s))
    
    # Container penalty: penalize points too close to boundary
    radii = np.sqrt(np.sum(pts ** 2, axis=1))
    container_viol = np.maximum(0, radii - (1.0 - 0.01))
    cont_energy = np.sum(container_viol ** 2) * 100.0
    
    return pair_energy + cont_energy


def _riesz_gradient(pts, s=20.0, eps=1e-8):
    """
    Analytical gradient of the Riesz energy.
    d/dp_i |p_i - p_j|^{-s} = -s * |p_i - p_j|^{-s-2} * (p_i - p_j)
    """
    n = len(pts)
    grad = np.zeros_like(pts)
    
    # Pair contributions
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 2)
    dist_sq = np.sum(diff ** 2, axis=2)  # (n, n)
    np.fill_diagonal(dist_sq, np.inf)
    dist = np.sqrt(dist_sq)
    
    # For each pair (i,j): contribution to grad[i] = -s * dist^{-s-2} * (p_i - p_j)
    # Sum over j for each i
    inv_dist_power = np.zeros_like(dist_sq)
    mask = dist_sq < np.inf
    inv_dist_power[mask] = dist_sq[mask] ** (-(s + 1) / 2.0)
    
    # grad[i] = -s * sum_j inv_dist_power[i,j] * diff[i,j]
    # = -s * sum_j dist^{-s-2} * (p_i - p_j)
    grad_pairs = -s * np.sum(inv_dist_power[:, :, np.newaxis] * diff, axis=1)
    grad += grad_pairs
    
    # Container gradient
    radii = np.sqrt(np.sum(pts ** 2, axis=1))
    radii_safe = np.maximum(radii, 1e-12)
    container_viol = np.maximum(0, radii - (1.0 - 0.01))
    grad_cont = 200.0 * container_viol[:, np.newaxis] * pts / radii_safe[:, np.newaxis]
    grad += grad_cont
    
    return grad


def _ring_init(n, seed=42):
    """
    Initialize points in concentric rings with hexagonal-like angular spacing.
    This gives a structured starting point closer to the optimal packing.
    """
    rng = np.random.default_rng(seed)
    
    # Known good packing has roughly: 1 center, rings of 6, 12, 18, 13 (total ~50)
    # Let's use a more systematic approach: distribute points in rings
    # Optimal packing radius ~0.1258, so diameter ~0.2516
    # In unit disk, we can fit rings at radii approximately: 0, 0.25, 0.5, 0.75, 0.9
    
    ring_radii = [0.0, 0.24, 0.48, 0.72, 0.92]
    ring_counts = [1, 6, 12, 18, 13]
    
    pts = []
    for r, count in zip(ring_radii, ring_counts):
        if count == 1:
            pts.append([0.0, 0.0])
        else:
            angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
            # Add small random perturbation to break symmetry
            angles += rng.uniform(-0.05, 0.05, count)
            for a in angles:
                pts.append([r * np.cos(a), r * np.sin(a)])
    
    pts = np.array(pts[:n])
    
    # Ensure all points are inside the unit disk
    radii = np.sqrt(np.sum(pts ** 2, axis=1))
    for i in range(n):
        if radii[i] > 0.99:
            pts[i] *= 0.99 / radii[i]
    
    return pts


def _adaptive_gradient_descent(pts, s=20.0, max_iter=5000, lr_init=0.01):
    """
    Gradient descent with adaptive step size on the Riesz energy.
    """
    n = len(pts)
    best_pts = pts.copy()
    best_score = _compute_score(best_pts)
    
    lr = lr_init
    patience = 0
    no_improve = 0
    
    for it in range(max_iter):
        if time.time() > DEADLINE - 20:
            break
        
        grad = _riesz_gradient(pts, s=s)
        
        # Adaptive step size: reduce if energy increases
        new_pts = pts - lr * grad
        
        # Keep points inside the unit disk
        radii = np.sqrt(np.sum(new_pts ** 2, axis=1))
        for i in range(n):
            if radii[i] > 0.995:
                new_pts[i] *= 0.995 / radii[i]
        
        # Compute score
        new_score = _compute_score(new_pts)
        
        if new_score > best_score:
            best_pts = new_pts.copy()
            best_score = new_score
            no_improve = 0
            # Increase learning rate slightly
            lr = min(lr * 1.01, 0.05)
        else:
            no_improve += 1
            # Decrease learning rate
            lr *= 0.95
            if lr < 1e-6:
                lr = 1e-6
        
        # Accept the move if it doesn't decrease the score too much (simulated annealing flavor)
        if new_score >= best_score * 0.99 or no_improve < 50:
            pts = new_pts
        else:
            # Revert and reduce step
            lr *= 0.8
        
        if no_improve > 200 and it > 1000:
            break
    
    return best_pts, best_score


def _lbfgs_polish(pts, r_target, maxiter=300):
    """
    Polish using L-BFGS-B to minimize a smooth penalty for violating radius r_target.
    """
    n = len(pts)
    
    def objective(flat):
        p = flat.reshape(n, 2)
        # Container violation
        radii = np.sqrt(np.sum(p ** 2, axis=1))
        cont_viol = np.maximum(0, radii + r_target - 1.0)
        cont_pen = np.sum(cont_viol ** 2) * 1000.0
        
        # Pair violation
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist, np.inf)
        pair_viol = np.maximum(0, 2 * r_target - dist)
        pair_pen = np.sum(pair_viol ** 2) * 1000.0
        
        return cont_pen + pair_pen
    
    def gradient(flat):
        p = flat.reshape(n, 2)
        grad = np.zeros_like(flat)
        
        # Container gradient
        radii = np.sqrt(np.sum(p ** 2, axis=1))
        radii_safe = np.maximum(radii, 1e-12)
        cont_viol = np.maximum(0, radii + r_target - 1.0)
        grad_cont = 2000.0 * cont_viol[:, np.newaxis] * p / radii_safe[:, np.newaxis]
        grad[:n * 2] = grad_cont.flatten()
        
        # Pair gradient
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        dist = np.sqrt(dist_sq)
        
        pair_viol = np.maximum(0, 2 * r_target - dist)
        mask = pair_viol > 0
        
        # grad[i] += 2 * 1000 * sum_j pair_viol[i,j] * (p_i - p_j) / dist[i,j]
        inv_dist = np.zeros_like(dist_sq)
        mask_safe = mask & (dist_sq > 1e-12)
        inv_dist[mask_safe] = 1.0 / np.sqrt(dist_sq[mask_safe])
        
        grad_pairs = 2000.0 * np.sum((pair_viol * inv_dist)[:, :, np.newaxis] * diff, axis=1)
        grad[:n * 2] += grad_pairs.flatten()
        
        return grad
    
    result = minimize(
        objective,
        pts.flatten(),
        method='L-BFGS-B',
        jac=gradient,
        options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10}
    )
    return result.x.reshape(n, 2)


def construct():
    t0 = time.time()
    
    # Start with ring-based initialization
    best_pts = _ring_init(N, seed=42)
    best_score = _compute_score(best_pts)
    
    # Phase 1: Riesz kernel gradient descent to spread points
    if time.time() - t0 < DEADLINE - 40:
        gd_pts, gd_score = _adaptive_gradient_descent(best_pts, s=20.0, max_iter=3000, lr_init=0.005)
        if gd_score > best_score:
            best_pts = gd_pts
            best_score = gd_score
    
    # Phase 2: L-BFGS-B polish with the current best radius
    if time.time() - t0 < DEADLINE - 15:
        r_target = best_score
        polished = _lbfgs_polish(best_pts, r_target, maxiter=200)
        polished_score = _compute_score(polished)
        if polished_score > best_score:
            best_pts = polished
            best_score = polished_score
    
    # Phase 3: Try a slightly larger target radius and polish again
    if time.time() - t0 < DEADLINE - 10:
        r_target = best_score * 1.001
        polished = _lbfgs_polish(best_pts, r_target, maxiter=150)
        polished_score = _compute_score(polished)
        if polished_score > best_score:
            best_pts = polished
            best_score = polished_score
    
    # Phase 4: Final fine polish
    if time.time() - t0 < DEADLINE - 5:
        r_target = best_score
        polished = _lbfgs_polish(best_pts, r_target, maxiter=100)
        polished_score = _compute_score(polished)
        if polished_score > best_score:
            best_pts = polished
            best_score = polished_score
    
    # Validate: ensure all points are inside and finite
    radii = np.sqrt(np.sum(best_pts ** 2, axis=1))
    if np.any(~np.isfinite(best_pts)) or np.any(radii >= 1.0):
        # Fallback to ring init
        best_pts = _ring_init(N, seed=42)
    
    return best_pts
