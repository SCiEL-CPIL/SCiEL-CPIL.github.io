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
    
    # Distance to boundary
    d_bound = 1.0 - np.linalg.norm(P, axis=1)
    r_bound = np.min(d_bound)
    
    # Pairwise distances
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair_dist = np.min(dist)
    r_pair = min_pair_dist / 2.0
    
    return min(r_pair, r_bound)

def _get_ring_initialization(n_points=50):
    """
    Initialize points in concentric rings.
    Ring 0: center (1 point)
    Ring k: k*6 points (hexagonal pattern)
    """
    points = []
    remaining = n_points
    
    # Center point
    points.append([0.0, 0.0])
    remaining -= 1
    
    # Add rings
    ring = 1
    while remaining > 0:
        n_in_ring = min(6 * ring, remaining)
        # Radius for this ring - approximate spacing
        base_radius = 0.12 * ring
        for i in range(n_in_ring):
            angle = 2 * np.pi * i / n_in_ring + (np.pi / 6 if ring % 2 == 0 else 0)
            x = base_radius * np.cos(angle)
            y = base_radius * np.sin(angle)
            points.append([x, y])
        remaining -= n_in_ring
        ring += 1
    
    return np.array(points[:n_points])

def _get_hex_initial_points():
    """Initialization using hexagonal lattice."""
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

def _riesz_energy(P, s=8.0, eps=1e-6):
    """
    Riesz kernel repulsive energy: E = sum_{i<j} |c_i - c_j|^{-s}
    plus boundary repulsion.
    """
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, eps)
    
    # Upper triangle only to avoid double counting
    iu = np.triu_indices(N, k=1)
    pairwise_dist = dist[iu]
    
    # Repulsive kernel
    E_pair = np.sum(pairwise_dist ** (-s))
    
    # Boundary repulsion: push points inward if too close to boundary
    dist_origin = np.linalg.norm(P, axis=1)
    boundary_violation = np.maximum(0, dist_origin - (1.0 - 0.01))
    E_boundary = np.sum(boundary_violation ** 2)
    
    return E_pair + 10.0 * E_boundary

def _riesz_grad(P, s=8.0, eps=1e-6):
    """
    Gradient of Riesz kernel energy.
    """
    grad = np.zeros_like(P)
    
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, eps)
    
    # d/dP_i of |P_i - P_j|^{-s} = -s * |P_i - P_j|^{-s-2} * (P_i - P_j)
    # Sum over j != i
    inv_dist_power = dist ** (-(s + 2))
    np.fill_diagonal(inv_dist_power, 0)
    
    # grad_i = -s * sum_j (P_i - P_j) * |P_i - P_j|^{-s-2}
    # = -s * (sum_j (P_i - P_j) * inv_dist_power_ij)
    # Note: diff[i,j] = P_i - P_j
    term = (diff * inv_dist_power[:, :, np.newaxis]).sum(axis=1)
    grad = -s * term
    
    # Boundary gradient
    dist_origin = np.linalg.norm(P, axis=1)
    boundary_violation = np.maximum(0, dist_origin - (1.0 - 0.01))
    inv_dist_origin = np.zeros_like(dist_origin)
    np.divide(1.0, dist_origin, out=inv_dist_origin, where=dist_origin > 0)
    grad += 20.0 * boundary_violation[:, np.newaxis] * inv_dist_origin[:, np.newaxis] * P
    
    return grad

def _smooth_energy(P, r, beta=50.0):
    """
    Smooth log-sum-exp energy function for differentiable optimization.
    """
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    violations_cc = 2.0 * r - dist
    dist_origin = np.linalg.norm(P, axis=1)
    violations_cb = dist_origin + r - 1.0
    
    def _lse(v, beta):
        v_scaled = beta * v
        v_max = np.max(v_scaled)
        if not np.isfinite(v_max) or v_max == -np.inf:
            return 0.0
        return (1.0 / beta) * (v_max + np.log(np.sum(np.exp(v_scaled - v_max))))
    
    E_cc = _lse(violations_cc, beta)
    E_cb = _lse(violations_cb, beta)
    
    return E_cc + E_cb

def _smooth_energy_grad(P, r, beta=50.0):
    """
    Gradient of the smooth log-sum-exp energy.
    """
    grad = np.zeros_like(P)
    
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    violations_cc = 2.0 * r - dist
    v_scaled_cc = beta * violations_cc
    v_max_cc = np.max(v_scaled_cc)
    if np.isfinite(v_max_cc) and v_max_cc != -np.inf:
        weights_cc = np.exp(v_scaled_cc - v_max_cc)
        weights_cc /= np.sum(weights_cc)
        inv_dist = np.zeros_like(dist)
        np.divide(1.0, dist, out=inv_dist, where=dist > 0)
        
        W_cc = weights_cc * inv_dist
        sum_W = np.sum(W_cc, axis=1)
        term1 = (P.T @ W_cc).T
        term2 = P * sum_W[:, np.newaxis]
        grad += term2 - term1
    
    dist_origin = np.linalg.norm(P, axis=1)
    violations_cb = dist_origin + r - 1.0
    v_scaled_cb = beta * violations_cb
    v_max_cb = np.max(v_scaled_cb)
    if np.isfinite(v_max_cb) and v_max_cb != -np.inf:
        weights_cb = np.exp(v_scaled_cb - v_max_cb)
        weights_cb /= np.sum(weights_cb)
        
        inv_dist_origin = np.zeros_like(dist_origin)
        np.divide(1.0, dist_origin, out=inv_dist_origin, where=dist_origin > 0)
        
        grad += (weights_cb * inv_dist_origin)[:, np.newaxis] * P
    
    return grad

def _optimize_riesz(P_init, s=8.0, max_iter=500, lr0=0.001, decay=0.995, time_limit=5.0):
    """
    Optimize using Riesz kernel energy with adaptive step size.
    """
    start_time = time.time()
    P = P_init.copy()
    best_P = P.copy()
    best_score = _compute_score(P)
    
    lr = lr0
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        E = _riesz_energy(P, s)
        G = _riesz_grad(P, s)
        
        # Adaptive step size: reduce if energy increases
        P_new = P - lr * G
        E_new = _riesz_energy(P_new, s)
        
        if E_new > E * 1.01:
            lr *= 0.8
        else:
            lr = min(lr * 1.02, lr0)
        
        P = P_new
        
        # Track best by score
        score = _compute_score(P)
        if score > best_score:
            best_score = score
            best_P = P.copy()
    
    return best_P, best_score

def _optimize_smooth(P_init, r_target, max_iter=300, beta=50.0, time_limit=4.0):
    """
    Optimize positions P to minimize smooth energy for a fixed r_target.
    """
    start_time = time.time()
    
    def objective(P_flat):
        P = P_flat.reshape(N, 2)
        return _smooth_energy(P, r_target, beta)
    
    def grad(P_flat):
        P = P_flat.reshape(N, 2)
        return _smooth_energy_grad(P, r_target, beta).flatten()
        
    try:
        res = minimize(objective, P_init.flatten(), jac=grad, method='L-BFGS-B', 
                       options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 20})
        P_opt = res.x.reshape(N, 2)
        return P_opt
    except Exception:
        return P_init

def _check_feasible(P, r, tol=1e-8):
    """Check if packing with radius r is feasible."""
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    
    dist_origin = np.linalg.norm(P, axis=1)
    max_origin = np.max(dist_origin)
    
    return (min_pair >= 2.0 * r - tol) and (max_origin + r <= 1.0 + tol)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s buffer for safety
    
    np.random.seed(42)
    
    # Multiple initialization strategies
    inits = []
    
    # 1. Ring initialization
    P_ring = _get_ring_initialization(N)
    inits.append(P_ring)
    
    # 2. Hexagonal lattice
    P_hex = _get_hex_initial_points()
    inits.append(P_hex)
    
    # 3. Perturbed ring
    P_ring_pert = P_ring + 0.01 * np.random.randn(N, 2)
    inits.append(P_ring_pert)
    
    # 4. Random within unit disk
    angles = np.random.uniform(0, 2 * np.pi, N)
    radii = np.sqrt(np.random.uniform(0, 0.8, N))
    P_rand = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    inits.append(P_rand)
    
    P_best = inits[0].copy()
    score_best = _compute_score(P_best)
    
    # Phase 1: Riesz kernel optimization from multiple starts
    for init_idx, P_init in enumerate(inits):
        if time.time() > deadline - 30.0:
            break
        
        # Try different s values
        for s in [6.0, 8.0, 10.0]:
            if time.time() > deadline - 30.0:
                break
            
            P_opt, score = _optimize_riesz(P_init, s=s, max_iter=300, lr0=0.0005, 
                                           decay=0.995, time_limit=3.0)
            if score > score_best:
                P_best = P_opt.copy()
                score_best = score
    
    # Phase 2: Binary search on r with smooth energy polish
    r_low = score_best
    r_high = 0.135
    
    for iteration in range(10):
        if time.time() > deadline - 10.0:
            break
        
        r_mid = (r_low + r_high) / 2.0
        
        best_P_for_r = None
        best_energy = np.inf
        
        n_restarts = 3
        for restart in range(n_restarts):
            if time.time() > deadline - 10.0:
                break
            
            if restart == 0:
                P_init = P_best.copy()
            else:
                sigma = 0.003 * (restart + 1)
                P_init = P_best + sigma * np.random.randn(N, 2)
            
            P_opt = _optimize_smooth(P_init, r_mid, max_iter=200, beta=50.0, time_limit=2.5)
            E = _smooth_energy(P_opt, r_mid, 50.0)
            
            if E < best_energy:
                best_energy = E
                best_P_for_r = P_opt
        
        if best_P_for_r is None:
            break
        
        if _check_feasible(best_P_for_r, r_mid):
            r_low = r_mid
            score = _compute_score(best_P_for_r)
            if score > score_best:
                P_best = best_P_for_r.copy()
                score_best = score
        else:
            r_high = r_mid
    
    # Phase 3: Final high-precision polish
    if time.time() < deadline - 5.0:
        r_target = score_best
        for restart in range(3):
            if time.time() > deadline - 3.0:
                break
            
            if restart == 0:
                P_init = P_best.copy()
            else:
                P_init = P_best + 0.001 * restart * np.random.randn(N, 2)
            
            P_opt = _optimize_smooth(P_init, r_target, max_iter=500, beta=100.0, time_limit=2.0)
            score = _compute_score(P_opt)
            if score > score_best:
                P_best = P_opt
                score_best = score
    
    # Ensure all points are inside the unit circle
    norms = np.linalg.norm(P_best, axis=1)
    mask = norms > 1.0 - 1e-10
    if np.any(mask):
        P_best[mask] *= (1.0 - 1e-10) / norms[mask]
    
    return P_best
# EVOLVE-BLOCK-END
