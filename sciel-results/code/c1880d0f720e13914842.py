"""
Packing 50 equal circles in a unit circle.
Strategy: Hexagonal lattice initialization + Simulated Annealing + Smooth Gradient Polish.
"""
import numpy as np
from scipy.optimize import minimize
import time

N = 50

def _score(points):
    """Compute the raw score: minimum radius."""
    if len(points) != N:
        return -1.0
    if not np.all(np.isfinite(points)):
        return -1.0
    
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair_dist = np.min(dist)
    
    # Boundary distances
    norms = np.sqrt(np.sum(points**2, axis=1))
    min_margin = np.min(1.0 - norms)
    
    return min(min_pair_dist / 2.0, min_margin)

def _softmin_logsumexp(dists, beta):
    """Soft minimum via log-sum-exp."""
    neg_beta = -beta * dists
    max_val = np.max(neg_beta)
    if not np.isfinite(max_val):
        return -np.inf
    log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta - max_val)))
    return -log_sum_exp / beta

def _smooth_objective_and_grad(params, beta):
    """
    params: flat array of length 2*N
    Returns (objective, gradient) for smooth approximation of -min_radius.
    """
    pts = params.reshape(N, 2)
    
    # Pairwise distances
    iu = np.triu_indices(N, k=1)
    diff = pts[iu[0]] - pts[iu[1]]
    pair_dists = np.sqrt(np.sum(diff**2, axis=1))
    
    # Boundary distances
    radii = np.hypot(pts[:, 0], pts[:, 1])
    boundary_dists = 1.0 - radii
    
    all_dists = np.concatenate([pair_dists, boundary_dists])
    scaled = all_dists / 2.0
    
    # Soft min
    soft_min_val = _softmin_logsumexp(scaled, beta)
    objective = -soft_min_val
    
    # Weights for gradient
    neg_beta_scaled = -beta * scaled
    max_val = np.max(neg_beta_scaled)
    weights = np.exp(neg_beta_scaled - max_val)
    w_sum = np.sum(weights)
    if w_sum < 1e-300:
        return objective, np.zeros_like(params)
    weights /= w_sum
    
    grad_scaled = -beta * weights
    n_pairs = len(pair_dists)
    
    grad_pts = np.zeros_like(pts)
    
    # Pair contributions
    for k in range(n_pairs):
        w = grad_scaled[k] / 2.0
        d = pair_dists[k]
        if d > 1e-14:
            i, j = iu[0][k], iu[1][k]
            d_dij = (pts[i] - pts[j]) / d
            grad_pts[i] += w * d_dij
            grad_pts[j] -= w * d_dij
    
    # Boundary contributions
    for i in range(N):
        w = grad_scaled[n_pairs + i] / 2.0
        r = radii[i]
        if r > 1e-14:
            d_dr = pts[i] / r
            grad_pts[i] += w * d_dr
    
    return objective, grad_pts.flatten()

def _hexagonal_init():
    """
    Generate a hexagonal lattice initialization clipped to the unit circle.
    """
    # Estimate spacing
    # Area of unit circle = pi. Area per circle ~ pi * r^2. 
    # For N=50, r~0.125. Area ~ 0.05. Density ~ 20.
    # Hex spacing d = sqrt(2 * area / sqrt(3)) approx.
    # Let's just try a few spacings and pick the one that fits best or use a standard one.
    
    best_pts = None
    best_score = -1
    
    # Try a range of spacings
    for d in np.linspace(0.2, 0.35, 100):
        # Generate lattice points
        # Hex lattice basis: (0,0), (d, 0), (d/2, d*sqrt(3)/2)
        # We need to cover the unit circle.
        # Max radius is 1.
        # x range: -1 to 1. y range: -1 to 1.
        
        # Number of steps in x
        nx = int(2 / d) + 2
        # Number of steps in y
        ny = int(2 / (d * np.sqrt(3)/2)) + 2
        
        pts = []
        for i in range(-nx, nx+1):
            for j in range(-ny, ny+1):
                x = i * d
                y = j * d * np.sqrt(3) / 2
                if i % 2 != 0:
                    y += d * np.sqrt(3) / 4
                if np.sqrt(x*x + y*y) <= 1.0:
                    pts.append([x, y])
        
        pts = np.array(pts)
        if len(pts) < N:
            continue
        
        # We have more than N points. We need to select N that maximize the minimum distance.
        # A simple heuristic: sort by distance to origin (center out) and take the first N?
        # Or sort by "local density"?
        # Actually, for a hex lattice, the points are already well-packed.
        # We can just take the N points closest to the origin? No, that leaves the edge empty.
        # We want to maximize the minimum distance.
        # Let's just take a random subset? No, that's bad.
        # Let's take the N points with the largest "clearance" to the boundary?
        
        # Better approach: If we have more than N, we can just keep the first N in a spiral order?
        # Or, we can run a quick local optimization on the first N.
        
        # Let's try sorting by distance to origin and taking the first N.
        dists = np.linalg.norm(pts, axis=1)
        indices = np.argsort(dists)
        candidate = pts[indices[:N]]
        
        # Ensure we have exactly N
        if len(candidate) < N:
            continue
            
        s = _score(candidate)
        if s > best_score:
            best_score = s
            best_pts = candidate.copy()
            
    if best_pts is None:
        # Fallback: random
        rng = np.random.default_rng(42)
        best_pts = rng.uniform(-0.9, 0.9, size=(N, 2))
        
    return best_pts

def _simulated_annealing(points, deadline):
    """
    Run Simulated Annealing to improve the packing.
    """
    current_pts = points.copy()
    current_score = _score(current_pts)
    best_pts = current_pts.copy()
    best_score = current_score
    
    # Parameters
    T = 0.1  # Initial temperature
    T_min = 1e-6
    alpha = 0.999  # Cooling rate
    step_size = 0.1
    
    rng = np.random.default_rng(123)
    
    # Precompute indices for fast distance calculation?
    # For N=50, O(N^2) is 2500 ops. Very fast.
    
    while time.time() < deadline and T > T_min:
        # Random move: perturb k points
        k = rng.integers(1, 4)  # 1 to 3 points
        indices = rng.choice(N, size=k, replace=False)
        
        old_pts = current_pts[indices].copy()
        
        # Perturb
        perturbations = rng.normal(0, step_size, size=(k, 2))
        current_pts[indices] += perturbations
        
        # Clamp to unit circle (soft clamp? or hard?)
        # Hard clamp helps avoid invalid states, but might get stuck.
        # Let's just evaluate score. If a point is outside, margin is negative, score drops.
        # The score function handles this: min(1 - |c|) will be negative.
        
        new_score = _score(current_pts)
        
        # Metropolis criterion
        if new_score > current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_pts = current_pts.copy()
        else:
            # Accept with probability exp((new - old) / T)
            # Since we maximize, delta = new - old is negative.
            delta = new_score - current_score
            if delta < 0:
                p = np.exp(delta / T)
                if rng.random() < p:
                    current_score = new_score
                else:
                    # Reject: revert
                    current_pts[indices] = old_pts
        
        T *= alpha
        
    return best_pts, best_score

def _polish(points, deadline):
    """
    Final polish using gradient descent on the smooth surrogate.
    """
    current_pts = points.copy()
    current_score = _score(current_pts)
    
    # Multi-stage beta annealing
    beta_stages = [50.0, 100.0, 200.0, 500.0, 1000.0]
    
    for beta in beta_stages:
        if time.time() > deadline:
            break
        
        try:
            result = minimize(
                lambda x, b=beta: _smooth_objective_and_grad(x, b)[0],
                lambda x, b=beta: _smooth_objective_and_grad(x, b)[1],
                current_pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-14, 'maxls': 100}
            )
            
            if time.time() > deadline:
                break
            
            new_pts = result.x.reshape(N, 2)
            
            if not np.all(np.isfinite(new_pts)):
                continue
            
            # Clamp to unit circle
            norms = np.hypot(new_pts[:, 0], new_pts[:, 1])
            mask = norms > 1.0
            if np.any(mask):
                new_pts[mask] *= (1.0 - 1e-12) / norms[mask]
            
            new_score = _score(new_pts)
            
            if new_score > current_score:
                current_pts = new_pts
                current_score = new_score
                
        except Exception:
            continue
            
    return current_pts, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # 1. Initialization
    init_pts = _hexagonal_init()
    init_score = _score(init_pts)
    
    best_pts = init_pts.copy()
    best_score = init_score
    
    # 2. Simulated Annealing
    # We have ~100s for SA.
    sa_deadline = start_time + 100.0
    if time.time() < sa_deadline:
        sa_pts, sa_score = _simulated_annealing(init_pts, sa_deadline)
        if sa_score > best_score:
            best_score = sa_score
            best_pts = sa_pts.copy()
    
    # 3. Polish
    polish_deadline = deadline - 5.0
    if time.time() < polish_deadline:
        polished_pts, polished_score = _polish(best_pts, polish_deadline)
        if polished_score > best_score:
            best_score = polished_score
            best_pts = polished_pts.copy()
    
    # Final validation and clamping
    best_pts = np.array(best_pts, dtype=np.float64)
    norms = np.linalg.norm(best_pts, axis=1)
    mask = norms > 1.0
    if np.any(mask):
        best_pts[mask] *= (1.0 - 1e-12) / norms[mask]
    
    return best_pts
