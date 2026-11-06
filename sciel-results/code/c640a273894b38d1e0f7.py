import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _score_raw(centers):
    """
    Compute the actual score (minimum radius) for a set of centers.
    Returns 0.0 if invalid.
    """
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    
    # Distance from origin
    norms = np.sqrt(np.sum(centers**2, axis=1))
    
    # Check all inside unit circle
    if np.any(norms > 1.0 + 1e-9):
        return 0.0
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    min_pair_dist = np.min(dists)
    min_edge_dist = np.min(1.0 - norms)
    
    r = min(min_pair_dist / 2.0, min_edge_dist)
    
    # If any circle is outside, r would be negative or the validity check fails.
    if min_edge_dist < -1e-9:
        return 0.0
        
    return max(r, 0.0)

def _soft_min_objective(x, beta=200.0):
    """
    Smooth approximation of the negative score using log-sum-exp.
    We want to maximize r = min(constraints).
    Minimize -soft_min(constraints).
    
    Constraints:
    1. Pairwise distances / 2
    2. Distance to boundary (1 - norm)
    """
    centers = x.reshape(N, 2)
    
    # Boundary constraints
    norms = np.sqrt(np.sum(centers**2, axis=1))
    edge_constraints = 1.0 - norms
    
    # Pairwise constraints
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    # Take upper triangle to avoid duplicates
    iu = np.triu_indices(N, 1)
    pair_constraints = dists[iu] / 2.0
    
    # Combine all constraints
    all_constraints = np.concatenate([edge_constraints, pair_constraints])
    
    # Soft min: -1/beta * log(sum(exp(-beta * x_k)))
    # Numerically stable version:
    # Let y_k = -beta * x_k
    # soft_min = -1/beta * (max(y) + log(sum(exp(y - max(y)))))
    
    y = -beta * all_constraints
    max_y = np.max(y)
    
    # Handle case where max_y is very large (constraints very negative)
    # exp(y - max_y) is safe because y - max_y <= 0
    exp_sum = np.sum(np.exp(y - max_y))
    
    # If exp_sum is 0 (shouldn't happen with finite inputs), handle gracefully
    if exp_sum == 0:
        return 0.0
        
    log_sum_exp = max_y + np.log(exp_sum)
    soft_min_val = -log_sum_exp / beta
    
    # We want to maximize soft_min_val, so minimize -soft_min_val
    return -soft_min_val

def _generate_hexagonal_initial(r_init):
    """
    Generate a hexagonal lattice packing with given circle radius.
    Returns N points closest to origin.
    """
    points = []
    spacing = 2 * r_init
    # Generate hexagonal lattice points
    # Range needs to be large enough to cover unit circle
    # Max radius 1, so i,j up to ~ 1/spacing
    limit = int(1.0 / spacing) + 2
    
    for j in range(-limit, limit + 1):
        for i in range(-limit, limit + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            r = np.sqrt(x*x + y*y)
            # Only keep points that can fit a circle of radius r_init
            if r <= 1.0 - r_init - 1e-9:
                points.append([x, y])
    
    if len(points) < N:
        # Try a smaller radius to get more points? Or just return None
        # If we can't fit N points with this spacing, the packing is denser.
        # But for initialization, we just need N valid distinct points.
        # If the lattice is too sparse, we might have fewer than N points inside.
        # In that case, we can generate more points or use a different init.
        return None
    
    points = np.array(points)
    # Sort by distance from origin and take N
    norms = np.sqrt(np.sum(points**2, axis=1))
    idx = np.argsort(norms, kind='stable')[:N]
    return points[idx]

def _generate_ring_initial():
    """
    Generate a ring-based initial configuration.
    """
    centers = []
    # Center
    centers.append([0.0, 0.0])
    
    # Rings: 1 + 6 + 12 + 18 + 13 = 50
    # Radii should be roughly proportional to sqrt(index) for uniform density,
    # but for packing, we want them spaced by ~2r.
    # Let's try fixed radii that work well for this count.
    ring_configs = [
        (6, 0.25),
        (12, 0.50),
        (18, 0.75),
        (13, 0.92)
    ]
    
    for count, r in ring_configs:
        angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
        for a in angles:
            centers.append([r * np.cos(a), r * np.sin(a)])
    
    centers = np.array(centers[:N])
    return centers

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s kill
    
    best_score = 0.0
    best_centers = None
    
    # List of initial configurations
    initial_configs = []
    
    # 1. Hexagonal lattices with different assumed radii
    for r_init in [0.10, 0.11, 0.12, 0.13, 0.14]:
        cfg = _generate_hexagonal_initial(r_init)
        if cfg is not None:
            initial_configs.append(cfg)
            
    # 2. Ring layout
    ring_cfg = _generate_ring_initial()
    initial_configs.append(ring_cfg)
    
    # 3. Random valid configurations
    rng = np.random.default_rng(42)
    for _ in range(5):
        cfg = rng.uniform(-0.9, 0.9, size=(N, 2))
        # Project inside circle to ensure validity start
        norms = np.sqrt(np.sum(cfg**2, axis=1))
        mask = norms > 0.9
        if np.any(mask):
            cfg[mask] *= 0.9 / norms[mask, np.newaxis]
        initial_configs.append(cfg)
    
    # Optimization parameters
    betas = [100.0, 200.0, 400.0]
    
    # Optimize each initial configuration
    for idx, x0 in enumerate(initial_configs):
        if time.time() > deadline - 10.0:
            break
            
        # Try with different beta values
        for beta in betas:
            if time.time() > deadline - 5.0:
                break
                
            try:
                # L-BFGS-B is good for this
                result = minimize(
                    _soft_min_objective,
                    x0.flatten(),
                    args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
                )
                
                centers = result.x.reshape(N, 2)
                
                # Check validity and score
                score = _score_raw(centers)
                
                if score > best_score:
                    best_score = score
                    best_centers = centers
                    
            except Exception:
                continue
    
    # If we have a good solution, try to refine it further
    if best_centers is not None and time.time() < deadline - 15.0:
        # Use a very high beta for the final polish to sharpen the minimum
        # Also try perturbing the best solution to escape any remaining local traps
        for _ in range(5):
            if time.time() > deadline - 5.0:
                break
            
            # Perturb best solution
            noise = rng.normal(0, 0.05, size=best_centers.shape)
            perturbed = best_centers + noise
            # Project inside
            norms = np.sqrt(np.sum(perturbed**2, axis=1))
            mask = norms > 0.99
            if np.any(mask):
                perturbed[mask] *= 0.99 / norms[mask, np.newaxis]
            
            try:
                result = minimize(
                    _soft_min_objective,
                    perturbed.flatten(),
                    args=(1000.0,),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
                )
                
                centers = result.x.reshape(N, 2)
                score = _score_raw(centers)
                
                if score > best_score:
                    best_score = score
                    best_centers = centers
                    
            except Exception:
                continue
    
    # Final high-precision polish
    if best_centers is not None and time.time() < deadline - 2.0:
        try:
            result = minimize(
                _soft_min_objective,
                best_centers.flatten(),
                args=(5000.0,),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 100}
            )
            
            centers = result.x.reshape(N, 2)
            score = _score_raw(centers)
            
            if score > best_score:
                best_score = score
                best_centers = centers
                
        except Exception:
            pass
    
    if best_centers is None:
        # Fallback: return a valid random configuration
        cfg = rng.uniform(-0.8, 0.8, size=(N, 2))
        norms = np.sqrt(np.sum(cfg**2, axis=1))
        mask = norms > 0.95
        if np.any(mask):
            cfg[mask] *= 0.95 / norms[mask, np.newaxis]
        return cfg
        
    return best_centers
