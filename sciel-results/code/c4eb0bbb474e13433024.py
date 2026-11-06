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
    
    if min_edge_dist < -1e-9:
        return 0.0
        
    return max(r, 0.0)

def _soft_min_objective(x, beta=200.0):
    """
    Smooth approximation of the negative score using log-sum-exp.
    """
    centers = x.reshape(N, 2)
    
    # Boundary constraints
    norms = np.sqrt(np.sum(centers**2, axis=1))
    edge_constraints = 1.0 - norms
    
    # Pairwise constraints
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    iu = np.triu_indices(N, 1)
    pair_constraints = dists[iu] / 2.0
    
    all_constraints = np.concatenate([edge_constraints, pair_constraints])
    
    y = -beta * all_constraints
    max_y = np.max(y)
    exp_sum = np.sum(np.exp(y - max_y))
    
    if exp_sum == 0:
        return 0.0
        
    log_sum_exp = max_y + np.log(exp_sum)
    soft_min_val = -log_sum_exp / beta
    
    return -soft_min_val

def _epigraph_objective(x):
    """
    Objective for epigraph formulation.
    x = [centers (2N), t]
    We want to maximize t, so minimize -t.
    """
    centers = x[:2*N].reshape(N, 2)
    t = x[2*N]
    return -t

def _epigraph_constraints(x):
    """
    Constraints for epigraph formulation.
    Returns list of constraint dicts.
    Constraint: g(x) >= 0
    1. d_ij / 2 - t >= 0 for all i < j
    2. 1 - |c_i| - t >= 0 for all i
    """
    centers = x[:2*N].reshape(N, 2)
    t = x[2*N]
    
    # Pairwise constraints
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    iu = np.triu_indices(N, 1)
    pair_slacks = dists[iu] / 2.0 - t
    
    # Boundary constraints
    norms = np.sqrt(np.sum(centers**2, axis=1))
    edge_slacks = 1.0 - norms - t
    
    # Combine into a single vector constraint or list of dicts
    # SLSQP accepts a list of dicts with 'fun', 'type': 'ineq'
    # Or a single function returning an array if we use 'cons' in a specific way?
    # Standard SLSQP expects a list of dicts.
    
    constraints = []
    # We can group them, but SLSQP handles individual constraints better or a vector if we define fun to return vector.
    # Let's define a single function returning the vector of all slacks.
    # Wait, scipy.optimize.minimize cons can be a list of dicts.
    # Let's just return the array of slacks in a single dict? No, 'fun' must return a scalar or array?
    # If 'fun' returns an array, 'type' must be 'ineq' and it's treated as element-wise >= 0.
    
    all_slacks = np.concatenate([pair_slacks, edge_slacks])
    
    return [{'type': 'ineq', 'fun': lambda x: np.concatenate([
        (np.sqrt(np.sum((x[:2*N].reshape(N, 2)[:, np.newaxis, :] - x[:2*N].reshape(N, 2)[np.newaxis, :, :])**2, axis=2))[np.triu_indices(N, 1)] / 2.0 - x[2*N]),
        (1.0 - np.sqrt(np.sum(x[:2*N].reshape(N, 2)**2, axis=1)) - x[2*N])
    ])}]

def _generate_hexagonal_initial(r_init):
    """
    Generate a hexagonal lattice packing with given circle radius.
    Returns N points closest to origin.
    """
    points = []
    spacing = 2 * r_init
    limit = int(1.0 / spacing) + 2
    
    for j in range(-limit, limit + 1):
        for i in range(-limit, limit + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            r = np.sqrt(x*x + y*y)
            if r <= 1.0 - r_init - 1e-9:
                points.append([x, y])
    
    if len(points) < N:
        return None
    
    points = np.array(points)
    norms = np.sqrt(np.sum(points**2, axis=1))
    idx = np.argsort(norms, kind='stable')[:N]
    return points[idx]

def _generate_ring_initial():
    """
    Generate a ring-based initial configuration.
    """
    centers = []
    centers.append([0.0, 0.0])
    
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

def _polish_with_epigraph(centers, t_init, time_limit=10.0):
    """
    Refine centers and t using SLSQP on the epigraph formulation.
    """
    start_time = time.time()
    
    # Initial guess for x: [centers_flat, t]
    x0 = np.concatenate([centers.flatten(), [t_init]])
    
    # Bounds: centers should be within [-1, 1], t >= 0
    bounds = [(-1.0, 1.0)] * (2 * N) + [(0.0, 1.0)]
    
    try:
        result = minimize(
            _epigraph_objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=_epigraph_constraints,
            options={'maxiter': 500, 'ftol': 1e-12}
        )
        
        if result.success or result.fun < -t_init:
            new_centers = result.x[:2*N].reshape(N, 2)
            new_t = result.x[2*N]
            
            # Validate
            score = _score_raw(new_centers)
            if score > t_init:
                return new_centers, score
    except Exception:
        pass
    
    return centers, t_init

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    best_score = 0.0
    best_centers = None
    
    initial_configs = []
    
    for r_init in [0.10, 0.11, 0.12, 0.13, 0.14]:
        cfg = _generate_hexagonal_initial(r_init)
        if cfg is not None:
            initial_configs.append(cfg)
            
    ring_cfg = _generate_ring_initial()
    initial_configs.append(ring_cfg)
    
    rng = np.random.default_rng(42)
    for _ in range(5):
        cfg = rng.uniform(-0.9, 0.9, size=(N, 2))
        norms = np.sqrt(np.sum(cfg**2, axis=1))
        mask = norms > 0.9
        if np.any(mask):
            cfg[mask] *= 0.9 / norms[mask, np.newaxis]
        initial_configs.append(cfg)
    
    betas = [100.0, 200.0, 400.0]
    
    # Phase 1: Soft-min optimization
    for idx, x0 in enumerate(initial_configs):
        if time.time() > deadline - 30.0:
            break
            
        for beta in betas:
            if time.time() > deadline - 25.0:
                break
                
            try:
                result = minimize(
                    _soft_min_objective,
                    x0.flatten(),
                    args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
                )
                
                centers = result.x.reshape(N, 2)
                score = _score_raw(centers)
                
                if score > best_score:
                    best_score = score
                    best_centers = centers
                    
            except Exception:
                continue

    # Phase 2: Epigraph Polish
    if best_centers is not None and time.time() < deadline - 10.0:
        # Try to polish the best solution found so far
        polished_centers, polished_score = _polish_with_epigraph(
            best_centers, best_score, time_limit=15.0
        )
        
        if polished_score > best_score:
            best_score = polished_score
            best_centers = polished_centers
            
        # Try polishing with a slightly perturbed version to escape local minima
        if time.time() < deadline - 5.0:
            perturbed = best_centers + rng.normal(0, 0.005, size=(N, 2))
            norms = np.sqrt(np.sum(perturbed**2, axis=1))
            mask = norms > 0.99
            if np.any(mask):
                perturbed[mask] *= 0.99 / norms[mask, np.newaxis]
                
            p_centers, p_score = _polish_with_epigraph(
                perturbed, best_score, time_limit=10.0
            )
            
            if p_score > best_score:
                best_score = p_score
                best_centers = p_centers

    # Final validation
    if best_centers is not None:
        final_score = _score_raw(best_centers)
        if final_score > 0.0 and np.all(np.isfinite(best_centers)):
            return best_centers
    
    fallback = _generate_hexagonal_initial(0.12)
    if fallback is not None:
        return fallback
    
    return _generate_ring_initial()

# EVOLVE-BLOCK-END
