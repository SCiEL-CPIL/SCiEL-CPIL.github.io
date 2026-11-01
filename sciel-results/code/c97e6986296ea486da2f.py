import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 145.0  # Safety margin for 150s limit

def _compute_score(centers):
    """Compute the packing score for given centers."""
    n = len(centers)
    if n < 2:
        return 0.0
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    norms = np.sqrt(np.sum(centers * centers, axis=1))
    min_bound = np.min(1.0 - norms)
    return min(min_pair / 2.0, min_bound)

def _soft_min_obj_and_grad(params, n, beta):
    """
    Objective: Minimize the log-sum-exp soft minimum of clearances.
    This is equivalent to maximizing the approximate minimum clearance.
    """
    centers = params.reshape(n, 2)
    
    # Calculate all clearances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    pair_dists = dists[i_idx, j_idx]
    pair_clearances = pair_dists / 2.0
    
    norms = np.sqrt(np.sum(centers * centers, axis=1))
    bound_clearances = 1.0 - norms
    
    # Combine all clearances into a single vector
    # To avoid numerical issues with exp, we subtract the max clearance (which is min of -clearances)
    # Actually, we are computing sum(exp(-beta * c_k)).
    # Let c_min_approx be the minimum clearance.
    # To stabilize exp(-beta * c_k), we can factor out exp(-beta * c_ref).
    # Let c_ref = min(clearances). Then exp(-beta * c_k) = exp(-beta * (c_k - c_ref)) * exp(-beta * c_ref).
    # The term exp(-beta * c_ref) is constant for the gradient direction but part of the value.
    # However, for gradient calculation, we just need the weights w_k = exp(-beta * c_k) / S.
    # The factor exp(-beta * c_ref) cancels out in the weights.
    
    all_clearances = np.concatenate([pair_clearances, bound_clearances])
    
    # Numerical stability: shift by min clearance
    c_min = np.min(all_clearances)
    shifted = -beta * (all_clearances - c_min)
    
    # Clip to prevent overflow/underflow issues in exp, though shifted <= 0
    # If beta is large and c_k is much larger than c_min, shifted is very negative -> exp is 0.
    # If c_k is c_min, shifted is 0 -> exp is 1.
    exp_terms = np.exp(shifted)
    S = np.sum(exp_terms)
    
    if S == 0 or not np.isfinite(S):
        return 0.0, np.zeros_like(params)
        
    # Objective value: (1/beta) * ln(S * exp(-beta * c_min)) = (1/beta) * (ln(S) - beta * c_min) = -c_min + (1/beta)*ln(S)
    obj_val = -c_min + (1.0 / beta) * np.log(S)
    
    # Weights w_k = exp(-beta * c_k) / S = exp(shifted) / S
    weights = exp_terms / S
    
    # Gradient calculation
    grad = np.zeros_like(centers)
    
    # Pairwise contributions
    # d(obj)/d(c_i) = - sum_{k in pairs involving i} w_k * (1/2) * (c_i - c_j)/||c_i - c_j||
    # Note: w_k is associated with clearance c_k.
    # For pair (i, j), clearance c_k = ||c_i - c_j|| / 2.
    # grad_c_k = (c_i - c_j) / ||c_i - c_j|| for c_i part, and (c_j - c_i) / ||c_i - c_j|| for c_j part.
    # The factor 1/2 is in the clearance definition, so grad of clearance is (c_i-c_j)/d.
    # Wait, c_k = 0.5 * d. grad(c_k) = 0.5 * (c_i-c_j)/d.
    
    for k in range(len(i_idx)):
        i, j = i_idx[k], j_idx[k]
        w = weights[k]
        if w < 1e-10:
            continue
        d = pair_dists[k]
        if d < 1e-10:
            continue
        vec = (centers[i] - centers[j]) / d
        # Contribution to grad from this clearance
        # d(obj)/d(c_i) += -w * grad(c_k)
        grad[i] -= w * 0.5 * vec
        grad[j] += w * 0.5 * vec
        
    # Boundary contributions
    # d(obj)/d(c_i) += -w * grad(c_bound)
    # c_bound = 1 - ||c_i||. grad(c_bound) = -c_i / ||c_i||.
    for i in range(n):
        w = weights[len(i_idx) + i]
        if w < 1e-10:
            continue
        norm = norms[i]
        if norm < 1e-10:
            continue
        vec = centers[i] / norm
        grad[i] -= w * (-vec) # -w * (-vec) = w * vec
        
    return obj_val, grad

def _get_hex_lattice(n):
    """Generate a hexagonal lattice and pick n points inside unit circle."""
    # Estimate radius to fit n points
    # Area ~ n * pi * r^2 <= pi * 1^2 => r ~ 1/sqrt(n)
    r_est = 0.12
    s = 2 * r_est
    
    points = []
    max_range = int(2.0 / s) + 3
    for j in range(-max_range, max_range + 1):
        for i in range(-max_range, max_range + 1):
            x = i * s + (j % 2) * s / 2.0
            y = j * s * np.sqrt(3) / 2.0
            if np.hypot(x, y) <= 0.95:
                points.append([x, y])
    
    if len(points) < n:
        # Fallback to random if lattice doesn't have enough
        rng = np.random.default_rng(42)
        while len(points) < n:
            angle = rng.uniform(0, 2 * np.pi)
            radius = rng.uniform(0, 0.9)
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            points.append([x, y])
    
    points = np.array(points)
    # Sort by distance to center to keep the most central ones
    dists = np.linalg.norm(points, axis=1)
    indices = np.argsort(dists)
    return points[indices[:n]]

def _get_ring_initialization(n, seed=0):
    """Generate initial points on concentric rings."""
    rng = np.random.default_rng(seed)
    points = []
    
    # Center point
    points.append([0.0, 0.0])
    
    # Ring 1: 6 points
    for i in range(6):
        angle = 2 * np.pi * i / 6
        r = 0.25
        points.append([r * np.cos(angle), r * np.sin(angle)])
        
    # Ring 2: 12 points
    for i in range(12):
        angle = 2 * np.pi * i / 12 + np.pi / 12
        r = 0.5
        points.append([r * np.cos(angle), r * np.sin(angle)])
        
    # Ring 3: 18 points
    for i in range(18):
        angle = 2 * np.pi * i / 18
        r = 0.75
        points.append([r * np.cos(angle), r * np.sin(angle)])
        
    # Ring 4: remaining
    remaining = n - len(points)
    for i in range(remaining):
        angle = 2 * np.pi * i / max(remaining, 1)
        r = 0.9
        points.append([r * np.cos(angle), r * np.sin(angle)])
        
    return np.array(points[:n])

def _get_random_initialization(n, seed=0):
    """Generate random initial points."""
    rng = np.random.default_rng(seed)
    points = []
    for _ in range(n):
        angle = rng.uniform(0, 2 * np.pi)
        radius = rng.uniform(0, 0.95)
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        points.append([x, y])
    return np.array(points)

def _optimize_from_start(centers, start_time, deadline):
    """
    Optimize from a single start using annealed beta for soft-min objective.
    """
    best_centers = centers.copy()
    best_score = _compute_score(best_centers)
    
    # Annealing schedule for beta
    # Start with small beta (smooth) and increase to focus on min
    current_beta = 10.0
    
    # We will run L-BFGS-B for a certain number of steps or time, then increase beta
    # and continue from the current best point.
    
    while time.time() < deadline - 5:
        # Determine how much time to spend on this beta level
        # Just run for a fixed number of iterations or until time limit
        try:
            res = minimize(
                _soft_min_obj_and_grad,
                best_centers.flatten(),
                args=(N, current_beta),
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 100, 'gtol': 1e-8}
            )
            
            new_centers = res.x.reshape(N, 2)
            
            if not np.all(np.isfinite(new_centers)):
                break
                
            new_score = _compute_score(new_centers)
            
            if new_score > best_score:
                best_score = new_score
                best_centers = new_centers.copy()
            else:
                # If no improvement, we might be stuck or beta is too high
                # Try a small perturbation to escape local minima
                rng = np.random.default_rng()
                best_centers += rng.normal(0, 0.001, size=best_centers.shape)
                # Ensure inside circle
                norms = np.linalg.norm(best_centers, axis=1)
                mask = norms > 0.99
                if np.any(mask):
                    best_centers[mask] *= 0.99 / norms[mask]
                    
        except Exception:
            break
            
        # Increase beta to sharpen the minimum
        current_beta *= 1.5
        if current_beta > 1000:
            break
            
    return best_centers, best_score

def _construct():
    """Main construction function."""
    start_time = time.time()
    
    # Generate multiple initializations
    initializations = [
        _get_hex_lattice(N),
        _get_ring_initialization(N, seed=0),
        _get_random_initialization(N, seed=42),
        _get_random_initialization(N, seed=123),
        _get_ring_initialization(N, seed=1),
        _get_random_initialization(N, seed=999),
    ]
    
    best_centers = None
    best_score = 0.0
    
    # Optimize from each initialization
    for init in initializations:
        if time.time() > DEADLINE - 10:
            break
            
        centers, score = _optimize_from_start(init, start_time, DEADLINE)
        
        if score > best_score:
            best_score = score
            best_centers = centers.copy()
            
    if best_centers is None:
        return initializations[0]
        
    # Final polish: Use a very high beta to refine the minimum clearance
    if time.time() < DEADLINE - 2:
        try:
            # Use a high beta for final polishing
            res = minimize(
                _soft_min_obj_and_grad,
                best_centers.flatten(),
                args=(N, 1000.0),
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 200, 'gtol': 1e-10}
            )
            new_centers = res.x.reshape(N, 2)
            new_score = _compute_score(new_centers)
            if new_score > best_score:
                best_score = new_score
                best_centers = new_centers
        except Exception:
            pass

    return best_centers

def construct():
    """Entry point: returns array of shape (N, 2) with circle centers."""
    return _construct()
