"""Rewritten program for packing 50 equal circles in a unit circle.
Uses soft-min objective with L-BFGS-B optimization and high-beta polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 145.0  # Safety margin for 150s limit


def _score(centers):
    """Compute the packing score for given centers."""
    n = len(centers)
    if n < 2:
        return 0.0
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    norms = np.sqrt(np.sum(centers * centers, axis=1))
    min_bound = np.min(1.0 - norms)
    return min(min_pair, min_bound)


def _soft_min_obj_and_grad(centers_flat, beta):
    """
    Soft-min objective: -softmin of all clearances.
    Clearances = pair distances/2 and (1 - norm) for each point.
    Minimizing -softmin maximizes the minimum clearance.
    """
    n = N
    centers = centers_flat.reshape(n, 2)
    
    # Compute all pair distances / 2
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    i_idx, j_idx = np.triu_indices(n, k=1)
    pair_clearances = dists[i_idx, j_idx] / 2.0
    
    # Boundary clearances
    norms = np.sqrt(np.sum(centers * centers, axis=1))
    bound_clearances = 1.0 - norms
    
    # All clearances
    all_clearances = np.concatenate([pair_clearances, bound_clearances])
    
    # Soft min: -log(sum(exp(-beta * x))) / beta
    # Use log-sum-exp trick for numerical stability
    shifted = -beta * all_clearances
    max_shifted = np.max(shifted)
    log_sum = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
    soft_min_val = -log_sum / beta
    
    # Gradient of soft_min w.r.t. centers
    # d(soft_min)/dx = -sum(exp(-beta*x_i) * dx_i/dx) / sum(exp(-beta*x_i))
    exp_vals = np.exp(shifted - max_shifted)
    weights = exp_vals / np.sum(exp_vals)  # softmax weights
    
    grad = np.zeros_like(centers)
    
    # Pair clearance gradients
    for k in range(len(pair_clearances)):
        w = weights[k]
        if w < 1e-15:
            continue
        i, j = i_idx[k], j_idx[k]
        d = dists[i, j]
        if d > 1e-10:
            vec = (centers[i] - centers[j]) / d
            # d(pair_clearance_k)/d(centers[i]) = vec / 2
            # d(pair_clearance_k)/d(centers[j]) = -vec / 2
            grad[i] += w * vec * 0.5
            grad[j] -= w * vec * 0.5
    
    # Boundary clearance gradients
    for i in range(n):
        w = weights[n * (n - 1) // 2 + i]
        if w < 1e-15:
            continue
        if norms[i] > 1e-10:
            # d(bound_clearance_i)/d(centers[i]) = -centers[i]/norm
            grad[i] -= w * centers[i] / norms[i]
    
    # Objective is -soft_min, so negate gradient
    obj = -soft_min_val
    grad = -grad
    
    return obj, grad.flatten()


def _optimize_lbfgs(centers, beta, maxiter=200, gtol=1e-8):
    """Run L-BFGS-B on the soft-min objective."""
    try:
        res = minimize(
            _soft_min_obj_and_grad,
            centers.flatten(),
            args=(beta,),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': maxiter, 'gtol': gtol}
        )
        if res.success or res.status in [1, 2, 4]:
            new_centers = res.x.reshape(N, 2)
            if np.all(np.isfinite(new_centers)):
                return new_centers
    except Exception:
        pass
    return centers


def _get_hex_lattice(n):
    """Generate hexagonal lattice points inside unit circle."""
    r_est = 0.125
    s = 2 * r_est
    
    points = []
    max_range = int(2.0 / s) + 3
    for j in range(-max_range, max_range + 1):
        for i in range(-max_range, max_range + 1):
            x = i * s + (j % 2) * s / 2.0
            y = j * s * np.sqrt(3) / 2.0
            if np.hypot(x, y) <= 0.98:
                points.append([x, y])
    
    if len(points) == 0:
        rng = np.random.default_rng(42)
        points = [[rng.uniform(-0.9, 0.9), rng.uniform(-0.9, 0.9)] for _ in range(n)]
    
    points = np.array(points)
    
    if len(points) < n:
        rng = np.random.default_rng(42)
        while len(points) < n:
            angle = rng.uniform(0, 2 * np.pi)
            radius = rng.uniform(0, 0.9)
            points = np.vstack([points, [[radius * np.cos(angle), radius * np.sin(angle)]]])
    
    # Sort by distance from center, take closest n
    dists = np.linalg.norm(points, axis=1)
    indices = np.argsort(dists)
    return points[indices[:n]]


def _get_ring_init(n, seed=0):
    """Ring-based initialization matching known optimal structure."""
    rng = np.random.default_rng(seed)
    points = []
    
    # Center
    points.append([0.0, 0.0])
    
    # Ring 1: 6 points
    for i in range(6):
        angle = 2 * np.pi * i / 6
        r = 0.252
        points.append([r * np.cos(angle), r * np.sin(angle)])
    
    # Ring 2: 12 points
    for i in range(12):
        angle = 2 * np.pi * i / 12 + np.pi / 12
        r = 0.504
        points.append([r * np.cos(angle), r * np.sin(angle)])
    
    # Ring 3: 18 points
    for i in range(18):
        angle = 2 * np.pi * i / 18
        r = 0.756
        points.append([r * np.cos(angle), r * np.sin(angle)])
    
    # Ring 4: remaining
    remaining = n - len(points)
    for i in range(remaining):
        angle = 2 * np.pi * i / max(remaining, 1) + np.pi / max(remaining, 1)
        r = 0.91
        points.append([r * np.cos(angle), r * np.sin(angle)])
    
    return np.array(points[:n])


def _get_random_init(n, seed=0):
    """Random initialization."""
    rng = np.random.default_rng(seed)
    points = []
    for _ in range(n):
        angle = rng.uniform(0, 2 * np.pi)
        radius = rng.uniform(0, 0.95)
        points.append([radius * np.cos(angle), radius * np.sin(angle)])
    return np.array(points)


def _polish(centers, start_time, deadline, beta_start=50.0, beta_end=500.0, n_steps=10):
    """
    Progressive beta polish: start with moderate beta, increase to high beta.
    High beta makes soft-min approximate true min, refining the exact bottleneck.
    """
    best = centers.copy()
    best_score = _score(best)
    
    betas = np.linspace(beta_start, beta_end, n_steps)
    
    for beta in betas:
        if time.time() > deadline - 3:
            break
        
        new_centers = _optimize_lbfgs(best, beta, maxiter=100, gtol=1e-9)
        new_score = _score(new_centers)
        
        if new_score > best_score:
            best_score = new_score
            best = new_centers.copy()
    
    return best, best_score


def _feasibility_search(centers, start_time, deadline):
    """
    Try to improve by finding a feasible configuration with larger radius.
    Uses iterative: increase target radius, solve feasibility via soft-min.
    """
    best = centers.copy()
    best_score = _score(best)
    
    # Try increasing the minimum clearance
    target = best_score * 1.001
    max_tries = 50
    
    for _ in range(max_tries):
        if time.time() > deadline - 5:
            break
        
        # Use soft-min with high beta to push all clearances above target
        beta = 200.0
        new_centers = _optimize_lbfgs(best, beta, maxiter=50, gtol=1e-10)
        new_score = _score(new_centers)
        
        if new_score > best_score:
            best_score = new_score
            best = new_centers.copy()
            target = best_score * 1.0005
        else:
            # Slightly perturb to escape local minimum
            rng = np.random.default_rng(int(time.time() * 1000) % 10000)
            perturbed = best + rng.normal(0, 0.001, size=best.shape)
            # Keep inside circle
            norms = np.linalg.norm(perturbed, axis=1)
            mask = norms > 0.999
            if np.any(mask):
                perturbed[mask] *= 0.999 / norms[mask]
            best = perturbed
    
    return best, best_score


def _construct():
    """Main construction function."""
    start_time = time.time()
    
    # Generate multiple initializations
    initializations = [
        _get_hex_lattice(N),
        _get_ring_init(N, seed=0),
        _get_random_init(N, seed=42),
        _get_ring_init(N, seed=1),
        _get_random_init(N, seed=123),
    ]
    
    best_centers = None
    best_score = 0.0
    
    # Optimize from each initialization
    for init in initializations:
        if time.time() > DEADLINE - 40:
            break
        
        # First: moderate beta optimization
        centers = _optimize_lbfgs(init, beta=20.0, maxiter=200, gtol=1e-8)
        
        # Then: progressive polish
        centers, score = _polish(centers, start_time, DEADLINE - 20, 
                                   beta_start=50.0, beta_end=500.0, n_steps=8)
        
        if score > best_score:
            best_score = score
            best_centers = centers.copy()
    
    if best_centers is None:
        return initializations[0]
    
    # Final high-beta polish on the best result
    if time.time() < DEADLINE - 10:
        best_centers, best_score = _polish(best_centers, start_time, DEADLINE - 5,
                                            beta_start=300.0, beta_end=2000.0, n_steps=10)
    
    # Feasibility search for final improvement
    if time.time() < DEADLINE - 3:
        best_centers, best_score = _feasibility_search(best_centers, start_time, DEADLINE - 1)
    
    return best_centers


def construct():
    """Entry point: returns array of shape (N, 2) with circle centers."""
    return _construct()
