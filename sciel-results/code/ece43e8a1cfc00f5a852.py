"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree
import sys

def _compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Check bounds
    if np.any(np.abs(centers) > 0.5 + 1e-12):
        return 0.0
    # Minimum distance to boundary
    boundary_dists = 0.5 - np.abs(centers)
    min_boundary = np.min(boundary_dists)
    if min_boundary < 0:
        return 0.0
    # Minimum pairwise distance
    tree = cKDTree(centers)
    pairs = tree.query_pairs(1.0)  # rough upper bound
    if len(pairs) == 0:
        min_pair = float('inf')
    else:
        dists = []
        for i, j in pairs:
            d = np.linalg.norm(centers[i] - centers[j])
            dists.append(d)
        min_pair = min(dists)
    r = min(min_pair / 2.0, min_boundary)
    return max(r, 0.0)

def _objective(centers, r_target):
    """Objective: maximize minimum of (pairwise_dist/2, boundary_dist) - r_target.
    We minimize the negative, or equivalently maximize the radius."""
    centers = np.asarray(centers, dtype=np.float64).reshape(N, 2)
    # Boundary distances
    boundary_dists = 0.5 - np.abs(centers)
    min_boundary = np.min(boundary_dists)
    # Pairwise distances
    tree = cKDTree(centers)
    pairs = tree.query_pairs(1.0)
    if len(pairs) == 0:
        min_pair = 1.0
    else:
        min_d = 1.0
        for i, j in pairs:
            d = np.linalg.norm(centers[i] - centers[j])
            if d < min_d:
                min_d = d
        min_pair = min_d
    r = min(min_pair / 2.0, min_boundary)
    # We want to maximize r, so minimize -r
    return -r

def _soft_objective(centers, beta=50.0):
    """Smooth surrogate using log-sum-exp for the min."""
    centers = np.asarray(centers, dtype=np.float64).reshape(N, 2)
    # Boundary distances
    boundary_dists = 0.5 - np.abs(centers)
    # Pairwise distances / 2
    tree = cKDTree(centers)
    pairs = tree.query_pairs(1.0)
    pair_vals = []
    for i, j in pairs:
        d = np.linalg.norm(centers[i] - centers[j]) / 2.0
        pair_vals.append(d)
    if len(pair_vals) == 0:
        pair_vals = [1.0]
    pair_vals = np.array(pair_vals)
    all_vals = np.concatenate([boundary_dists, pair_vals])
    # Soft min via log-sum-exp: min(x) ≈ -(1/beta) * log(sum(exp(-beta*x)))
    # But we want to maximize the min, so we minimize -soft_min
    # soft_min = -(1/beta) * log(sum(exp(-beta * x)))
    # We want to maximize soft_min, so minimize -soft_min
    m = np.min(all_vals)
    if m < -1.0:  # avoid overflow
        return -m
    exp_vals = np.exp(-beta * (all_vals - m))
    soft_min = m - np.log(np.sum(exp_vals)) / beta
    return -soft_min

def _gradient_descent_polish(centers, max_iter=5000, lr_init=0.01, lr_decay=0.999):
    """Polish using subgradient method on the soft objective."""
    centers = np.array(centers, dtype=np.float64).copy()
    lr = lr_init
    best_r = _compute_radius(centers)
    best_centers = centers.copy()
    
    for it in range(max_iter):
        obj = _soft_objective(centers, beta=100.0)
        # Numerical gradient
        eps = 1e-7
        grad = np.zeros_like(centers)
        for i in range(N):
            for j in range(2):
                c_plus = centers.copy()
                c_minus = centers.copy()
                c_plus[i, j] += eps
                c_minus[i, j] -= eps
                grad[i, j] = (_soft_objective(c_plus, beta=100.0) - _soft_objective(c_minus, beta=100.0)) / (2 * eps)
        centers = centers - lr * grad
        # Project to square
        centers = np.clip(centers, -0.5, 0.5)
        lr *= lr_decay
        r = _compute_radius(centers)
        if r > best_r:
            best_r = r
            best_centers = centers.copy()
    return best_centers

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Multi-start with different initial configurations
    best_centers = None
    best_r = 0.0
    
    # Start 1: Grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    init1 = grid[:N].astype(np.float64)
    
    # Start 2: Hexagonal lattice
    def hex_init(n):
        centers = []
        # Hexagonal packing
        dx = 0.1
        dy = dx * np.sqrt(3) / 2
        x = -0.5 + dx/2
        row = 0
        while len(centers) < n:
            y = -0.5 + dy/2 + row * dy
            offset = 0 if row % 2 == 0 else dx/2
            xx = -0.5 + dx/2 + offset
            while xx < 0.5 and len(centers) < n:
                centers.append([xx, y])
                xx += dx
            row += 1
        return np.array(centers[:n], dtype=np.float64)
    
    init2 = hex_init(N)
    
    # Start 3: Random
    np.random.seed(42)
    init3 = np.random.uniform(-0.45, 0.45, size=(N, 2))
    
    inits = [init1, init2, init3]
    
    for init in inits:
        if time.time() > deadline - 30:
            break
        
        # Simulated annealing / random restarts
        centers = init.copy()
        r = _compute_radius(centers)
        
        # Try L-BFGS-B optimization
        try:
            result = minimize(
                _objective,
                centers.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            opt_centers = result.x.reshape(N, 2)
            opt_r = _compute_radius(opt_centers)
            if opt_r > r:
                r = opt_r
                centers = opt_centers
        except Exception:
            pass
        
        if r > best_r:
            best_r = r
            best_centers = centers.copy()
        
        # Polish
        if time.time() < deadline - 10:
            polished = _gradient_descent_polish(centers, max_iter=2000, lr_init=0.005)
            pol_r = _compute_radius(polished)
            if pol_r > best_r:
                best_r = pol_r
                best_centers = polished.copy()
    
    # Final polish on best
    if best_centers is not None and time.time() < deadline - 5:
        final_polished = _gradient_descent_polish(best_centers, max_iter=5000, lr_init=0.002)
        final_r = _compute_radius(final_polished)
        if final_r > best_r:
            best_r = final_r
            best_centers = final_polished
    
    if best_centers is None:
        return init1
    
    # Validate
    r_final = _compute_radius(best_centers)
    if r_final <= 0:
        return init1
    
    return best_centers
# EVOLVE-BLOCK-END
