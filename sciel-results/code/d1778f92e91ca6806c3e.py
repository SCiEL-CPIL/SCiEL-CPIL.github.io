"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree

def _compute_score(points):
    """Compute the packing radius for given points."""
    if len(points) != N:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    # Check inside container
    dists = np.linalg.norm(points, axis=1)
    if np.any(dists >= 1.0):
        return 0.0
    # Check pairwise distances
    tree = cKDTree(points)
    pairs = tree.query_pairs(2.0)  # pairs with distance < 2.0
    if len(pairs) == 0:
        min_pair_dist = float('inf')
    else:
        pair_dists = []
        for i, j in pairs:
            d = np.linalg.norm(points[i] - points[j])
            pair_dists.append(d)
        min_pair_dist = min(pair_dists) if pair_dists else float('inf')
    
    r = min(min_pair_dist / 2.0, 1.0 - np.max(dists))
    return r

def _initial_hexagonal():
    """Generate initial hexagonal lattice packing."""
    # Start with a reasonable hexagonal packing
    spacing = 2.0 / (np.sqrt(N) * 0.7)
    while True:
        k = int(1.0 / spacing) + 3
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        mask = radius <= 1.0 - spacing / 2.0
        points = points[mask]
        if len(points) >= N:
            # Sort by radius to prefer central points, then take N
            points = points[np.argsort(radius, kind="stable")[:N]]
            return points
        spacing *= 0.95

def _soft_min_penalty(points, beta=50.0):
    """Compute soft-min of all pairwise distances and boundary distances."""
    # Pairwise distances
    tree = cKDTree(points)
    pairs = tree.query_pairs(1.0)  # Only consider close pairs for efficiency
    if len(pairs) > 0:
        pair_dists = np.array([np.linalg.norm(points[i] - points[j]) for i, j in pairs])
        # Soft min
        if len(pair_dists) > 0:
            pair_min = -np.logaddexp.reduce(-pair_dists / beta) * beta
        else:
            pair_min = float('inf')
    else:
        pair_min = float('inf')
    
    # Boundary distances
    dists = np.linalg.norm(points, axis=1)
    boundary_dists = 1.0 - dists
    if len(boundary_dists) > 0:
        boundary_min = -np.logaddexp.reduce(-boundary_dists / beta) * beta
    else:
        boundary_min = float('inf')
    
    return min(pair_min, boundary_min)

def construct():
    """Construct 300 circle centers in unit disk maximizing minimum radius."""
    t0 = time.time()
    deadline = t0 + 110.0  # Safety margin before 120s limit
    
    # Initial hexagonal packing
    points = _initial_hexagonal()
    best_points = points.copy()
    best_score = _compute_score(points)
    
    # Optimization using scipy minimize with a custom objective
    # We minimize the negative of a soft-min approximation of the packing radius
    
    def objective(x):
        if time.time() > deadline - 5.0:
            return 1e6
        pts = x.reshape(N, 2)
        # Penalty for going outside
        dists = np.linalg.norm(pts, axis=1)
        outside_penalty = np.sum(np.maximum(0, dists - 1.0)**2) * 100
        # Soft min of pairwise and boundary distances
        beta = 30.0
        # Pairwise: use KDTree for efficiency
        try:
            tree = cKDTree(pts)
            pairs = tree.query_pairs(0.5)
            if len(pairs) > 0:
                pair_dists = np.array([np.linalg.norm(pts[i] - pts[j]) for i, j in pairs])
                pair_soft_min = -np.logaddexp.reduce(-pair_dists / beta) * beta
            else:
                pair_soft_min = 1.0
        except:
            pair_soft_min = 0.0
        
        boundary_dists = 1.0 - dists
        boundary_soft_min = -np.logaddexp.reduce(-boundary_dists / beta) * beta
        
        soft_min = min(pair_soft_min, boundary_soft_min)
        return -soft_min + outside_penalty
    
    def gradient(x):
        if time.time() > deadline - 5.0:
            return np.ones_like(x)
        pts = x.reshape(N, 2)
        beta = 30.0
        
        # Compute gradient numerically or analytically
        # For simplicity, use finite differences with small step
        eps = 1e-7
        grad = np.zeros_like(x)
        f0 = objective(x)
        
        for i in range(0, len(x), 2):
            for di in range(2):
                x[i + di] += eps
                f_plus = objective(x)
                x[i + di] -= 2 * eps
                f_minus = objective(x)
                x[i + di] += eps
                grad[i + di] = (f_plus - f_minus) / (2 * eps)
        
        return grad
    
    # Use L-BFGS-B for optimization
    x0 = points.flatten()
    
    # Try multiple restarts with different initial conditions
    for restart in range(3):
        if time.time() > deadline - 30.0:
            break
        
        if restart == 0:
            x0 = points.flatten()
        else:
            # Add some random perturbation
            np.random.seed(42 + restart)
            x0 = points.flatten() + np.random.normal(0, 0.01, size=N * 2)
        
        try:
            result = minimize(objective, x0, method='L-BFGS-B', 
                            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8,
                                   'maxls': 20, 'disp': False})
            if time.time() > deadline - 10.0:
                break
            
            new_points = result.x.reshape(N, 2)
            # Clip to ensure inside unit disk
            dists = np.linalg.norm(new_points, axis=1)
            mask = dists > 1.0
            if np.any(mask):
                new_points[mask] = new_points[mask] / dists[mask, None] * 0.999
            
            new_score = _compute_score(new_points)
            if new_score > best_score:
                best_score = new_score
                best_points = new_points.copy()
        except Exception as e:
            pass
    
    # Final refinement: local search on the best points
    if time.time() < deadline - 15.0:
        # Simple iterative improvement: move each point to increase min distance
        for iteration in range(100):
            if time.time() > deadline - 10.0:
                break
            improved = False
            for i in range(N):
                if time.time() > deadline - 10.0:
                    break
                # Try small moves in random directions
                for _ in range(5):
                    direction = np.random.randn(2)
                    direction /= np.linalg.norm(direction)
                    for step in [0.001, 0.0005, 0.002]:
                        new_pt = best_points[i] + direction * step
                        if np.linalg.norm(new_pt) < 1.0:
                            old_score = _compute_score(best_points)
                            best_points[i] = new_pt
                            new_score = _compute_score(best_points)
                            if new_score > old_score:
                                improved = True
                                break
                            else:
                                best_points[i] -= direction * step
            if not improved:
                break
    
    return best_points
# EVOLVE-BLOCK-END

