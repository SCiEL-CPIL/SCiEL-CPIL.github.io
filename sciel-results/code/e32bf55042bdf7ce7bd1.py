"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START
def _hexagonal_initial(n=50):
    """Create initial hexagonal lattice points inside unit disk."""
    # Use a denser hexagonal lattice and select best n points
    best_points = None
    best_score = -1
    
    for base_spacing in [0.22, 0.23, 0.24, 0.25, 0.26]:
        spacing = base_spacing
        for _ in range(20):
            k = int(1.0 / spacing) + 2
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            points = np.array(rows)
            radius = np.hypot(points[:, 0], points[:, 1])
            mask = radius <= 1.0 - spacing / 2.0
            points = points[mask]
            if len(points) >= n:
                # Select n points with best packing
                # Sort by distance from center, take innermost n
                dists = np.hypot(points[:, 0], points[:, 1])
                idx = np.argsort(dists)[:n]
                pts = points[idx]
                # Compute score
                min_dist = 1e10
                for i in range(n):
                    for j in range(i+1, n):
                        d = np.linalg.norm(pts[i] - pts[j])
                        if d < min_dist:
                            min_dist = d
                    if np.linalg.norm(pts[i]) > 1.0:
                        min_dist = 0
                score = min(min_dist / 2.0, 1.0 - np.max(np.linalg.norm(pts, axis=1)))
                if score > best_score:
                    best_score = score
                    best_points = pts.copy()
            spacing *= 0.97
    
    if best_points is None:
        # Fallback: uniform random
        rng = np.random.default_rng(42)
        best_points = np.zeros((n, 2))
        for i in range(n):
            while True:
                angle = rng.uniform(0, 2*np.pi)
                r = np.sqrt(rng.uniform(0, 1))
                pt = np.array([r*np.cos(angle), r*np.sin(angle)])
                if np.all(np.linalg.norm(pt - best_points[:i], axis=1) > 0.15):
                    best_points[i] = pt
                    break
    return best_points


def _compute_score(points):
    """Compute the packing score: min(min_pair_dist/2, min(1-|c_i|))."""
    n = len(points)
    if n == 0:
        return 0.0
    
    # Min distance from any center to boundary
    dists_to_boundary = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = np.min(dists_to_boundary)
    
    if min_boundary <= 0:
        return 0.0
    
    # Min pairwise distance using cdist for speed
    D = cdist(points, points)
    # Mask out diagonal
    np.fill_diagonal(D, np.inf)
    min_pair = np.min(D)
    
    return min(min_pair / 2.0, min_boundary)


def _smooth_objective(x, beta=50.0):
    """Smooth surrogate: maximize t s.t. t <= d_ij/2 for all i<j, t <= 1-|c_i| for all i.
    We minimize -t + penalty for violations using log-sum-exp."""
    points = x.reshape(-1, 2)
    n = len(points)
    
    # Pairwise distances
    D = cdist(points, points)
    np.fill_diagonal(D, np.inf)
    pair_dists = D[D > 0]
    
    # Boundary distances
    boundary_dists = 1.0 - np.linalg.norm(points, axis=1)
    
    # All constraints: pair_dists/2 and boundary_dists
    all_dists = np.concatenate([pair_dists / 2.0, boundary_dists])
    
    # Log-sum-exp for smooth minimum
    max_d = np.max(all_dists)
    if max_d <= 0:
        return 1.0  # Large penalty
    
    log_sum = np.log(np.sum(np.exp(-beta * (all_dists - max_d)))) / beta + max_d
    # We want to maximize the minimum, so minimize negative smooth min
    return -log_sum


def _polish(points, time_budget=20.0):
    """High-precision polish using scipy minimize with smooth objective."""
    start_time = time.time()
    deadline = start_time + time_budget
    
    x0 = points.flatten()
    
    best_points = points.copy()
    best_score = _compute_score(points)
    
    for beta in [30.0, 50.0, 100.0, 200.0]:
        if time.time() > deadline - 5:
            break
        
        result = minimize(
            _smooth_objective,
            x0,
            args=(beta,),
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        
        if result.fun < 0:  # Valid improvement in smooth objective
            new_points = result.x.reshape(-1, 2)
            new_score = _compute_score(new_points)
            if new_score > best_score:
                best_points = new_points
                best_score = new_score
                x0 = result.x
    
    return best_points, best_score


def _simulated_annealing(points, time_budget=100.0):
    """Run simulated annealing to improve the packing."""
    n = len(points)
    points = points.copy()
    best_points = points.copy()
    best_score = _compute_score(points)
    
    start_time = time.time()
    deadline = start_time + time_budget
    
    # Initial temperature
    T = 0.01
    T_min = 1e-10
    cooling_rate = 0.9995
    step_scale = 0.05
    
    rng = np.random.default_rng(12345)
    
    while time.time() < deadline and T > T_min:
        # Propose a random move
        idx = rng.integers(0, n)
        move_type = rng.random()
        
        if move_type < 0.7:
            # Random displacement
            delta = (rng.random(2) - 0.5) * 2 * step_scale * T * 10
            new_points = points.copy()
            new_points[idx] = points[idx] + delta
        elif move_type < 0.85:
            # Swap two points
            jdx = rng.integers(0, n)
            if jdx == idx:
                continue
            new_points = points.copy()
            new_points[idx], new_points[jdx] = points[jdx].copy(), points[idx].copy()
        else:
            # Reflect a point through origin (with noise)
            new_points = points.copy()
            new_points[idx] = -points[idx] + (rng.random(2) - 0.5) * 0.01
        
        # Check validity
        if np.any(np.linalg.norm(new_points, axis=1) > 1.0):
            continue
        
        new_score = _compute_score(new_points)
        
        # Accept or reject
        if new_score >= best_score or rng.random() < np.exp((new_score - best_score) / T):
            points = new_points
            if new_score > best_score:
                best_score = new_score
                best_points = new_points.copy()
        
        T *= cooling_rate
    
    return best_points, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Generate initial guess
    points = _hexagonal_initial(N)
    current_score = _compute_score(points)
    
    # Run simulated annealing
    points, current_score = _simulated_annealing(points, time_budget=80.0)
    
    # If time remains, try a few more restarts with different seeds
    elapsed = time.time() - start_time
    if elapsed < deadline - 30:
        for seed in [7, 13, 42, 99]:
            if time.time() > deadline - 35:
                break
            rng = np.random.default_rng(seed)
            # Perturb the best solution
            perturbed = points + (rng.random((N, 2)) - 0.5) * 0.02
            # Ensure inside unit circle
            norms = np.linalg.norm(perturbed, axis=1)
            mask = norms > 1.0
            if np.any(mask):
                perturbed[mask] /= norms[mask][:, None] * 1.01
            p, s = _simulated_annealing(perturbed, time_budget=min(15, deadline - 35 - (time.time() - start_time)))
            if s > current_score:
                points = p
                current_score = s
    
    # Final high-precision polish pass
    remaining = deadline - 10 - (time.time() - start_time)
    if remaining > 5:
        points, current_score = _polish(points, time_budget=remaining)
    
    return points
# EVOLVE-BLOCK-END
