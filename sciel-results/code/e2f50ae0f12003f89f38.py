"""Program for packing 50 equal circles in a unit circle; SEARCH approach with deadline guard."""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 50
DEADLINE = time.time() + 140.0  # 140s safety margin out of 150s


def compute_radius(points):
    """Compute the packing radius for given centres."""
    points = np.asarray(points, dtype=np.float64)
    # Distance to boundary
    boundary_dists = 1.0 - np.linalg.norm(points, axis=1)
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    return min(boundary_dists.min(), min_pair_dist / 2.0)


def fast_radius(points):
    """Fast radius computation using only nearest neighbors via cdist-like approach."""
    points = np.asarray(points, dtype=np.float64)
    boundary_dists = 1.0 - np.linalg.norm(points, axis=1)
    # Use squared distances for speed, take sqrt only for min
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    sq_dists = np.sum(diff * diff, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    min_pair_dist = np.sqrt(np.min(sq_dists))
    return min(boundary_dists.min(), min_pair_dist / 2.0)


def seed_hexagonal():
    """Generate a hexagonal lattice packing as a seed."""
    spacing = 0.2
    points_list = []
    # Generate hex lattice points
    for j in range(-5, 6):
        for i in range(-5, 6):
            x = i * spacing + (0.5 * spacing if j % 2 else 0)
            y = j * spacing * np.sqrt(3) / 2
            if np.sqrt(x * x + y * y) < 1.0:
                points_list.append([x, y])
    
    points = np.array(points_list)
    # Sort by distance from center to prefer central points
    dists = np.linalg.norm(points, axis=1)
    points = points[np.argsort(dists)]
    
    # Select N points with largest minimum spacing
    if len(points) >= N:
        # Greedy selection: pick point farthest from all selected
        selected = [points[np.argmin(dists)]]
        while len(selected) < N:
            sel = np.array(selected)
            min_dists = np.min(np.linalg.norm(points[:, np.newaxis, :] - sel[np.newaxis, :, :], axis=2), axis=1)
            selected.append(points[np.argmax(min_dists)])
        return np.array(selected)
    return points


def seed_random():
    """Generate a random valid packing."""
    # Try random points with rejection
    for _ in range(10000):
        points = np.random.uniform(-1, 1, size=(N, 2))
        # Scale to fit inside unit circle
        norms = np.linalg.norm(points, axis=1)
        max_norm = norms.max()
        if max_norm > 0:
            points *= (0.95 / max_norm)
        r = compute_radius(points)
        if r > 0.1:
            break
    return points


def optimize_from_seed(seed_points, max_iter=5000, timeout=15.0):
    """Local optimization using L-BFGS-B to maximize the packing radius with smooth epigraph."""
    start_time = time.time()
    
    # Use epigraph formulation: maximize t subject to:
    #   t <= 1 - |x_i| for all i
    #   t <= |x_i - x_j|/2 for all i<j
    # We minimize -t with smooth constraints using log-sum-exp barrier
    
    beta = 80.0  # sharpness
    
    def objective(x):
        points = x.reshape(N, 2)
        boundary_dists = 1.0 - np.linalg.norm(points, axis=1)
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        sq_dists = np.sum(diff * diff, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        dists = np.sqrt(sq_dists)
        
        # Smooth min for boundary: approx min(boundary_dists)
        b_min = -np.log(np.sum(np.exp(-beta * boundary_dists))) / beta
        # Smooth min for pairs: approx min(dists/2)
        d_min = -np.log(np.sum(np.exp(-beta * dists))) / beta / 2.0
        
        # Use soft-min of the two to get a smooth surrogate for r = min(b_min, d_min)
        # softmin(a,b) = -1/beta * log(exp(-beta*a) + exp(-beta*b))
        r = -np.log(np.exp(-beta * b_min) + np.exp(-beta * d_min)) / beta
        return -r
    
    def jac(x):
        points = x.reshape(N, 2)
        boundary_dists = 1.0 - np.linalg.norm(points, axis=1)
        norms = np.linalg.norm(points, axis=1)
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        sq_dists = np.sum(diff * diff, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        dists = np.sqrt(sq_dists)
        
        b_min = -np.log(np.sum(np.exp(-beta * boundary_dists))) / beta
        d_min = -np.log(np.sum(np.exp(-beta * dists))) / beta / 2.0
        
        # Gradient of softmin
        eb = np.exp(-beta * b_min)
        ed = np.exp(-beta * d_min)
        denom = eb + ed
        wb = eb / denom  # weight for boundary term
        wd = ed / denom  # weight for pair term
        
        # Gradient of b_min w.r.t. points
        gb = np.zeros_like(points)
        exp_b = np.exp(-beta * boundary_dists)
        sum_exp_b = np.sum(exp_b)
        # d(-log(sum(exp(-beta*b_i))))/d(b_i) = -beta * exp_b_i / sum_exp_b
        # d(b_i)/d(x_i) = -x_i / |x_i|
        for i in range(N):
            if norms[i] > 1e-10:
                gb[i] = wb * (-beta * exp_b[i] / sum_exp_b) * (-points[i] / norms[i])
        
        # Gradient of d_min w.r.t. points
        gd = np.zeros_like(points)
        exp_d = np.exp(-beta * dists)
        sum_exp_d = np.sum(exp_d)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                if dists[i, j] > 1e-10:
                    # d(dists[i,j])/d(x_i) = (x_i - x_j) / dists[i,j]
                    # d(-log(sum(exp(-beta*d))))/d(d_ij) = -beta * exp_d_ij / sum_exp_d
                    grad_dij = -beta * exp_d[i, j] / sum_exp_d
                    gd[i] += wd * grad_dij * (points[i] - points[j]) / dists[i, j]
        
        grad = gb + gd
        return grad.flatten()
    
    # Constraints: points must be inside unit circle
    bounds = [(-1.0, 1.0)] * (2 * N)
    
    result = minimize(
        objective,
        seed_points.flatten(),
        jac=jac,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': max_iter, 'maxfun': max_iter * 3, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    
    elapsed = time.time() - start_time
    optimized = result.x.reshape(N, 2)
    
    # Ensure points are strictly inside
    norms = np.linalg.norm(optimized, axis=1)
    too_far = norms >= 1.0
    if np.any(too_far):
        optimized[too_far] *= 0.999 / norms[too_far]
    
    return optimized, elapsed


def local_refinement(points, n_steps=1000, step_size=0.01):
    """Fast coordinate descent local refinement using cached radius."""
    points = points.copy()
    current_r = fast_radius(points)
    for step in range(n_steps):
        if time.time() > DEADLINE - 5.0:
            break
        i = random.randint(0, N - 1)
        best_new_r = current_r
        best_new_point = points[i].copy()
        for dx in [-step_size, 0, step_size]:
            for dy in [-step_size, 0, step_size]:
                if dx == 0 and dy == 0:
                    continue
                new_point = points[i].copy()
                new_point[0] += dx
                new_point[1] += dy
                if np.linalg.norm(new_point) >= 1.0:
                    continue
                points[i] = new_point
                new_r = fast_radius(points)
                if new_r > best_new_r:
                    best_new_r = new_r
                    best_new_point = new_point.copy()
                else:
                    points[i] = points[i] - np.array([dx, dy])
        if best_new_r > current_r:
            points[i] = best_new_point
            current_r = best_new_r
    return points


def search_portfolio():
    """Search approach: generate varied seeds and keep the best."""
    best_points = None
    best_radius = 0.0
    
    # Start with hexagonal seed
    seed_hex = seed_hexagonal()
    r_hex = fast_radius(seed_hex)
    best_points = seed_hex.copy()
    best_radius = r_hex
    
    # Generate multiple random seeds
    n_seeds = 10
    for i in range(n_seeds):
        if time.time() > DEADLINE - 30.0:
            break
        seed = seed_random()
        r = fast_radius(seed)
        if r > best_radius:
            best_points = seed.copy()
            best_radius = r
    
    # Try optimizing from several seeds
    seeds_to_optimize = [best_points.copy()]
    
    # Add hexagonal and a few random ones
    seeds_to_optimize.append(seed_hex.copy())
    for i in range(5):
        if time.time() > DEADLINE - 60.0:
            break
        s = seed_random()
        seeds_to_optimize.append(s)
    
    for seed in seeds_to_optimize:
        if time.time() > DEADLINE - 20.0:
            break
        try:
            optimized, elapsed = optimize_from_seed(seed, max_iter=3000, timeout=10.0)
            r = fast_radius(optimized)
            if r > best_radius:
                best_points = optimized.copy()
                best_radius = r
        except Exception:
            pass
    
    # Final local refinement
    if time.time() < DEADLINE - 5.0:
        best_points = local_refinement(best_points, n_steps=800, step_size=0.003)
        best_radius = fast_radius(best_points)
    
    # Final high-precision polish: one more L-BFGS-B with tighter tolerance
    if time.time() < DEADLINE - 3.0:
        try:
            polished, _ = optimize_from_seed(best_points, max_iter=1000, timeout=5.0)
            r_polished = fast_radius(polished)
            if r_polished > best_radius:
                best_points = polished.copy()
                best_radius = r_polished
        except Exception:
            pass
    
    return best_points


def construct():
    """Construct centres of N equal circles inside the unit circle."""
    global DEADLINE
    DEADLINE = time.time() + 130.0
    
    np.random.seed(42)
    random.seed(42)
    
    points = search_portfolio()
    
    # Final validation and clipping
    points = np.asarray(points, dtype=np.float64)
    norms = np.linalg.norm(points, axis=1)
    # Ensure strictly inside
    for i in range(N):
        if norms[i] >= 1.0:
            points[i] *= 0.999 / norms[i]
    
    # Ensure finite values
    if not np.all(np.isfinite(points)):
        # Fallback to hexagonal
        points = seed_hexagonal()
    
    return points


if __name__ == "__main__":
    points = construct()
    print("Final radius:", compute_radius(points))
