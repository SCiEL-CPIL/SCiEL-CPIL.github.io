"""Circle packing: 50 equal circles in unit circle.
Strategy: ring-based init + L-BFGS-B optimization with soft-min objective."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50
DEADLINE = 150.0  # wall-clock seconds
START_TIME = time.time()


def _remaining_time():
    return DEADLINE - (time.time() - START_TIME)


def _compute_score(centers):
    """Compute the packing radius for given centers."""
    n = len(centers)
    # Distance to boundary
    dists_boundary = 1.0 - np.linalg.norm(centers, axis=1)
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists, axis=1)
    r = min(np.min(dists_boundary), np.min(min_pair) / 2.0)
    return r


def _soft_min_obj_and_grad(centers, beta=100.0):
    """
    Soft-min objective using log-sum-exp for smoothness.
    We want to maximize r = min(min_boundary, min_pair/2).
    Equivalently, minimize -r.
    
    Using soft-min: r_soft = -(1/beta) * log(sum(exp(-beta * r_i)))
    where r_i are the individual constraints (boundary dists and pair dists/2).
    """
    n = len(centers)
    # Boundary distances
    dists_boundary = 1.0 - np.linalg.norm(centers, axis=1)
    
    # Pairwise distances / 2
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    pair_half_dists = dists / 2.0
    # Remove diagonal (inf)
    mask = ~np.eye(n, dtype=bool)
    pair_vals = pair_half_dists[mask]
    
    # All constraint values
    all_vals = np.concatenate([dists_boundary, pair_vals])
    
    # Soft min: -(1/beta) * log(sum(exp(-beta * x)))
    # For numerical stability, subtract max
    max_val = np.max(all_vals)
    shifted = -beta * (all_vals - max_val)
    # Clip for stability
    shifted = np.clip(shifted, -100, 100)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min_val = max_val - log_sum / beta
    
    obj = -soft_min_val  # We minimize this (i.e., maximize soft_min)
    
    # Gradient computation (numerical would be slow; let's do analytic)
    # Actually, let's use numerical gradient for simplicity and correctness
    # But that's O(n) evaluations. Let's do analytic.
    
    # d(soft_min)/d(centers) = sum_i (exp(-beta * (x_i - max_val)) / sum_j exp(-beta*(x_j-max_val))) * (-beta) * dx_i/d(centers)
    # = -sum_i w_i * dx_i/d(centers) where w_i = exp(-beta*(x_i-max_val)) / sum(exp(...))
    
    weights = np.exp(shifted)
    weights /= np.sum(weights)
    
    grad = np.zeros_like(centers)
    
    # Boundary contributions: x_i = 1 - |c_i|, dx_i/dc_i = -c_i/|c_i|
    for i in range(n):
        norm_c = np.linalg.norm(centers[i])
        if norm_c > 1e-10:
            grad[i] += -weights[i] * (-centers[i] / norm_c)
        # else derivative is undefined, skip
    
    # Pair contributions: for pair (i,j), x = |c_i - c_j|/2, dx/dc_i = (c_i-c_j)/|c_i-c_j|, dx/dc_j = (c_j-c_i)/|c_j-c_i|
    # Each pair appears once in the flattened array
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            d = centers[i] - centers[j]
            dist = np.linalg.norm(d)
            if dist > 1e-10:
                w = weights[idx]
                grad[i] += -w * (d / dist)
                grad[j] += -w * (-d / dist)
            idx += 1
    
    return obj, grad


def _ring_init(n_circles):
    """Initialize with center + concentric rings."""
    # 1 center + rings of 6, 12, 18, 13 = 50
    # Ring radii approximately: 0, r1, r2, r3, r4
    # For optimal packing, rings are roughly at: 0, 0.25, 0.5, 0.75, 0.98 (times optimal radius scale)
    # Let's use approximate ring structure
    rings = [
        (1, 0.0),      # center
        (6, 0.25),     # ring 1
        (12, 0.50),    # ring 2
        (18, 0.75),    # ring 3
        (13, 0.97),    # ring 4
    ]
    centers = []
    for count, radius in rings:
        if count == 1 and radius == 0.0:
            centers.append([0.0, 0.0])
        else:
            angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
            for a in angles:
                centers.append([radius * np.cos(a), radius * np.sin(a)])
    
    centers = np.array(centers[:n_circles])
    return centers


def _gradient_ascent(centers, max_time=30.0, beta_start=50.0, beta_end=200.0):
    """
    Optimize using L-BFGS-B with increasing beta for sharpness.
    """
    result_centers = centers.copy()
    best_score = _compute_score(result_centers)
    
    # Multiple stages with increasing beta
    betas = [50.0, 100.0, 200.0, 500.0]
    for beta in betas:
        if _remaining_time() < 10.0:
            break
        
        def objective(x):
            c = x.reshape(-1, 2)
            obj, grad = _soft_min_obj_and_grad(c, beta=beta)
            return obj, grad
        
        x0 = result_centers.flatten()
        try:
            res = minimize(objective, x0, method='L-BFGS-B', jac=True,
                          options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-15})
            result_centers = res.x.reshape(-1, 2)
        except Exception:
            pass
        
        score = _compute_score(result_centers)
        if score > best_score:
            best_score = score
            result_centers = result_centers.copy()
    
    return result_centers, best_score


def construct():
    best_centers = None
    best_score = 0.0
    
    # Strategy 1: Ring-based initialization
    centers = _ring_init(N)
    centers, score = _gradient_ascent(centers)
    if score > best_score:
        best_score = score
        best_centers = centers.copy()
    
    # Strategy 2: Perturbed ring initialization
    if _remaining_time() > 30.0:
        np.random.seed(42)
        for trial in range(5):
            if _remaining_time() < 15.0:
                break
            centers = _ring_init(N)
            # Add small random perturbation
            centers += np.random.randn(N, 2) * 0.02
            # Clip to stay in unit circle
            norms = np.linalg.norm(centers, axis=1)
            mask = norms > 0.95
            centers[mask] *= 0.95 / norms[mask, np.newaxis]
            
            centers, score = _gradient_ascent(centers, max_time=20.0)
            if score > best_score:
                best_score = score
                best_centers = centers.copy()
    
    # Strategy 3: Hexagonal lattice with optimization
    if _remaining_time() > 30.0:
        # Generate hex lattice points
        spacing = 0.25
        points = []
        for i in range(-5, 6):
            for j in range(-5, 6):
                x = i * spacing + (j % 2) * spacing * 0.5
                y = j * spacing * np.sqrt(3) / 2
                if np.hypot(x, y) < 0.95:
                    points.append([x, y])
        points = np.array(points)
        # Select N closest to origin
        if len(points) >= N:
            radii = np.linalg.norm(points, axis=1)
            idx = np.argsort(radii)[:N]
            centers = points[idx].copy()
            centers, score = _gradient_ascent(centers, max_time=20.0)
            if score > best_score:
                best_score = score
                best_centers = centers.copy()
    
    # Final polish: high beta optimization
    if best_centers is not None and _remaining_time() > 5.0:
        centers = best_centers.copy()
        
        def final_objective(x):
            c = x.reshape(-1, 2)
            obj, grad = _soft_min_obj_and_grad(c, beta=1000.0)
            return obj, grad
        
        x0 = centers.flatten()
        try:
            res = minimize(final_objective, x0, method='L-BFGS-B', jac=True,
                          options={'maxiter': 200, 'ftol': 1e-16, 'gtol': 1e-16})
            polished = res.x.reshape(-1, 2)
            score = _compute_score(polished)
            if score > best_score:
                best_score = score
                best_centers = polished
        except Exception:
            pass
    
    return best_centers


# Test
if __name__ == "__main__":
    centers = construct()
    print("Score:", _compute_score(centers))
    print("Shape:", centers.shape)
