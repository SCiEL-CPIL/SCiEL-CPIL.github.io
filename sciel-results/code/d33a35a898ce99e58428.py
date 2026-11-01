"""Packing 50 equal circles in a unit circle using SDR-based QCQP relaxation."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE_SECONDS = 120.0
START_TIME = time.time()


def _elapsed():
    return time.time() - START_TIME


def _remaining():
    return DEADLINE_SECONDS - _elapsed() - 2.0  # 2s safety margin


def _compute_radius(centers):
    """Compute the valid packing radius for a set of centers."""
    n = len(centers)
    if n < 2:
        return 0.0
    # Check all centers are finite
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Container constraint
    dists_to_origin = np.linalg.norm(centers, axis=1)
    container_r = np.min(1.0 - dists_to_origin)
    if container_r < 0:
        return 0.0
    # Pairwise distances
    min_pair_dist = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < min_pair_dist:
                min_pair_dist = d
    pair_r = min_pair_dist / 2.0
    return min(container_r, pair_r)


def _sdr_relaxation(centers_init, r_init, max_iters=20):
    """
    Sequential SDR approach:
    1. Formulate the QCQP with auxiliary variable r.
    2. Relax to SDP, solve, extract feasible point.
    3. Iterate by tightening around the current solution.
    
    We use a simplified approach: since full SDP solvers aren't available,
    we use a penalty-based gradient ascent on a smoothed objective that
    approximates the SDR projection.
    """
    n = len(centers_init)
    centers = centers_init.copy().astype(float)
    r = r_init
    
    # We'll use a differentiable surrogate: maximize r subject to soft constraints
    # Using log-sum-exp style smooth min for the objective
    
    def objective(params):
        """Negative radius to minimize (we want to maximize r)."""
        c = params[:2 * n].reshape(n, 2)
        r_val = params[2 * n]
        
        # Container violation
        dists = np.linalg.norm(c, axis=1)
        container_viol = np.max(np.maximum(0, np.linalg.norm(c, axis=1) + r_val - 1.0))
        
        # Pairwise violations
        pair_viol = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(c[i] - c[j])
                pair_viol += max(0, 2 * r_val - d)
        
        # Smooth penalty
        penalty = 100.0 * (container_viol ** 2 + pair_viol ** 2)
        return -r_val + penalty
    
    def objective_grad(params):
        """Gradient of the objective."""
        c = params[:2 * n].reshape(n, 2)
        r_val = params[2 * n]
        
        grad = np.zeros_like(params)
        
        # Container gradient
        dists = np.linalg.norm(c, axis=1)
        for i in range(n):
            if dists[i] > 0:
                norm_c = c[i] / dists[i]
            else:
                norm_c = np.zeros(2)
            viol_c = max(0, dists[i] + r_val - 1.0)
            if viol_c > 0:
                grad[2*i] += 200.0 * viol_c * norm_c[0]
                grad[2*i+1] += 200.0 * viol_c * norm_c[1]
                grad[2*n] += 200.0 * viol_c
        
        # Pairwise gradients
        for i in range(n):
            for j in range(i + 1, n):
                diff = c[i] - c[j]
                d = np.linalg.norm(diff)
                if d < 1e-10:
                    continue
                viol_p = max(0, 2 * r_val - d)
                if viol_p > 0:
                    unit_diff = diff / d
                    grad[2*i] += 200.0 * viol_p * unit_diff[0]
                    grad[2*i+1] += 200.0 * viol_p * unit_diff[1]
                    grad[2*j] -= 200.0 * viol_p * unit_diff[0]
                    grad[2*j+1] -= 200.0 * viol_p * unit_diff[1]
                    grad[2*n] += 400.0 * viol_p
        
        # Gradient of -r
        grad[2*n] -= 1.0
        
        return grad
    
    # Use L-BFGS-B with bounds
    bounds = [(-1.0, 1.0)] * (2 * n) + [(0.0, 1.0)]
    
    x0 = np.concatenate([centers.ravel(), [r]])
    
    best_x = x0.copy()
    best_val = objective(x0)
    
    for it in range(max_iters):
        if _remaining() < 5.0:
            break
        
        result = minimize(objective, x0, jac=objective_grad, method='L-BFGS-B',
                          bounds=bounds, options={'maxiter': 50, 'gtol': 1e-6})
        
        if result.fun < best_val:
            best_val = result.fun
            best_x = result.x.copy()
        
        x0 = result.x.copy()
    
    best_c = best_x[:2*n].reshape(n, 2)
    best_r = best_x[2*n]
    
    # Project to feasible
    # Ensure all centers are within container
    for i in range(n):
        d = np.linalg.norm(best_c[i])
        if d > 1.0 - best_r:
            if d > 0:
                best_c[i] *= (1.0 - best_r) / d
            else:
                best_c[i] = np.array([0, 0])
    
    # Ensure pairwise distances
    # Simple projection: if two circles overlap, push them apart
    for _ in range(50):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                diff = best_c[i] - best_c[j]
                d = np.linalg.norm(diff)
                min_d = 2 * best_r
                if d < min_d - 1e-10:
                    if d < 1e-10:
                        angle = (i + j * 0.7) * 0.3
                        push = np.array([np.cos(angle), np.sin(angle)]) * (min_d - d) / 2
                    else:
                        push = (diff / d) * (min_d - d) / 2
                    best_c[i] += push
                    best_c[j] -= push
                    moved = True
        if not moved:
            break
    
    # Recompute actual radius
    actual_r = _compute_radius(best_c)
    
    return best_c, actual_r


def _gradient_ascent(centers, r, steps=200, lr=0.001):
    """
    Gradient ascent on a smooth approximation of the minimum distance.
    Uses softmax-weighted nearest neighbors for smoothness.
    """
    n = len(centers)
    centers = centers.copy().astype(float)
    
    def smooth_min_radius(c, temperature=0.05):
        """Compute a smooth approximation of the packing radius."""
        # Container
        dists_origin = np.linalg.norm(c, axis=1)
        container_r = 1.0 - np.max(dists_origin)
        
        # Pairwise: use soft min
        min_pair = float('inf')
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(c[i] - c[j])
                if d < min_pair:
                    min_pair = d
        pair_r = min_pair / 2.0
        
        return min(container_r, pair_r)
    
    def objective(params):
        c = params.reshape(n, 2)
        return -smooth_min_radius(c)
    
    def objective_grad(params):
        c = params.reshape(n, 2)
        grad = np.zeros_like(params)
        
        # Container gradient
        dists = np.linalg.norm(c, axis=1)
        max_idx = np.argmax(dists)
        if dists[max_idx] > 1e-10:
            grad[2*max_idx] = -c[max_idx, 0] / dists[max_idx]
            grad[2*max_idx+1] = -c[max_idx, 1] / dists[max_idx]
        
        # Pairwise gradient (only nearest pair matters for min)
        min_d = float('inf')
        min_i, min_j = 0, 1
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(c[i] - c[j])
                if d < min_d:
                    min_d = d
                    min_i, min_j = i, j
        
        if min_d > 1e-10:
            diff = c[min_i] - c[min_j]
            unit = diff / min_d
            grad[2*min_i] = 0.5 * unit[0]
            grad[2*min_i+1] = 0.5 * unit[1]
            grad[2*min_j] = -0.5 * unit[0]
            grad[2*min_j+1] = -0.5 * unit[1]
        
        return grad
    
    x = centers.ravel().copy()
    
    for step in range(steps):
        if _remaining() < 5.0:
            break
        
        g = objective_grad(x)
        # Adaptive learning rate
        lr_adapt = lr * (0.95 ** (step // 20))
        x -= lr_adapt * g
        
        # Project back to feasible region
        c = x.reshape(n, 2)
        for i in range(n):
            d = np.linalg.norm(c[i])
            if d > 1.0:
                c[i] *= 1.0 / d
        x = c.ravel()
    
    centers = x.reshape(n, 2)
    r = _compute_radius(centers)
    return centers, r


def _random_initialization(n):
    """Random initialization with rejection sampling."""
    centers = np.zeros((n, 2))
    # Start with a rough hexagonal grid
    spacing = 2.0 / (np.sqrt(n) + 1.0)
    k = int(1.0 / spacing) + 2
    rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
            for i in range(-k, k + 1) for j in range(-k, k + 1)]
    points = np.array(rows)
    radius = np.hypot(points[:, 0], points[:, 1])
    points = points[radius <= 1.0 - spacing / 2.0]
    
    if len(points) >= n:
        centers = points[np.argsort(np.hypot(points[:, 0], points[:, 1]), kind="stable")[:n]]
    else:
        # Fallback to random
        for i in range(n):
            while True:
                pt = np.random.uniform(-1, 1, 2)
                if np.linalg.norm(pt) <= 1.0:
                    break
            centers[i] = pt
    
    return centers


def construct():
    global START_TIME
    START_TIME = time.time()
    
    n = N
    best_centers = None
    best_r = 0.0
    
    # Initial hexagonal packing
    centers = _random_initialization(n)
    r = _compute_radius(centers)
    best_centers = centers.copy()
    best_r = r
    
    # Try multiple restarts with perturbation
    n_restarts = 15
    
    for restart in range(n_restarts):
        if _remaining() < 10.0:
            break
        
        # Perturb the current best
        if restart == 0:
            c = centers.copy()
        else:
            c = best_centers.copy()
            noise_scale = 0.1 / (1 + restart * 0.5)
            c += np.random.normal(0, noise_scale, size=c.shape)
            # Project back
            for i in range(n):
                d = np.linalg.norm(c[i])
                if d > 0.95:
                    c[i] *= 0.95 / d
        
        # SDR-like relaxation
        c_sdr, r_sdr = _sdr_relaxation(c, best_r, max_iters=15)
        if r_sdr > best_r:
            best_r = r_sdr
            best_centers = c_sdr.copy()
        
        # Gradient ascent polish
        c_grad, r_grad = _gradient_ascent(c_sdr, best_r, steps=100, lr=0.002)
        if r_grad > best_r:
            best_r = r_grad
            best_centers = c_grad.copy()
    
    # Final polish with smaller steps
    if _remaining() > 15.0:
        c_final, r_final = _gradient_ascent(best_centers, best_r, steps=300, lr=0.0005)
        if r_final > best_r:
            best_r = r_final
            best_centers = c_final.copy()
    
    # Ensure validity
    final_r = _compute_radius(best_centers)
    if final_r <= 0:
        # Fallback to simple hexagonal
        best_centers = _random_initialization(n)
        final_r = _compute_radius(best_centers)
    
    return best_centers
