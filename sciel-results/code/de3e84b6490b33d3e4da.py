"""Pack 200 equal circles in a unit square to maximize common radius.

Strategy:
1. Initialize with a hexagonal lattice pattern, which is a strong local optimum structure.
2. Use a smooth soft-min surrogate for the minimum distance constraint.
3. Apply L-BFGS-B optimization with an increasing beta schedule. Low beta allows
   for larger, smoother moves to escape local minima, while high beta sharpens
   the approximation to the true min function, enabling precise convergence.
4. Multi-start with slight perturbations to explore nearby basins.
5. Final polishing phase with very high beta and tight tolerances.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 200
_TIME_LIMIT = 135  # Safety margin before 150s kill

# Precompute pair indices for efficiency
i_idx, j_idx = np.triu_indices(N, k=1)
NUM_PAIRS = len(i_idx)


def _get_radius(pts):
    """Calculate the maximum possible radius for given centers."""
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < N:
        return 0.0
    if np.any(~np.isfinite(pts)):
        return 0.0
    
    # Check bounds
    if np.any(pts[:, 0] < -0.5) or np.any(pts[:, 0] > 0.5) or \
       np.any(pts[:, 1] < -0.5) or np.any(pts[:, 1] > 0.5):
        return 0.0

    # Wall distances
    min_x = np.min(pts[:, 0])
    max_x = np.max(pts[:, 0])
    min_y = np.min(pts[:, 1])
    max_y = np.max(pts[:, 1])
    
    wall_dist = min(min_x + 0.5, 0.5 - max_x, min_y + 0.5, 0.5 - max_y)

    # Pairwise distances
    # Use cdist for speed, but it's memory intensive. 
    # For N=200, 200x200 matrix is 40k floats, which is fine.
    from scipy.spatial.distance import cdist
    D = cdist(pts, pts)
    np.fill_diagonal(D, np.inf)
    min_pair = np.min(D) / 2.0

    r = min(wall_dist, min_pair)
    return r if r > 0 else 0.0


def _objective_and_grad(x, beta):
    """
    Objective function and gradient for L-BFGS-B.
    Minimizes -soft_min(constraints).
    """
    pts = x.reshape((N, 2))

    C = np.empty(4 * N + NUM_PAIRS)

    # Wall values
    C[0:N] = pts[:, 0] + 0.5
    C[N:2*N] = 0.5 - pts[:, 0]
    C[2*N:3*N] = pts[:, 1] + 0.5
    C[3*N:4*N] = 0.5 - pts[:, 1]

    # Pair values
    dx = pts[i_idx, 0] - pts[j_idx, 0]
    dy = pts[i_idx, 1] - pts[j_idx, 1]
    dists = np.sqrt(dx**2 + dy**2)
    C[4*N:] = dists / 2.0

    # Soft min computation
    # y = -beta * C
    # soft_min = - (max_y + log(sum(exp(y - max_y)))) / beta
    y = -beta * C
    max_y = np.max(y)
    
    if not np.isfinite(max_y):
        min_val = np.min(C)
        return -min_val, np.zeros_like(x)

    shifted = y - max_y
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)

    if sum_exp == 0:
        min_val = np.min(C)
        return -min_val, np.zeros_like(x)

    soft_min_val = -(max_y + np.log(sum_exp)) / beta
    obj_val = -soft_min_val

    # Weights for gradient
    weights = exp_shifted / sum_exp

    # Gradient computation
    grad_pts = np.zeros((N, 2))

    # Wall gradients
    # d/dx (x + 0.5) = 1, d/dx (0.5 - x) = -1
    grad_pts[:, 0] += weights[0:N] - weights[N:2*N]
    grad_pts[:, 1] += weights[2*N:3*N] - weights[3*N:4*N]

    # Pair gradients
    # d/dx_i (dist/2) = (x_i - x_j) / (2 * dist)
    valid = dists > 1e-10
    dx_safe = np.where(valid, dx, 1.0)
    dy_safe = np.where(valid, dy, 1.0)
    dists_safe = np.where(valid, dists, 1.0)

    grad_pair_x_i = np.where(valid, dx_safe / (2.0 * dists_safe), 0.0)
    grad_pair_x_j = np.where(valid, -dx_safe / (2.0 * dists_safe), 0.0)
    grad_pair_y_i = np.where(valid, dy_safe / (2.0 * dists_safe), 0.0)
    grad_pair_y_j = np.where(valid, -dy_safe / (2.0 * dists_safe), 0.0)

    pair_weights = weights[4*N:]

    np.add.at(grad_pts[:, 0], i_idx, grad_pair_x_i * pair_weights)
    np.add.at(grad_pts[:, 0], j_idx, grad_pair_x_j * pair_weights)
    np.add.at(grad_pts[:, 1], i_idx, grad_pair_y_i * pair_weights)
    np.add.at(grad_pts[:, 1], j_idx, grad_pair_y_j * pair_weights)

    # Chain rule: objective is -soft_min, soft_min = -(max+log(sum))/beta
    # d obj / dx = - d soft_min / dx
    # d soft_min / dx = sum(w_k * d C_k / dx) * (-1/beta)  <-- Wait.
    # Let's re-derive carefully.
    # S = -1/beta * (max_y + log(sum(exp(y_k))))
    # dS/dx = -1/beta * (0 + (1/sum) * sum(exp(y_k) * dy_k/dx))
    # dy_k/dx = -beta * dC_k/dx
    # dS/dx = -1/beta * (1/sum) * sum(exp(y_k) * (-beta) * dC_k/dx)
    # dS/dx = (1/sum) * sum(exp(y_k) * dC_k/dx) = sum(w_k * dC_k/dx)
    # We want to MINIMIZE -S.
    # d(-S)/dx = - dS/dx = - sum(w_k * dC_k/dx)
    
    grad_pts = -grad_pts

    return obj_val, grad_pts.flatten()


def _generate_hex_initial(spacing=0.078, seed=42):
    """Generate a centered hexagonal lattice packing, select N best points."""
    rng = np.random.RandomState(seed)
    points = []
    row_h = spacing * np.sqrt(3) / 2

    y = -1.0
    row_idx = 0
    while y < 1.0:
        shift = (spacing / 2.0) if row_idx % 2 == 1 else 0.0
        x = -1.0 + shift
        while x < 1.0:
            points.append([x, y])
            x += spacing
        y += row_h
        row_idx += 1

    pts = np.array(points)

    # Filter points inside the square
    mask = (pts[:, 0] >= -0.5) & (pts[:, 0] <= 0.5) & \
           (pts[:, 1] >= -0.5) & (pts[:, 1] <= 0.5)
    pts = pts[mask]

    if len(pts) < N:
        return rng.uniform(-0.45, 0.45, (N, 2))

    # Add small jitter to break symmetry
    pts = pts + rng.randn(len(pts), 2) * 0.002
    pts = np.clip(pts, -0.5, 0.5)

    # Select N points closest to the center to ensure a dense core
    dists_to_center = np.sum(pts**2, axis=1)
    indices = np.argsort(dists_to_center)
    pts = pts[indices[:N]]

    return pts


def _run_optimization(pts, time_limit):
    """
    Run L-BFGS-B optimization with an increasing beta schedule.
    """
    t_start = time.time()
    current_pts = pts.copy()
    current_r = _get_radius(current_pts)
    
    # Beta schedule: start smooth, end sharp
    # Each step gets a portion of the time limit
    # We will iterate through betas and potentially repeat if time allows
    betas = [20, 50, 100, 200, 500, 1000, 2000]
    
    # We'll do multiple passes if time permits
    max_passes = 5 
    pass_count = 0
    
    while time.time() - t_start < time_limit:
        if pass_count >= max_passes:
            break
        pass_count += 1
        for beta in betas:
            if time.time() - t_start > time_limit:
                break
            
            try:
                x0 = current_pts.flatten()
                
                # Define function and gradient
                def fun_and_grad(x):
                    return _objective_and_grad(x, beta)
                
                # L-BFGS-B options
                # maxiter: limit iterations to avoid hanging on a single beta
                # ftol, gtol: tight tolerances for precision
                res = minimize(fun_and_grad, x0, method='L-BFGS-B',
                              jac=True, bounds=[(-0.5, 0.5)] * (2 * N),
                              options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10, 'disp': False})
                
                new_pts = res.x.reshape(N, 2)
                new_pts = np.clip(new_pts, -0.5, 0.5)
                
                new_r = _get_radius(new_pts)
                
                if new_r > current_r:
                    current_r = new_r
                    current_pts = new_pts
                    
            except Exception as e:
                # If optimization fails, skip to next beta
                continue

    return current_pts, current_r


def construct():
    t0 = time.time()
    best_r = 0.0
    best_pts = None

    # --- Phase 1: Multi-start with Hexagonal Lattice ---
    # Try a few different spacings and seeds to find a good local basin
    initialisations = [
        ("hex_0.078", _generate_hex_initial(0.078, seed=42)),
        ("hex_0.080", _generate_hex_initial(0.080, seed=123)),
        ("hex_0.076", _generate_hex_initial(0.076, seed=777)),
        ("hex_0.079", _generate_hex_initial(0.079, seed=999)),
    ]

    # Allocate time: 20% for initial multi-start, 80% for final polish
    time_per_start = (_TIME_LIMIT * 0.2) / len(initialisations)
    
    for name, init_pts in initialisations:
        if time.time() - t0 > _TIME_LIMIT * 0.2:
            break
        try:
            # Quick optimization for this start
            # Use a shorter time limit for initial exploration
            pts_opt, r_opt = _run_optimization(init_pts, time_per_start)
            
            if r_opt > best_r:
                best_r = r_opt
                best_pts = pts_opt
        except Exception:
            pass

    # Fallback
    if best_pts is None:
        best_pts = _generate_hex_initial(0.078, seed=42)
        best_r = _get_radius(best_pts)

    # --- Phase 2: Final Polish ---
    # Use remaining time to refine the best candidate
    remaining_time = _TIME_LIMIT - (time.time() - t0)
    if remaining_time > 5:
        try:
            pts_polish, r_polish = _run_optimization(best_pts, remaining_time)
            if r_polish > best_r:
                best_r = r_polish
                best_pts = pts_polish
        except Exception:
            pass

    if best_r <= 0:
        best_pts = _generate_hex_initial(0.078, seed=42)

    return best_pts
