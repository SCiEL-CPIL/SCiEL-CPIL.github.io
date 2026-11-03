"""Pack 200 equal circles in a unit square using parameterized hexagonal lattice with boundary distortion."""
import numpy as np
import time
from scipy.spatial import KDTree
from scipy.optimize import minimize

N = 200
_TIME_LIMIT = 140  # seconds, leave margin before 150s kill


def _compute_radius(pts):
    """Calculate the maximum possible radius for given centers."""
    if len(pts) == 0:
        return 0.0

    # Distance to walls
    min_x = np.min(pts[:, 0])
    max_x = np.max(pts[:, 0])
    min_y = np.min(pts[:, 1])
    max_y = np.max(pts[:, 1])

    wall_dist = min(min_x + 0.5, 0.5 - max_x, min_y + 0.5, 0.5 - max_y)

    # Min pairwise distance
    tree = KDTree(pts)
    pairs = tree.query_pairs(r=1.0)
    if len(pairs) == 0:
        min_pair_dist = np.inf
    else:
        min_pair_dist = np.inf
        for i, j in pairs:
            d = np.linalg.norm(pts[i] - pts[j])
            if d < min_pair_dist:
                min_pair_dist = d
        if min_pair_dist == np.inf:
            D = np.zeros((len(pts), len(pts)))
            np.fill_diagonal(D, np.inf)
            for i in range(len(pts)):
                for j in range(i+1, len(pts)):
                    D[i, j] = D[j, i] = np.linalg.norm(pts[i] - pts[j])
            min_pair_dist = np.min(D)

    return min(wall_dist, min_pair_dist / 2.0)


def _hex_lattice_indices(n_rows, n_cols):
    """Generate hex lattice indices for a grid of n_rows x n_cols points."""
    indices = []
    for row in range(n_rows):
        for col in range(n_cols):
            indices.append((row, col))
    return np.array(indices, dtype=float)


def _distort_lattice(params, base_lattice):
    """
    Apply distortion to base lattice points.
    params: [a1x, a1y, a2x, a2y, fx1, fy1, fx2, fy2, fx3, fy3]
    where a1, a2 are lattice vectors, and f terms are Fourier distortion coefficients.
    base_lattice: (n, 2) array of lattice coordinates (i, j)
    """
    a1 = np.array([params[0], params[1]])
    a2 = np.array([params[2], params[3]])
    
    # Fourier distortion coefficients
    fx1, fy1 = params[4], params[5]
    fx2, fy2 = params[6], params[7]
    fx3, fy3 = params[8], params[9]
    
    # Base positions
    base = base_lattice[:, 0:1] * a1[None, :] + base_lattice[:, 1:2] * a2[None, :]
    
    # Distortion: low-order Fourier modes
    i = base_lattice[:, 0]
    j = base_lattice[:, 1]
    n_i = len(base_lattice)
    
    # Normalize i, j to [0, 1] for Fourier modes
    i_max = max(i.max(), 1)
    j_max = max(j.max(), 1)
    i_norm = i / i_max
    j_norm = j / j_max
    
    # Radial coordinate from center
    i_center = (i - i_max / 2) / (i_max / 2 + 1e-10)
    j_center = (j - j_max / 2) / (j_max / 2 + 1e-10)
    r = np.sqrt(i_center**2 + j_center**2)
    
    # Distortion fields
    dx = (fx1 * np.sin(2 * np.pi * i_norm) + 
          fx2 * np.sin(4 * np.pi * i_norm) + 
          fx3 * np.sin(2 * np.pi * j_norm))
    dy = (fy1 * np.sin(2 * np.pi * j_norm) + 
          fy2 * np.sin(4 * np.pi * j_norm) + 
          fy3 * np.sin(2 * np.pi * i_norm))
    
    # Add radial distortion to push boundary points inward
    radial_scale = 1.0 - 0.1 * r**2
    base = base * radial_scale[:, None]
    
    distorted = base + np.column_stack([dx, dy])
    return distorted


def _generate_hex_base(n_rows, n_cols):
    """Generate base hex lattice coordinates (i, j) for n_rows x n_cols."""
    base = []
    for row in range(n_rows):
        for col in range(n_cols):
            base.append([float(row), float(col)])
    return np.array(base)


def _objective(params, base_lattice):
    """
    Objective: negative of the minimum radius.
    We want to maximize r, so minimize -r.
    """
    pts = _distort_lattice(params, base_lattice)
    
    # Clip to square
    pts[:, 0] = np.clip(pts[:, 0], -0.5, 0.5)
    pts[:, 1] = np.clip(pts[:, 1], -0.5, 0.5)
    
    r = _compute_radius(pts)
    return -r


def _optimize_lattice(base_lattice, x0, time_budget=30.0):
    """
    Optimize lattice parameters using L-BFGS-B.
    """
    t_start = time.time()
    
    bounds = [(-0.5, 0.5)] * 10
    
    result = minimize(
        _objective,
        x0,
        args=(base_lattice,),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 5000}
    )
    
    elapsed = time.time() - t_start
    return result.x, -result.fun


def _refine_with_free_optimization(pts, time_budget=30.0):
    """
    After lattice optimization, do a free optimization of all points
    to fine-tune the packing.
    """
    t_start = time.time()
    n = len(pts)
    
    def _soft_min_and_grad(x, beta=200.0):
        pts = x.reshape(-1, 2)
        
        # Wall distances
        wall_vals = np.concatenate([
            pts[:, 0] + 0.5,
            0.5 - pts[:, 0],
            pts[:, 1] + 0.5,
            0.5 - pts[:, 1]
        ])
        
        # Pair distances using KDTree
        tree = KDTree(pts)
        pairs = tree.query_pairs(r=0.3)
        
        if len(pairs) == 0:
            pair_vals = np.array([1.0])
        else:
            pair_vals = np.array([np.linalg.norm(pts[i] - pts[j]) / 2.0 for i, j in pairs])
        
        all_vals = np.concatenate([wall_vals, pair_vals])
        
        # Soft min
        neg_beta_vals = -beta * all_vals
        max_neg = np.max(neg_beta_vals)
        shifted = neg_beta_vals - max_neg
        exp_shifted = np.exp(shifted)
        sum_exp = np.sum(exp_shifted)
        
        soft_min_val = -(max_neg + np.log(sum_exp)) / beta
        
        # Gradient
        grad = np.zeros_like(pts)
        weights = exp_shifted / sum_exp
        
        # Wall gradients
        for i in range(n):
            grad[i, 0] += weights[4*i] * 1.0
            grad[i, 0] += weights[4*i+1] * (-1.0)
            grad[i, 1] += weights[4*i+2] * 1.0
            grad[i, 1] += weights[4*i+3] * (-1.0)
        
        # Pair gradients
        if len(pairs) > 0:
            for k, (i, j) in enumerate(pairs):
                w = weights[4*n + k]
                diff = pts[i] - pts[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    continue
                grad[i] += w * diff / (2.0 * dist)
                grad[j] -= w * diff / (2.0 * dist)
        
        return soft_min_val, grad.flatten()
    
    x0 = pts.flatten()
    bounds = [(-0.5, 0.5)] * (2 * n)
    
    result = minimize(
        _soft_min_and_grad,
        x0,
        args=(200.0,),
        jac=True,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-10}
    )
    
    pts_opt = result.x.reshape(-1, 2)
    pts_opt[:, 0] = np.clip(pts_opt[:, 0], -0.5, 0.5)
    pts_opt[:, 1] = np.clip(pts_opt[:, 1], -0.5, 0.5)
    
    return pts_opt


def _force_polish(pts, r_target, max_iters=1000):
    """
    Force-directed local optimization to ensure all constraints are satisfied.
    """
    pts = pts.copy()
    n = len(pts)
    current_lr = 0.001
    
    for it in range(max_iters):
        if time.time() > _TIME_LIMIT - 5:
            break
        
        forces = np.zeros_like(pts)
        tree = KDTree(pts)
        
        for i in range(n):
            # Wall forces
            if pts[i, 0] < -0.5 + r_target:
                forces[i, 0] += 10.0 * (r_target - (pts[i, 0] + 0.5))
            if pts[i, 0] > 0.5 - r_target:
                forces[i, 0] -= 10.0 * (pts[i, 0] - (0.5 - r_target))
            if pts[i, 1] < -0.5 + r_target:
                forces[i, 1] += 10.0 * (r_target - (pts[i, 1] + 0.5))
            if pts[i, 1] > 0.5 - r_target:
                forces[i, 1] -= 10.0 * (pts[i, 1] - (0.5 - r_target))
            
            # Repulsive forces from close neighbors
            neighbors = tree.query_ball_point(pts[i], 2 * r_target + 0.01)
            for j in neighbors:
                if i == j:
                    continue
                diff = pts[i] - pts[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    forces[i] += np.random.randn(2) * 0.001
                    continue
                if dist < 2 * r_target:
                    overlap = 2 * r_target - dist
                    force_mag = 5.0 * overlap / dist
                    forces[i] += force_mag * diff
        
        pts += current_lr * forces
        pts[:, 0] = np.clip(pts[:, 0], -0.5, 0.5)
        pts[:, 1] = np.clip(pts[:, 1], -0.5, 0.5)
        
        current_lr *= 0.999
        
        if np.max(np.abs(forces)) < 1e-8:
            break
    
    return pts


def construct():
    t0 = time.time()
    np.random.seed(42)
    
    best_r = 0.0
    best_pts = None
    
    # Try different hex lattice configurations
    # For 200 circles, we can try various row/col combinations
    configs = [
        (14, 15),  # 210 points, take 200
        (15, 14),  # 210 points
        (13, 16),  # 208 points
        (16, 13),  # 208 points
        (12, 17),  # 204 points
        (17, 12),  # 204 points
        (10, 20),  # 200 points exactly
        (20, 10),  # 200 points exactly
        (11, 18),  # 198, pad
        (18, 11),  # 198, pad
    ]
    
    for n_rows, n_cols in configs:
        elapsed = time.time() - t0
        remaining = _TIME_LIMIT - 10 - elapsed
        if remaining < 20:
            break
        
        # Generate base lattice
        base_lattice = _generate_hex_base(n_rows, n_cols)
        
        # If we have more points than needed, subsample
        if len(base_lattice) > N:
            # Take every other point in a staggered pattern
            idx = np.arange(N)
            base_lattice = base_lattice[idx]
        elif len(base_lattice) < N:
            # Pad by duplicating some points with small offsets
            n_pad = N - len(base_lattice)
            pad_pts = base_lattice[:n_pad] + np.random.randn(n_pad, 2) * 0.001
            base_lattice = np.vstack([base_lattice, pad_pts])
        
        # Initial parameters: hexagonal lattice with spacing ~1/sqrt(200)
        spacing = 1.0 / np.sqrt(N)
        a1 = np.array([spacing, 0.0])
        a2 = np.array([spacing * 0.5, spacing * np.sqrt(3) / 2])
        
        # Center the lattice
        center = np.array([0.0, 0.0])
        
        x0 = np.concatenate([a1, a2, np.zeros(6)])
        
        # Optimize lattice parameters
        time_budget = min(remaining * 0.4, 40.0)
        params_opt, r_lattice = _optimize_lattice(base_lattice, x0, time_budget)
        
        # Get the actual points
        pts = _distort_lattice(params_opt, base_lattice)
        pts[:, 0] = np.clip(pts[:, 0], -0.5, 0.5)
        pts[:, 1] = np.clip(pts[:, 1], -0.5, 0.5)
        
        r_current = _compute_radius(pts)
        
        # Refine with free optimization
        elapsed = time.time() - t0
        remaining = _TIME_LIMIT - 10 - elapsed
        if remaining > 25 and r_current > best_r:
            time_budget = min(remaining * 0.5, 45.0)
            pts_refined = _refine_with_free_optimization(pts, time_budget)
            r_refined = _compute_radius(pts_refined)
            
            if r_refined > r_current:
                r_current = r_refined
                pts = pts_refined
        
        # Force polish
        elapsed = time.time() - t0
        remaining = _TIME_LIMIT - 10 - elapsed
        if remaining > 15:
            r_target = r_current * 1.001
            pts_polished = _force_polish(pts, r_target, max_iters=500)
            r_polished = _compute_radius(pts_polished)
            
            if r_polished > r_current:
                r_current = r_polished
                pts = pts_polished
        
        if r_current > best_r:
            best_r = r_current
            best_pts = pts.copy()
    
    # Final polish on best result
    if best_pts is not None:
        elapsed = time.time() - t0
        remaining = _TIME_LIMIT - 5 - elapsed
        if remaining > 20:
            # Do another round of free optimization
            time_budget = min(remaining * 0.6, 50.0)
            pts_final = _refine_with_free_optimization(best_pts, time_budget)
            r_final = _compute_radius(pts_final)
            
            if r_final > best_r:
                best_r = r_final
                best_pts = pts_final
        
        # Final force polish
        elapsed = time.time() - t0
        remaining = _TIME_LIMIT - 5 - elapsed
        if remaining > 10:
            r_target = best_r * 1.001
            pts_final = _force_polish(best_pts, r_target, max_iters=800)
            r_final = _compute_radius(pts_final)
            
            if r_final > best_r:
                best_r = r_final
                best_pts = pts_final
    
    # Fallback: simple grid
    if best_pts is None or len(best_pts) != N:
        rows, cols = 14, 15
        dx = 1.0 / cols
        dy = 1.0 / rows
        xs = -0.5 + (np.arange(cols) + 0.5) * dx
        ys = -0.5 + (np.arange(rows) + 0.5) * dy
        grid = np.array([(x, y) for y in ys for x in xs])
        best_pts = grid[:N]
    
    # Ensure exactly N points
    if len(best_pts) > N:
        best_pts = best_pts[:N]
    elif len(best_pts) < N:
        # Pad with points near existing ones
        n_pad = N - len(best_pts)
        pad_pts = best_pts[:n_pad] + np.random.randn(n_pad, 2) * 0.001
        best_pts = np.vstack([best_pts, pad_pts])
    
    # Final validation and clipping
    best_pts[:, 0] = np.clip(best_pts[:, 0], -0.5, 0.5)
    best_pts[:, 1] = np.clip(best_pts[:, 1], -0.5, 0.5)
    
    return best_pts


if __name__ == "__main__":
    pts = construct()
    r = _compute_radius(pts)
    print(f"Best Radius: {r}")
