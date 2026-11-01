"""Improved packing algorithm for 300 equal circles in a unit circle."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree


def _hex_init(N):
    """Generate hexagonal lattice points, select N closest to origin."""
    s = np.sqrt(2.0 * np.pi / (N * np.sqrt(3.0)))
    s *= 0.95

    best_points = None
    for attempt in range(20):
        k = int(1.0 / s) + 3
        pts = []
        for j in range(-k, k + 1):
            y = j * s * np.sqrt(3.0) / 2.0
            if abs(y) > 1.0:
                continue
            x_offset = (j % 2) * s / 2.0
            for i in range(-k, k + 1):
                x = (i + x_offset) * s
                r = np.hypot(x, y)
                if r <= 1.0 - s / 2.0:
                    pts.append([x, y])
        pts = np.array(pts)
        if len(pts) >= N:
            dists = np.hypot(pts[:, 0], pts[:, 1])
            idx = np.argsort(dists, kind='stable')[:N]
            if best_points is None:
                best_points = pts[idx]
        s *= 0.98
    return best_points if best_points is not None else np.random.randn(N, 2) * 0.5


def _compute_radius(pts):
    """Compute the feasible radius for a set of points."""
    n = len(pts)
    if n < 2:
        min_pair = 1.0
    else:
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0

    dists_boundary = 1.0 - np.hypot(pts[:, 0], pts[:, 1])
    min_boundary = np.min(dists_boundary)

    return min(min_pair, min_boundary)


def _soft_min_value(pts, temperature=0.01):
    """
    Smooth surrogate for the min radius using log-sum-exp.
    We want to maximize r = min(min_pair/2, min_boundary).
    Equivalently, minimize -r.
    """
    n = len(pts)

    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_vals = dists / 2.0

    boundary_vals = 1.0 - np.hypot(pts[:, 0], pts[:, 1])

    all_vals = np.concatenate([pair_vals.flatten(), boundary_vals])

    all_vals = np.clip(all_vals, -100, 100)
    min_val = np.min(all_vals)
    exp_vals = np.exp((all_vals - min_val) / temperature)
    soft_min = min_val - temperature * np.log(np.sum(exp_vals))

    return -soft_min


def _soft_min_grad(pts, temperature=0.01):
    """Gradient of the soft-min objective."""
    n = len(pts)

    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2) + 1e-12)
    np.fill_diagonal(dists, np.inf)
    pair_vals = dists / 2.0

    boundary_vals = 1.0 - np.hypot(pts[:, 0], pts[:, 1])

    all_vals = np.concatenate([pair_vals.flatten(), boundary_vals])
    all_vals = np.clip(all_vals, -100, 100)

    min_val = np.min(all_vals)
    exp_vals = np.exp((all_vals - min_val) / temperature)
    weights = exp_vals / np.sum(exp_vals)

    grad = np.zeros_like(pts)

    pair_weights = weights[:n * (n - 1)]
    pair_weights = pair_weights.reshape(n, n)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            w = pair_weights[i, j]
            if dists[i, j] > 1e-12:
                grad[i] += (pts[i] - pts[j]) / (2.0 * dists[i, j]) * w * (-temperature)

    boundary_weights = weights[n * (n - 1):]
    for i in range(n):
        r = np.hypot(pts[i, 0], pts[i, 1])
        if r > 1e-12:
            grad[i] += -pts[i] / r * boundary_weights[i] * (-temperature)

    return grad


def _ring_init(N):
    """
    Generate an initial packing using concentric rings.
    This exploits the circular symmetry of the container.
    """
    # Estimate number of rings needed
    # Area per ring ~ 2*pi*r*dr
    # We want to place N points.
    # Let's try a simple heuristic:
    # Ring k has radius r_k and holds n_k points.
    # Constraint: 2*pi*r_k / n_k >= 2*r_c (where r_c is circle radius)
    # And sum n_k = N.
    
    # Let's just generate a hex lattice but force it into rings?
    # Or better, construct rings explicitly.
    
    # Start with a single point at center
    points = [[0.0, 0.0]]
    remaining = N - 1
    
    # Estimate ring radii
    # Assume r_c ~ 0.05
    # Ring 1: r=0.1, capacity ~ 2*pi*0.1/0.1 = 6
    # Ring 2: r=0.2, capacity ~ 12
    # Ring k: r=k*0.1, capacity ~ 6*k
    
    # Total capacity for K rings: 6 * K*(K+1)/2 = 3*K*(K+1)
    # 3*K^2 ~ 300 => K ~ 10
    
    r_c_est = 0.053
    ring_radius = r_c_est
    ring_idx = 0
    
    while remaining > 0 and ring_radius < 0.95:
        # Number of points in this ring
        # Circumference = 2*pi*r
        # Spacing = 2*r_c
        # n = floor(2*pi*r / (2*r_c)) = floor(pi*r/r_c)
        n_in_ring = int(np.floor(np.pi * ring_radius / r_c_est))
        if n_in_ring < 1:
            n_in_ring = 1
        
        if n_in_ring > remaining:
            n_in_ring = remaining
        
        # Place n_in_ring points at ring_radius
        # Angular offset to stagger rings
        angle_offset = np.pi / n_in_ring if ring_idx % 2 == 0 else 0.0
        for i in range(n_in_ring):
            angle = 2 * np.pi * i / n_in_ring + angle_offset
            x = ring_radius * np.cos(angle)
            y = ring_radius * np.sin(angle)
            points.append([x, y])
        
        remaining -= n_in_ring
        ring_idx += 1
        ring_radius += 2 * r_c_est * np.cos(np.pi / max(n_in_ring, 3))  # Approx next ring radius
    
    pts = np.array(points)
    
    # If we have too many, trim (shouldn't happen)
    if len(pts) > N:
        dists = np.hypot(pts[:, 0], pts[:, 1])
        idx = np.argsort(dists, kind='stable')[:N]
        pts = pts[idx]
        
    # If we have too few, add random points (shouldn't happen with this heuristic)
    while len(pts) < N:
        angle = np.random.uniform(0, 2*np.pi)
        r = np.random.uniform(0, 0.9)
        pts = np.vstack([pts, [[r*np.cos(angle), r*np.sin(angle)]]])
        
    return pts


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill

    # Try both hex and ring initializations, pick the better one
    pts_hex = _hex_init(N)
    pts_ring = _ring_init(N)
    
    r_hex = _compute_radius(pts_hex)
    r_ring = _compute_radius(pts_ring)
    
    if r_ring > r_hex:
        pts = pts_ring
        best_r = r_ring
    else:
        pts = pts_hex
        best_r = r_hex
    
    best_pts = pts.copy()
    
    # Optimization with soft-min
    temperatures = [0.005, 0.01, 0.02, 0.05]
    
    for temp in temperatures:
        if time.time() > deadline - 30.0:
            break
        
        def objective(x):
            p = x.reshape(N, 2)
            return _soft_min_value(p, temperature=temp)
        
        def gradient(x):
            p = x.reshape(N, 2)
            return _soft_min_grad(p, temperature=temp).flatten()
        
        x0 = pts.flatten()
        
        try:
            result = minimize(
                objective,
                x0,
                jac=gradient,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
            )
            
            if result.success or result.fun < objective(x0):
                new_pts = result.x.reshape(N, 2)
                new_r = _compute_radius(new_pts)
                if new_r > best_r:
                    best_pts = new_pts
                    best_r = new_r
                    pts = new_pts
        
        except Exception:
            pass
        
        if time.time() > deadline - 20.0:
            break
    
    # Final polish
    if time.time() < deadline - 15.0:
        temp = 0.002
        def objective(x):
            p = x.reshape(N, 2)
            return _soft_min_value(p, temperature=temp)
        
        def gradient(x):
            p = x.reshape(N, 2)
            return _soft_min_grad(p, temperature=temp).flatten()
        
        x0 = pts.flatten()
        try:
            result = minimize(
                objective,
                x0,
                jac=gradient,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 100}
            )
            if result.success:
                new_pts = result.x.reshape(N, 2)
                new_r = _compute_radius(new_pts)
                if new_r > best_r:
                    best_pts = new_pts
                    best_r = new_r
        except Exception:
            pass
    
    # Ensure validity
    dists = np.hypot(best_pts[:, 0], best_pts[:, 1])
    mask = dists <= 1.0
    if not np.all(mask):
        for i in range(N):
            if dists[i] > 1.0:
                best_pts[i] = best_pts[i] / dists[i] * 0.999
    
    return best_pts
# EVOLVE-BLOCK-END
