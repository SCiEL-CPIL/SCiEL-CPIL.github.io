"""Improved packing for 50 equal circles in a unit circle using ring-based structure + DE + L-BFGS-B + binary search on radius."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import cdist

def _compute_score(points):
    """Compute the packing score: min(min_pair_dist/2, min_boundary_dist)."""
    if not np.all(np.isfinite(points)):
        return 0.0
    if points.shape != (N, 2):
        return 0.0
    radii = np.hypot(points[:, 0], points[:, 1])
    if np.any(radii > 1.0 + 1e-12):
        return 0.0
    min_boundary = np.min(1.0 - radii)
    if min_boundary < 0:
        return 0.0
    d = cdist(points, points)
    iu = np.triu_indices(N, k=1)
    min_pair = np.min(d[iu]) / 2.0
    return min(min_pair, min_boundary)

def _energy(points, s=8.0):
    """Repulsive Riesz kernel energy with boundary penalty."""
    d = cdist(points, points)
    iu = np.triu_indices(N, k=1)
    pair_d = d[iu]
    # Avoid division by zero
    pair_d = np.maximum(pair_d, 1e-10)
    E_pairs = np.sum(pair_d ** (-s))
    # Boundary penalty: push points inside
    radii = np.hypot(points[:, 0], points[:, 1])
    E_boundary = np.sum(np.maximum(0.0, radii - 0.99) ** 4)
    return E_pairs + 100.0 * E_boundary

def _energy_grad(points, s=8.0):
    """Gradient of the energy function."""
    n = len(points)
    grad = np.zeros_like(points)
    d = cdist(points, points)
    # Pairwise repulsion gradient
    for i in range(n):
        for j in range(i + 1, n):
            diff = points[i] - points[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-10:
                continue
            # d/dxi sum |xi-xj|^-s = -s * |xi-xj|^(-s-2) * (xi-xj)
            factor = -s * dist ** (-s - 2)
            grad[i] += factor * diff
            grad[j] -= factor * diff
    # Boundary gradient
    radii = np.hypot(points[:, 0], points[:, 1])
    mask = radii > 0.99
    if np.any(mask):
        r = radii[mask]
        factor = 4.0 * 100.0 * (r - 0.99) ** 3
        grad[mask] += factor[:, np.newaxis] * points[mask] / r[:, np.newaxis]
    return grad

def _get_ring_initialization():
    """Generate a ring-based initial configuration."""
    # Use a known good structure: center + rings
    # For 50 circles, try: 1 center, then rings of 6, 12, 18, 13
    # or 1 + 6 + 12 + 18 + 13 = 50
    points = [np.array([0.0, 0.0])]
    
    # Ring 1: 6 points
    r1 = 0.35
    for k in range(6):
        angle = 2 * np.pi * k / 6
        points.append(np.array([r1 * np.cos(angle), r1 * np.sin(angle)]))
    
    # Ring 2: 12 points
    r2 = 0.65
    for k in range(12):
        angle = 2 * np.pi * k / 12 + np.pi / 12
        points.append(np.array([r2 * np.cos(angle), r2 * np.sin(angle)]))
    
    # Ring 3: 18 points
    r3 = 0.85
    for k in range(18):
        angle = 2 * np.pi * k / 18
        points.append(np.array([r3 * np.cos(angle), r3 * np.sin(angle)]))
    
    # Ring 4: 13 points
    r4 = 0.95
    for k in range(13):
        angle = 2 * np.pi * k / 13 + np.pi / 13
        points.append(np.array([r4 * np.cos(angle), r4 * np.sin(angle)]))
    
    return np.array(points[:N])

def _get_hex_initial_points():
    """Generate hexagonal lattice initial points."""
    spacing = 0.24
    points = []
    for j in range(-10, 11):
        for i in range(-10, 11):
            x = i * spacing + (j % 2) * spacing / 2
            y = j * spacing * np.sqrt(3) / 2
            if np.hypot(x, y) < 1.0 - spacing / 2:
                points.append([x, y])
    points = np.array(points)
    # Select N closest to center
    radii = np.hypot(points[:, 0], points[:, 1])
    idx = np.argsort(radii, kind='stable')[:N]
    return points[idx]

def _optimize_positions(points, deadline):
    """Local optimization using L-BFGS-B on a smooth surrogate objective."""
    def objective(x_flat):
        pts = x_flat.reshape(N, 2)
        d = cdist(pts, pts)
        iu = np.triu_indices(N, k=1)
        pair_d = d[iu]
        radii = np.hypot(pts[:, 0], pts[:, 1])
        boundary_d = 1.0 - radii
        all_d = np.concatenate([pair_d, boundary_d])
        # Soft min
        beta = 80.0
        scaled = all_d / 2.0
        max_val = np.max(-beta * scaled)
        log_sum_exp = max_val + np.log(np.sum(np.exp(-beta * scaled - max_val)))
        soft_min = -log_sum_exp / beta
        return -soft_min
    
    x0 = points.flatten()
    try:
        result = minimize(
            objective, x0, method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
        )
        if time.time() > deadline:
            return points
        opt_pts = result.x.reshape(N, 2)
        # Clamp to unit circle
        radii = np.hypot(opt_pts[:, 0], opt_pts[:, 1])
        mask = radii > 1.0
        if np.any(mask):
            opt_pts[mask] /= radii[mask, np.newaxis]
        return opt_pts
    except:
        return points

def _global_search_ring_params(deadline):
    """Use differential evolution to optimize ring parameters."""
    # Parameters: ring radii and angular offsets for each ring
    # Structure: 1 center + rings with n_k points each
    # Total = 1 + sum(n_k) = 50, so sum(n_k) = 49
    # Try: 6, 12, 18, 13
    ring_counts = [6, 12, 18, 13]
    
    def decode(params):
        points = [np.array([0.0, 0.0])]
        for k, nk in enumerate(ring_counts):
            r = params[k]  # ring radius
            offset = params[len(ring_counts) + k]  # angular offset
            for m in range(nk):
                angle = 2 * np.pi * m / nk + offset
                points.append(np.array([r * np.cos(angle), r * np.sin(angle)]))
        return np.array(points[:N])
    
    def objective(params):
        pts = decode(params)
        score = _compute_score(pts)
        return -score
    
    # Bounds for each ring radius (0.2 to 0.98) and offset (0 to 2*pi)
    bounds = [(0.2, 0.98)] * 4 + [(0, 2 * np.pi)] * 4
    
    try:
        result = differential_evolution(
            objective, bounds, maxiter=50, popsize=15,
            tol=1e-8, seed=42, polish=True,
            updating='immediate', init='sobol'
        )
        if time.time() > deadline:
            return _get_ring_initialization()
        best_params = result.x
        best_pts = decode(best_params)
        return best_pts
    except:
        return _get_ring_initialization()

def construct():
    t_start = time.time()
    deadline = t_start + 140.0  # 10s safety margin before 150s kill
    
    best_points = None
    best_score = 0.0
    
    # Phase 1: Ring-based global search
    if time.time() < deadline - 30:
        ring_pts = _global_search_ring_params(deadline - 30)
        ring_pts = _optimize_positions(ring_pts, deadline - 25)
        score = _compute_score(ring_pts)
        if score > best_score:
            best_score = score
            best_points = ring_pts.copy()
    
    # Phase 2: Hexagonal lattice + local optimization
    if time.time() < deadline - 20:
        hex_pts = _get_hex_initial_points()
        hex_pts = _optimize_positions(hex_pts, deadline - 15)
        score = _compute_score(hex_pts)
        if score > best_score:
            best_score = score
            best_points = hex_pts.copy()
    
    # Phase 3: Binary search on radius with feasibility check
    if best_points is not None and time.time() < deadline - 10:
        # Try to improve by binary searching on the target radius
        lo = best_score
        hi = 0.13  # slightly above known best
        
        for _ in range(8):
            if time.time() > deadline - 5:
                break
            mid = (lo + hi) / 2
            # Check if feasible with radius mid using local optimization
            test_pts = best_points.copy()
            # Scale points slightly to try to fit larger circles
            for trial in range(3):
                test_pts = _optimize_positions(test_pts, deadline - 5)
                s = _compute_score(test_pts)
                if s >= mid:
                    lo = mid
                    if s > best_score:
                        best_score = s
                        best_points = test_pts.copy()
                    break
                else:
                    hi = mid
                    break
    
    # Phase 4: Final polish with tighter tolerance
    if best_points is not None and time.time() < deadline - 3:
        def objective_tight(x_flat):
            pts = x_flat.reshape(N, 2)
            d = cdist(pts, pts)
            iu = np.triu_indices(N, k=1)
            pair_d = d[iu]
            radii = np.hypot(pts[:, 0], pts[:, 1])
            boundary_d = 1.0 - radii
            all_d = np.concatenate([pair_d, boundary_d])
            beta = 150.0
            scaled = all_d / 2.0
            max_val = np.max(-beta * scaled)
            log_sum_exp = max_val + np.log(np.sum(np.exp(-beta * scaled - max_val)))
            soft_min = -log_sum_exp / beta
            return -soft_min
        
        try:
            result = minimize(
                objective_tight, best_points.flatten(), method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-13}
            )
            polished = result.x.reshape(N, 2)
            radii = np.hypot(polished[:, 0], polished[:, 1])
            mask = radii > 1.0
            if np.any(mask):
                polished[mask] /= radii[mask, np.newaxis]
            s = _compute_score(polished)
            if s > best_score:
                best_points = polished
                best_score = s
        except:
            pass
    
    if best_points is None:
        # Fallback
        best_points = _get_hex_initial_points()
    
    # Final validation
    final_score = _compute_score(best_points)
    if final_score <= 0:
        # Return a valid fallback
        best_points = _get_ring_initialization()
        final_score = _compute_score(best_points)
    
    return best_points

# EVOLVE-BLOCK-END
