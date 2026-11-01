"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

DEADLINE_FRACTION = 0.85  # safety margin before 150s limit


def _score(points):
    """Compute the packing radius: min(min_pair_dist/2, min(1 - |c_i|))."""
    if len(points) < 2:
        return 0.0
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    # Distance to boundary
    radii = np.sqrt(np.sum(points ** 2, axis=1))
    min_boundary = np.min(1.0 - radii)
    return min(min_pair, min_boundary)


def _objective_and_grad(x, N):
    """
    Objective: maximize r = min(min_pair/2, min(1-|c|)).
    We minimize -r. Using a smooth approximation: soft-min of all constraints.
    
    Actually, let's use a direct approach: minimize the negative of the minimum constraint.
    For smoothness, we can use: -min(constraints) ≈ soft-min with large beta.
    
    But for gradient-based optimization, we can just compute the gradient numerically
    or analytically. Let's use a simple approach: minimize the negative of the score,
    with the score being the minimum of all pairwise half-distances and boundary distances.
    
    For the gradient, we need d(-min_i g_i)/dx where g_i are the constraints.
    The subgradient of min is -1 on the argmin set. So d(-min)/dx = +1 on argmin set.
    
    Let's compute the score and its subgradient.
    """
    points = x.reshape(N, 2)
    
    # Compute all pairwise distances and boundary distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2) + 1e-15)
    np.fill_diagonal(dists, np.inf)
    half_dists = dists / 2.0
    
    radii = np.sqrt(np.sum(points ** 2, axis=1) + 1e-15)
    boundary_dists = 1.0 - radii
    
    # All constraints
    all_constraints = np.concatenate([half_dists[~np.isinf(half_dists)].flatten(), boundary_dists])
    
    # Find the minimum constraint(s)
    min_val = np.min(all_constraints)
    
    # Find which constraints are at the minimum (within tolerance)
    tol = 1e-10
    min_mask = all_constraints <= min_val + tol
    
    # Build subgradient
    grad = np.zeros_like(x)
    
    # Count how many constraints are at minimum
    # Pairwise constraints: index i*N + j for i<j
    # Boundary constraints: index N*(N-1)/2 + i for i in range(N)
    n_pairs = N * (N - 1) // 2
    
    # Process pairwise constraints
    pair_idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            if min_mask[pair_idx]:
                # d(half_dist_ij)/dx = (x_i - x_j) / (2 * dist_ij) for each component
                # But we're minimizing -min, so gradient is + (x_i - x_j) / (2 * dist_ij)
                d = dists[i, j]
                if d > 1e-15:
                    grad[i] += (points[i] - points[j]) / (2.0 * d)
                    grad[j] += (points[j] - points[i]) / (2.0 * d)
            pair_idx += 1
    
    # Process boundary constraints
    for i in range(N):
        idx = n_pairs + i
        if min_mask[idx]:
            # d(1 - |c_i|)/dx = -x_i / |c_i|
            r = radii[i]
            if r > 1e-15:
                grad[i] += -points[i] / r
            else:
                # If at origin, boundary dist is 1, gradient is 0
                pass
    
    # Objective value: -min_val
    return -min_val, grad


def _hex_lattice_init(N, angle=0.0):
    """Generate hexagonal lattice points, rotated by angle, clipped to unit disk."""
    # Estimate spacing
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                rows.append([x, y])
        points = np.array(rows)
        
        # Rotate
        if angle != 0.0:
            c, s = np.cos(angle), np.sin(angle)
            rot = np.array([[c, -s], [s, c]])
            points = points @ rot.T
        
        radii = np.hypot(points[:, 0], points[:, 1])
        mask = radii <= 1.0 - spacing / 2.0
        points = points[mask]
        
        if len(points) >= N:
            # Sort by radius to prefer center points
            radii = np.hypot(points[:, 0], points[:, 1])
            idx = np.argsort(radii, kind="stable")[:N]
            return points[idx]
        spacing *= 0.98


def _ring_init(N):
    """Generate points on concentric rings."""
    points = [[0.0, 0.0]]
    r = 0.0
    ring_num = 0
    while len(points) < N:
        ring_num += 1
        # Approximate number of points on ring: 2*pi*r / (2*r_est)
        # Use a heuristic: ~6*ring_num points
        n_on_ring = max(6, 6 * ring_num)
        r = ring_num * 0.03  # rough estimate
        for k in range(n_on_ring):
            angle = 2.0 * np.pi * k / n_on_ring
            points.append([r * np.cos(angle), r * np.sin(angle)])
    
    points = np.array(points[:N])
    return points


def _random_init(N, seed=42):
    """Generate random points inside unit disk."""
    rng = np.random.RandomState(seed)
    points = []
    for _ in range(N):
        r = np.sqrt(rng.rand())
        theta = 2.0 * np.pi * rng.rand()
        points.append([r * np.cos(theta), r * np.sin(theta)])
    return np.array(points)


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_FRACTION * 150.0  # 127.5s
    
    # Try multiple initializations
    best_points = None
    best_score = -np.inf
    
    # 1. Hex lattice (original)
    try:
        pts = _hex_lattice_init(N, angle=0.0)
        sc = _score(pts)
        if sc > best_score:
            best_score = sc
            best_points = pts.copy()
    except Exception:
        pass
    
    # 2. Rotated hex lattices
    for angle in [np.pi / 6, np.pi / 3, np.pi / 4]:
        if time.time() > deadline:
            break
        try:
            pts = _hex_lattice_init(N, angle=angle)
            sc = _score(pts)
            if sc > best_score:
                best_score = sc
                best_points = pts.copy()
        except Exception:
            pass
    
    # 3. Ring initialization
    if time.time() < deadline:
        try:
            pts = _ring_init(N)
            sc = _score(pts)
            if sc > best_score:
                best_score = sc
                best_points = pts.copy()
        except Exception:
            pass
    
    # 4. Random initialization with a few seeds
    for seed in [42, 123, 777]:
        if time.time() > deadline:
            break
        try:
            pts = _random_init(N, seed=seed)
            sc = _score(pts)
            if sc > best_score:
                best_score = sc
                best_points = pts.copy()
        except Exception:
            pass
    
    if best_points is None:
        # Fallback
        best_points = _hex_lattice_init(N, angle=0.0)
        best_score = _score(best_points)
    
    # Now polish the best point set using gradient ascent on the score
    # We minimize -score using the subgradient method
    
    x = best_points.flatten().copy()
    current_score = best_score
    
    # Use scipy minimize with BFGS or L-BFGS-B
    # We'll do a few rounds of optimization
    
    remaining_time = deadline - time.time()
    if remaining_time > 5.0:
        try:
            # Use L-BFGS-B with the objective and gradient
            result = minimize(
                _objective_and_grad,
                x,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 500, 'gtol': 1e-8, 'maxls': 50}
            )
            new_points = result.x.reshape(N, 2)
            new_score = _score(new_points)
            if new_score > current_score:
                best_points = new_points
                best_score = new_score
        except Exception:
            pass
    
    # If we still have time, do a second round of polishing
    remaining_time = deadline - time.time()
    if remaining_time > 10.0:
        try:
            x = best_points.flatten().copy()
            result = minimize(
                _objective_and_grad,
                x,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'gtol': 1e-10, 'maxls': 50}
            )
            new_points = result.x.reshape(N, 2)
            new_score = _score(new_points)
            if new_score > best_score:
                best_points = new_points
                best_score = new_score
        except Exception:
            pass
    
    # Final check: ensure all points are inside the unit disk
    radii = np.hypot(best_points[:, 0], best_points[:, 1])
    mask = radii <= 1.0
    if not np.all(mask):
        # Clip points that are slightly outside
        for i in range(N):
            if radii[i] > 1.0:
                best_points[i] *= (1.0 - 1e-6) / radii[i]
    
    return best_points
# EVOLVE-BLOCK-END
