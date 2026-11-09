import numpy as np
import time
from scipy.spatial import ConvexHull

N = 50

def _fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, seed=None):
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

def _compute_facet_data(points):
    """
    Compute convex hull facets and their circumradii.
    Returns: (all_radii, circumcenters, simplices, valid_mask)
    all_radii: array of circumradii (radians) for each valid facet
    circumcenters: array of circumcenter unit vectors
    simplices: array of facet vertex indices
    valid_mask: boolean mask of valid facets
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return None, None, None, None

    simplices = hull.simplices  # (F, 3)
    v1 = points[simplices[:, 0]]
    v2 = points[simplices[:, 1]]
    v3 = points[simplices[:, 2]]

    n_vec = np.cross(v2 - v1, v3 - v1)
    norms = np.linalg.norm(n_vec, axis=1)
    valid = norms > 1e-10
    if not np.any(valid):
        return None, None, None, None

    idx_valid = np.where(valid)[0]
    n_valid = n_vec[idx_valid]
    norms_valid = norms[idx_valid]
    v1_valid = v1[idx_valid]
    v2_valid = v2[idx_valid]
    v3_valid = v3[idx_valid]

    c_circ = n_valid / norms_valid[:, None]

    # Ensure circumcenter points outward (same side as centroid)
    centroids = (v1_valid + v2_valid + v3_valid) / 3.0
    dots = np.sum(c_circ * centroids, axis=1)
    flip = dots < 0
    c_circ[flip] = -c_circ[flip]

    cos_theta = np.sum(c_circ * v1_valid, axis=1)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    radii = np.arccos(cos_theta)

    return radii, c_circ, simplices[idx_valid], valid

def _covering_radius(points):
    """Return the covering radius in radians."""
    radii, _, _, _ = _compute_facet_data(points)
    if radii is None:
        return np.pi
    return np.max(radii)

def _smooth_covering_radius(points, beta=50.0):
    """
    Smooth surrogate for the max circumradius using soft-max.
    Returns (smooth_value, all_radii, circumcenters, simplices)
    """
    radii, c_circ, simplices, valid = _compute_facet_data(points)
    if radii is None:
        return np.pi, None, None, None

    # Soft-max approximation
    max_r = np.max(radii)
    shifted = radii - max_r
    weights = np.exp(beta * shifted)
    weights /= np.sum(weights)
    smooth_val = max_r + np.log(np.sum(weights)) / beta

    return smooth_val, radii, c_circ, simplices

def _gradient_step(points, beta=50.0, learning_rate=0.01):
    """
    Compute a gradient step to reduce the smooth covering radius.
    Aggregates gradients from all active facets (weighted by soft-max weights).
    Returns new points after the step.
    """
    radii, c_circ, simplices, valid = _compute_facet_data(points)
    if radii is None:
        return points.copy()

    max_r = np.max(radii)
    shifted = radii - max_r
    weights = np.exp(beta * shifted)
    total_w = np.sum(weights)
    weights /= total_w

    # Gradient of circumradius w.r.t. vertices:
    # For a facet with circumcenter c and vertices v1, v2, v3,
    # the gradient of the circumradius angle at vertex vi is:
    # d(r)/d(vi) = -c / sin(theta) + vi * cos(theta)/sin(theta)
    # where theta is the circumradius angle.
    # More precisely, if cos(theta) = c . vi, then:
    # d(theta)/d(vi) = -(c - vi*cos(theta)) / sin(theta)
    # = -(c - vi*(c.vi)) / sin(theta)

    grad = np.zeros_like(points)
    sin_theta = np.sqrt(np.maximum(1.0 - radii**2, 1e-12))  # sin(arccos(cos)) = sqrt(1-cos^2)
    cos_theta = np.cos(radii)

    for i in range(len(radii)):
        w = weights[i]
        if w < 1e-8:
            continue
        s = sin_theta[i]
        if s < 1e-10:
            continue
        c = c_circ[i]
        ct = cos_theta[i]
        # Gradient contribution for each vertex in this facet
        for k in range(3):
            vi = points[simplices[i, k]]
            # d(theta)/d(vi) = -(c - vi * ct) / s
            g = -(c - vi * ct) / s
            grad[simplices[i, k]] += w * g

    # Project gradient onto tangent space of sphere
    # For each point, remove radial component
    for i in range(N):
        radial = np.dot(grad[i], points[i])
        grad[i] -= radial * points[i]

    # Take step
    new_points = points - learning_rate * grad
    # Normalize back to sphere
    norms = np.linalg.norm(new_points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    new_points /= norms

    return new_points

def _line_search(points, lr, beta=50.0, max_backtracks=20):
    """
    Perform backtracking line search to find a good learning rate.
    Returns (new_points, used_lr, improved)
    """
    current_val, _, _, _ = _smooth_covering_radius(points, beta)
    current_cr = _covering_radius(points)

    for bt in range(max_backtracks):
        new_pts = _gradient_step(points, beta, lr)
        new_val, _, _, _ = _smooth_covering_radius(new_pts, beta)
        new_cr = _covering_radius(new_pts)

        if new_cr < current_cr or new_val < current_val:
            return new_pts, lr, True
        lr *= 0.5
        if lr < 1e-8:
            break

    return points, lr, False

def _local_optimize(points, beta=50.0, max_iters=500, lr_init=0.05, deadline=None):
    """
    Local optimization using gradient descent with line search.
    """
    best_pts = points.copy()
    best_cr = _covering_radius(best_pts)
    lr = lr_init

    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break

        # Try to improve
        new_pts, lr, improved = _line_search(points, lr, beta)

        if improved:
            points = new_pts
            cr = _covering_radius(points)
            if cr < best_cr:
                best_cr = cr
                best_pts = points.copy()
        else:
            # Try smaller step
            lr *= 0.5
            if lr < 1e-10:
                break

        # Occasionally add small random perturbation to escape local minima
        if it % 100 == 50:
            idx = np.random.randint(N)
            noise = np.random.randn(3) * 0.01
            p_new = points[idx] + noise
            n_new = np.linalg.norm(p_new)
            if n_new > 1e-10:
                points[idx] = p_new / n_new

    return best_pts, best_cr

def _repulsive_energy(points, k=5):
    """
    Compute a repulsive energy that encourages uniform distribution.
    Uses inverse distance power for top-k nearest neighbors.
    """
    # Compute pairwise distances
    dots = np.dot(points, points.T)
    dots = np.clip(dots, -1.0, 1.0)
    dists = np.arccos(dots)  # angular distances

    # For each point, sum of top-k nearest neighbor repulsion
    energy = 0.0
    for i in range(N):
        # Exclude self
        d = dists[i].copy()
        d[i] = np.pi
        # Get k nearest
        nn_idx = np.argsort(d)[:k]
        nn_dists = d[nn_idx]
        # Repulsive: 1/d^2
        energy += np.sum(1.0 / np.maximum(nn_dists, 0.01)**2)

    return energy

def _repulsive_gradient(points, k=5):
    """
    Compute gradient of repulsive energy.
    """
    grad = np.zeros_like(points)
    dots = np.dot(points, points.T)
    dots = np.clip(dots, -1.0, 1.0)
    dists = np.arccos(dots)

    for i in range(N):
        d = dists[i].copy()
        d[i] = np.pi
        nn_idx = np.argsort(d)[:k]
        for j in nn_idx:
            d_ij = dists[i, j]
            if d_ij < 0.01:
                continue
            # d(1/d^2)/d(theta) = -2/d^3
            # d(theta)/d(p_i) = -(p_j - p_i * (p_i.p_j)) / sin(theta)
            sin_theta = np.sin(d_ij)
            if sin_theta < 1e-10:
                continue
            cos_theta = dots[i, j]
            g = -2.0 / d_ij**3 * (-(points[j] - points[i] * cos_theta) / sin_theta)
            grad[i] += g

    # Project to tangent space
    for i in range(N):
        radial = np.dot(grad[i], points[i])
        grad[i] -= radial * points[i]

    return grad

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    np.random.seed(42)

    best_points = None
    best_radius = np.pi

    def update_best(pts):
        nonlocal best_points, best_radius
        cr = _covering_radius(pts)
        if cr < best_radius:
            best_radius = cr
            best_points = pts.copy()
        return cr

    # Phase 1: Multiple diverse initializations with local optimization
    initializations = [
        _fibonacci_sphere(N),
        _random_sphere(N, seed=1),
        _random_sphere(N, seed=2),
        _random_sphere(N, seed=3),
        # Slightly perturbed fibonacci
        _fibonacci_sphere(N) + np.random.randn(N, 3) * 0.05,
        _random_sphere(N, seed=4),
        _random_sphere(N, seed=5),
        _random_sphere(N, seed=6),
    ]

    for init_idx, init_pts in enumerate(initializations):
        if time.time() > deadline - 30:
            break

        # Normalize
        norms = np.linalg.norm(init_pts, axis=1, keepdims=True)
        init_pts = init_pts / np.where(norms < 1e-10, 1.0, norms)

        # Check it's not in a closed hemisphere
        # Simple check: not all points on one side of any plane through origin
        # For now, just proceed and let the optimizer handle it

        cr = update_best(init_pts)

        # Local optimization
        opt_pts, opt_cr = _local_optimize(init_pts, beta=50.0, max_iters=400, lr_init=0.05, deadline=deadline - 25)
        update_best(opt_pts)

        # If we have time, do a second round with different beta
        if time.time() < deadline - 30:
            opt_pts2, opt_cr2 = _local_optimize(opt_pts, beta=100.0, max_iters=200, lr_init=0.02, deadline=deadline - 20)
            update_best(opt_pts2)

    # Phase 2: Repulsive energy refinement on best solution
    if best_points is not None and time.time() < deadline - 15:
        pts = best_points.copy()
        lr = 0.01
        for it in range(200):
            if time.time() > deadline - 10:
                break
            grad = _repulsive_gradient(pts, k=5)
            new_pts = pts - lr * grad
            norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
            new_pts /= np.where(norms < 1e-10, 1.0, norms)

            new_cr = _covering_radius(new_pts)
            if new_cr < best_radius:
                best_radius = new_cr
                best_points = new_pts.copy()
                pts = new_pts.copy()
            else:
                lr *= 0.9

    # Phase 3: Final gradient polish
    if best_points is not None and time.time() < deadline - 5:
        pts = best_points.copy()
        lr = 0.01
        for it in range(300):
            if time.time() > deadline - 2:
                break
            new_pts, lr, improved = _line_search(pts, lr, beta=100.0, max_backtracks=10)
            if improved:
                pts = new_pts
                cr = _covering_radius(pts)
                if cr < best_radius:
                    best_radius = cr
                    best_points = pts.copy()
            else:
                lr *= 0.5
                if lr < 1e-10:
                    break

    # Final validation and normalization
    if best_points is None:
        best_points = _fibonacci_sphere(N)

    best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)

    # Ensure no coincident points
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(best_points[i] - best_points[j])
            if d < 1e-6:
                # Separate them
                angle = np.arccos(np.clip(np.dot(best_points[i], best_points[j]), -1, 1))
                if angle < 1e-4:
                    # Rotate point j slightly
                    axis = np.cross(best_points[i], best_points[j])
                    if np.linalg.norm(axis) < 1e-10:
                        axis = np.array([1.0, 0.0, 0.0])
                    axis /= np.linalg.norm(axis)
                    theta = 0.01
                    c, s = np.cos(theta), np.sin(theta)
                    R = np.array([
                        [c, -s, 0],
                        [s, c, 0],
                        [0, 0, 1]
                    ])
                    # Rotate around axis
                    R = c * np.eye(3) + (1-c) * np.outer(axis, axis) + s * np.array([
                        [0, -axis[2], axis[1]],
                        [axis[2], 0, -axis[0]],
                        [-axis[1], axis[0], 0]
                    ])
                    best_points[j] = R @ best_points[j]
                    best_points[j] /= np.linalg.norm(best_points[j])

    # Final normalization
    best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)

    return best_points
