"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.spatial import ConvexHull

N = 130

# EVOLVE-BLOCK-START

def _covering_radius(points):
    """
    Compute the covering radius in radians using vectorized circumcenter calculation.
    Returns (max_radius, worst_face_index, worst_face_vertices) for targeted optimization.
    """
    if len(points) < 3:
        return np.pi / 2, -1, None

    norms = np.linalg.norm(points, axis=1, keepdims=True)
    if np.any(norms < 1e-12):
        return np.pi, -1, None
    points = points / norms

    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi / 2, -1, None

    faces = hull.simplices  # shape (F, 3)
    F = len(faces)
    if F == 0:
        return np.pi / 2, -1, None

    # Extract face vertices
    p0 = points[faces[:, 0]]  # (F, 3)
    p1 = points[faces[:, 1]]
    p2 = points[faces[:, 2]]

    # Compute circumcenter for each face
    A = p1 - p0
    B = p2 - p0

    A_A = np.einsum('ij,ij->i', A, A)
    A_B = np.einsum('ij,ij->i', A, B)
    B_B = np.einsum('ij,ij->i', B, B)

    det = A_A * B_B - A_B * A_B
    safe_det = np.where(np.abs(det) < 1e-12, 1e-12, det)

    u = (0.5 * A_A * B_B - 0.5 * A_B * B_B) / safe_det
    v = (0.5 * A_B * A_A - 0.5 * A_A * B_B) / safe_det

    O = p0 + u[:, np.newaxis] * A + v[:, np.newaxis] * B

    norm_O = np.linalg.norm(O, axis=1, keepdims=True)
    safe_norm = np.where(norm_O < 1e-12, 1e-12, norm_O)
    C = O / safe_norm

    degenerate = np.abs(det) < 1e-12
    if np.any(degenerate):
        centroids = (p0 + p1 + p2) / 3.0
        cent_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        safe_cent_norm = np.where(cent_norms < 1e-12, 1e-12, cent_norms)
        C[degenerate] = centroids[degenerate] / safe_cent_norm[degenerate]

    # Angular distances
    d0 = np.arccos(np.clip(np.einsum('ij,ij->i', C, p0), -1, 1))
    d1 = np.arccos(np.clip(np.einsum('ij,ij->i', C, p1), -1, 1))
    d2 = np.arccos(np.clip(np.einsum('ij,ij->i', C, p2), -1, 1))

    r_face = np.maximum(np.maximum(d0, d1), d2)
    max_radius = float(np.max(r_face))
    worst_idx = int(np.argmax(r_face))

    return max_radius, worst_idx, faces[worst_idx].copy()

def _covering_radius_fast(points):
    """Fast version that only returns the radius value."""
    cr, _, _ = _covering_radius(points)
    return cr

def _optimize_from_start(start_points, time_limit, rng, initial_step=0.08):
    """
    Local optimization with targeted moves toward worst facet centroid.
    """
    points = start_points.copy()
    current_cr, worst_idx, worst_face = _covering_radius(points)
    best_points = points.copy()
    best_cr = current_cr

    start_time = time.time()
    step_size = initial_step
    min_step = 1e-6
    max_iter = 500000
    no_improve_count = 0
    no_improve_limit = 150

    for _ in range(max_iter):
        if time.time() - start_time > time_limit:
            break

        # Strategy: target the worst facet's vertices with repulsion
        if worst_face is not None and worst_idx >= 0:
            # Move one of the worst facet's vertices away from the circumcenter
            target_idx = int(rng.choice(worst_face))
            # Direction: away from the centroid of the worst face
            centroid = np.mean(points[worst_face], axis=0)
            centroid_norm = np.linalg.norm(centroid)
            if centroid_norm > 1e-12:
                direction = points[target_idx] - centroid / centroid_norm
            else:
                direction = rng.normal(size=3)
            direction_norm = np.linalg.norm(direction)
            if direction_norm < 1e-12:
                direction = rng.normal(size=3)
                direction_norm = np.linalg.norm(direction)
            direction /= direction_norm
        else:
            # Random fallback
            target_idx = int(rng.integers(N))
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)

        improved = False
        for sign in [1, -1]:
            new_point = points[target_idx] + sign * step_size * direction
            new_norm = np.linalg.norm(new_point)
            if new_norm < 1e-12:
                continue
            new_point /= new_norm

            old_point = points[target_idx]
            points[target_idx] = new_point

            new_cr, new_worst_idx, new_worst_face = _covering_radius(points)

            if new_cr < current_cr - 1e-13:
                current_cr = new_cr
                worst_idx = new_worst_idx
                worst_face = new_worst_face
                improved = True
                if new_cr < best_cr:
                    best_cr = new_cr
                    best_points = points.copy()
                step_size = min(step_size * 1.02, 0.2)
                no_improve_count = 0
                break
            else:
                points[target_idx] = old_point

        if not improved:
            no_improve_count += 1
            step_size = max(step_size * 0.97, min_step)

            if no_improve_count > no_improve_limit:
                # Escape: random larger perturbation
                step_size = rng.uniform(0.03, 0.12)
                no_improve_count = 0

        if step_size < min_step * 1.01:
            break

    return best_points, best_cr

def _spectral_init(rng, n_points=N):
    """
    Generate initial points using a spectral/repulsion approach.
    Start with random points and apply a few rounds of repulsion-based relaxation.
    """
    points = rng.normal(size=(n_points, 3))
    points /= np.linalg.norm(points, axis=1, keepdims=True)

    # Simple repulsion relaxation
    for _ in range(50):
        # Compute pairwise distances
        # For efficiency, just do a few gradient steps
        for i in range(n_points):
            # Sum of repulsion from all other points
            diff = points[i] - points  # (n, 3)
            dists = np.linalg.norm(diff, axis=1)
            # Avoid division by zero
            dists[dists < 1e-12] = 1e-12
            # Repulsion force: stronger when closer
            forces = diff / (dists[:, np.newaxis] ** 2)
            forces[i] = 0  # No self-repulsion
            grad = np.sum(forces, axis=0)
            # Project onto tangent plane
            proj = grad - np.dot(grad, points[i]) * points[i]
            points[i] += 0.01 * proj
            points[i] /= np.linalg.norm(points[i])

    return points

def construct():
    total_deadline = 145.0
    start_time = time.time()

    rng = np.random.default_rng(42)

    best_points = None
    best_cr = np.pi

    # 1. Fibonacci spiral start (deterministic, good baseline)
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    fib_points, fib_cr = _optimize_from_start(fib_points, 20.0, rng, initial_step=0.08)
    if fib_cr < best_cr:
        best_cr = fib_cr
        best_points = fib_points.copy()

    # 2. Spectral/repulsion initialized start
    elapsed = time.time() - start_time
    if total_deadline - elapsed > 10.0:
        spec_points = _spectral_init(rng)
        spec_points, spec_cr = _optimize_from_start(spec_points, 15.0, rng, initial_step=0.06)
        if spec_cr < best_cr:
            best_cr = spec_cr
            best_points = spec_points.copy()

    # 3. Multiple random restarts
    elapsed = time.time() - start_time
    remaining = total_deadline - elapsed - 5.0
    if remaining < 2.0:
        if best_points is not None:
            norms = np.linalg.norm(best_points, axis=1, keepdims=True)
            best_points = best_points / norms
        return best_points

    num_restarts = int(min(8, max(2, remaining / 12.0)))
    time_per_restart = remaining / num_restarts

    for i_restart in range(num_restarts):
        elapsed = time.time() - start_time
        if elapsed > total_deadline - 5.0:
            break

        random_points = rng.normal(size=(N, 3))
        random_points /= np.linalg.norm(random_points, axis=1, keepdims=True)

        alloc_time = min(time_per_restart, total_deadline - elapsed - 5.0)
        if alloc_time < 1.0:
            break

        init_step = rng.uniform(0.04, 0.12)
        rand_points, rand_cr = _optimize_from_start(random_points, alloc_time, rng, initial_step=init_step)

        if rand_cr < best_cr:
            best_cr = rand_cr
            best_points = rand_points.copy()

    # Final normalization
    if best_points is not None:
        norms = np.linalg.norm(best_points, axis=1, keepdims=True)
        best_points = best_points / norms

    return best_points

# EVOLVE-BLOCK-END
