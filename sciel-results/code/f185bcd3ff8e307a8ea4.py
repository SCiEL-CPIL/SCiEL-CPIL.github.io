"""Improved program for covering the sphere with 130 equal caps.
Combines Riemannian L-BFGS-B optimization with convex-hull-based covering radius.
"""
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time
import math

N = 130


def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms


def _fibonacci_sphere(n, seed_noise=0.0, seed=42):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = math.pi * (1.0 + math.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    if seed_noise > 0:
        rng = np.random.default_rng(seed)
        noise = rng.normal(0, seed_noise, (n, 3))
        pts = pts + noise
        pts = _normalize(pts)
    return pts


def _covering_radius_deg(pts):
    """Compute covering radius in degrees via convex hull facets."""
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return 180.0

    max_rad = 0.0
    for simplex in hull.simplices:
        verts = pts[simplex]
        if len(verts) < 3:
            continue
        v0, v1, v2 = verts[0], verts[1], verts[2]
        edge1 = v1 - v0
        edge2 = v2 - v0
        normal = np.cross(edge1, edge2)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-15:
            continue
        normal = normal / norm_len

        centroid = np.mean(verts, axis=0)
        if np.dot(normal, centroid) < 0:
            normal = -normal

        for v in verts:
            cos_ang = np.clip(np.dot(v, normal), -1.0, 1.0)
            ang = math.acos(cos_ang)
            if ang > max_rad:
                max_rad = ang

    return math.degrees(max_rad)


def _thomson_init(n, seed=42):
    """Thomson (electrostatic) initialization via gradient descent on repulsion energy."""
    rng = np.random.default_rng(seed)
    P = rng.standard_normal((n, 3))
    P = _normalize(P)

    def energy_and_grad(P_flat):
        P = P_flat.reshape(-1, 3)
        P = _normalize(P)

        dots = P @ P.T
        dist_sq = 2.0 - 2.0 * dots
        np.fill_diagonal(dist_sq, 1.0)

        inv_dist_sq = 1.0 / np.maximum(dist_sq, 1e-10)
        np.fill_diagonal(inv_dist_sq, 0.0)
        energy = np.sum(inv_dist_sq)

        grad = np.zeros_like(P)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                d_sq = dist_sq[i, j]
                if d_sq < 1e-10:
                    continue
                grad[i] += 4.0 * P[j] / (d_sq * d_sq)

        grad_dot_P = np.sum(grad * P, axis=1, keepdims=True)
        grad_tangent = grad - grad_dot_P * P

        return energy, grad_tangent.flatten()

    try:
        res = minimize(
            energy_and_grad,
            P.flatten(),
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
        )
        if res.x is not None:
            P = res.x.reshape(-1, 3)
            P = _normalize(P)
    except Exception:
        pass

    return P


def _riemannian_optimize(P_init, max_iters=200, tol=1e-8):
    """
    Riemannian L-BFGS-B optimization on the sphere.
    Minimizes a smooth surrogate of the covering radius.
    """
    P = _normalize(P_init.copy())

    # Precompute a fixed set of sample points on the sphere for the surrogate
    rng = np.random.default_rng(12345)
    n_samples = 3000
    theta = rng.uniform(0, math.pi, n_samples)
    phi = rng.uniform(0, 2 * math.pi, n_samples)
    samples = np.column_stack([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ])

    temperature = 0.03

    def objective_and_grad(P_flat):
        P = P_flat.reshape(-1, 3)
        P = _normalize(P)

        # Compute distances from each sample to each point
        dots = samples @ P.T
        dots = np.clip(dots, -1.0 + 1e-12, 1.0 - 1e-12)
        dists = np.arccos(dots)  # (n_samples, N)
        min_dists = np.min(dists, axis=1)  # (n_samples,)

        # Soft max surrogate
        t = 1.0 / temperature
        max_d = np.max(min_dists)
        exp_shifted = np.exp(t * (min_dists - max_d))
        log_sum = np.log(np.sum(exp_shifted))
        value = max_d + log_sum / t

        # Gradient w.r.t. min_dists
        weights = exp_shifted / np.sum(exp_shifted)  # (n_samples,)

        # Gradient w.r.t. P
        nearest_idx = np.argmin(dists, axis=1)  # (n_samples,)
        grad_P = np.zeros((N, 3))

        for s in range(n_samples):
            j = nearest_idx[s]
            w = weights[s]
            if w < 1e-15:
                continue
            d_val = dots[s, j]
            denom = math.sqrt(max(1.0 - d_val * d_val, 1e-15))
            grad_P[j] -= w * samples[s] / denom

        # Project gradient to tangent space (Riemannian gradient)
        grad_dot_P = np.sum(grad_P * P, axis=1, keepdims=True)
        grad_tangent = grad_P - grad_dot_P * P

        return value, grad_tangent.flatten()

    try:
        res = minimize(
            objective_and_grad,
            P.flatten(),
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': max_iters, 'ftol': tol, 'gtol': 1e-7, 'disp': False}
        )
        if res.x is not None:
            P = res.x.reshape(-1, 3)
            P = _normalize(P)
    except Exception:
        pass

    return P


def _polish_with_hull(P_init, max_iters=50):
    """
    Local refinement using convex hull facet information.
    Moves points to reduce the largest circumradius.
    """
    P = _normalize(P_init.copy())
    best_P = P.copy()
    best_radius = _covering_radius_deg(P)

    for it in range(max_iters):
        try:
            hull = ConvexHull(P)
        except Exception:
            break

        # Find the facet with the largest circumradius
        max_facet_rad = 0.0
        max_facet_idx = -1
        max_facet_verts = None

        for idx, simplex in enumerate(hull.simplices):
            verts = P[simplex]
            if len(verts) < 3:
                continue
            v0, v1, v2 = verts[0], verts[1], verts[2]
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
            norm_len = np.linalg.norm(normal)
            if norm_len < 1e-15:
                continue
            normal = normal / norm_len
            centroid = np.mean(verts, axis=0)
            if np.dot(normal, centroid) < 0:
                normal = -normal

            facet_rad = 0.0
            for v in verts:
                cos_ang = np.clip(np.dot(v, normal), -1.0, 1.0)
                ang = math.acos(cos_ang)
                if ang > facet_rad:
                    facet_rad = ang

            if facet_rad > max_facet_rad:
                max_facet_rad = facet_rad
                max_facet_idx = idx
                max_facet_verts = simplex.copy()

        if max_facet_idx < 0 or max_facet_rad < math.radians(11.0):
            break

        # Move the vertex of the worst facet toward the facet center
        # to reduce the circumradius
        verts = P[max_facet_verts]
        v0, v1, v2 = verts[0], verts[1], verts[2]
        edge1 = v1 - v0
        edge2 = v2 - v0
        normal = np.cross(edge1, edge2)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-15:
            break
        normal = normal / norm_len
        centroid = np.mean(verts, axis=0)
        if np.dot(normal, centroid) < 0:
            normal = -normal

        # Find which vertex is farthest from the normal direction
        max_dist = 0.0
        worst_vert_idx = 0
        for i, v in enumerate(verts):
            cos_ang = np.clip(np.dot(v, normal), -1.0, 1.0)
            ang = math.acos(cos_ang)
            if ang > max_dist:
                max_dist = ang
                worst_vert_idx = i

        # Move this vertex slightly toward the centroid direction
        # in the tangent plane
        worst_v = P[max_facet_verts[worst_vert_idx]]
        move_dir = centroid - worst_v
        # Project to tangent plane
        move_dir -= np.dot(move_dir, worst_v) * worst_v
        move_norm = np.linalg.norm(move_dir)
        if move_norm > 1e-10:
            move_dir = move_dir / move_norm
            step = 0.01
            P[max_facet_verts[worst_vert_idx]] = worst_v + step * move_dir
            P = _normalize(P)

        radius = _covering_radius_deg(P)
        if radius < best_radius:
            best_radius = radius
            best_P = P.copy()
        else:
            P = best_P.copy()

    return best_P


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit

    # Initial configurations
    best_points = _fibonacci_sphere(N)
    best_radius = _covering_radius_deg(best_points)

    # Try Thomson initialization
    if time.time() < deadline - 40.0:
        thomson_P = _thomson_init(N, seed=42)
        radius = _covering_radius_deg(thomson_P)
        if radius < best_radius:
            best_radius = radius
            best_points = thomson_P.copy()

    # Multi-restart Riemannian optimization
    n_restarts = 3
    for restart in range(n_restarts):
        if time.time() > deadline - 30.0:
            break

        if restart == 0:
            current = best_points.copy()
        else:
            rng = np.random.default_rng(restart * 777 + 42)
            noise = rng.randn(N, 3) * 0.05
            current = best_points + noise
            current = _normalize(current)

        # Riemannian L-BFGS-B optimization
        optimized = _riemannian_optimize(current, max_iters=100, tol=1e-8)

        # Evaluate
        radius = _covering_radius_deg(optimized)
        if radius < best_radius:
            best_radius = radius
            best_points = optimized.copy()

    # Final polish pass if time permits
    if time.time() < deadline - 15.0:
        polished = _polish_with_hull(best_points, max_iters=30)
        radius = _covering_radius_deg(polished)
        if radius < best_radius:
            best_points = polished

    # Ensure output is valid
    best_points = _normalize(best_points)

    return best_points
