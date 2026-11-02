import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time
import warnings

N = 130
TOTAL_TIME_LIMIT = 145.0  # safety margin before 150s kill


def _covering_radius_from_hull(points):
    """Compute exact covering radius in radians from convex hull facets."""
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi
    max_radius = 0.0
    for simplex in hull.simplices:
        verts = points[simplex]
        a, b, c = verts[0], verts[1], verts[2]
        normal = np.cross(b - a, c - a)
        norm_val = np.linalg.norm(normal)
        if norm_val < 1e-12:
            continue
        normal = normal / norm_val
        centroid = np.mean(verts, axis=0)
        if np.dot(normal, centroid) < 0:
            normal = -normal
        dots = np.clip(np.dot(verts, normal), -1.0, 1.0)
        angles = np.arccos(dots)
        facet_radius = np.max(angles)
        max_radius = max(max_radius, facet_radius)
    return max_radius


def _spherical_circumcenter(a, b, c):
    """
    Compute the spherical circumcenter of triangle (a,b,c) on the unit sphere.
    Returns the unit vector direction of the circumcenter, or None if degenerate.
    The circumcenter is the point equidistant (angularly) from a, b, c.
    In 3D, it lies in the plane spanned by a, b, c, i.e., along the normal to the
    triangle's plane (for a planar triangle inscribed in a sphere, the circumcenter
    projects to the intersection of the plane with the sphere's radius).
    For a spherical triangle, the circumcenter direction is the normalized
    vector: a*(sin(B)...) -- but a robust approach: the circumcenter of the
    planar triangle abc, projected onto the sphere, gives the spherical circumcenter
    only if the triangle is small. A more robust method: solve for the point x
    on the sphere equidistant from a, b, c.
    
    Robust method: the circumcenter direction is proportional to
    a * (|b|^2 |c|^2 - ...) -- actually for unit vectors, the circumcenter of the
    great-circle triangle is along the vector:
    x = a * sin(2*angle_bc) + b * sin(2*angle_ca) + c * sin(2*angle_ab)
    where angle_bc is the angle at vertex b, etc.
    
    Simpler robust approach: the circumcenter lies in the plane of a,b,c.
    It is the unit vector u in span{a,b,c} such that u.a = u.b = u.c.
    Solve the linear system: u = alpha*a + beta*b + gamma*c with u.a=u.b=u.c and |u|=1.
    Equivalently, u is along the normal to the plane only for right triangles.
    
    Use the formula: circumcenter_dir = normalize(a*(dot(b,c)-dot(a,a)*dot(a, ...))
    Actually the cleanest: the circumcenter of the planar triangle with vertices
    a,b,c (unit vectors) projected onto the unit sphere is NOT the spherical
    circumcenter in general. 
    
    Correct spherical circumcenter: the point x on S^2 minimizing max(angle(x,a),angle(x,b),angle(x,c))
    subject to x in the plane of a,b,c. This is the same as the circumcenter of the
    planar triangle formed by a,b,c in 3D, normalized.
    
    Planar circumcenter formula:
    Let A=a, B=b, C=c.
    d = 2*(A*(B[1]*C[2]-C[1]*B[2]) - B*(A[1]*C[2]-C[1]*A[2]) + C*(A[1]*B[2]-B[1]*A[2]))
    ux = (|A|^2*(B[1]*C[2]-C[1]*B[2]) - |B|^2*(A[1]*C[2]-C[1]*A[2]) + |C|^2*(A[1]*B[2]-B[1]*A[2])) / d
    uy = (|A|^2*(B[2]*C[0]-C[2]*B[0]) - |B|^2*(A[2]*C[0]-C[2]*A[0]) + |C|^2*(A[2]*B[0]-B[2]*A[0])) / d
    uz = (|A|^2*(B[0]*C[1]-C[0]*B[1]) - |B|^2*(A[0]*C[1]-C[0]*A[1]) + |C|^2*(A[0]*B[1]-B[0]*A[1])) / d
    Then circumcenter = (ux, uy, uz), normalize to unit length.
    """
    A, B, C = a, b, c
    a2 = np.dot(A, A)
    b2 = np.dot(B, B)
    c2 = np.dot(C, C)
    
    # 2 * cross(B-A, C-A) dot ... actually use the standard formula
    # d = 2 * (A x (B x C)) ... let me use the explicit formula
    d = 2.0 * (A[0] * (B[1] * C[2] - C[1] * B[2]) -
               A[1] * (B[0] * C[2] - C[0] * B[2]) +
               A[2] * (B[0] * C[1] - C[0] * B[1]))
    # Wait, that's not right. Let me use the standard 2D circumcenter formula in 3D.
    # For points A, B, C in 3D (coplanar), the circumcenter is:
    # Solve: |X - A|^2 = |X - B|^2 = |X - C|^2
    # 2 X . (B - A) = |B|^2 - |A|^2
    # 2 X . (C - A) = |C|^2 - |A|^2
    # Plus X in plane of A,B,C.
    # Use: X = A + t*(B-A) + s*(C-A), solve for t,s.
    # |A + t(B-A) + s(C-A) - A|^2 = |A + t(B-A) + s(C-A) - B|^2
    # |t(B-A) + s(C-A)|^2 = |t(B-A) + s(C-A) - (B-A)|^2
    # Let u = B-A, v = C-A
    # |t u + s v|^2 = |t u + s v - u|^2
    # t^2|u|^2 + 2ts u.v + s^2|v|^2 = t^2|u|^2 + 2ts u.v + s^2|v|^2 - 2t|u|^2 - 2s u.v + |u|^2
    # 0 = -2t|u|^2 - 2s u.v + |u|^2
    # t|u|^2 + s u.v = |u|^2/2  ... (1)
    # Similarly for C:
    # |t u + s v|^2 = |t u + s v - v|^2
    # 0 = -2t u.v - 2s|v|^2 + |v|^2
    # t u.v + s|v|^2 = |v|^2/2  ... (2)
    # Solve:
    uu = np.dot(u, u)
    vv = np.dot(v, v)
    uv = np.dot(u, v)
    det = uu * vv - uv * uv
    if abs(det) < 1e-14:
        return None
    t = (vv * (uu / 2.0) - uv * (vv / 2.0)) / det
    s = (uu * (vv / 2.0) - uv * (uu / 2.0)) / det
    # Wait, let me redo:
    # From (1): t*uu + s*uv = uu/2
    # From (2): t*uv + s*vv = vv/2
    # Cramer's rule:
    # t = ( (uu/2)*vv - (vv/2)*uv ) / det = (vv*(uu - uv)) / (2*det)
    # s = ( uu*(vv/2) - uv*(uu/2) ) / det = (uu*(vv - uv)) / (2*det)
    t = (vv * (uu - uv)) / (2.0 * det)
    s = (uu * (vv - uv)) / (2.0 * det)
    X = A + t * (B - A) + s * (C - A)
    norm = np.linalg.norm(X)
    if norm < 1e-12:
        return None
    return X / norm


def _soft_covering_radius_and_grad(points, grid_points, sigma=0.03):
    """
    Vectorized soft covering radius (log-sum-exp based) and its gradient.
    """
    dots = grid_points @ points.T
    dots = np.clip(dots, -1.0, 1.0)
    angles = np.arccos(dots)  # (M, N)

    # Soft-min: for each grid point, weighted average using softmax of -angles/sigma
    neg_scaled = -angles / sigma  # (M, N)
    max_neg = np.max(neg_scaled, axis=1, keepdims=True)
    exp_neg = np.exp(neg_scaled - max_neg)  # (M, N)
    weights = exp_neg / np.sum(exp_neg, axis=1, keepdims=True)  # (M, N)

    # Soft-min values: (M,)
    soft_min_vals = np.sum(weights * angles, axis=1)

    # Soft-max of soft-min values
    scaled_max = soft_min_vals / sigma
    max_scaled = np.max(scaled_max)
    exp_max = np.exp(scaled_max - max_scaled)
    max_weights = exp_max / np.sum(exp_max)  # (M,)
    soft_max_val = sigma * (max_scaled + np.log(np.sum(exp_max)))

    # Gradient
    denom = np.sqrt(np.maximum(1.0 - dots * dots, 1e-18))  # (M, N)
    coeff = max_weights[:, np.newaxis] * weights  # (M, N)
    alpha = coeff / denom  # (M, N)

    term1 = -(alpha.T @ grid_points)  # (N, 3)
    beta = alpha * dots  # (M, N)
    beta_sum = np.sum(beta, axis=0)  # (N,)
    term2 = beta_sum[:, np.newaxis] * points  # (N, 3)

    grad = term1 + term2  # (N, 3)

    return soft_max_val, grad


def _repulsive_force_step(pts, step_size):
    """One step of repulsive force optimization on the sphere."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    weights = 1.0 / dist_sq
    np.fill_diagonal(weights, 0.0)
    forces = np.sum(weights[:, :, np.newaxis] * diff, axis=1)
    pts = pts + step_size * forces
    norms = np.linalg.norm(pts, axis=1)
    norms = np.where(norms < 1e-12, 1.0, norms)
    pts = pts / norms[:, np.newaxis]
    return pts


def _lloyd_refinement(pts, grid_points, start_time, time_limit, max_iters=20):
    """
    Lloyd's algorithm (CVT) refinement with damping.
    Moves each point to the spherical centroid of its Voronoi cell (approximated by grid points).
    """
    pts = pts.copy()
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break

        # Assign each grid point to the nearest configuration point
        dots = grid_points @ pts.T
        nearest_idx = np.argmax(dots, axis=1)  # Max dot product = Min angle

        # Calculate spherical centroids
        new_pts = np.zeros_like(pts)
        for j in range(N):
            mask = (nearest_idx == j)
            count = np.sum(mask)
            if count > 0:
                centroid = np.sum(grid_points[mask], axis=0)
                norm = np.linalg.norm(centroid)
                if norm > 1e-12:
                    new_pts[j] = centroid / norm
                else:
                    new_pts[j] = pts[j]
            else:
                new_pts[j] = pts[j]

        # Damping to prevent oscillation
        alpha = 0.5
        pts = alpha * new_pts + (1 - alpha) * pts

        # Renormalize
        norms = np.linalg.norm(pts, axis=1)
        norms = np.where(norms < 1e-12, 1.0, norms)
        pts = pts / norms[:, np.newaxis]

        # Check convergence
        diff = np.linalg.norm(pts - new_pts, axis=1)
        if np.max(diff) < 1e-7:
            break

    return pts


def _exact_local_polish(points, start_time, time_limit):
    """
    High-precision polish: for the worst facet, move its vertices toward
    the facet's spherical circumcenter to reduce its circumradius.
    """
    pts = points.copy()
    for it in range(500):
        if time.time() - start_time > time_limit:
            break
        try:
            hull = ConvexHull(pts)
        except Exception:
            break

        # Find the worst facet
        worst_facet = None
        worst_radius = -1.0
        worst_normal = None
        worst_circum = None

        for simplex in hull.simplices:
            verts = pts[simplex]
            a, b, c = verts[0], verts[1], verts[2]
            normal = np.cross(b - a, c - a)
            norm_val = np.linalg.norm(normal)
            if norm_val < 1e-12:
                continue
            normal = normal / norm_val
            centroid = np.mean(verts, axis=0)
            if np.dot(normal, centroid) < 0:
                normal = -normal
            dots = np.clip(np.dot(verts, normal), -1.0, 1.0)
            angles = np.arccos(dots)
            facet_radius = np.max(angles)
            if facet_radius > worst_radius:
                worst_radius = facet_radius
                worst_facet = simplex
                worst_normal = normal
                # Compute spherical circumcenter
                circum = _spherical_circumcenter(a, b, c)
                if circum is not None:
                    worst_circum = circum
                else:
                    worst_circum = normal

        if worst_facet is None:
            break

        # Move the vertices of the worst facet toward the circumcenter
        move_amount = 0.002
        for idx in worst_facet:
            target = worst_circum
            # Move in tangent direction toward target
            tangent = target - np.dot(pts[idx], target) * pts[idx]
            tangent_norm = np.linalg.norm(tangent)
            if tangent_norm > 1e-12:
                pts[idx] = pts[idx] + move_amount * tangent / tangent_norm

        # Renormalize
        norms = np.linalg.norm(pts, axis=1)
        norms = np.where(norms < 1e-12, 1.0, norms)
        pts = pts / norms[:, np.newaxis]

    return pts


def construct():
    start_time = time.time()
    time_limit = TOTAL_TIME_LIMIT

    # 1. Initialization: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    # 2. Repulsive Force Optimization (quick phase)
    step_size = 0.05
    cooling = 0.998
    repulsive_iters = 0
    while time.time() - start_time < time_limit * 0.08 and repulsive_iters < 200:
        pts = _repulsive_force_step(pts, step_size)
        step_size *= cooling
        repulsive_iters += 1
        if step_size < 1e-6:
            break

    # 3. Generate evaluation grid
    N_eval = 20000
    i_eval = np.arange(N_eval) + 0.5
    z_eval = 1.0 - 2.0 * i_eval / N_eval
    phi_eval = np.pi * (1.0 + np.sqrt(5.0)) * i_eval
    r_eval = np.sqrt(1.0 - z_eval * z_eval)
    grid_points = np.column_stack([r_eval * np.cos(phi_eval), r_eval * np.sin(phi_eval), z_eval])

    # 4. Lloyd refinement
    if time.time() - start_time < time_limit * 0.25:
        pts = _lloyd_refinement(pts, grid_points, start_time, time_limit * 0.25, max_iters=30)

    # 5. Soft surrogate gradient descent with sigma annealing
    if time.time() - start_time < time_limit * 0.5:
        sigmas = [0.1, 0.05, 0.03, 0.02, 0.01]
        for sigma in sigmas:
            if time.time() - start_time > time_limit * 0.5:
                break

            def objective_and_grad(x_flat, sigma=sigma):
                p = x_flat.reshape(N, 3)
                norms = np.linalg.norm(p, axis=1, keepdims=True)
                norms = np.where(norms < 1e-12, 1.0, norms)
                p = p / norms
                val, grad = _soft_covering_radius_and_grad(p, grid_points, sigma=sigma)
                # Project gradient to tangent space
                grad_tangent = grad - np.sum(grad * p, axis=1, keepdims=True) * p
                return val, grad_tangent.flatten()

            x0 = pts.flatten()
            try:
                res = minimize(
                    objective_and_grad,
                    x0,
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-7, 'disp': False}
                )
                if res.x is not None:
                    pts = res.x.reshape(N, 3)
                    norms = np.linalg.norm(pts, axis=1, keepdims=True)
                    norms = np.where(norms < 1e-12, 1.0, norms)
                    pts = pts / norms
            except Exception:
                pass

    # 6. Final Lloyd pass
    if time.time() - start_time < time_limit * 0.75:
        pts = _lloyd_refinement(pts, grid_points, start_time, time_limit * 0.75, max_iters=20)

    # 7. Exact local polish
    if time.time() - start_time < time_limit - 5.0:
        pts = _exact_local_polish(pts, start_time, time_limit - 5.0)

    # Final normalization
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    pts = pts / norms

    return pts
