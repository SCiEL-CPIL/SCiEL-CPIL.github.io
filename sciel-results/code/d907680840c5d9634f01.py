"""Optimize 130 points on the unit sphere to minimize the covering radius."""
import numpy as np
import time
from scipy.spatial import ConvexHull, Delaunay
from scipy.optimize import minimize

N = 130
TIME_LIMIT = 150.0
SAFETY_MARGIN = 12.0


def _normalize(P):
    P = np.asarray(P, dtype=float)
    norms = np.linalg.norm(P, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return P / norms


def _covering_radius_exact(P):
    """Exact covering radius in radians via Delaunay triangulation of the points."""
    try:
        # Delaunay in 3D gives tetrahedra; the boundary faces are the convex hull facets
        tri = Delaunay(P)
        # Boundary facets: triangles with only one adjacent tetrahedron
        # tri.simplices are tetrahedra (4 points each)
        # tri.neighbors: for each tetrahedron, 4 neighbors (-1 for boundary)
        # Collect boundary triangles
        boundary_tris = []
        seen = set()
        for i, tet in enumerate(tri.simplices):
            nbrs = tri.neighbors[i]
            for j in range(4):
                if nbrs[j] == -1:
                    # The face opposite vertex j is a boundary face
                    face = [tet[k] for k in range(4) if k != j]
                    key = tuple(sorted(face))
                    if key not in seen:
                        seen.add(key)
                        boundary_tris.append(face)
        if not boundary_tris:
            return np.pi
        boundary_tris = np.array(boundary_tris, dtype=int)
        # For each boundary triangle, compute circumradius
        # Circumradius of triangle with vertices a, b, c:
        # R = |a||b||c| / (4 * area) where area = 0.5*|cross(b-a, c-a)|
        # Since points are on unit sphere, |a|=|b|=|c|=1
        # R = 1 / (4 * area)
        a = P[boundary_tris[:, 0]]
        b = P[boundary_tris[:, 1]]
        c = P[boundary_tris[:, 2]]
        cross = np.cross(b - a, c - a)
        area = 0.5 * np.linalg.norm(cross, axis=1)
        # Avoid division by zero
        area[area < 1e-15] = 1e-15
        R = 1.0 / (4.0 * area)
        # The circumradius in angular terms: for a triangle on the unit sphere,
        # the covering radius contribution is the angular circumradius
        # For a spherical triangle, the circumradius theta satisfies sin(theta) = R (for the planar circumradius)
        # Actually, the angular distance from the center of the empty cap to the triangle vertices
        # The cap center is the circumcenter of the triangle projected to the sphere
        # The angular circumradius is arcsin(R) where R is the planar circumradius... 
        # More precisely, for points on the unit sphere, the circumcenter of the triangle
        # (in 3D) projected to the sphere gives the cap center, and the angular radius
        # is the angle between the cap center and any vertex.
        # Planar circumradius R (distance from circumcenter to vertex in 3D)
        # Since |vertex| = 1 and circumcenter is at distance d from origin,
        # the angular radius theta satisfies: cos(theta) = d / 1 = d
        # and R^2 = d^2 + 1 - 2*d*1*cos(theta)... 
        # Actually simpler: the circumcenter C of the triangle in 3D space.
        # |C| = d. The angle between C and a vertex v: cos(theta) = C·v / (|C||v|) = C·v / d
        # The circumradius in angular terms is theta = arccos(C·v / d)
        # Alternatively, for a spherical triangle, the circumradius can be computed as:
        # If the planar circumradius is R (distance from circumcenter to vertex),
        # and the circumcenter is at distance d from origin, then
        # R^2 = d^2 + 1 - 2*d*cos(theta) where theta is the angular radius
        # So cos(theta) = (d^2 + 1 - R^2) / (2*d)
        # But we need d. For a triangle on the unit sphere, the circumcenter C satisfies
        # |C - a| = |C - b| = |C - c| = R
        # C = alpha*a + beta*b + gamma*c (in the plane of the triangle)
        # Actually, the circumcenter of a triangle on the unit sphere:
        # The normal to the plane of the triangle is n = cross(b-a, c-a)/|cross(b-a, c-a)|
        # The circumcenter C = t*n for some t (since it's equidistant from a, b, c)
        # |C - a|^2 = |C|^2 - 2*C·a + |a|^2 = t^2 - 2*t*(n·a) + 1 = R^2
        # For all three vertices, n·a = n·b = n·c (since they're coplanar)
        # Let h = n·a (the height of the plane from origin)
        # t^2 - 2*t*h + 1 = R^2
        # The circumradius R is the distance from C to a vertex.
        # The angular radius theta: cos(theta) = C·a / (|C|*|a|) = t*h / |t| = h (if t>0)
        # So theta = arccos(h) where h = n·a (the dot product of the unit normal with a vertex)
        # Wait, this is the distance from origin to the plane, which is the same for all three vertices.
        # The angular circumradius is arccos(h) where h = n·a.
        # But we need to make sure the cap is empty (the circumcenter is on the correct side).
        # The covering radius is the maximum over all boundary facets of the angular circumradius.
        normals = cross / np.linalg.norm(cross, axis=1, keepdims=True)
        h = np.sum(normals * a, axis=1)
        # The angular circumradius
        theta = np.arccos(np.clip(h, -1.0, 1.0))
        return np.max(theta)
    except Exception:
        return np.pi


def _covering_radius_smooth(P, beta):
    """Smooth covering radius using log-sum-exp over facet circumradii."""
    try:
        tri = Delaunay(P)
        boundary_tris = []
        seen = set()
        for i, tet in enumerate(tri.simplices):
            nbrs = tri.neighbors[i]
            for j in range(4):
                if nbrs[j] == -1:
                    face = [tet[k] for k in range(4) if k != j]
                    key = tuple(sorted(face))
                    if key not in seen:
                        seen.add(key)
                        boundary_tris.append(face)
        if not boundary_tris:
            return np.pi, None
        boundary_tris = np.array(boundary_tris, dtype=int)
        a = P[boundary_tris[:, 0]]
        b = P[boundary_tris[:, 1]]
        c = P[boundary_tris[:, 2]]
        cross = np.cross(b - a, c - a)
        norms = np.linalg.norm(cross, axis=1, keepdims=True)
        norms[norms < 1e-15] = 1e-15
        normals = cross / norms
        h = np.sum(normals * a, axis=1)
        # theta = arccos(h)
        # Smooth max: (1/beta) * log(sum(exp(beta * theta)))
        # For numerical stability, use log-sum-exp
        theta = np.arccos(np.clip(h, -1.0, 1.0))
        max_theta = np.max(theta)
        if max_theta < 1e-10:
            return 0.0, theta
        # Log-sum-exp
        shifted = beta * (theta - max_theta)
        log_sum = max_theta * beta + np.log(np.sum(np.exp(shifted)))
        smooth_val = log_sum / beta
        return smooth_val, theta
    except Exception:
        return np.pi, None


def _objective_and_grad(P_flat, beta):
    """Objective and gradient via finite differences (expensive but robust)."""
    P = P_flat.reshape(-1, 3)
    P = _normalize(P)
    val, theta = _covering_radius_smooth(P, beta)
    if theta is None:
        return val, np.zeros_like(P_flat)
    # Compute gradient via finite differences on the smooth objective
    # This is expensive, so use a small step
    eps = 1e-6
    grad = np.zeros_like(P_flat)
    # Only compute gradient for a subset of points to save time
    # Actually, let's use the analytic gradient if possible
    # The smooth objective is (1/beta) * log(sum_k exp(beta * theta_k))
    # where theta_k = arccos(h_k) and h_k = n_k · a_k
    # d(theta_k)/d(P) is complex. Let's use finite differences on the full objective.
    # But that's O(N) evaluations. Instead, let's just return the value and use
    # a derivative-free method or a simpler gradient.
    # For now, return zero gradient and use a different optimization approach.
    return val, grad


def _repulsive_force_init(n_iter=200, dt=0.01):
    """Initialize points using repulsive forces to spread them on the sphere."""
    # Start with a random or Fibonacci initialization
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    P = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    P = _normalize(P)
    
    for it in range(n_iter):
        # Compute repulsive forces
        # F_i = sum_{j != i} (P_i - P_j) / |P_i - P_j|^2
        diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist_sq, 1.0)
        # Force: (P_i - P_j) / |P_i - P_j|^2
        force = np.zeros_like(P)
        for i in range(N):
            for j in range(N):
                if i != j:
                    force[i] += (P[i] - P[j]) / dist_sq[i, j]
        P += dt * force
        P = _normalize(P)
    return P


def _local_refine(P, time_budget):
    """Refine points using gradient descent on the smooth covering radius."""
    start_time = time.time()
    beta = 50.0  # Temperature for log-sum-exp
    
    P = _normalize(P)
    best_P = P.copy()
    best_val = _covering_radius_exact(P)
    
    # Use a simple iterative improvement: perturb points and accept if covering radius decreases
    # This is more robust than gradient descent for this non-smooth problem
    step_size = 0.05
    for it in range(5000):
        if time.time() - start_time > time_budget:
            break
        # Try to improve by moving each point slightly
        improved = False
        for i in range(N):
            if time.time() - start_time > time_budget:
                break
            # Try moving point i in a random direction
            for attempt in range(10):
                direction = np.random.randn(3)
                direction /= np.linalg.norm(direction)
                # Try a small step
                new_P = P.copy()
                new_P[i] = P[i] + step_size * direction
                new_P[i] /= np.linalg.norm(new_P[i])
                
                # Evaluate covering radius
                val = _covering_radius_exact(new_P)
                if val < best_val - 1e-10:
                    P = new_P
                    best_P = P.copy()
                    best_val = val
                    improved = True
                    break
        if not improved:
            # Decrease step size
            step_size *= 0.9
            if step_size < 1e-5:
                break
    
    return best_P, best_val


def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT - SAFETY_MARGIN
    
    # Step 1: Repulsive force initialization
    P = _repulsive_force_init(n_iter=100, dt=0.005)
    
    # Step 2: Local refinement
    time_spent = time.time() - start_time
    remaining = deadline - time_spent
    if remaining > 5:
        P, best_val = _local_refine(P, remaining - 2)
    
    # Ensure validity: normalize and check
    P = _normalize(P)
    
    return P
