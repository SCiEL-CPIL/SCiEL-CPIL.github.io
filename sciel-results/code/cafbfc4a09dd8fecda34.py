"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 150

def _log_energy(points):
    """Compute log(1/d) energy = -sum log(d) for all pairs."""
    # Compute all pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # Set diagonal to 1 to avoid log(0)
    np.fill_diagonal(dists, 1.0)
    # Compute sum of log(1/d) = -sum log(d)
    # Use fsum for precision
    vals = -np.log(dists)
    np.fill_diagonal(vals, 0.0)
    # Sum upper triangle only
    iu = np.triu_indices(N, k=1)
    return math.fsum(vals[iu].tolist())

def _log_energy_grad(points):
    """Gradient of log energy w.r.t. points.
    E = sum_{i<j} log(1/|xi-xj|) = -sum_{i<j} log|xi-xj|
    dE/dxi = sum_{j!=i} (xi-xj)/|xi-xj|^2
    """
    grad = np.zeros_like(points)
    for i in range(N):
        diff = points[i] - points  # (N, 3)
        dists = np.linalg.norm(diff, axis=1)
        dists[i] = 1.0  # avoid div by zero
        # grad_i = sum_{j!=i} (xi-xj)/|xi-xj|^2
        for j in range(N):
            if j != i:
                grad[i] += diff[j] / (dists[j] ** 2)
    return grad

def _project_to_sphere(points):
    """Normalize each point to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return points / norms

def _fibonacci_sphere(n):
    """Generate n points on unit sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _geodesic_dome(n):
    """Generate approximately uniform points using geodesic dome (subdivided icosahedron)."""
    # Start with icosahedron
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)
    ]
    
    # Subdivide until we have enough points
    while len(verts) < n:
        new_faces = []
        mid_cache = {}
        
        def midpoint(v1, v2):
            key = (min(v1, v2), max(v1, v2))
            if key not in mid_cache:
                mid = (verts[v1] + verts[v2]) / 2.0
                mid = mid / np.linalg.norm(mid)
                mid_cache[key] = len(verts)
                verts = np.vstack([verts, mid.reshape(1, -1)])
            return mid_cache[key]
        
        for f in faces:
            a, b, c = f
            ab = midpoint(a, b)
            bc = midpoint(b, c)
            ca = midpoint(c, a)
            new_faces.extend([(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)])
        
        faces = new_faces
    
    # Return first n vertices
    return verts[:n]

def construct():
    deadline = time.time() + 145.0  # Safety margin before 150s limit
    
    # Try multiple initializations and optimization strategies
    best_points = _fibonacci_sphere(N)
    best_energy = _log_energy(best_points)
    
    # Initializations to try
    inits = [
        _fibonacci_sphere(N),
        _geodesic_dome(N),
        np.random.RandomState(42).randn(N, 3),
    ]
    inits[2] = inits[2] / np.linalg.norm(inits[2], axis=1, keepdims=True)
    
    for init in inits:
        if time.time() > deadline:
            break
        
        # Use L-BFGS-B with projection
        def objective_and_grad(x):
            pts = _project_to_sphere(x)
            return _log_energy(pts), _log_energy_grad(pts)
        
        try:
            res = minimize(
                lambda x: _log_energy(_project_to_sphere(x)),
                init,
                jac=lambda x: _log_energy_grad(_project_to_sphere(x)),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 10000}
            )
            pts = _project_to_sphere(res.x)
            e = _log_energy(pts)
            if e < best_energy:
                best_energy = e
                best_points = pts
        except Exception:
            pass
    
    # Final polish with tighter tolerance
    if time.time() < deadline - 5.0:
        try:
            res = minimize(
                lambda x: _log_energy(_project_to_sphere(x)),
                best_points,
                jac=lambda x: _log_energy_grad(_project_to_sphere(x)),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-18, 'gtol': 1e-14, 'maxfun': 50000}
            )
            pts = _project_to_sphere(res.x)
            e = _log_energy(pts)
            if e < best_energy:
                best_energy = e
                best_points = pts
        except Exception:
            pass
    
    return _project_to_sphere(best_points)
