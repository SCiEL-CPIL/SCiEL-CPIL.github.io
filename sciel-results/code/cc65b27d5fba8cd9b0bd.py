"""Spherical covering with 50 points via symplectic potential minimization."""
import numpy as np
from scipy.spatial import ConvexHull

N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return pts / norms

def _potential_gradient(pts):
    """Calculate the gradient of the potential energy sum 1/d^4."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, 1e-10)
    
    # Gradient of 1/d^4 is -4*(p_i-p_j)/d^6
    # Force is -Gradient, so Force is 4*(p_i-p_j)/d^6
    # We return the gradient (which points towards other particles, increasing energy)
    factor = 4.0 / (dist_sq ** 3)
    factor = np.where(np.eye(N, dtype=bool), 0.0, factor)
    
    grad = np.sum(factor[:, :, np.newaxis] * diff, axis=1)
    return grad

def _compute_covering_radius(pts):
    """Compute the covering radius in degrees for points on the unit sphere."""
    try:
        hull = ConvexHull(pts)
        max_radius = 0.0
        for simplex in hull.simplices:
            verts = pts[simplex]
            a, b, c = verts[0], verts[1], verts[2]
            edge1 = b - a
            edge2 = c - a
            normal = np.cross(edge1, edge2)
            norm = np.linalg.norm(normal)
            if norm < 1e-12:
                continue
            normal = normal / norm
            centroid = (a + b + c) / 3.0
            if np.dot(centroid, normal) < 0:
                normal = -normal
            d_a = np.arccos(np.clip(np.dot(normal, a), -1, 1))
            d_b = np.arccos(np.clip(np.dot(normal, b), -1, 1))
            d_c = np.arccos(np.clip(np.dot(normal, c), -1, 1))
            facet_radius = max(d_a, d_b, d_c)
            if facet_radius > max_radius:
                max_radius = facet_radius
        return np.degrees(max_radius)
    except Exception:
        # Fallback: dense grid
        n_theta = 100
        n_phi = 200
        thetas = np.linspace(0, np.pi, n_theta)
        phis = np.linspace(0, 2 * np.pi, n_phi)
        TH, PH = np.meshgrid(thetas, phis, indexing='ij')
        Y = np.zeros((n_theta, n_phi, 3))
        Y[:, :, 0] = np.sin(TH) * np.cos(PH)
        Y[:, :, 1] = np.sin(TH) * np.sin(PH)
        Y[:, :, 2] = np.cos(TH)
        Y = Y.reshape(-1, 3)
        dots = Y @ pts.T
        min_dots = np.max(dots, axis=1)
        max_angular = np.arccos(np.clip(min_dots, -1, 1))
        return np.degrees(np.max(max_angular))

def construct():
    """
    Minimize repulsive potential on the unit sphere using damped velocity Verlet
    and Riemannian gradient descent polish.
    """
    # Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts = _normalize(pts)
    
    # Phase 1: Damped Velocity Verlet
    dt = 0.01
    n_steps = 2000
    gamma = 0.5
    
    vel = np.zeros((N, 3))
    forces = -_potential_gradient(pts)
    
    for step in range(n_steps):
        vel = (vel + 0.5 * dt * forces) * (1.0 - gamma * dt)
        pts = pts + dt * vel
        pts = _normalize(pts)
        
        forces = -_potential_gradient(pts)
        vel = (vel + 0.5 * dt * forces) * (1.0 - gamma * dt)
        
    # Phase 2: Riemannian Gradient Descent Polish
    step_size = 0.01
    for k in range(500):
        grad = _potential_gradient(pts)
        for i in range(N):
            g = grad[i]
            p = pts[i]
            g_tan = g - np.dot(g, p) * p
            pts[i] = pts[i] - step_size * g_tan
            pts[i] = pts[i] / np.linalg.norm(pts[i])
        step_size *= 0.99
        
    pts = _normalize(pts)
    return pts
# EVOLVE-BLOCK-END
