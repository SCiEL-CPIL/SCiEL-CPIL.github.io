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


def _fibonacci_sphere(n, seed=0):
    """Generate n points on sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return pts


def _compute_forces(pts, beta=5.0):
    """
    Compute forces for potential U = sum_{i<j} exp(-beta * d_ij^2)
    Force on i from j: 2*beta*(p_i - p_j)*exp(-beta*|p_i-p_j|^2)
    Plus radial constraint force.
    """
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 0.0)
    
    weights = 2.0 * beta * np.exp(-beta * dist_sq)
    np.fill_diagonal(weights, 0.0)
    
    forces = np.sum(weights[:, :, np.newaxis] * diff, axis=1)
    
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms_safe = np.where(norms < 1e-10, 1.0, norms)
    radial_forces = -50.0 * (norms - 1.0) * pts / norms_safe
    forces += radial_forces
    
    return forces


def _anneal(pts, n_steps=10000, dt_start=0.005, dt_end=0.001, 
            beta_start=2.0, beta_end=15.0, damping=0.98):
    """Run annealing schedule with velocity Verlet."""
    vel = np.zeros_like(pts)
    
    for step in range(n_steps):
        t = step / max(n_steps - 1, 1)
        dt = dt_start * (dt_end / dt_start) ** t
        beta = beta_start + (beta_end - beta_start) * t
        
        F = _compute_forces(pts, beta=beta)
        vel_half = vel + 0.5 * dt * F
        pts_new = _normalize(pts + dt * vel_half)
        F_new = _compute_forces(pts_new, beta=beta)
        vel = (vel_half + 0.5 * dt * F_new) * damping
        pts = pts_new
    
    return _normalize(pts)


def construct():
    """
    Use symplectic velocity Verlet integration to minimize a repulsive potential
    on the unit sphere, aiming to find a good covering configuration.
    """
    best_pts = None
    best_radius = np.inf
    
    for seed in range(3):
        rng = np.random.RandomState(seed)
        pts = _fibonacci_sphere(N, seed=seed)
        pts += rng.randn(N, 3) * 0.01
        pts = _normalize(pts)
        
        # Phase 1: Coarse
        pts = _anneal(pts, n_steps=8000, dt_start=0.005, dt_end=0.002,
                      beta_start=2.0, beta_end=8.0, damping=0.97)
        
        # Phase 2: Fine
        pts = _anneal(pts, n_steps=8000, dt_start=0.002, dt_end=0.0005,
                      beta_start=8.0, beta_end=20.0, damping=0.99)
        
        pts = _normalize(pts)
        
        try:
            radius = _compute_covering_radius(pts)
        except Exception:
            radius = np.inf
        
        if radius < best_radius:
            best_radius = radius
            best_pts = pts.copy()
    
    return best_pts
# EVOLVE-BLOCK-END
