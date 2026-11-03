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


def _compute_forces(pts, power):
    """Vectorized force calculation for repulsive potential."""
    # pts shape (N, 3)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0) # Avoid division by zero
    dist = np.sqrt(dist_sq)
    
    # Force magnitude: 1/dist^(power+1)
    # We want force vector F_ij = (1/dist^(power+1)) * (diff / dist) = diff / dist^(power+2)
    # Wait, potential V = 1/dist^power. Force = -grad V = power / dist^(power+1) * r_hat.
    # So magnitude is proportional to 1/dist^(power+1).
    force_mag = 1.0 / (dist ** (power + 1))
    np.fill_diagonal(force_mag, 0.0)
    
    forces = np.sum(force_mag[:, :, np.newaxis] * diff, axis=1)
    return forces


def construct():
    """
    Use symplectic velocity Verlet integration to minimize a repulsive potential
    on the unit sphere, aiming to find a good covering configuration.
    """
    # Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Symplectic integrator parameters
    # Using Coulomb-like potential (power=1) for better uniform distribution
    power = 1 
    
    # Multi-stage integration with cooling
    n_steps = 20000
    vel = np.zeros((N, 3))
    
    # Schedule: dt decreases, gamma increases
    # Start with large dt and low damping, end with small dt and high damping
    dt_start = 0.01
    dt_end = 0.001
    gamma_start = 0.01
    gamma_end = 0.1
    
    for step in range(n_steps):
        progress = step / n_steps
        dt = dt_start * (1.0 - progress) + dt_end * progress
        gamma = gamma_start * (1.0 - progress) + gamma_end * progress
        
        # Compute forces
        forces = _compute_forces(pts, power)
        
        # Add centripetal force to keep points on sphere
        r_i = np.linalg.norm(pts, axis=1, keepdims=True)
        r_i_safe = np.where(r_i > 1e-10, r_i, 1.0)
        radial_force = -2.0 * (r_i - 1.0) * (pts / r_i_safe)
        forces += radial_force
        
        # Velocity Verlet step
        vel_half = vel + 0.5 * dt * forces
        
        pts_new = pts + dt * vel_half
        
        # Normalize to unit sphere
        pts_new = _normalize(pts_new)
        
        # Compute forces at new position
        forces_new = _compute_forces(pts_new, power)
        
        r_new = np.linalg.norm(pts_new, axis=1, keepdims=True)
        r_new_safe = np.where(r_new > 1e-10, r_new, 1.0)
        radial_force_new = -2.0 * (r_new - 1.0) * (pts_new / r_new_safe)
        forces_new += radial_force_new
        
        vel_new = vel_half + 0.5 * dt * forces_new
        
        # Apply damping
        vel = vel_new * (1.0 - gamma * dt)
        pts = pts_new
    
    # Final normalization
    pts = _normalize(pts)
    
    return pts
# EVOLVE-BLOCK-END
