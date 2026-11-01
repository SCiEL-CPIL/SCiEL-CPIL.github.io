"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
import math

def _spherical_grid(n_lat=300, n_lon=600):
    """Generate a dense grid of points on the unit sphere."""
    # Use lat/lon grid
    lats = np.linspace(-np.pi/2 + 1e-10, np.pi/2 - 1e-10, n_lat)
    lons = np.linspace(0, 2*np.pi, n_lon, endpoint=False)
    LA, LO = np.meshgrid(lats, lons, indexing='ij')
    x = np.cos(LA) * np.cos(LO)
    y = np.cos(LA) * np.sin(LO)
    z = np.sin(LA)
    pts = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    # Normalize to be safe
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    return pts

def _covering_radius(pts):
    """Compute covering radius in radians using dense grid approximation."""
    # Use KDTree for fast nearest neighbor
    tree = cKDTree(pts)
    # Sample points on sphere
    # For efficiency, use a moderate grid
    grid = _spherical_grid(n_lat=200, n_lon=400)
    dists, _ = tree.query(grid)
    max_dist = np.max(dists)
    # Convert chord distance to angular distance
    # chord = 2*sin(theta/2), so theta = 2*arcsin(chord/2)
    # But cKDTree returns Euclidean distance
    # For unit vectors, Euclidean dist d corresponds to angle theta where d = 2*sin(theta/2)
    # So theta = 2*arcsin(d/2)
    # But we want to be careful: if d > 2, clamp
    max_dist = np.clip(max_dist, 0, 2)
    theta = 2 * np.arcsin(max_dist / 2.0)
    return theta

def _objective_and_grad(x_flat):
    """Objective function and gradient for L-BFGS-B."""
    pts = x_flat.reshape(-1, 3)
    # Normalize
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    pts_norm = pts / norms
    
    # Compute covering radius
    radius = _covering_radius(pts_norm)
    
    # Gradient: approximate by finite differences? No, let's use a simpler approach.
    # Actually, for L-BFGS-B we can just use the function value.
    # But L-BFGS-B can work without gradient (it will use finite differences internally).
    # Let's just return the value.
    return radius

def construct():
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    x0_flat = x0.ravel()
    
    # Optimize using L-BFGS-B
    # We'll minimize the covering radius
    # Since L-BFGS-B doesn't have a built-in gradient for this, it will use numerical gradients.
    # This might be slow, but let's try with a limited number of iterations.
    
    def obj(x):
        pts = x.reshape(-1, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        pts = pts / norms
        return _covering_radius(pts)
    
    result = minimize(obj, x0_flat, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
    
    if result.success or result.fun < _covering_radius(x0):
        x_opt = result.x.reshape(-1, 3)
    else:
        x_opt = x0
    
    # Normalize final result
    norms = np.linalg.norm(x_opt, axis=1, keepdims=True)
    x_opt = x_opt / norms
    
    return x_opt
# EVOLVE-BLOCK-END
