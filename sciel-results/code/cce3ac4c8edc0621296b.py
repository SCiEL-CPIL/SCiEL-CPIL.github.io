"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150

# EVOLVE-BLOCK-START

def _log_energy(points):
    """Compute logarithmic energy sum_{i<j} log(1/|xi-xj|)."""
    n = points.shape[0]
    # Compute all pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    # Set diagonal to 1 to avoid log(0)
    np.fill_diagonal(dists, 1.0)
    # Sum log(1/d) = -sum log(d) for i<j
    # Only take upper triangle
    iu = np.triu_indices(n, k=1)
    return -np.sum(np.log(dists[iu]))

def _log_energy_grad(points):
    """Compute gradient of logarithmic energy w.r.t. points."""
    n = points.shape[0]
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, 1.0)
    
    # Gradient of -log(|xi-xj|) w.r.t. xi is (xi-xj)/|xi-xj|^2
    # For each point i, grad_i = sum_{j != i} (xi - xj) / |xi - xj|^2
    # Note: the energy is sum_{i<j} -log(|xi-xj|)
    # d/dxi sum_{j} -log(|xi-xj|) = sum_{j != i} (xi - xj)/|xi-xj|^2
    
    # dists[i,j] = |xi - xj|
    # We need (xi-xj)/|xi-xj|^2 for each pair
    # diff[i,j] = xi - xj
    # inv_dists_sq = 1 / dists^2
    inv_dists_sq = 1.0 / (dists * dists)
    np.fill_diagonal(inv_dists_sq, 0.0)
    
    # grad[i] = sum_j diff[i,j] * inv_dists_sq[i,j]
    grad = np.sum(diff * inv_dists_sq[:, :, np.newaxis], axis=1)
    return grad

def _project_to_sphere(points):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms

def _make_geodesic_dome(N_target):
    """Create a geodesic dome (subdivided icosahedron) with approximately N_target vertices."""
    # Start with icosahedron
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ]
    
    # Subdivide until we have close to N_target vertices
    # An icosahedron has 12 vertices. Each subdivision multiplies faces by 4.
    # Vertices after subdivision s: 10*4^s + 2
    # s=0: 12, s=1: 42, s=2: 162, s=3: 642
    # We want ~150, so s=2 gives 162 which is close.
    
    def subdivide(faces, verts):
        new_faces = []
        mid_cache = {}
        
        def midpoint(v1, v2):
            key = (min(v1, v2), max(v1, v2))
            if key in mid_cache:
                return mid_cache[key]
            mid = (verts[v1] + verts[v2]) / 2.0
            mid = mid / np.linalg.norm(mid)
            idx = len(verts)
            verts.append(mid)
            mid_cache[key] = idx
            return idx
        
        for face in faces:
            a, b, c = face
            ab = midpoint(a, b)
            bc = midpoint(b, c)
            ca = midpoint(c, a)
            new_faces.append([a, ab, ca])
            new_faces.append([b, bc, ab])
            new_faces.append([c, ca, bc])
            new_faces.append([ab, bc, ca])
        
        return new_faces, np.array(verts)
    
    verts_list = [list(v) for v in verts]
    faces_list = [list(f) for f in faces]
    
    # Subdivide twice to get 162 vertices
    for _ in range(2):
        faces_list, verts_arr = subdivide(faces_list, verts_list)
        verts_list = [list(v) for v in verts_arr]
    
    # Now we have 162 vertices. Remove 12 to get 150.
    # Remove vertices that are most "redundant" - those closest to their neighbors
    # Actually, let's just take the first 150 for simplicity, or better, remove the 12 with smallest min-distance to others
    
    verts_arr = np.array(verts_list)
    n = len(verts_arr)
    
    # Compute min distance from each vertex to any other
    diff = verts_arr[:, np.newaxis, :] - verts_arr[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dists = np.min(dists, axis=1)
    
    # Remove the 12 vertices with smallest min-distance (most redundant)
    remove_idx = np.argsort(min_dists)[:n - N_target]
    keep_idx = np.setdiff1d(np.arange(n), remove_idx)
    
    return verts_arr[keep_idx]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s deadline, 10s safety margin
    
    # Start with geodesic dome
    points = _make_geodesic_dome(N)
    
    # Optimize using L-BFGS-B with projection
    best_energy = _log_energy(points)
    best_points = points.copy()
    
    def objective(x):
        pts = x.reshape(N, 3)
        pts = _project_to_sphere(pts)
        return _log_energy(pts)
    
    def gradient(x):
        pts = x.reshape(N, 3)
        pts = _project_to_sphere(pts)
        grad = _log_energy_grad(pts)
        # Project gradient to tangent space
        # For a point x on sphere, tangent space is perpendicular to x
        # grad_tangent = grad - (grad . x) * x
        grad_tangent = grad - np.sum(grad * pts, axis=1, keepdims=True) * pts
        return grad_tangent.flatten()
    
    # Use L-BFGS-B
    try:
        result = minimize(
            objective,
            points.flatten(),
            jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'disp': False}
        )
        
        if time.time() < deadline:
            new_points = _project_to_sphere(result.x.reshape(N, 3))
            new_energy = _log_energy(new_points)
            if new_energy < best_energy:
                best_energy = new_energy
                best_points = new_points
    except Exception:
        pass
    
    # If time remains, do a second round of optimization from multiple random perturbations
    if time.time() < deadline - 20.0:
        for trial in range(5):
            if time.time() >= deadline - 10.0:
                break
            
            # Perturb the best points slightly
            perturbed = best_points + 0.01 * np.random.randn(N, 3)
            perturbed = _project_to_sphere(perturbed)
            
            try:
                result = minimize(
                    objective,
                    perturbed.flatten(),
                    jac=gradient,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'disp': False}
                )
                
                new_points = _project_to_sphere(result.x.reshape(N, 3))
                new_energy = _log_energy(new_points)
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_points = new_points
            except Exception:
                pass
    
    # Final polish with tighter tolerance
    if time.time() < deadline - 5.0:
        try:
            result = minimize(
                objective,
                best_points.flatten(),
                jac=gradient,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12, 'disp': False}
            )
            
            new_points = _project_to_sphere(result.x.reshape(N, 3))
            new_energy = _log_energy(new_points)
            if new_energy < best_energy:
                best_energy = new_energy
                best_points = new_points
        except Exception:
            pass
    
    # Ensure all points are on the unit sphere and valid
    best_points = _project_to_sphere(best_points)
    
    # Check for near-duplicate points
    diff = best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    if np.min(np.fill_diagonal(dists, np.inf)) < 1e-9:
        # Fallback to geodesic dome
        best_points = _make_geodesic_dome(N)
        best_points = _project_to_sphere(best_points)
    
    return best_points

# EVOLVE-BLOCK-END
