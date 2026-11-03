"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import math
from scipy.optimize import minimize


def _project_to_sphere(points):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return points / norms


def _log_energy(points):
    """Compute logarithmic energy with fsum for accuracy."""
    n = len(points)
    # Compute all pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # Use fsum for the upper triangle
    upper_idx = np.triu_indices(n, k=1)
    vals = np.log(1.0 / dists[upper_idx])
    return math.fsum(vals)


def _compute_gradients(points):
    """Compute the gradient of log energy for all points, vectorized."""
    n = len(points)
    # Compute pairwise differences and distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]  # (n, n, 3)
    d2 = np.sum(diff ** 2, axis=2)  # (n, n)
    
    # Set diagonal to 1 to avoid division by zero
    np.fill_diagonal(d2, 1.0)
    
    # Gradient: grad_i = -sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    scale = 1.0 / d2  # (n, n)
    scale[range(n), range(n)] = 0.0  # Zero out diagonal
    
    grad = -np.sum(diff * scale[:, :, np.newaxis], axis=1)  # (n, 3)
    
    # Project onto tangent space
    tangent_grad = np.zeros_like(grad)
    for i in range(n):
        p = points[i]
        g = grad[i]
        g_tangent = g - np.dot(g, p) * p
        tangent_grad[i] = g_tangent
    
    return tangent_grad


def _tangent_descent(points, max_iter=2000, tol=1e-12):
    """
    Gradient descent on the sphere's tangent space with adaptive step size.
    """
    n = len(points)
    points = points.copy()
    current_energy = _log_energy(points)
    
    for iteration in range(max_iter):
        # Compute gradient in tangent space
        tangent_grad = _compute_gradients(points)
        
        # Adaptive step size with line search
        step = 0.01
        best_energy = current_energy
        best_points = points
        
        for trial in range(20):
            trial_points = points - step * tangent_grad
            trial_points = _project_to_sphere(trial_points)
            
            # Check minimum distance
            diff_t = trial_points[:, np.newaxis, :] - trial_points[np.newaxis, :, :]
            dists_t = np.linalg.norm(diff_t, axis=2)
            np.fill_diagonal(dists_t, np.inf)
            if np.min(dists_t) < 1e-9:
                step *= 0.5
                continue
            
            new_energy = _log_energy(trial_points)
            if new_energy < best_energy:
                best_energy = new_energy
                best_points = trial_points
                step *= 1.5  # Increase step if we made progress
            else:
                step *= 0.5
        
        if best_energy < current_energy:
            current_energy = best_energy
            points = best_points
        else:
            break
        
        # Check convergence
        if iteration > 0 and iteration % 50 == 0:
            if abs(current_energy - best_energy) < tol:
                break
    
    return points, current_energy


def construct():
    """
    Construct 150 points on the unit sphere using a geodesic dome (subdivided icosahedron)
    as initial positions, then refine with tangent-space optimization.
    """
    # Step 1: Create a geodesic dome with enough vertices
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    ico_verts = np.array([
        [-1,  phi, 0], [ 1,  phi, 0], [-1, -phi, 0], [ 1, -phi, 0],
        [0, -1,  phi], [0,  1,  phi], [0, -1, -phi], [0,  1, -phi],
        [ phi, 0, -1], [ phi, 0,  1], [-phi, 0, -1], [-phi, 0,  1]
    ], dtype=float)
    ico_verts = ico_verts / np.linalg.norm(ico_verts, axis=1, keepdims=True)
    
    # Icosahedron faces (triangles)
    ico_faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ]
    
    def subdivide(vertices, faces):
        """Subdivide a triangular mesh by adding midpoints."""
        new_vertices = list(vertices)
        new_faces = []
        edge_cache = {}
        
        def get_midpoint(i, j):
            key = (min(i, j), max(i, j))
            if key in edge_cache:
                return edge_cache[key]
            mid = (vertices[i] + vertices[j]) / 2.0
            mid = mid / np.linalg.norm(mid)
            new_vertices.append(mid)
            idx = len(new_vertices) - 1
            edge_cache[key] = idx
            return idx
        
        for face in faces:
            a, b, c = face
            ab = get_midpoint(a, b)
            bc = get_midpoint(b, c)
            ca = get_midpoint(c, a)
            new_faces.append([a, ab, ca])
            new_faces.append([b, bc, ab])
            new_faces.append([c, ca, bc])
            new_faces.append([ab, bc, ca])
        
        return np.array(new_vertices), new_faces
    
    # Subdivide to get more than N vertices
    verts, faces = subdivide(ico_verts, ico_faces)  # 42 vertices
    verts, faces = subdivide(verts, faces)  # 162 vertices
    
    # Select N vertices from the geodesic dome
    if len(verts) > N:
        # Use greedy selection to maintain uniform spacing
        selected = [0]  # Start with first vertex
        remaining = list(range(1, len(verts)))
        
        while len(selected) < N:
            # For each remaining vertex, compute min distance to selected
            best_idx = -1
            best_min_dist = -1
            
            for idx in remaining:
                dists_to_selected = np.linalg.norm(verts[idx] - verts[selected], axis=1)
                min_dist = np.min(dists_to_selected)
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_idx = idx
            
            selected.append(best_idx)
            remaining.remove(best_idx)
        
        verts = verts[selected]
    elif len(verts) < N:
        # Add Fibonacci sphere points to reach N
        n_extra = N - len(verts)
        i = np.arange(n_extra) + 0.5
        z = 1.0 - 2.0 * i / n_extra
        phi_angle = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        extra = np.column_stack([r * np.cos(phi_angle), r * np.sin(phi_angle), z])
        verts = np.vstack([verts, extra])
    
    # Step 2: Refine with multiple restarts
    best_points = verts.copy()
    best_energy = _log_energy(best_points)
    
    # Try multiple restarts with different perturbations
    for restart in range(12):
        current = best_points.copy()
        if restart > 0:
            rng = np.random.RandomState(12345 + restart * 137)
            scale = 0.005 * (1 + restart * 0.3)
            current = current + rng.normal(0, scale, size=current.shape)
            current = _project_to_sphere(current)
        
        refined, energy = _tangent_descent(current, max_iter=1500)
        
        if energy < best_energy:
            best_energy = energy
            best_points = refined
    
    # Final refinement with smaller steps
    final_points, final_energy = _tangent_descent(best_points, max_iter=3000, tol=1e-14)
    if final_energy < best_energy:
        best_points = final_points
        best_energy = final_energy
    
    # Final projection to ensure unit sphere
    best_points = _project_to_sphere(best_points)
    
    return best_points
# EVOLVE-BLOCK-END
