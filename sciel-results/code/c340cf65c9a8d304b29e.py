"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import math


def _project_to_sphere(points):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return points / norms


def _log_energy(points):
    """Compute the logarithmic energy sum_{i<j} log(1/|x_i - x_j|)."""
    n = len(points)
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(points[i] - points[j])
            if d < 1e-15:
                return float('inf')
            total += math.log(1.0 / d)
    return total


def _log_energy_fast(points):
    """Faster computation using vectorized distance matrix."""
    n = len(points)
    # Compute all pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # Use fsum for the upper triangle
    upper_idx = np.triu_indices(n, k=1)
    vals = np.log(1.0 / dists[upper_idx])
    return math.fsum(vals)


def _tangent_basis(point):
    """Compute an orthonormal basis for the tangent space at a point on the sphere."""
    p = point / np.linalg.norm(point)
    # Choose a reference vector not parallel to p
    if abs(p[0]) < 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    else:
        ref = np.array([0.0, 1.0, 0.0])
    t1 = ref - np.dot(ref, p) * p
    t1 = t1 / np.linalg.norm(t1)
    t2 = np.cross(p, t1)
    return t1, t2


def _newton_step(points, step_size=0.01, max_iter=200):
    """
    Apply a constrained Newton-like solver on the sphere's tangent space.
    We use gradient descent with adaptive step size in the tangent space.
    The gradient of the log energy w.r.t. point i is:
        grad_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2 * (-1) ... 
    Actually, for E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log|x_i - x_j|
    dE/dx_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2 * (-1)
             = -sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    But we need to project this gradient onto the tangent space.
    """
    n = len(points)
    points = points.copy()
    
    for iteration in range(max_iter):
        # Compute gradient in R^3
        grad = np.zeros_like(points)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                diff = points[i] - points[j]
                d2 = np.dot(diff, diff)
                if d2 < 1e-20:
                    continue
                grad[i] -= diff / d2
        
        # Project gradient onto tangent space
        tangent_grad = np.zeros_like(points)
        for i in range(n):
            p = points[i]
            p_norm = np.linalg.norm(p)
            if p_norm < 1e-15:
                continue
            p_unit = p / p_norm
            # Project: g_tangent = g - (g . p_unit) * p_unit
            g = grad[i]
            g_tangent = g - np.dot(g, p_unit) * p_unit
            tangent_grad[i] = g_tangent
        
        # Adaptive step size with line search
        best_step = step_size
        best_energy = _log_energy_fast(points)
        
        for trial in range(20):
            step = step_size * (0.5 ** trial)
            # Move in tangent direction and re-project
            new_points = points - step * tangent_grad
            new_points = _project_to_sphere(new_points)
            
            # Check for minimum distance
            min_d = np.inf
            for i in range(n):
                for j in range(i + 1, n):
                    d = np.linalg.norm(new_points[i] - new_points[j])
                    if d < min_d:
                        min_d = d
            if min_d < 1e-9:
                continue
            
            new_energy = _log_energy_fast(new_points)
            if new_energy < best_energy:
                best_energy = new_energy
                best_step = step
        
        if best_step == 0:
            break
        
        points = points - best_step * tangent_grad
        points = _project_to_sphere(points)
        
        # Check convergence
        if iteration > 0:
            prev_energy = _log_energy_fast(points)
            if abs(prev_energy - best_energy) < 1e-12:
                break
    
    return points


def construct():
    """
    Construct 150 points on the unit sphere using a geodesic dome (subdivided icosahedron)
    as initial positions, then refine with a tangent-space Newton solver.
    """
    # Step 1: Create a high-resolution geodesic dome (subdivided icosahedron)
    # Start with icosahedron vertices
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
    
    # Subdivide the icosahedron multiple times to get ~150 vertices
    # Each subdivision multiplies faces by 4, so:
    # Level 0: 12 vertices, 20 faces
    # Level 1: 42 vertices, 80 faces
    # Level 2: 162 vertices, 320 faces
    # Level 3: 642 vertices, 1280 faces
    # We want ~150 vertices, so level 2 gives 162 which is close.
    
    def subdivide(vertices, faces):
        """Subdivide a triangular mesh by adding midpoints and creating 4 new triangles per face."""
        new_vertices = list(vertices)
        new_faces = []
        edge_cache = {}
        
        def get_midpoint(i, j):
            key = (min(i, j), max(i, j))
            if key in edge_cache:
                return edge_cache[key]
            mid = (vertices[i] + vertices[j]) / 2.0
            mid = mid / np.linalg.norm(mid)  # Project to sphere
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
    
    # Subdivide twice to get 162 vertices
    verts, faces = subdivide(ico_verts, ico_faces)
    verts, faces = subdivide(verts, faces)
    
    # We have 162 vertices, need 150. Remove the 12 least "important" ones.
    # Strategy: remove vertices that are closest to other vertices (redundant)
    if len(verts) > N:
        # Compute for each vertex the minimum distance to any other vertex
        min_dists = np.full(len(verts), np.inf)
        for i in range(len(verts)):
            for j in range(len(verts)):
                if i == j:
                    continue
                d = np.linalg.norm(verts[i] - verts[j])
                if d < min_dists[i]:
                    min_dists[i] = d
        
        # Remove vertices with smallest min_dist (most redundant)
        remove_indices = np.argsort(min_dists)[:len(verts) - N]
        keep_mask = np.ones(len(verts), dtype=bool)
        keep_mask[remove_indices] = False
        verts = verts[keep_mask]
    
    # Ensure we have exactly N points
    if len(verts) < N:
        # Pad with Fibonacci spiral points if needed
        n_extra = N - len(verts)
        i = np.arange(n_extra) + 0.5 + len(verts)
        z = 1.0 - 2.0 * i / N
        phi_angle = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        extra = np.column_stack([r * np.cos(phi_angle), r * np.sin(phi_angle), z])
        verts = np.vstack([verts, extra])
    elif len(verts) > N:
        verts = verts[:N]
    
    # Step 2: Refine with tangent-space Newton solver
    # Use multiple restarts with different initial perturbations
    best_points = verts.copy()
    best_energy = _log_energy_fast(best_points)
    
    # Try a few iterations of Newton refinement
    for restart in range(5):
        current = best_points.copy()
        if restart > 0:
            # Add small random perturbation
            rng = np.random.RandomState(42 + restart)
            current = current + rng.normal(0, 0.01, size=current.shape)
            current = _project_to_sphere(current)
        
        refined = _newton_step(current, step_size=0.005, max_iter=100)
        energy = _log_energy_fast(refined)
        
        if energy < best_energy:
            best_energy = energy
            best_points = refined
    
    # Final projection to ensure unit sphere
    best_points = _project_to_sphere(best_points)
    
    return best_points
# EVOLVE-BLOCK-END
