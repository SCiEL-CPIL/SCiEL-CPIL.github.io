"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import distance_matrix, Delaunay
import math

N = 100

# EVOLVE-BLOCK-START
def _log_energy_and_grad(pts):
    """
    Compute logarithmic energy E = sum_{i<j} log(1/|xi - xj|) = -sum_{i<j} log(|xi - xj|)
    and its gradient with respect to pts (N x 3).
    """
    pts = np.asarray(pts, dtype=np.float64)
    n = pts.shape[0]
    
    # Compute pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    
    # Avoid log(0) by clipping
    dist_safe = np.maximum(dist, 1e-15)
    
    # Energy: sum over i<j of -log(dist[i,j])
    iu = np.triu_indices(n, k=1)
    d_upper = dist_safe[iu]
    energy = -np.sum(np.log(d_upper))
    
    # Gradient: grad[i] = -sum_{j != i} (pts[i] - pts[j]) / ||pts[i] - pts[j]||^2
    dist_sq = dist_safe ** 2
    
    W = np.zeros_like(dist_sq)
    np.fill_diagonal(W, 0.0)
    W = 1.0 / dist_sq
    np.fill_diagonal(W, 0.0)
    
    row_sum = W.sum(axis=1)
    grad = -pts * row_sum[:, np.newaxis] + W @ pts
    
    return energy, grad

def _project_to_sphere(pts):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _build_mesh_connectivity(pts):
    """
    Build a fixed mesh connectivity (edges) from initial points using Delaunay triangulation.
    Returns list of edges (pairs of indices).
    """
    # Use Delaunay triangulation on the 3D points
    try:
        tri = Delaunay(pts)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                j = (i + 1) % 3
                edge = tuple(sorted([simplex[i], simplex[j]]))
                edges.add(edge)
        return list(edges)
    except Exception:
        # Fallback: connect each point to its k nearest neighbors
        from scipy.spatial import cKDTree
        tree = cKDTree(pts)
        _, indices = tree.query(pts, k=8)  # 7 neighbors + self
        edges = set()
        for i in range(len(pts)):
            for j in indices[i]:
                if i != j:
                    edge = tuple(sorted([i, j]))
                    edges.add(edge)
        return list(edges)

def _mesh_aware_gradient_descent(pts, edges, max_iter=3000, lr=0.01, decay=0.999, tol=1e-10):
    """
    Gradient descent on the sphere with mesh-aware regularization.
    The mesh connectivity provides structural constraints that help avoid bad local minima.
    """
    pts = _project_to_sphere(pts.copy())
    current_energy, current_grad = _log_energy_and_grad(pts)
    
    # Precompute edge list as arrays
    edge_i = np.array([e[0] for e in edges])
    edge_j = np.array([e[1] for e in edges])
    
    # Build sparse representation of mesh Laplacian for regularization
    n = len(pts)
    # Mesh regularization: penalize deviation from equal edge lengths
    # This keeps the mesh "rigid" in a soft sense
    
    for it in range(max_iter):
        # Compute gradient
        energy, grad = _log_energy_and_grad(pts)
        
        # Project gradient to tangent space
        grad_dot = np.einsum('ij,ij->i', grad, pts)
        grad_tangent = grad - grad_dot[:, np.newaxis] * pts
        
        # Mesh regularization gradient: encourage uniform edge lengths
        # For each edge, the gradient contribution to keep edge length ~ constant
        edge_vecs = pts[edge_i] - pts[edge_j]
        edge_lens = np.linalg.norm(edge_vecs, axis=1)
        edge_lens_safe = np.maximum(edge_lens, 1e-10)
        
        # Target edge length: average edge length
        target_len = np.mean(edge_lens)
        
        # Regularization gradient: pull edges toward target length
        reg_grad = np.zeros_like(pts)
        for idx in range(len(edge_i)):
            vi = edge_i[idx]
            vj = edge_j[idx]
            d = edge_vecs[idx] / edge_lens_safe[idx]
            # Gradient of (len - target)^2 w.r.t. vi and vj
            factor = 2.0 * (edge_lens_safe[idx] - target_len)
            reg_grad[vi] += factor * d
            reg_grad[vj] -= factor * d
        
        # Combine gradients
        combined_grad = grad_tangent + 0.1 * reg_grad
        
        # Project combined gradient to tangent space
        combined_dot = np.einsum('ij,ij->i', combined_grad, pts)
        combined_tangent = combined_grad - combined_dot[:, np.newaxis] * pts
        
        # Adaptive learning rate
        step = lr * combined_tangent
        
        # Line search
        alpha = 1.0
        improved = False
        for _ in range(20):
            pts_new = pts + alpha * step
            pts_new = _project_to_sphere(pts_new)
            new_energy, _ = _log_energy_and_grad(pts_new)
            
            if new_energy < energy:
                pts = pts_new
                current_energy = new_energy
                improved = True
                break
            else:
                alpha *= 0.5
        
        if not improved:
            lr *= 0.5
            if lr < 1e-8:
                break
        else:
            lr = min(lr * 1.01, 0.1)
        
        # Check convergence
        if np.linalg.norm(combined_tangent) < tol:
            break
    
    return pts, current_energy

def _generate_geodesic_initial(n_points=100):
    """
    Generate a geodesic sphere-like initial configuration.
    Start with an icosahedron and subdivide.
    """
    # Icosahedron vertices
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Icosahedron faces
    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ]
    
    # Subdivide faces to get more points
    # We'll do multiple levels of subdivision
    current_verts = verts.copy()
    current_faces = [f[:] for f in faces]
    
    target = n_points
    while len(current_verts) < target:
        new_faces = []
        edge_midpoints = {}
        
        for face in current_faces:
            a, b, c = face
            edges = [(a, b), (b, c), (c, a)]
            
            new_verts = [a, b, c]
            for e in edges:
                key = tuple(sorted(e))
                if key not in edge_midpoints:
                    # Create midpoint
                    mid = (current_verts[e[0]] + current_verts[e[1]]) / 2.0
                    mid = mid / np.linalg.norm(mid)
                    edge_midpoints[key] = len(current_verts)
                    current_verts = np.vstack([current_verts, mid])
                new_verts.append(edge_midpoints[key])
            
            # 4 new triangles
            ab, bc, ca = new_verts[3], new_verts[4], new_verts[5]
            new_faces.append([a, ab, ca])
            new_faces.append([b, bc, ab])
            new_faces.append([c, ca, bc])
            new_faces.append([ab, bc, ca])
        
        current_faces = new_faces
    
    # If we have too many, keep first n_points
    if len(current_verts) > target:
        current_verts = current_verts[:target]
    
    return current_verts

def construct():
    # Generate geodesic initial configuration
    pts_init = _generate_geodesic_initial(N)
    pts_init = _project_to_sphere(pts_init)
    
    best_energy = np.inf
    best_pts = pts_init.copy()
    
    n_restarts = 10
    
    for restart in range(n_restarts):
        if restart == 0:
            x0 = pts_init.copy()
        elif restart == 1:
            # Random rotation
            from scipy.spatial.transform import Rotation
            rot = Rotation.random(random_state=restart)
            x0 = rot.apply(pts_init)
        else:
            # Perturbed geodesic
            noise_scale = 0.05 * restart
            noise = np.random.normal(0, noise_scale, size=(N, 3))
            x0 = pts_init + noise
            x0 = _project_to_sphere(x0)
        
        # Build mesh connectivity from current configuration
        edges = _build_mesh_connectivity(x0)
        
        try:
            res_pts, res_energy = _mesh_aware_gradient_descent(x0, edges, max_iter=2000, lr=0.01)
            if res_energy < best_energy:
                best_energy = res_energy
                best_pts = res_pts
        except Exception as e:
            pass
    
    # Final projection to ensure unit sphere
    best_pts = _project_to_sphere(best_pts)
    
    # Verify no points are too close
    dist = distance_matrix(best_pts, best_pts)
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    
    if min_dist < 1e-9:
        # Fallback to Fibonacci spiral
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
    # Ensure finite values
    if not np.all(np.isfinite(best_pts)):
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
    return best_pts
# EVOLVE-BLOCK-END
