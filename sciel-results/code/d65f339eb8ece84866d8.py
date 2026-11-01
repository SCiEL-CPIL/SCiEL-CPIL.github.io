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
    
    # Compute pairwise distances efficiently
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
    dist = np.sqrt(np.maximum(dist_sq, 1e-30))
    
    # Avoid log(0) by clipping
    dist_safe = np.maximum(dist, 1e-15)
    
    # Energy: sum over i<j of -log(dist[i,j])
    iu = np.triu_indices(n, k=1)
    d_upper = dist_safe[iu]
    energy = -math.fsum(np.log(d_upper).tolist())
    
    # Gradient: grad[i] = -sum_{j != i} (pts[i] - pts[j]) / ||pts[i] - pts[j]||^2
    dist_sq_safe = np.maximum(dist_sq, 1e-30)
    
    W = np.zeros_like(dist_sq_safe)
    np.fill_diagonal(W, 0.0)
    W = 1.0 / dist_sq_safe
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
    Build a Delaunay triangulation on the sphere and extract edge connectivity.
    Returns adjacency matrix (N x N boolean) and list of edges.
    """
    # Use Delaunay triangulation in 3D (convex hull)
    try:
        tri = Delaunay(pts)
        simplices = tri.simplices  # shape (M, 3)
        
        # Extract edges from simplices
        edges = set()
        for simplex in simplices:
            i, j, k = simplex
            edges.add((min(i,j), max(i,j)))
            edges.add((min(i,k), max(i,k)))
            edges.add((min(j,k), max(j,k)))
        
        # Build adjacency matrix
        adj = np.zeros((N, N), dtype=bool)
        for i, j in edges:
            adj[i, j] = True
            adj[j, i] = True
        
        return adj, list(edges)
    except Exception:
        # Fallback: k-nearest neighbors graph
        k = 8
        dist_mat = distance_matrix(pts, pts)
        np.fill_diagonal(dist_mat, np.inf)
        adj = np.zeros((N, N), dtype=bool)
        for i in range(N):
            nn_idx = np.argsort(dist_mat[i])[:k]
            for j in nn_idx:
                adj[i, j] = True
                adj[j, i] = True
        edges = [(i, j) for i in range(N) for j in range(i+1, N) if adj[i, j]]
        return adj, edges

def _mesh_aware_gradient(pts, adj):
    """
    Compute gradient with mesh-aware weighting: neighbors in the mesh get
    slightly higher weight to enforce structural consistency.
    """
    energy, grad_base = _log_energy_and_grad(pts)
    
    # Compute base gradient
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
    dist_sq_safe = np.maximum(dist_sq, 1e-30)
    
    W = np.zeros_like(dist_sq_safe)
    np.fill_diagonal(W, 0.0)
    W = 1.0 / dist_sq_safe
    np.fill_diagonal(W, 0.0)
    
    # Apply mesh-aware weighting: boost weights for mesh edges
    # This encourages the optimizer to respect the structural constraints
    boost = 1.0 + 0.3 * adj.astype(np.float64)
    W_mesh = W * boost
    np.fill_diagonal(W_mesh, 0.0)
    
    row_sum = W_mesh.sum(axis=1)
    grad = -pts * row_sum[:, np.newaxis] + W_mesh @ pts
    
    return energy, grad

def _riemannian_cg_mesh(pts, adj, max_iter=3000, tol=1e-9):
    """
    Riemannian Conjugate Gradient with mesh-aware gradient.
    """
    pts = _project_to_sphere(pts.copy())
    current_energy, current_grad = _mesh_aware_gradient(pts, adj)
    
    # Project gradient to tangent space
    grad_dot = np.einsum('ij,ij->i', current_grad, pts)
    grad_tangent = current_grad - grad_dot[:, np.newaxis] * pts
    
    # Initial direction
    d = -grad_tangent
    
    for k in range(max_iter):
        # Adaptive line search
        alpha = 0.1
        improved = False
        best_alpha = 0.0
        best_energy = current_energy
        best_pts_new = pts.copy()
        best_grad_new = current_grad.copy()
        
        for backtrack in range(40):
            pts_new = pts + alpha * d
            pts_new = _project_to_sphere(pts_new)
            
            new_energy, new_grad = _mesh_aware_gradient(pts_new, adj)
            
            if new_energy < current_energy:
                best_alpha = alpha
                best_energy = new_energy
                best_pts_new = pts_new
                best_grad_new = new_grad
                improved = True
            else:
                alpha *= 0.7
            
            if alpha < 1e-10:
                break
        
        if not improved:
            break
            
        pts = best_pts_new
        current_energy = best_energy
        current_grad = best_grad_new
        
        # Check convergence
        grad_norm = np.linalg.norm(grad_tangent)
        if grad_norm < tol:
            break
            
        # Compute new tangent gradient
        grad_dot_new = np.einsum('ij,ij->i', current_grad, pts)
        grad_tangent_new = current_grad - grad_dot_new[:, np.newaxis] * pts
        
        # Fletcher-Reeves beta with damping
        num = np.sum(grad_tangent_new ** 2)
        den = np.sum(grad_tangent ** 2)
        if den < 1e-15:
            beta = 0.0
        else:
            beta = num / den
            beta = min(beta, 1.0)
            
        # Transport d to new tangent space
        d_dot_new = np.einsum('ij,ij->i', d, pts)
        d_tangent = d - d_dot_new[:, np.newaxis] * pts
        
        # New direction
        d = -grad_tangent_new + beta * d_tangent
        
        # Update grad_tangent
        grad_tangent = grad_tangent_new
        
    return pts, current_energy

def _structural_regularization(pts, adj, weight=0.01):
    """
    Add a small regularization term that encourages uniform edge lengths
    within the mesh, promoting structural consistency.
    """
    edges = []
    for i in range(N):
        for j in range(i+1, N):
            if adj[i, j]:
                edges.append((i, j))
    
    if not edges:
        return 0.0, np.zeros_like(pts)
    
    edge_lengths = []
    for i, j in edges:
        edge_lengths.append(np.linalg.norm(pts[i] - pts[j]))
    
    edge_lengths = np.array(edge_lengths)
    mean_len = np.mean(edge_lengths)
    
    # Regularization: penalize deviation from mean edge length
    reg = weight * np.sum((edge_lengths - mean_len)**2)
    
    # Gradient of regularization
    grad_reg = np.zeros_like(pts)
    for idx, (i, j) in enumerate(edges):
        diff = pts[i] - pts[j]
        dist = np.linalg.norm(diff)
        if dist > 1e-15:
            factor = weight * 2 * (edge_lengths[idx] - mean_len) / dist
            grad_reg[i] += factor * diff
            grad_reg[j] -= factor * diff
    
    return reg, grad_reg

def _fibonacci_spiral(n=N, offset=0.5):
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def construct():
    # Generate initial configuration using Fibonacci spiral
    pts0 = _fibonacci_spiral(N, offset=0.5)
    
    # Build mesh connectivity from initial configuration
    adj, edges = _build_mesh_connectivity(pts0)
    
    # Run Riemannian CG with mesh-aware gradient
    best_pts, best_energy = _riemannian_cg_mesh(pts0, adj, max_iter=5000, tol=1e-10)
    
    # Additional refinement passes with structural regularization
    for pass_num in range(3):
        try:
            # Add structural regularization
            reg, grad_reg = _structural_regularization(best_pts, adj, weight=0.005)
            
            # Combine gradients
            _, grad_base = _log_energy_and_grad(best_pts)
            grad_combined = grad_base + grad_reg
            
            # Project to tangent space
            grad_dot = np.einsum('ij,ij->i', grad_combined, best_pts)
            grad_tangent = grad_combined - grad_dot[:, np.newaxis] * best_pts
            
            # Take a small step
            step_size = 0.01
            pts_new = best_pts + step_size * grad_tangent
            pts_new = _project_to_sphere(pts_new)
            
            # Evaluate
            new_energy, _ = _log_energy_and_grad(pts_new)
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_pts = pts_new
        except Exception:
            pass
    
    # Final refinement with plain RCG (no mesh weighting) for fine-tuning
    try:
        final_pts, final_energy = _riemannian_cg_mesh(best_pts, adj, max_iter=3000, tol=1e-11)
        if final_energy < best_energy:
            best_energy = final_energy
            best_pts = final_pts
    except Exception:
        pass
    
    # Final projection to ensure unit sphere
    best_pts = _project_to_sphere(best_pts)
    
    # Verify no points are too close
    dist = distance_matrix(best_pts, best_pts)
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    
    if min_dist < 1e-9:
        # If points are too close, fall back to Fibonacci spiral
        best_pts = _fibonacci_spiral(N)
        
    # Ensure finite values
    if not np.all(np.isfinite(best_pts)):
        best_pts = _fibonacci_spiral(N)
        
    return best_pts
# EVOLVE-BLOCK-END
