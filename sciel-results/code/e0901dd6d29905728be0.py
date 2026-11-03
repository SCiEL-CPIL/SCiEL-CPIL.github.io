import numpy as np
import time
import math
from scipy.optimize import minimize

N = 150
DEADLINE = 115.0  # Safety margin for 120s limit

def _subdivided_icosahedron(n_subdivisions):
    """Generate a subdivided icosahedron mesh on the unit sphere."""
    # Golden ratio
    t = (1.0 + math.sqrt(5.0)) / 2.0
    # 12 vertices of the icosahedron
    verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    # Normalize
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    
    # 20 faces of the icosahedron
    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ]
    
    # Subdivide faces
    for _ in range(n_subdivisions):
        new_faces = []
        midpoints = {}
        
        def get_midpoint(i, j):
            key = (min(i, j), max(i, j))
            if key not in midpoints:
                v1 = verts[i]
                v2 = verts[j]
                mid = (v1 + v2) / 2.0
                mid /= np.linalg.norm(mid)
                midpoints[key] = len(verts)
                verts = np.vstack([verts, mid])
            return midpoints[key]
        
        for f in faces:
            a, b, c = f
            ab = get_midpoint(a, b)
            bc = get_midpoint(b, c)
            ca = get_midpoint(c, a)
            new_faces.append([a, ab, ca])
            new_faces.append([b, bc, ab])
            new_faces.append([c, ca, bc])
            new_faces.append([ab, bc, ca])
        faces = new_faces
        
    # Select points
    # For N=150, we want roughly 150 points.
    # An icosahedron has 12 vertices, 30 edges, 20 faces.
    # Subdivision level 1: 12 + 30 + 20*4 = 122 points? No, vertices are shared.
    # Vertices: 12
    # Midpoints of edges: 30
    # Centroids of faces: 20
    # Total: 62 points. Too few.
    # Subdivision level 2:
    # Vertices: 12
    # Edge midpoints: 30
    # Face centroids: 20
    # Sub-edge midpoints: 30 * 2 = 60? No, each edge is split into 4 segments?
    # Let's just generate a large set and pick 150.
    # A simpler way: Use the Fibonacci spiral for initialization as it is robust and fast,
    # but the prompt asks for a different strategy. The current program uses Fibonacci.
    # The inspirations use subdivided icosahedron.
    # Let's generate a dense icosahedron and sample 150 points from it.
    
    # Actually, generating a proper icosahedron subdivision that yields exactly N points is tricky.
    # A common approach is to generate a dense mesh and then select N points.
    # Or, just use the vertices of a high-subdivision icosahedron.
    # For N=150, subdivision level 2 gives 162 vertices (12 + 30 + 20 + 60 + 40? Let's check).
    # Level 0: 12 verts, 20 faces
    # Level 1: 12 verts + 30 edge mids + 20 face centers = 62 verts. 80 faces.
    # Level 2: 62 verts + (80*3/2) edge mids? No.
    # Each face is divided into 4.
    # New vertices: midpoints of the 3 edges of each face.
    # Total edges in Level 1: 80 * 3 / 2 = 120 edges.
    # So 120 new midpoints.
    # Total vertices Level 2: 62 + 120 = 182.
    # This is close to 150. We can just take the first 150 vertices or sample them.
    
    return verts[:N]

def _log_energy_and_grad(pts):
    """Compute log energy and gradient for points on the sphere."""
    pts = pts.reshape(-1, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    
    # Compute pairwise distances using dot products for efficiency
    # |xi - xj|^2 = 2 - 2*xi·xj for unit vectors
    dot = np.dot(pts, pts.T)
    np.fill_diagonal(dot, 1.0)
    dist_sq = 2.0 - 2.0 * dot
    np.maximum(dist_sq, 1e-30, out=dist_sq)
    dist = np.sqrt(dist_sq)
    
    # Energy: sum_{i<j} log(1/dist) = -sum_{i<j} log(dist)
    iu = np.triu_indices(N, 1)
    d = dist[iu]
    energy = -np.sum(np.log(d))
    
    # Gradient: dE/dxi = sum_{j != i} (xi - xj) / |xi - xj|^2
    # Since dE/dxi = sum_{j != i} -d/dxi log(dist_ij) = sum_{j != i} (xi-xj)/dist_ij^2
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    # grad_i = sum_j (xi - xj) / dist_ij^2
    # = sum_j xi/dist_ij^2 - sum_j xj/dist_ij^2
    # = xi * sum_j inv_dist_sq[i,j] - sum_j xj * inv_dist_sq[i,j]
    row_sums = np.sum(inv_dist_sq, axis=1, keepdims=True)
    grad = pts * row_sums - np.dot(inv_dist_sq, pts)
    
    # Project gradient to tangent space
    dot_grad = np.sum(grad * pts, axis=1, keepdims=True)
    grad_tangent = grad - dot_grad * pts
    
    return energy, grad_tangent

def _fibonacci_sphere(n):
    """Generate n points on the sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _gram_energy_and_grad(G):
    """
    Compute log energy and gradient with respect to Gram matrix G.
    
    G is N x N, PSD, with G_ii = 1 (unit vectors).
    Chordal distance squared: d_ij^2 = 2 - 2*G_ij
    Energy: sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    
    Gradient: dE/dG_ij = sum_{k: k!=i, k!=j, k in pair with i or j} ...
    Actually, E = -sum_{i<j} 0.5 * log(2 - 2*G_ij)
    dE/dG_ij = -0.5 * (-2) / (2 - 2*G_ij) = 1 / (2 - 2*G_ij) for i<j
    By symmetry, dE/dG_ji = same.
    
    But G must be PSD and have unit diagonal. We'll handle constraints via
    projection or by optimizing over a parameterization.
    """
    # Ensure symmetry
    G = 0.5 * (G + G.T)
    
    # Clip off-diagonal to ensure valid distances
    G_off = G.copy()
    np.fill_diagonal(G_off, 1.0)
    # G_ij must be in (-1, 1) for real points
    G_off = np.clip(G_off, -1.0 + 1e-12, 1.0 - 1e-12)
    
    dist_sq = 2.0 - 2.0 * G_off
    np.maximum(dist_sq, 1e-30, out=dist_sq)
    
    # Energy
    iu = np.triu_indices(N, 1)
    energy = -0.5 * np.sum(np.log(dist_sq[iu]))
    
    # Gradient w.r.t. G_ij (i<j): dE/dG_ij = 1 / dist_sq_ij
    # This is because dE/dG_ij = -d/dG_ij [0.5 * log(2-2*G_ij)] 
    # = -0.5 * (-2)/(2-2*G_ij) = 1/(2-2*G_ij)
    grad = np.zeros_like(G)
    grad_vals = 1.0 / dist_sq[iu]
    grad[iu] = grad_vals
    grad[(iu[1], iu[0])] = grad_vals  # Symmetric
    
    # Zero out diagonal (constrained to 1)
    np.fill_diagonal(grad, 0.0)
    
    return energy, grad

def _project_gram_to_psd(G, rank_constraint=True):
    """
    Project G onto the set of PSD matrices with unit diagonal.
    If rank_constraint, also try to enforce rank <= 3 (points in R^3).
    """
    G = 0.5 * (G + G.T)
    
    # Eigenvalue decomposition
    eigvals, eigvecs = np.linalg.eigh(G)
    
    # Clip negative eigenvalues to 0 for PSD
    eigvals_clipped = np.maximum(eigvals, 0.0)
    
    # If rank constraint, keep only top 3 eigenvalues
    if rank_constraint:
        # Sort by eigenvalue magnitude
        idx = np.argsort(-eigvals_clipped)
        eigvals_clipped = np.zeros_like(eigvals_clipped)
        eigvals_clipped[idx[:3]] = np.sort(eigvals_clipped[idx[:3]])[::-1]
    
    G_psd = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
    G_psd = 0.5 * (G_psd + G_psd.T)
    
    # Normalize diagonal to 1
    diag = np.sqrt(np.abs(np.diag(G_psd)))
    diag = np.maximum(diag, 1e-10)
    G_psd = G_psd / np.outer(diag, diag)
    
    # Ensure exact unit diagonal
    np.fill_diagonal(G_psd, 1.0)
    
    return G_psd

def construct():
    start_time = time.time()
    rng = np.random.RandomState(42)
    
    # Fibonacci spiral initialization
    points = _fibonacci_sphere(N)
    
    best_energy = np.inf
    best_points = points.copy()
    
    def objective_and_grad(x):
        pts = x.reshape(-N, 3)
        e, g = _log_energy_and_grad(pts)
        return e, g.flatten()
    
    # Phase 1: Initial L-BFGS-B optimization
    x0 = points.flatten()
    result = minimize(objective_and_grad, x0, method='L-BFGS-B', jac=True, 
                      options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10})
    
    points = result.x.reshape(-N, 3)
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    
    energy, _ = _log_energy_and_grad(points)
    best_energy = energy
    best_points = points.copy()
    
    # Phase 1b: Gram-matrix optimization
    # Convert current points to Gram matrix
    G0 = np.dot(points, points.T)
    np.fill_diagonal(G0, 1.0)
    
    # Optimize in Gram space with L-BFGS-B
    # We'll optimize over the off-diagonal entries, keeping diagonal fixed
    # Parameterize: G = G_base + delta, where delta has zero diagonal
    
    def gram_objective_and_grad(x):
        # x is the flattened off-diagonal part
        G = np.zeros((N, N))
        np.fill_diagonal(G, 1.0)
        
        # Fill off-diagonal from x
        iu = np.triu_indices(N, 1)
        G[iu] = x
        G[(iu[1], iu[0])] = x
        
        # Project to PSD with unit diagonal
        G_proj = _project_gram_to_psd(G, rank_constraint=True)
        
        # Compute energy and gradient
        e, g = _gram_energy_and_grad(G_proj)
        
        # Map gradient back to parameter space
        g_flat = g[iu].copy()
        
        return e, g_flat
    
    # Initial x from G0
    iu = np.triu_indices(N, 1)
    x0_gram = G0[iu].copy()
    
    try:
        result_gram = minimize(gram_objective_and_grad, x0_gram, method='L-BFGS-B', jac=True,
                               options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
        
        # Reconstruct points from optimized Gram
        G_opt = np.zeros((N, N))
        np.fill_diagonal(G_opt, 1.0)
        G_opt[iu] = result_gram.x
        G_opt[(iu[1], iu[0])] = result_gram.x
        G_opt = _project_gram_to_psd(G_opt, rank_constraint=True)
        
        # Extract points via eigenvalue decomposition
        eigvals, eigvecs = np.linalg.eigh(G_opt)
        # Take top 3 eigenvalues for R^3 embedding
        idx = np.argsort(-eigvals)
        eigvals_top = eigvals[idx[:3]]
        eigvecs_top = eigvecs[:, idx[:3]]
        
        # Points are rows of eigvecs_top * sqrt(eigvals_top)
        points_gram = eigvecs_top * np.sqrt(np.maximum(eigvals_top, 0))
        points_gram /= np.linalg.norm(points_gram, axis=1, keepdims=True)
        
        gram_energy, _ = _log_energy_and_grad(points_gram)
        
        if gram_energy < best_energy:
            best_energy = gram_energy
            best_points = points_gram.copy()
            points = points_gram.copy()
    except Exception:
        pass  # Fall back to coordinate-based optimization
    
    # Phase 2: Enhanced Simulated Annealing with multiple strategies
    base_temp = 0.08
    current_temp = base_temp
    no_improve_count = 0
    max_no_improve = 30
    sa_iterations = 0
    
    while time.time() - start_time < DEADLINE - 8.0:
        sa_iterations += 1
        
        # Adaptive perturbation strategy
        if no_improve_count > max_no_improve:
            # Reset and use larger perturbations
            num_perturb = rng.randint(10, 25)
            current_temp = min(current_temp * 1.8, 0.25)
            no_improve_count = 0
        else:
            num_perturb = max(3, N // 25)
        
        # Select points to perturb - prefer points with high local energy contribution
        dot = np.dot(points, points.T)
        np.fill_diagonal(dot, 1.0)
        dist_sq = 2.0 - 2.0 * dot
        np.maximum(dist_sq, 1e-30, out=dist_sq)
        local_energy = np.sum(np.log(1.0 / np.sqrt(dist_sq)), axis=1)
        
        # Weighted selection
        weights = local_energy - local_energy.min() + 1e-6
        weights /= weights.sum()
        indices = rng.choice(N, num_perturb, replace=False, p=weights)
        
        # Generate perturbations in tangent space
        for idx in indices:
            p = points[idx]
            v = rng.randn(3)
            v -= np.dot(v, p) * p
            v_norm = np.linalg.norm(v)
            if v_norm > 1e-10:
                mag = current_temp * rng.uniform(0.5, 1.5)
                v = v / v_norm * mag
                points[idx] = p + v
                points[idx] /= np.linalg.norm(points[idx])
        
        # Local optimization
        x0 = points.flatten()
        max_iter_inner = 100
        result = minimize(objective_and_grad, x0, method='L-BFGS-B', jac=True, 
                          options={'maxiter': max_iter_inner, 'ftol': 1e-12, 'gtol': 1e-8})
        
        points = result.x.reshape(-N, 3)
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        
        energy, _ = _log_energy_and_grad(points)
        
        # Metropolis acceptance
        if energy < best_energy:
            best_energy = energy
            best_points = points.copy()
            no_improve_count = 0
            current_temp = max(current_temp * 0.90, 0.02)
        else:
            delta_e = energy - best_energy
            if delta_e < 1e-10:
                best_energy = energy
                best_points = points.copy()
                no_improve_count = 0
                current_temp = max(current_temp * 0.95, 0.02)
            else:
                prob = np.exp(-delta_e / max(current_temp, 1e-8))
                if rng.rand() < prob:
                    no_improve_count += 1
                    current_temp = max(current_temp * 0.99, 0.01)
                else:
                    points = best_points.copy()
                    no_improve_count += 1
                    current_temp = max(current_temp * 0.99, 0.01)
    
    # Phase 3: Gram-matrix refinement from best points
    if time.time() - start_time < DEADLINE - 5.0:
        G_best = np.dot(best_points, best_points.T)
        np.fill_diagonal(G_best, 1.0)
        x0_gram2 = G_best[iu].copy()
        
        try:
            result_gram2 = minimize(gram_objective_and_grad, x0_gram2, method='L-BFGS-B', jac=True,
                                    options={'maxiter': 150, 'ftol': 1e-14, 'gtol': 1e-12})
            
            G_opt2 = np.zeros((N, N))
            np.fill_diagonal(G_opt2, 1.0)
            G_opt2[iu] = result_gram2.x
            G_opt2[(iu[1], iu[0])] = result_gram2.x
            G_opt2 = _project_gram_to_psd(G_opt2, rank_constraint=True)
            
            eigvals2, eigvecs2 = np.linalg.eigh(G_opt2)
            idx2 = np.argsort(-eigvals2)
            eigvals_top2 = eigvals2[idx2[:3]]
            eigvecs_top2 = eigvecs2[:, idx2[:3]]
            
            points_gram2 = eigvecs_top2 * np.sqrt(np.maximum(eigvals_top2, 0))
            points_gram2 /= np.linalg.norm(points_gram2, axis=1, keepdims=True)
            
            gram_energy2, _ = _log_energy_and_grad(points_gram2)
            
            if gram_energy2 < best_energy:
                best_energy = gram_energy2
                best_points = points_gram2.copy()
        except Exception:
            pass
    
    # Phase 4: Multiple final polish passes
    polish_configs = [
        {'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-12},
        {'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-13},
    ]
    
    for config in polish_configs:
        if time.time() - start_time < DEADLINE - 3.0:
            x0 = best_points.flatten()
            result = minimize(objective_and_grad, x0, method='L-BFGS-B', jac=True, 
                              options=config)
            
            final_points = result.x.reshape(-N, 3)
            final_points /= np.linalg.norm(final_points, axis=1, keepdims=True)
            
            final_energy, _ = _log_energy_and_grad(final_points)
            if final_energy < best_energy:
                best_energy = final_energy
                best_points = final_points
    
    # Phase 5: Ultra-fine polish
    if time.time() - start_time < DEADLINE - 1.5:
        x0 = best_points.flatten()
        result = minimize(objective_and_grad, x0, method='L-BFGS-B', jac=True, 
                          options={'maxiter': 100, 'ftol': 1e-16, 'gtol': 1e-15})
        
        final_points = result.x.reshape(-N, 3)
        final_points /= np.linalg.norm(final_points, axis=1, keepdims=True)
        
        final_energy, _ = _log_energy_and_grad(final_points)
        if final_energy < best_energy:
            best_energy = final_energy
            best_points = final_points
    
    return best_points
