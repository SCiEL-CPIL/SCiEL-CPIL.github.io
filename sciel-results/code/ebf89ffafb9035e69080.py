import numpy as np
import time
from scipy.optimize import minimize
from scipy.sparse.csgraph import laplacian
from scipy.sparse import csr_matrix

N = 14
D = 3

def get_pairwise_sq_dists(points):
    """Calculate squared pairwise distances for N points in D dimensions."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    iu = np.triu_indices(N, k=1)
    return sq_dists[iu]

def raw_score(points):
    """Calculate the actual score: (max_dist / min_dist)^2."""
    sq_dists = get_pairwise_sq_dists(points)
    min_sq = np.min(sq_dists)
    if min_sq < 1e-12:
        return np.inf
    max_sq = np.max(sq_dists)
    return (max_sq / min_sq)

def objective_smooth(x, tau=100.0):
    """
    Smoothed objective function using log-sum-exp for max and min.
    Minimizes log(max_sq_dist) - log(min_sq_dist).
    """
    points = x.reshape(N, D)
    sq_dists = get_pairwise_sq_dists(points)
    sq_dists = np.maximum(sq_dists, 1e-12)
    
    m_max = np.max(sq_dists)
    log_sum_exp_max = m_max + np.log(np.sum(np.exp(tau * (sq_dists - m_max)))) / tau
    
    m_min = np.min(sq_dists)
    log_sum_exp_min = m_min - np.log(np.sum(np.exp(-tau * (sq_dists - m_min)))) / tau
    
    return np.log(log_sum_exp_max) - np.log(log_sum_exp_min)

def get_truncated_tetrahedron_vertices():
    """
    Get vertices of a regular truncated tetrahedron.
    The truncated tetrahedron has 12 vertices.
    We need 14 points, so we'll add 2 more points (perhaps center points of faces or edges).
    """
    # Regular tetrahedron vertices
    a = 1.0
    tet_verts = np.array([
        [a, a, a],
        [a, -a, -a],
        [-a, a, -a],
        [-a, -a, a]
    ])
    
    # Truncation: cut off each vertex at 1/3 of the way to the center
    # This creates 12 new vertices (4 original vertices become faces, 4 new faces)
    # Actually, truncating a tetrahedron at 1/3 gives a truncated tetrahedron with 12 vertices
    
    # The 12 vertices of a regular truncated tetrahedron can be generated as:
    # Take all permutations of (0, 1, 2) and (0, 1, 4) with signs, scaled appropriately
    # Or more simply, use the known coordinates
    
    # A regular truncated tetrahedron has vertices that can be described as:
    # All even permutations of (0, 1, 2) and (0, -1, 4) with appropriate scaling
    # Let me use a different approach: the 12 vertices are the 12 edge midpoints of a cube
    # No wait, that's a cuboctahedron.
    
    # Let me use the actual truncated tetrahedron coordinates:
    # The 12 vertices can be generated from the tetrahedron by truncating at 1/3
    # For a regular tetrahedron with vertices at (1,1,1), (1,-1,-1), (-1,1,-1), (-1,-1,1):
    # The center is at (0,0,0)
    # Truncating at parameter t means new vertices at (1-t)*V_i + t*V_j for each edge (i,j)
    
    # For t = 1/3 (truncation at 1/3 of the way from vertex to opposite face):
    t = 1.0 / 3.0
    vertices = []
    for i in range(4):
        for j in range(i+1, 4):
            # Two new vertices on edge (i,j)
            v1 = (1-t) * tet_verts[i] + t * tet_verts[j]
            v2 = (1-t) * tet_verts[j] + t * tet_verts[i]
            vertices.append(v1)
            vertices.append(v2)
    
    vertices = np.array(vertices)
    # Scale to unit size
    vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
    
    return vertices

def get_basis_from_laplacian(adj_matrix):
    """
    Get orthonormal basis vectors from the null space of the graph Laplacian.
    Returns a matrix of shape (N, k) where k is the number of non-zero eigenvalues
    (or we take the first D=3 non-trivial eigenvectors).
    """
    # Compute Laplacian
    L = laplacian(adj_matrix, normed=False)
    
    # Get eigenvalues and eigenvectors
    eigvals, eigvecs = np.linalg.eigh(L.toarray())
    
    # Sort by eigenvalue (ascending)
    idx = np.argsort(eigvals)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    # The first eigenvector corresponds to eigenvalue 0 (constant vector)
    # We want the next D=3 eigenvectors for the 3D embedding
    # These give the "smoothest" variations on the graph
    
    # Take the first D non-trivial eigenvectors (skip the constant one)
    basis = eigvecs[:, 1:D+1]
    
    # Ensure orthonormality (they should already be orthonormal from eigh)
    # But let's QR decompose to be safe
    Q, R = np.linalg.qr(basis)
    
    return Q

def low_dim_objective(coeffs, basis, tau=50.0):
    """
    Objective in terms of low-dimensional coefficients.
    points = basis @ coeffs.reshape(N, D)
    """
    points = basis @ coeffs.reshape(N, D)
    return objective_smooth(points.flatten(), tau)

def construct():
    start_time = time.time()
    timeout = 145.0  # Safety margin before 150s limit
    
    best_score = np.inf
    best_points = None
    
    def update_best(pts):
        nonlocal best_score, best_points
        if pts is None:
            return
        pts = np.asarray(pts)
        if pts.shape != (N, D):
            return
        # Check distinctness
        sq_dists = get_pairwise_sq_dists(pts)
        if np.min(sq_dists) < 1e-10:
            return
        score = raw_score(pts)
        if score < best_score:
            best_score = score
            best_points = pts.copy()

    rng = np.random.default_rng(42)
    
    # --- Strategy 1: Low-dimensional basis optimization using truncated tetrahedron ---
    
    # Get the 12 vertices of truncated tetrahedron
    tt_verts = get_truncated_tetrahedron_vertices()
    
    # We need 14 points. Add 2 more points: center and one more
    center = np.array([0.0, 0.0, 0.0])
    # Add a point that is the average of all vertices (should be center)
    # Let's add two points along different axes
    p13 = np.array([1.0, 0.0, 0.0]) * 0.5
    p14 = np.array([0.0, 1.0, 0.0]) * 0.5
    
    # Create 14-point configuration
    init_14 = np.vstack([tt_verts, center, p13, p14])
    init_14 = init_14 / np.linalg.norm(init_14, axis=1, keepdims=True)
    
    # Create adjacency matrix for the graph
    # The truncated tetrahedron has 12 vertices, each connected to 3 others
    # Plus 2 additional points connected to nearby vertices
    adj = np.zeros((N, N))
    
    # Truncated tetrahedron edges: each vertex connected to 3 others
    # The 12 vertices come in pairs from the 6 edges of the original tetrahedron
    # Let me define the adjacency based on the structure
    for i in range(12):
        for j in range(i+1, 12):
            dist = np.linalg.norm(tt_verts[i] - tt_verts[j])
            if dist < 0.6:  # Threshold for adjacency
                adj[i, j] = 1
                adj[j, i] = 1
    
    # Connect the 2 additional points to nearby vertices
    for i in range(12):
        if np.linalg.norm(init_14[12] - init_14[i]) < 0.8:
            adj[12, i] = 1
            adj[i, 12] = 1
        if np.linalg.norm(init_14[13] - init_14[i]) < 0.8:
            adj[13, i] = 1
            adj[i, 13] = 1
    
    # Get basis from Laplacian
    adj_sparse = csr_matrix(adj)
    basis = get_basis_from_laplacian(adj_sparse)
    
    # Initial coefficients: project initial points onto the basis
    coeffs0 = basis.T @ init_14.flatten().reshape(-1)
    # Actually, we need to solve for coeffs such that basis @ coeffs = init_14
    # Since basis has shape (14, 3), we can't directly invert
    # Let's just use the initial points directly in coordinate space
    
    # Optimize in the low-dimensional space
    # coeffs has shape (N, D) = (14, 3)
    # points = basis @ coeffs
    # But basis is (14, 3), so basis @ coeffs gives (14, 3) only if coeffs is (3, 3)
    # Actually, we want points = basis @ coeffs where coeffs is (3, 3)
    # No, let me reconsider.
    
    # The idea is: points = B @ C where B is (N, D) and C is (D, D)
    # This constrains the points to lie in the span of the basis vectors
    # But that's only D^2 = 9 parameters, which is too few for 14 points in 3D
    
    # Alternative: use the basis vectors as a frame, and allow each point to be
    # a linear combination with different coefficients
    # points[i] = sum_j coeffs[i,j] * basis[j]
    # This is just a change of basis, not a constraint
    
    # Let me use a different approach: optimize directly in coordinate space
    # but use the truncated tetrahedron as initial guess
    
    # --- Direct coordinate optimization with multiple starting points ---
    
    coord_candidates = [init_14.flatten()]
    
    # Icosahedron + center + 1 point
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    
    center = np.array([0.0, 0.0, 0.0])
    v = rng.normal(size=3)
    v = v / np.linalg.norm(v)
    p14 = v * 1.0
    
    init_ico = np.vstack([ico, center, p14])
    coord_candidates.append(init_ico.flatten())
    
    # Random points on a sphere
    for i in range(5):
        pts = rng.normal(size=(N, D))
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        coord_candidates.append(pts.flatten())
        
    # Random points in a cube
    for i in range(5):
        pts = rng.uniform(-1, 1, size=(N, D))
        coord_candidates.append(pts.flatten())

    # Fibonacci sphere
    def fibonacci_sphere(n):
        points = []
        golden_angle = np.pi * (3. - np.sqrt(5.))
        for i in range(n):
            y = 1 - (i / float(n - 1)) * 2
            radius = np.sqrt(1 - y**2)
            theta = golden_angle * i
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            points.append([x, y, z])
        return np.array(points)
    
    fib = fibonacci_sphere(N)
    coord_candidates.append(fib.flatten())
    
    # Coordinate-space optimization
    n_coord_restarts = 80
    for i in range(n_coord_restarts):
        if time.time() - start_time > timeout - 20:
            break
            
        if i < len(coord_candidates):
            x0 = coord_candidates[i]
        else:
            x0 = rng.uniform(-1, 1, size=(N * D))
            
        try:
            # Multi-stage optimization
            res1 = minimize(
                objective_smooth, 
                x0, 
                method='L-BFGS-B', 
                args=(10.0,),
                options={'maxiter': 100, 'ftol': 1e-6, 'gtol': 1e-4}
            )
            if res1.x is not None:
                pts1 = res1.x.reshape(N, D)
                update_best(pts1)
                
                if time.time() - start_time < timeout - 15:
                    res2 = minimize(
                        objective_smooth, 
                        res1.x, 
                        method='L-BFGS-B', 
                        args=(50.0,),
                        options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6}
                    )
                    if res2.x is not None:
                        pts2 = res2.x.reshape(N, D)
                        update_best(pts2)
                        
                        if time.time() - start_time < timeout - 10:
                            res3 = minimize(
                                objective_smooth, 
                                res2.x, 
                                method='L-BFGS-B', 
                                args=(200.0,),
                                options={'maxiter': 400, 'ftol': 1e-10, 'gtol': 1e-8}
                            )
                            if res3.x is not None:
                                pts3 = res3.x.reshape(N, D)
                                update_best(pts3)
                                
        except Exception:
            pass

    # --- Final polish: use the best points found and do a high-precision coordinate optimization ---
    if best_points is not None and time.time() - start_time < timeout - 2:
        try:
            res_final = minimize(
                objective_smooth, 
                best_points.flatten(), 
                method='L-BFGS-B', 
                args=(500.0,),
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res_final.x is not None:
                pts_final = res_final.x.reshape(N, D)
                update_best(pts_final)
        except Exception:
            pass

    if best_points is None:
        # Fallback
        best_points = coord_candidates[0].reshape(N, D)
        best_score = raw_score(best_points)

    return best_points

if __name__ == "__main__":
    pts = construct()
    score = raw_score(pts)
    print(f"Score: {score}")
