import numpy as np
import time
from scipy.optimize import minimize

N = 14
D = 3

# Precompute pairwise indices to avoid recomputing on every call
_IU = np.triu_indices(N, k=1)

def get_pairwise_sq_dists(points):
    """Calculate squared pairwise distances for N points in D dimensions."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    return sq_dists[_IU]

def objective_epigraph(x, tau=50.0):
    """
    Epigraph formulation: minimize t subject to all sq_dists <= t,
    and maximize min sq_dist. We use a smooth approximation:
    - Soft max of sq_dists as proxy for t
    - Soft min of sq_dists as proxy for minimum distance
    - Objective: soft_max - lambda * soft_min (we want to minimize ratio)
    
    Actually, let's use: minimize (soft_max / soft_min) via log formulation
    but with an auxiliary variable approach for better conditioning.
    
    Here we use: minimize log(soft_max) - log(soft_min) which is equivalent
    to the log-ratio but computed with separate soft-max and soft-min.
    """
    points = x.reshape(N, D)
    sq_dists = get_pairwise_sq_dists(points)
    sq_dists = np.maximum(sq_dists, 1e-12)
    
    # Soft max
    m_max = np.max(sq_dists)
    log_sum_exp_max = m_max + np.log(np.sum(np.exp(tau * (sq_dists - m_max)))) / tau
    
    # Soft min  
    m_min = np.min(sq_dists)
    log_sum_exp_min = m_min - np.log(np.sum(np.exp(-tau * (sq_dists - m_min)))) / tau
    
    return np.log(log_sum_exp_max) - np.log(log_sum_exp_min)

def build_truncated_tetrahedron_graph():
    """
    Build the adjacency matrix of the truncated tetrahedron graph.
    The truncated tetrahedron has 12 vertices, but we need 14.
    We'll use a 14-vertex graph with high symmetry: two tetrahedra sharing an edge,
    or a different high-symmetry graph.
    
    For 14 vertices, let's use the graph of the cuboctahedron (12 vertices) plus 2 more,
    or better: use a complete multipartite or other structured graph.
    
    Actually, for the Laplacian null space approach, we want a graph whose Laplacian
    has a null space that gives us good initial configurations. Let's use a complete
    graph K_14 (which has null space = constant vectors) combined with structure.
    
    Better approach: Use the Laplacian of a highly symmetric graph on 14 vertices.
    The icosahedron has 12 vertices. For 14, we can use a truncated octahedron variant
    or simply construct a graph with known symmetry.
    
    Let's use a simpler approach: create a graph where vertices are arranged in
    orbits under a symmetry group, and compute the Laplacian.
    """
    # Create a 14-vertex graph with D_3d-like symmetry
    # 6 + 6 + 2 structure (two hexagonal rings + 2 axial)
    # Adjacency: connect vertices within same ring (cycle) and between rings
    A = np.zeros((N, N))
    
    # Ring 1: vertices 0-5 (hexagonal cycle)
    for i in range(6):
        A[i, (i+1) % 6] = 1
        A[(i+1) % 6, i] = 1
    
    # Ring 2: vertices 6-11 (hexagonal cycle, offset by 30 degrees)
    for i in range(6):
        j = 6 + i
        A[j, 6 + (i+1) % 6] = 1
        A[6 + (i+1) % 6, j] = 1
    
    # Connect rings: each vertex in ring 1 connects to 2 nearest in ring 2
    for i in range(6):
        # Connect to offset positions
        A[i, 6 + i] = 1
        A[6 + i, i] = 1
        A[i, 6 + (i+1) % 6] = 1
        A[6 + (i+1) % 6, i] = 1
    
    # Axial vertices: 12 and 13
    # Connect to all ring vertices
    for i in range(12):
        A[12, i] = 1
        A[i, 12] = 1
        A[13, i] = 1
        A[i, 13] = 1
    
    # Connect axial vertices to each other
    A[12, 13] = 1
    A[13, 12] = 1
    
    return A

def get_laplacian_null_basis(A):
    """
    Get orthonormal basis for the null space of the Laplacian L = D - A.
    For a connected graph, the null space is 1-dimensional (constant vector).
    For our purposes, we'll use the eigenvectors corresponding to the smallest
    eigenvalues (including zero) to form a low-dimensional basis.
    """
    # Compute degree matrix
    degrees = A.sum(axis=1)
    D_mat = np.diag(degrees)
    L = D_mat - A
    
    # Compute eigenvalues and eigenvectors
    eigvals, eigvecs = np.linalg.eigh(L)
    
    # Take the first few eigenvectors (smallest eigenvalues)
    # These span the "slow" modes of the graph
    n_basis = min(8, N)  # Use up to 8 basis vectors
    basis = eigvecs[:, :n_basis]
    
    # Orthonormalize (they should already be orthonormal from eigh)
    return basis

def basis_to_points(coeffs, basis):
    """
    Convert coefficients to points using the basis.
    points = coeffs @ basis.T
    where coeffs is (N, k) and basis is (N, k), giving points as (k, N) -> transpose to (N, k)
    Actually: each point i is sum_j coeffs[i,j] * basis[j,:]
    So points = coeffs @ basis
    """
    # coeffs: (N, k), basis: (N, k) -> points[i] = sum_j coeffs[i,j] * basis[j]
    # This is: points = coeffs @ basis.T? No.
    # points[i, :] = sum_j coeffs[i,j] * basis[j, :]
    # This is matrix multiplication: points = coeffs @ basis.T? 
    # coeffs is (N, k), basis.T is (k, N) -> result is (N, N)? That's wrong.
    # Let me reconsider: basis[j] is a vector in R^N? No, basis is (N, k) where each column is a basis vector in R^N.
    # We want points in R^3, not R^N.
    # 
    # Better interpretation: The null space of the Laplacian gives us vectors in R^N.
    # We can interpret these as defining a low-dimensional embedding.
    # Let basis be (N, k) where each row is a point in R^k (the embedding coordinates).
    # Then we scale: points = coeffs @ basis where coeffs is (k, k) diagonal scaling?
    # 
    # Actually, the standard approach: if L has eigenvectors v_1, ..., v_k (each in R^N),
    # we can form an embedding where point i has coordinates (v_1[i], v_2[i], ..., v_k[i]).
    # Then we can scale each coordinate independently.
    #
    # So: basis is (N, k) where basis[i, j] = v_j[i]
    # Raw embedding: points_raw[i, j] = basis[i, j]
    # Scaled: points[i, j] = coeffs[j] * basis[i, j] where coeffs is (k,)
    # But we can also allow mixing: points = basis @ M where M is (k, 3)
    
    # Let's use: points = basis @ M where M is (k, 3)
    # coeffs is flattened M: shape (k*3,)
    M = coeffs.reshape(basis.shape[1], 3)
    points = basis @ M
    return points

def basis_objective(coeffs, basis, tau=50.0):
    """
    Objective function for basis representation.
    coeffs: flattened (k, 3) matrix
    """
    points = basis_to_points(coeffs, basis)
    sq_dists = get_pairwise_sq_dists(points)
    
    # Avoid log(0)
    sq_dists = np.maximum(sq_dists, 1e-12)
    
    # Soft max
    m_max = np.max(sq_dists)
    log_sum_exp_max = m_max + np.log(np.sum(np.exp(tau * (sq_dists - m_max)))) / tau
    
    # Soft min
    m_min = np.min(sq_dists)
    log_sum_exp_min = m_min - np.log(np.sum(np.exp(-tau * (sq_dists - m_min)))) / tau
    
    obj = np.log(log_sum_exp_max) - np.log(log_sum_exp_min)
    return obj

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
    
    # Avoid log(0)
    sq_dists = np.maximum(sq_dists, 1e-12)
    
    # Soft max of squared distances
    m_max = np.max(sq_dists)
    log_sum_exp_max = m_max + np.log(np.sum(np.exp(tau * (sq_dists - m_max)))) / tau
    
    # Soft min of squared distances
    m_min = np.min(sq_dists)
    log_sum_exp_min = m_min - np.log(np.sum(np.exp(-tau * (sq_dists - m_min)))) / tau
    
    # Objective is log(max) - log(min)
    return np.log(log_sum_exp_max) - np.log(log_sum_exp_min)

def objective_min_dist(x, target_max=None):
    """
    Maximize the minimum squared distance, optionally with a penalty
    for exceeding target_max on any pairwise distance.
    This helps in a two-phase approach: first minimize max, then maximize min.
    """
    points = x.reshape(N, D)
    sq_dists = get_pairwise_sq_dists(points)
    sq_dists = np.maximum(sq_dists, 1e-12)
    
    m_min = np.min(sq_dists)
    
    if target_max is not None:
        # Penalty for distances exceeding target_max
        excess = np.maximum(sq_dists - target_max, 0.0)
        penalty = np.sum(excess**2)
        return -np.log(m_min) + penalty
    else:
        return -np.log(m_min)

def gram_objective(v, tau=50.0, rank_penalty=100.0):
    """
    Objective in terms of Gram matrix vector.
    v is a flattened symmetric matrix of size N x N.
    We enforce PSD and rank <= 3 via eigenvalue penalties.
    """
    G = v.reshape(N, N)
    # Symmetrize
    G = (G + G.T) / 2.0
    
    # Compute eigenvalues for PSD and rank check
    eigvals = np.linalg.eigvalsh(G)
    
    # Penalty for negative eigenvalues (PSD constraint)
    neg_eigs = np.clip(-eigvals, 0, None)
    psd_penalty = rank_penalty * np.sum(neg_eigs**2)
    
    # Penalty for rank > 3 (keep only top 3 eigenvalues, penalize the rest)
    # Sort eigenvalues in descending order
    eigvals_sorted = np.sort(eigvals)[::-1]
    # Penalize eigenvalues beyond the 3rd
    rank_penalty_val = rank_penalty * np.sum(eigvals_sorted[3:]**2)
    
    # Compute squared distances from Gram matrix
    # d_ij^2 = G_ii + G_jj - 2*G_ij
    sq_dists_full = G[np.diag_indices_from(G)][:, np.newaxis] + G[np.diag_indices_from(G)][np.newaxis, :] - 2 * G
    iu = np.triu_indices(N, k=1)
    sq_dists = sq_dists_full[iu]
    
    # Avoid log(0)
    sq_dists = np.maximum(sq_dists, 1e-12)
    
    # Soft max and min
    m_max = np.max(sq_dists)
    log_sum_exp_max = m_max + np.log(np.sum(np.exp(tau * (sq_dists - m_max)))) / tau
    
    m_min = np.min(sq_dists)
    log_sum_exp_min = m_min - np.log(np.sum(np.exp(-tau * (sq_dists - m_min)))) / tau
    
    obj = np.log(log_sum_exp_max) - np.log(log_sum_exp_min)
    
    return obj + psd_penalty + rank_penalty_val

def gram_to_points(G):
    """Convert Gram matrix to point coordinates via eigendecomposition."""
    G = (G + G.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(G)
    # Keep top 3 eigenvalues
    top3 = np.argsort(eigvals)[::-1][:3]
    eigvals_top = eigvals[top3]
    eigvecs_top = eigvecs[:, top3]
    # Ensure non-negative
    eigvals_top = np.maximum(eigvals_top, 0)
    points = eigvecs_top * np.sqrt(eigvals_top)
    return points

def d3d_symmetric_points(r1, r2, r3):
    """
    Generate 14 points using D_{3d} symmetry with 3 independent radial parameters.
    
    D_{3d} has order 12. We use:
    - 6 points on a hexagonal ring at radius r1, height 0
    - 4 points on a square ring at radius r2, height h
    - 2 points on the z-axis at heights ±r3
    
    Total: 6 + 4 + 2 = 12... we need 14.
    
    Let me reconsider. D_{3d} has 12 elements. For 14 points, we can use:
    - 6 points from one orbit (hexagonal arrangement)
    - 6 points from another orbit (hexagonal arrangement at different height)
    - 2 points on the z-axis
    
    Or use a different symmetry group. Let me use D_{6h} which has 12 elements,
    and place points in orbits.
    
    Actually, let me use a simpler approach: place 14 points with 3-fold symmetry.
    14 = 6 + 6 + 2 (two hexagonal rings + 2 axial points)
    """
    # 6 points on hexagonal ring 1 at height -h
    angles1 = np.linspace(0, 2*np.pi, 6, endpoint=False)
    pts1 = np.column_stack([
        r1 * np.cos(angles1),
        r1 * np.sin(angles1),
        np.full(6, -r3)
    ])
    
    # 6 points on hexagonal ring 2 at height +h
    angles2 = np.linspace(np.pi/6, 2*np.pi + np.pi/6, 6, endpoint=False)
    pts2 = np.column_stack([
        r2 * np.cos(angles2),
        r2 * np.sin(angles2),
        np.full(6, r3)
    ])
    
    # 2 points on z-axis
    pts3 = np.array([[0, 0, -r3*1.5], [0, 0, r3*1.5]])
    
    return np.vstack([pts1, pts2, pts3])

def d3d_objective(params):
    """Objective for D_{3d} symmetric configuration."""
    r1, r2, r3 = params
    if r1 < 0 or r2 < 0 or r3 < 0:
        return 1e10
    pts = d3d_symmetric_points(r1, r2, r3)
    return raw_score(pts)

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
        score = raw_score(pts)
        if score < best_score:
            best_score = score
            best_points = pts.copy()
    
    rng = np.random.default_rng(42)
    
    # --- Strategy 0: Laplacian null space basis optimization ---
    # Build the graph and get basis vectors
    A = build_truncated_tetrahedron_graph()
    basis = get_laplacian_null_basis(A)
    k = basis.shape[1]
    
    # Optimize the (k, 3) mixing matrix using the basis representation
    # Initial: identity-like scaling
    M_init = np.eye(k, 3) * 0.5
    coeffs_init = M_init.flatten()
    
    n_basis_restarts = 30
    for i in range(n_basis_restarts):
        if time.time() - start_time > timeout - 60:
            break
        
        if i == 0:
            c0 = coeffs_init.copy()
        else:
            c0 = rng.normal(size=k * 3) * 0.5
        
        try:
            # Multi-stage optimization in basis space
            res_b1 = minimize(
                basis_objective,
                c0,
                args=(basis, 10.0),
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-6, 'gtol': 1e-4}
            )
            if res_b1.x is not None:
                pts_b1 = basis_to_points(res_b1.x, basis)
                update_best(pts_b1)
                
                if time.time() - start_time < timeout - 40:
                    res_b2 = minimize(
                        basis_objective,
                        res_b1.x,
                        args=(basis, 50.0),
                        method='L-BFGS-B',
                        options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6}
                    )
                    if res_b2.x is not None:
                        pts_b2 = basis_to_points(res_b2.x, basis)
                        update_best(pts_b2)
                        
                        if time.time() - start_time < timeout - 20:
                            res_b3 = minimize(
                                basis_objective,
                                res_b2.x,
                                args=(basis, 200.0),
                                method='L-BFGS-B',
                                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
                            )
                            if res_b3.x is not None:
                                pts_b3 = basis_to_points(res_b3.x, basis)
                                update_best(pts_b3)
                                
        except Exception:
            pass
    
    # --- Strategy 1: D_{3d} symmetry-constrained optimization ---
    
    # Grid search over the 3D parameter space
    r1_vals = np.linspace(0.5, 2.0, 10)
    r2_vals = np.linspace(0.5, 2.0, 10)
    r3_vals = np.linspace(0.2, 1.5, 8)
    
    best_sym_score = np.inf
    best_sym_params = None
    
    for r1 in r1_vals:
        if time.time() - start_time > timeout - 50:
            break
        for r2 in r2_vals:
            for r3 in r3_vals:
                try:
                    score = d3d_objective([r1, r2, r3])
                    if score < best_sym_score:
                        best_sym_score = score
                        best_sym_params = [r1, r2, r3]
                except Exception:
                    pass
    
    # Refine with Nelder-Mead around the best grid point
    if best_sym_params is not None:
        try:
            res_sym = minimize(
                d3d_objective,
                best_sym_params,
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if res_sym.x is not None:
                pts_sym = d3d_symmetric_points(*res_sym.x)
                update_best(pts_sym)
        except Exception:
            pass
    
    # Also try several random initial points for Nelder-Mead
    for _ in range(10):
        if time.time() - start_time > timeout - 30:
            break
        r1_init = rng.uniform(0.5, 2.0)
        r2_init = rng.uniform(0.5, 2.0)
        r3_init = rng.uniform(0.2, 1.5)
        try:
            res_rand = minimize(
                d3d_objective,
                [r1_init, r2_init, r3_init],
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if res_rand.x is not None:
                pts_rand = d3d_symmetric_points(*res_rand.x)
                update_best(pts_rand)
        except Exception:
            pass
    
    # --- Strategy 2: Coordinate-space optimization ---
    
    # Initial candidates
    # 1. Icosahedron + center + 1 point
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
    coord_candidates = [init_ico.flatten()]
    
    # 2. Random points on a sphere
    for i in range(2):
        pts = rng.normal(size=(N, D))
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        coord_candidates.append(pts.flatten())
        
    # 3. Fibonacci sphere
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
    
    # 4. Two-ring configuration (hexagonal rings)
    angles1 = np.linspace(0, 2*np.pi, 7, endpoint=False)
    ring1 = np.column_stack([np.cos(angles1), np.sin(angles1), np.full(7, -0.5)])
    angles2 = np.linspace(np.pi/7, 2*np.pi + np.pi/7, 7, endpoint=False)
    ring2 = np.column_stack([np.cos(angles2), np.sin(angles2), np.full(7, 0.5)])
    two_ring = np.vstack([ring1, ring2])
    coord_candidates.append(two_ring.flatten())

    # Coordinate-space optimization
    n_coord_restarts = 40
    for i in range(n_coord_restarts):
        if time.time() - start_time > timeout - 20:
            break
            
        if i < len(coord_candidates):
            x0 = coord_candidates[i]
        else:
            x0 = rng.uniform(-1, 1, size=(N * D))
            
        try:
            # Multi-stage optimization with increasing tau
            res1 = minimize(
                objective_smooth, 
                x0, 
                method='L-BFGS-B', 
                args=(10.0,),
                options={'maxiter': 80, 'ftol': 1e-6, 'gtol': 1e-4}
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
                        options={'maxiter': 150, 'ftol': 1e-8, 'gtol': 1e-6}
                    )
                    if res2.x is not None:
                        pts2 = res2.x.reshape(N, D)
                        update_best(pts2)
                        
                        if time.time() - start_time < timeout - 8:
                            res3 = minimize(
                                objective_smooth, 
                                res2.x, 
                                method='L-BFGS-B', 
                                args=(200.0,),
                                options={'maxiter': 250, 'ftol': 1e-10, 'gtol': 1e-8}
                            )
                            if res3.x is not None:
                                pts3 = res3.x.reshape(N, D)
                                update_best(pts3)
                                
        except Exception:
            pass
    
    # --- Strategy 3: Two-phase refinement ---
    # Phase 1: Minimize maximum distance (with scale normalization)
    # Phase 2: Maximize minimum distance subject to max distance constraint
    if best_points is not None and time.time() - start_time < timeout - 10:
        try:
            # Normalize current best to have max distance = 1
            pts_norm = best_points.copy()
            sq_dists_norm = get_pairwise_sq_dists(pts_norm)
            max_sq_norm = np.max(sq_dists_norm)
            if max_sq_norm > 1e-12:
                pts_norm = pts_norm / np.sqrt(max_sq_norm)
            
            # Phase 1: Try to reduce max distance further
            res_p1 = minimize(
                objective_smooth,
                pts_norm.flatten(),
                method='L-BFGS-B',
                args=(500.0,),
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res_p1.x is not None:
                pts_p1 = res_p1.x.reshape(N, D)
                update_best(pts_p1)
            
            # Phase 2: From the best, try to maximize min distance
            # while keeping max distance bounded
            if best_points is not None:
                sq_dists_best = get_pairwise_sq_dists(best_points)
                max_sq_best = np.max(sq_dists_best)
                
                # Use min_dist objective with target_max slightly above current max
                target_max = max_sq_best * 1.01
                
                res_p2 = minimize(
                    objective_min_dist,
                    best_points.flatten(),
                    args=(target_max,),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                if res_p2.x is not None:
                    pts_p2 = res_p2.x.reshape(N, D)
                    update_best(pts_p2)
                    
        except Exception:
            pass

    # --- Final polish: use the best points found and do a high-precision coordinate optimization ---
    if best_points is not None and time.time() - start_time < timeout - 3:
        try:
            # Multiple polish attempts with different tau values
            for tau_final in [200.0, 500.0, 1000.0]:
                if time.time() - start_time > timeout - 2:
                    break
                res_final = minimize(
                    objective_smooth, 
                    best_points.flatten(), 
                    method='L-BFGS-B', 
                    args=(tau_final,),
                    options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                if res_final.x is not None:
                    pts_final = res_final.x.reshape(N, D)
                    update_best(pts_final)
        except Exception:
            pass
    
    # One more attempt: try to improve via small perturbations + re-optimization
    if best_points is not None and time.time() - start_time < timeout - 2:
        for _ in range(5):
            if time.time() - start_time > timeout - 1:
                break
            try:
                perturbed = best_points + rng.normal(size=(N, D)) * 0.01
                res_pert = minimize(
                    objective_smooth,
                    perturbed.flatten(),
                    method='L-BFGS-B',
                    args=(500.0,),
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                if res_pert.x is not None:
                    pts_pert = res_pert.x.reshape(N, D)
                    update_best(pts_pert)
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
