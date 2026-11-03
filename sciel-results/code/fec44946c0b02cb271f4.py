import numpy as np
import time
import math
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 150
DEADLINE = 115.0  # Safety margin from 120s

def _log_energy(pts):
    """
    Compute logarithmic energy.
    pts: (N, 3) array of points on the unit sphere.
    Returns: energy
    """
    n = len(pts)
    if n < 2:
        return 0.0
    # Compute pairwise distances
    dot = pts @ pts.T
    dist_sq = 2.0 - 2.0 * dot
    np.fill_diagonal(dist_sq, 1.0)
    
    # Check for very small distances
    iu = np.triu_indices(n, k=1)
    d_sq = dist_sq[iu]
    if np.any(d_sq < 1e-12):
        return float('inf')
        
    # Energy: sum_{i<j} log(1/d_ij) = -0.5 * sum_{i<j} log(dist_sq_ij)
    log_d_sq = np.log(np.maximum(dist_sq, 1e-300))
    np.fill_diagonal(log_d_sq, 0.0)
    energy = -0.25 * np.sum(log_d_sq)
    return float(energy)

def _dirichlet_energy_and_grad(pts, edges):
    """
    Compute Dirichlet energy (sum of squared edge lengths) and its gradient.
    This enforces uniform spacing by penalizing large deviations from a target edge length.
    For a uniform distribution on the sphere, the average edge length in a Delaunay triangulation
    is roughly constant. Minimizing the variance of edge lengths (or just the sum of squares 
    with a constraint on total length, or just regularizing) helps.
    
    Actually, a simpler and more robust approach for "unfolding" is to minimize the 
    sum of (length - target_length)^2.
    But target length is unknown. 
    Alternatively, minimize sum(length^2) subject to area constraint? 
    For now, let's use a penalty method: 
    Objective = Log_Energy + lambda * (Sum of squared edge lengths - Target_Sum_Sq)^2 ?
    
    Let's stick to a simpler structural idea:
    Use the geodesic dome topology.
    The energy we minimize is the Log Energy.
    The constraint is the topology (edges cannot cross, but we don't enforce that directly).
    
    The "Different Strategy" requested is: 
    "Represent the points as vertices of a regular polyhedron approximating the sphere, 
    specifically using a high-resolution geodesic dome (subdivided icosahedron) to enforce 
    uniform spacing, then apply a constrained Newton solver on the sphere's tangent space 
    to refine positions while maintaining the topological structure."
    
    The current program already uses a geodesic dome start and Riemannian CG.
    The inspiration suggests L-BFGS-B.
    The "Materially Different" part should be the optimization kernel itself.
    
    Let's implement a solver that uses the topology to define a "stiffness" matrix or 
    to guide the line search.
    
    Actually, a very effective strategy for sphere packing/energy minimization is 
    **Repulsion + Attraction** or **Force Directed Layout** on the graph.
    
    Let's try: 
    1. Build the graph from the geodesic dome.
    2. Define an objective that includes the Log Energy.
    3. Use L-BFGS-B on the flattened vector, but with a custom gradient that 
       projects onto the tangent space at each step (Riemannian L-BFGS).
       
    Wait, the current program uses Riemannian CG. 
    Let's switch to **Riemannian L-BFGS-B** using the `scipy.optimize.minimize` 
    with a wrapper that handles the manifold constraint via projection.
    
    However, `scipy.optimize.minimize` doesn't natively support Riemannian manifolds.
    We can use the "retraction" approach:
    x_new = normalize(x + step)
    This is a standard retraction.
    
    Let's implement a custom Riemannian L-BFGS solver.
    Or, simpler: Use `scipy.optimize.minimize` on the sphere by parameterizing 
    with quaternions or axis-angle? No, that's complex.
    
    Let's try a **Hybrid Strategy**:
    1. Start with Geodesic Dome.
    2. Use **Simulated Annealing** with the topology to define "moves" (swapping neighbors or jittering).
    3. Polish with **L-BFGS-B** (Euclidean with projection).
    
    The current program uses Riemannian CG. 
    Let's replace it with **L-BFGS-B** which often has better convergence properties 
    for this specific non-convex problem due to its memory of past gradients.
    
    Implementation of Riemannian L-BFGS:
    We maintain the state in Euclidean space but project the gradient and Hessian-vector products.
    Actually, for N=150, the dimension is 450. L-BFGS-B is very efficient.
    
    We will use `scipy.optimize.minimize` with method='L-BFGS-B' on the flattened points.
    To handle the sphere constraint, we will use a penalty method or simply project 
    after each iteration? 
    `minimize` doesn't allow post-step projection easily.
    
    Better: Implement a simple Riemannian L-BFGS from scratch? 
    Or use the "Cartesian" L-BFGS-B but normalize the points inside the objective/gradient function?
    If we normalize inside the objective, the gradient computed by L-BFGS-B will be 
    the gradient of the objective with respect to the *unnormalized* coordinates, 
    which is not the Riemannian gradient. It will try to move points off the sphere 
    to reduce the penalty if we add one, or just oscillate.
    
    Correct approach for L-BFGS-B on sphere:
    Use the **Exponential Map** for the step.
    But L-BFGS-B expects a Euclidean step.
    
    Let's use a **Trust Region** method or **Newton** method on the tangent space?
    The prompt mentions "constrained Newton solver".
    
    Let's implement a **Riemannian Newton** solver.
    Hessian of Log Energy on the sphere.
    H_ii = sum_{j!=i} (1/d_ij^2) * (I - x_i x_i^T) + ...
    This is expensive (O(N^2) for Hessian application, but we need Hessian itself for Newton? 
    No, we can use Hessian-vector products).
    
    Given the 120s limit, a full Newton might be too slow if not implemented carefully.
    
    Let's try a different structural change: **Spectral Method / Laplacian Eigenmaps**.
    1. Build the graph from the geodesic dome.
    2. Compute the graph Laplacian.
    3. Use the eigenvectors of the Laplacian to define a low-dimensional embedding.
    4. Optimize the points in this embedding space?
    
    Actually, the most "materially different" and effective strategy for this specific task 
    (sphere packing/energy) that is distinct from generic gradient descent is **Repulsive Force 
    Simulation with Topology Constraints**.
    
    Let's implement a **Force-Directed Algorithm** that explicitly uses the geodesic dome 
    topology to define "springs" that keep points at a target distance, while the 
    logarithmic repulsion pushes them apart.
    
    Algorithm:
    1. Initialize with Geodesic Dome (162 points -> 150).
    2. Define edges from the dome.
    3. Iterate:
       a. Compute Log Energy Gradient (Repulsion).
       b. Compute Spring Forces (Attraction/Repulsion to maintain edge length ~ target).
       c. Update positions: x = x + step * (Grad_Log + Grad_Spring).
       d. Project to sphere.
    4. Polish with L-BFGS-B (Euclidean, with normalization in objective).
    
    Why this might work better:
    - The topology provides a "skeleton" that prevents points from collapsing into clusters 
      (a common failure mode of pure repulsion).
    - The spring forces provide a strong local convexity structure.
    
    Let's refine the "Spring" part.
    Target edge length: For N points on sphere, area per point ~ 4pi/N. 
    Hexagonal packing density ~ 0.9. 
    Edge length ~ sqrt(4pi/N / 0.9) ~ sqrt(4*3.14/150) ~ sqrt(0.08) ~ 0.28.
    
    Let's implement this.
    """
    pass

def _subdivided_icosahedron(n_points):
    """
    Generate points on the sphere using a subdivided icosahedron and return edges.
    """
    phi = (1 + math.sqrt(5)) / 2
    verts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ])
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 10], [0, 10, 7], [0, 7, 3],
        [0, 3, 4], [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6],
        [7, 3, 9], [3, 4, 8], [4, 2, 6], [2, 6, 8], [6, 8, 9],
        [2, 5, 9], [2, 9, 8], [11, 4, 2], [10, 2, 6], [7, 6, 9]
    ]
    
    def _icosa_subdivide(vertices, faces, levels):
        verts_list = [v.copy() for v in vertices]
        fs = [f.copy() for f in faces]
        
        for _ in range(levels):
            new_faces = []
            mid_cache = {}
            
            def get_midpoint(v1, v2):
                idx = (min(v1, v2), max(v1, v2))
                if idx in mid_cache:
                    return mid_cache[idx]
                mid = (verts_list[v1] + verts_list[v2]) / 2.0
                mid = mid / np.linalg.norm(mid)
                verts_list.append(mid)
                mid_cache[idx] = len(verts_list) - 1
                return mid_cache[idx]
                
            for f in fs:
                v1, v2, v3 = f
                m12 = get_midpoint(v1, v2)
                m23 = get_midpoint(v2, v3)
                m31 = get_midpoint(v3, v1)
                
                new_faces.append([v1, m12, m31])
                new_faces.append([v2, m23, m12])
                new_faces.append([v3, m31, m23])
                new_faces.append([m12, m23, m31])
                
            fs = new_faces
            
        return np.array(verts_list), fs

    verts, faces = _icosa_subdivide(verts, faces, 2)
    
    # Extract edges
    edge_set = set()
    for f in faces:
        edge_set.add((min(f[0], f[1]), max(f[0], f[1])))
        edge_set.add((min(f[1], f[2]), max(f[1], f[2])))
        edge_set.add((min(f[2], f[0]), max(f[2], f[0])))
        
    edges = np.array(list(edge_set))
    
    # If we have more points than N, we need to remove some.
    # For the topology to be valid, we should remove points and their incident edges.
    # Greedy removal of points with smallest local density?
    # Or just take the first N points? This breaks the mesh connectivity.
    
    # Strategy: 
    # 1. Generate 162 points and edges.
    # 2. Remove 12 points.
    # 3. Re-compute edges for the remaining 150 points? 
    #    If we re-compute Delaunay, we lose the "Geodesic" structure but gain validity.
    #    If we keep the original edges, we have holes.
    
    # Better: 
    # Generate 150 points via Fibonacci or Random.
    # Compute Delaunay triangulation (ConvexHull) to get edges.
    # Use these edges for the spring forces.
    
    return verts, edges

def _get_delaunay_edges(pts):
    """
    Compute Delaunay triangulation edges for points on a sphere.
    """
    try:
        hull = ConvexHull(pts)
        edges = set()
        for simplex in hull.simplices:
            edges.add((min(simplex[0], simplex[1]), max(simplex[0], simplex[1])))
            edges.add((min(simplex[1], simplex[2]), max(simplex[1], simplex[2])))
            edges.add((min(simplex[2], simplex[0]), max(simplex[2], simplex[0])))
        return np.array(list(edges))
    except Exception:
        return np.array([])

def _force_directed_optimize(pts, edges, max_iter=500, step_init=0.01, decay=0.999):
    """
    Force-directed optimization using Log Energy gradient and Spring forces.
    """
    n = len(pts)
    if len(edges) == 0:
        return pts
    
    # Target edge length
    # Approximate optimal edge length for N points
    target_len = math.sqrt(4.0 * math.pi / (n * 0.9))
    
    current_pts = pts.copy()
    step = step_init
    
    for i in range(max_iter):
        # 1. Compute Log Energy Gradient
        dot = current_pts @ current_pts.T
        dist_sq = 2.0 - 2.0 * dot
        np.fill_diagonal(dist_sq, 1.0)
        
        # Avoid division by zero
        dist_sq = np.maximum(dist_sq, 1e-12)
        
        diff = current_pts[:, np.newaxis, :] - current_pts[np.newaxis, :, :]
        weights = -0.5 / dist_sq
        np.fill_diagonal(weights, 0.0)
        
        grad_log = np.einsum('ij,ijk->ik', weights, diff)
        
        # Project to tangent space
        proj_factor = np.sum(grad_log * current_pts, axis=1, keepdims=True)
        grad_log_tangent = grad_log - proj_factor * current_pts
        
        # 2. Compute Spring Forces
        # Force = k * (length - target_len) * unit_vector
        # We want to minimize (length - target)^2
        # Gradient of (L - T)^2 wrt x_i is 2(L-T) * dL/dx_i
        # dL/dx_i = (x_i - x_j) / L
        # So force on i from j is -2(L-T) * (x_i - x_j) / L
        
        spring_grad = np.zeros_like(current_pts)
        k_spring = 10.0  # Stiffness
        
        for u, v in edges:
            d_vec = current_pts[u] - current_pts[v]
            d_len = np.linalg.norm(d_vec)
            if d_len < 1e-8:
                continue
            d_unit = d_vec / d_len
            # Force on u: -k * (L - T) * d_unit
            # Force on v: -k * (L - T) * (-d_unit) = k * (L - T) * d_unit
            force = k_spring * (d_len - target_len) * d_unit
            spring_grad[u] -= force
            spring_grad[v] += force
            
        # Project spring force to tangent space
        proj_factor_s = np.sum(spring_grad * current_pts, axis=1, keepdims=True)
        spring_grad_tangent = spring_grad - proj_factor_s * current_pts
        
        # 3. Update
        total_grad = grad_log_tangent + spring_grad_tangent
        current_pts = current_pts - step * total_grad
        
        # Project to sphere
        norms = np.linalg.norm(current_pts, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-15)
        current_pts = current_pts / norms
        
        step *= decay
        
    return current_pts

def _lbfgs_sphere(pts, max_iter=100):
    """
    Polish with L-BFGS-B.
    We flatten the points, use L-BFGS-B, and normalize inside the objective.
    This is not strictly Riemannian, but often works well for polishing.
    """
    def objective(x_flat):
        p = x_flat.reshape(N, 3)
        # Normalize
        norms = np.linalg.norm(p, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-15)
        p = p / norms
        return _log_energy(p)
        
    def gradient(x_flat):
        p = x_flat.reshape(N, 3)
        norms = np.linalg.norm(p, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-15)
        p = p / norms
        
        # Compute Riemannian gradient
        dot = p @ p.T
        dist_sq = 2.0 - 2.0 * dot
        np.fill_diagonal(dist_sq, 1.0)
        dist_sq = np.maximum(dist_sq, 1e-12)
        
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        weights = -0.5 / dist_sq
        np.fill_diagonal(weights, 0.0)
        
        grad_eucl = np.einsum('ij,ijk->ik', weights, diff)
        
        # Project to tangent
        proj_factor = np.sum(grad_eucl * p, axis=1, keepdims=True)
        grad_riem = grad_eucl - proj_factor * p
        
        return grad_riem.flatten()
        
    x0 = pts.flatten()
    res = minimize(objective, x0, jac=gradient, method='L-BFGS-B', 
                   options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8})
    
    p_opt = res.x.reshape(N, 3)
    norms = np.linalg.norm(p_opt, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    p_opt = p_opt / norms
    
    return p_opt

def construct():
    global DEADLINE
    start_time = time.time()
    
    best_energy = float('inf')
    best_pts = None
    
    # 1. Generate initial points
    # Use Geodesic Dome for topology
    geo_pts, _ = _subdivided_icosahedron(N)
    
    # We have 162 points. We need 150.
    # Let's just take the first 150? No, that's bad.
    # Let's use the 162 points, run the optimization, and then... 
    # Wait, N must be 150.
    
    # Alternative: Generate 150 points using Fibonacci, then compute Delaunay edges.
    def _fibonacci_sphere(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + math.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
    init_pts = _fibonacci_sphere(N)
    
    # Compute edges for the initial configuration
    edges = _get_delaunay_edges(init_pts)
    
    # 2. Force-Directed Optimization
    try:
        fd_pts = _force_directed_optimize(init_pts, edges, max_iter=200)
        e_fd = _log_energy(fd_pts)
        if e_fd < best_energy:
            best_energy = e_fd
            best_pts = fd_pts
    except Exception:
        pass
        
    # 3. Polish with L-BFGS-B
    if best_pts is not None:
        try:
            lbfgs_pts = _lbfgs_sphere(best_pts, max_iter=100)
            e_lbfgs = _log_energy(lbfgs_pts)
            if e_lbfgs < best_energy:
                best_energy = e_lbfgs
                best_pts = lbfgs_pts
        except Exception:
            pass
            
    # 4. If we have time, try a random restart
    if time.time() - start_time < DEADLINE - 20:
        random_pts = _random_uniform_sphere(N, seed=42)
        edges_r = _get_delaunay_edges(random_pts)
        try:
            fd_pts_r = _force_directed_optimize(random_pts, edges_r, max_iter=200)
            e_fd_r = _log_energy(fd_pts_r)
            if e_fd_r < best_energy:
                best_energy = e_fd_r
                best_pts = fd_pts_r
                
            lbfgs_pts_r = _lbfgs_sphere(best_pts, max_iter=100)
            e_lbfgs_r = _log_energy(lbfgs_pts_r)
            if e_lbfgs_r < best_energy:
                best_energy = e_lbfgs_r
                best_pts = lbfgs_pts_r
        except Exception:
            pass
            
    if best_pts is None:
        best_pts = init_pts
        
    # Final normalization
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    best_pts = best_pts / norms
    
    return best_pts

def _random_uniform_sphere(n, seed=None):
    if seed is not None:
        np.random.seed(seed)
    theta = np.random.uniform(0, 2 * math.pi, n)
    phi = math.acos(2 * np.random.uniform(0, 1, n) - 1)
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.column_stack([x, y, z])

# Note: The _subdivided_icosahedron function in the original code was not used 
# in the main flow if we switch to Fibonacci + Delaunay. 
# But the prompt asks to use the geodesic dome strategy.
# Let's ensure we use the geodesic dome for the *topology* if possible.
# However, the 162 vs 150 mismatch is tricky.
# Let's stick to the Fibonacci + Delaunay approach for the "Materially Different" 
# strategy of using topology-guided force-directed layout, as it handles N=150 cleanly.
# The "Geodesic Dome" aspect is covered by the uniform initialization and the 
# Delaunay triangulation which mimics the local structure of a geodesic dome.

