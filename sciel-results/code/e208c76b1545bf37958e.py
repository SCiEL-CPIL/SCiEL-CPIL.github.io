import numpy as np
import time
from scipy.optimize import minimize

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

def get_basis_vectors():
    """
    Generate 14 orthonormal basis vectors in R^3 using the null space of the
    Laplacian of a candidate high-symmetry graph (truncated tetrahedron).
    
    The truncated tetrahedron has 12 vertices. We'll use its symmetry to
    generate basis vectors, then add 2 more to reach 14.
    
    For simplicity, we use the vertices of a regular icosahedron (12 vertices)
    plus 2 additional points, then orthogonalize to get 14 orthonormal vectors
    in R^3. Since we can't have 14 orthonormal vectors in R^3, we interpret
    this as: each point is a linear combination of 3 basis vectors (the
    standard basis), but we constrain the search to a low-dimensional
    parameterization.
    
    Actually, the directive says "linear combination of 14 orthonormal basis
    vectors derived from the null space of the Laplacian". This is mathematically
    impossible in R^3 (max 3 orthonormal vectors). 
    
    Reinterpreting: We'll use a low-dimensional parameterization where each
    point is defined by a small number of parameters, inspired by the symmetry
    of the truncated tetrahedron. The truncated tetrahedron has 12 vertices
    with high symmetry. We'll place 12 points at the vertices of a scaled
    truncated tetrahedron, and 2 additional points.
    
    For the 12 vertices of a truncated tetrahedron, we can use:
    - 4 triangles (from truncating the 4 vertices of a regular tetrahedron)
    - 4 hexagons (from the 6 edges, but truncated gives 4 hexagons? No, 6 hexagons)
    
    Actually, let me use a simpler approach: use the icosahedron (12 vertices)
    as the base, which has high symmetry, and add 2 more points.
    """
    # 12 vertices of a regular icosahedron
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    
    # 2 additional points (e.g., along z-axis)
    extra = np.array([[0, 0, 1.5], [0, 0, -1.5]])
    
    all_points = np.vstack([ico, extra])
    return all_points

def low_dim_objective(params):
    """
    Objective function for low-dimensional parameterization.
    params: array of shape (N, 3) representing coefficients for each point.
    We'll use a simple scaling approach: each point is scaled by a factor.
    """
    # Interpret params as scaling factors for the base configuration
    # This is a simplified approach - we scale each point individually
    base = get_basis_vectors()
    points = base * params[:, np.newaxis]
    
    # Ensure points are distinct
    sq_dists = get_pairwise_sq_dists(points)
    if np.min(sq_dists) < 1e-12:
        return 1e10
    
    return raw_score(points)

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
    
    # --- Strategy 0: Low-dimensional basis expansion optimization ---
    # Start with the icosahedron + 2 extra points configuration
    base = get_basis_vectors()
    
    # Try uniform scaling
    for scale in np.linspace(0.5, 2.0, 10):
        pts = base * scale
        update_best(pts)
    
    # Try individual scaling of each point
    # params[i] is the scaling factor for point i
    init_params = np.ones(N)
    
    # Optimize the scaling parameters
    try:
        res_scale = minimize(
            low_dim_objective,
            init_params,
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6}
        )
        if res_scale.x is not None:
            pts_scale = base * res_scale.x[:, np.newaxis]
            update_best(pts_scale)
    except Exception:
        pass
    
    # --- Strategy 1: Coordinate-space optimization ---
    
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
    for i in range(3):
        pts = rng.normal(size=(N, D))
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        coord_candidates.append(pts.flatten())
        
    # 3. Random points in a cube
    for i in range(3):
        pts = rng.uniform(-1, 1, size=(N, D))
        coord_candidates.append(pts.flatten())

    # 4. Fibonacci sphere
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
    n_coord_restarts = 50
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
                        
                        if time.time() - start_time < timeout - 10:
                            res3 = minimize(
                                objective_smooth, 
                                res2.x, 
                                method='L-BFGS-B', 
                                args=(200.0,),
                                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
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
