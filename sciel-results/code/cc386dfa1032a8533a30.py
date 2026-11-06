"""
Minimize logarithmic energy of 100 points on the unit sphere.
Uses L-BFGS-B with multiple diverse initializations and high-precision polishing.
"""
import numpy as np
import math
import time
from scipy.optimize import minimize

N = 100

def _energy_and_grad(pts):
    """
    Compute the logarithmic energy and its gradient for points on the sphere.
    Energy: sum_{i<j} log(1/|p_i - p_j|)
    Gradient: sum_{j!=i} (p_i - p_j) / |p_i - p_j|^2, projected to tangent space.
    """
    n = len(pts)
    # Compute pairwise differences
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    
    # Compute squared distances
    d2 = np.sum(diff * diff, axis=2)
    
    # Avoid division by zero for i=j (distance 0)
    # We will mask these out in summation
    np.fill_diagonal(d2, 1.0)
    
    # Distances
    d = np.sqrt(d2)
    
    # Energy calculation
    # We only need i < j
    mask_upper = np.triu(np.ones((n, n), dtype=bool), k=1)
    d_upper = d[mask_upper]
    
    # Check for extremely small distances (points coinciding)
    if np.any(d_upper < 1e-9):
        # This is bad, return high energy to push away
        return 1e18, np.zeros_like(pts)
        
    energy = -np.sum(np.log(d_upper))
    
    # Gradient calculation
    # grad_i = sum_{j != i} (p_i - p_j) / |p_i - p_j|^2
    # Note: diff[i,j] = p_i - p_j
    # So term is diff[i,j] / d2[i,j]
    
    inv_d2 = 1.0 / d2
    np.fill_diagonal(inv_d2, 0.0)
    
    # grad_i = sum_j (p_i - p_j) / d2_ij
    # Vectorized: sum over j (axis=1)
    grad = np.sum(inv_d2[:, :, np.newaxis] * diff, axis=1)
    
    # Project gradient onto tangent space
    # p_i is unit vector, so normal is p_i
    radial = np.sum(grad * pts, axis=1, keepdims=True)
    grad = grad - radial * pts
    
    return energy, grad

def _hessian_tangent(pts):
    """
    Compute the Hessian of the logarithmic energy restricted to the tangent space.
    For a single point i, the Hessian in tangent directions is:
    H_ij = sum_{k != i} [ (delta_ij * I - 2*(p_i - p_k)(p_i - p_k)^T / d2) / d2 
                          + 2*(p_i - p_k)(p_i - p_k)^T / d2^2 - 2*(p_i - p_k)(p_i - p_k)^T / d2^2 ]
    
    Actually, let's be more careful. The energy is E = sum_{i<j} -log|p_i - p_j|.
    For point i, the gradient contribution from pair (i,j) is (p_i - p_j)/|p_i - p_j|^2.
    
    The Hessian of -log|p_i - p_j| with respect to p_i (in R^3) is:
    H = I/d2 - 2*(p_i-p_j)(p_i-p_j)^T/d2^2
    
    When projected to tangent space at p_i, we use P = I - p_i p_i^T.
    The tangent Hessian is P * H * P.
    
    For computational efficiency, we'll compute the full Hessian block for each point.
    """
    n = len(pts)
    H = np.zeros((n, 3, n, 3))
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(d2, 1.0)
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d2_ij = d2[i, j]
            d_ij = diff[i, j]
            # Hessian of -log|p_i - p_j| w.r.t. p_i
            # = I/d2 - 2*d*d^T/d2^2
            H_block = np.eye(3) / d2_ij - 2.0 * np.outer(d_ij, d_ij) / (d2_ij * d2_ij)
            # Project to tangent space: P * H_block * P where P = I - p_i p_i^T
            P = np.eye(3) - np.outer(pts[i], pts[i])
            H_block = P @ H_block @ P
            H[i, :, j, :] = H_block
    
    return H

def _fibonacci_sphere(n):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _icosahedron_sphere(n):
    """
    Generate points on the unit sphere using an icosahedron-based approach.
    Start with icosahedron vertices and subdivide faces recursively.
    """
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2
    
    # Icosahedron vertices
    a = 1.0 / phi
    vertices = np.array([
        [-1, a, 0], [1, a, 0], [-1, -a, 0], [1, -a, 0],
        [0, -1, a], [0, 1, a], [0, -1, -a], [0, 1, -a],
        [a, 0, -1], [a, 0, 1], [-a, 0, -1], [-a, 0, 1]
    ], dtype=float)
    
    # Normalize
    vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
    
    # Icosahedron faces (12 faces... actually 20 faces)
    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ]
    
    # Subdivide faces to get more points
    # We need n points total. Start with 12 vertices.
    # Each subdivision of a face adds 3 new points (midpoints of edges).
    # But we need to be careful about shared edges.
    
    # For simplicity, let's use a different approach: 
    # Generate points by recursively subdividing and taking centroids
    if n <= 12:
        # Just use first n vertices
        return vertices[:n]
    
    # Use recursive subdivision
    # Each face has 3 edges. Midpoint of edge (a,b) is normalize((a+b)/2)
    # We'll track edges to avoid duplicates
    
    points = [v.copy() for v in vertices]
    point_set = set()
    for p in points:
        point_set.add(tuple(np.round(p, 10)))
    
    def get_midpoint(a, b):
        key = tuple(sorted([tuple(np.round(a, 10)), tuple(np.round(b, 10))]))
        if key in edge_midpoints:
            return edge_midpoints[key]
        mid = (a + b) / 2.0
        mid = mid / np.linalg.norm(mid)
        edge_midpoints[key] = mid
        points.append(mid)
        point_set.add(tuple(np.round(mid, 10)))
        return mid
    
    edge_midpoints = {}
    
    # Subdivide faces until we have enough points
    current_faces = [list(f) for f in faces]
    
    while len(points) < n:
        new_faces = []
        for face in current_faces:
            # Get the 3 vertices
            v0, v1, v2 = [points[i] for i in face]
            
            # Get midpoints of edges
            m01 = get_midpoint(v0, v1)
            m12 = get_midpoint(v1, v2)
            m20 = get_midpoint(v2, v0)
            
            # Create 4 sub-faces
            new_faces.append([v0, m01, m20])
            new_faces.append([m01, v1, m12])
            new_faces.append([m20, m12, v2])
            new_faces.append([m01, m12, m20])
            
            if len(points) >= n:
                break
        current_faces = new_faces
        if len(points) >= n:
            break
    
    # Return first n points
    return np.array(points[:n])

def _random_sphere(n, seed):
    """Generate random points on unit sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms

def _project_to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms

def _optimize_lbfgs(pts0, deadline, maxiter=5000):
    """
    Optimize using L-BFGS-B.
    """
    x0 = pts0.flatten()
    
    def objective(x):
        pts = x.reshape(-1, 3)
        pts = _project_to_sphere(pts)
        e, _ = _energy_and_grad(pts)
        return e

    def grad(x):
        pts = x.reshape(-1, 3)
        pts = _project_to_sphere(pts)
        _, g = _energy_and_grad(pts)
        return g.flatten()

    try:
        result = minimize(
            objective, x0, jac=grad, method='L-BFGS-B',
            options={'maxiter': maxiter, 'maxfun': maxiter * 10, 
                     'ftol': 1e-15, 'gtol': 1e-10, 'm': 20}
        )
        if result.success or result.fun < objective(x0):
            best_pts = _project_to_sphere(result.x.reshape(-1, 3))
            best_energy = _energy_and_grad(best_pts)[0]
            return best_pts, best_energy
        else:
            return pts0, _energy_and_grad(pts0)[0]
    except Exception:
        return pts0, _energy_and_grad(pts0)[0]

def _newton_polish(pts0, deadline, maxiter=100):
    """
    Newton-Raphson polish using the tangent-space Hessian.
    This should converge quadratically to machine precision.
    """
    pts = pts0.copy()
    n = len(pts)
    
    for it in range(maxiter):
        if time.time() > deadline - 5.0:
            break
        
        # Compute gradient
        e, grad = _energy_and_grad(pts)
        
        # Check convergence
        if np.max(np.abs(grad)) < 1e-14:
            break
        
        # Compute Hessian (this is expensive, so limit iterations)
        H = _hessian_tangent(pts)
        
        # For each point, solve the 3x3 linear system in tangent space
        # H_i * delta_i = -grad_i
        # where H_i is the 3x3 block for point i (but we need to handle coupling)
        
        # Actually, the full Hessian couples all points. Let's do a simplified version:
        # For each point i, the diagonal block H[i,:,i,:] gives the local curvature.
        # We'll use a damped Newton step with only the diagonal blocks.
        
        delta = np.zeros_like(pts)
        for i in range(n):
            H_i = H[i, :, i, :]
            # Add small regularization for numerical stability
            H_i_reg = H_i + 1e-10 * np.eye(3)
            try:
                step = np.linalg.solve(H_i_reg, -grad[i])
            except np.linalg.LinAlgError:
                step = -0.01 * grad[i]
            
            # Ensure step is in tangent space
            radial = np.dot(step, pts[i])
            step = step - radial * pts[i]
            
            delta[i] = step
        
        # Damped update: move by alpha * delta
        # Use line search to find good alpha
        best_alpha = 1.0
        best_e = e
        for alpha in [1.0, 0.5, 0.25, 0.1, 0.05, 0.01]:
            new_pts = pts + alpha * delta
            new_pts = _project_to_sphere(new_pts)
            new_e, _ = _energy_and_grad(new_pts)
            if new_e < best_e:
                best_e = new_e
                best_alpha = alpha
        
        pts = pts + best_alpha * delta
        pts = _project_to_sphere(pts)
        
        if best_alpha < 0.01:
            # Stuck, try smaller step
            break
    
    return pts, _energy_and_grad(pts)[0]

def _fine_polish(pts0, deadline, maxiter=10000):
    """
    High precision polishing using L-BFGS-B with very tight tolerances.
    """
    x0 = pts0.flatten()
    
    def objective(x):
        pts = x.reshape(-1, 3)
        pts = _project_to_sphere(pts)
        e, _ = _energy_and_grad(pts)
        return e

    def grad(x):
        pts = x.reshape(-1, 3)
        pts = _project_to_sphere(pts)
        _, g = _energy_and_grad(pts)
        return g.flatten()

    try:
        result = minimize(
            objective, x0, jac=grad, method='L-BFGS-B',
            options={'maxiter': maxiter, 'maxfun': maxiter * 10,
                     'ftol': 1e-16, 'gtol': 1e-12, 'm': 10}
        )
        if result.success or result.fun < objective(x0):
            best_pts = _project_to_sphere(result.x.reshape(-1, 3))
            best_energy = _energy_and_grad(best_pts)[0]
            return best_pts, best_energy
        else:
            return pts0, _energy_and_grad(pts0)[0]
    except Exception:
        return pts0, _energy_and_grad(pts0)[0]

def construct():
    """
    Construct 100 points on the unit sphere minimizing logarithmic energy.
    """
    start_time = time.time()
    # Safety margin: evaluator kills at 150s. Keep 10s buffer.
    deadline = start_time + 140.0

    best_pts = None
    best_energy = np.inf

    # 1. Diverse Initializations
    # Fibonacci sphere is a very strong heuristic for this problem.
    initializations = []
    
    # Primary: Fibonacci
    initializations.append(_fibonacci_sphere(N))
    
    # Icosahedron-based initialization
    try:
        initializations.append(_icosahedron_sphere(N))
    except Exception:
        pass
    
    # Secondary: Random starts with different seeds
    seeds = [42, 123, 777, 999, 2023]
    for s in seeds:
        initializations.append(_random_sphere(N, s))

    # 2. Optimization Loop
    for i, pts0 in enumerate(initializations):
        if time.time() > deadline - 20.0:
            break
        
        # Standard L-BFGS optimization
        pts_opt, e_opt = _optimize_lbfgs(pts0, deadline, maxiter=3000)
        
        if e_opt < best_energy:
            best_energy = e_opt
            best_pts = pts_opt.copy()
            
        # Early termination if we hit a very low energy (unlikely but possible)
        # The record is around -1083.37
        if best_energy < -1083.377:
            # We have beaten the published record (if true) or are very close.
            # Continue polishing below.
            pass

    if best_pts is None:
        # Fallback: return Fibonacci if optimization failed
        return _fibonacci_sphere(N)

    # 3. Polishing Phase
    # Use remaining time to refine the best candidate
    time_left = deadline - time.time()
    
    if time_left > 10.0:
        # First, try Newton polish (fast quadratic convergence)
        pts_newton, e_newton = _newton_polish(best_pts, deadline, maxiter=50)
        if e_newton < best_energy:
            best_energy = e_newton
            best_pts = pts_newton
        
        # Then, high precision L-BFGS polish
        time_left = deadline - time.time()
        if time_left > 5.0:
            pts_polished, e_polished = _fine_polish(best_pts, deadline, maxiter=5000)
            if e_polished < best_energy:
                best_energy = e_polished
                best_pts = pts_polished
        
        # Final Newton polish
        time_left = deadline - time.time()
        if time_left > 2.0:
            pts_final, e_final = _newton_polish(best_pts, deadline, maxiter=30)
            if e_final < best_energy:
                best_energy = e_final
                best_pts = pts_final

    # 4. Final Check and Return
    # Ensure points are strictly on the sphere and valid
    best_pts = _project_to_sphere(best_pts)
    
    return best_pts
