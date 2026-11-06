"""
Minimize logarithmic energy of 100 points on the unit sphere.
Uses Riemannian L-BFGS with exponential map retraction,
multiple diverse initializations, Newton polish, and local search.
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
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(d2, 1.0)
    d = np.sqrt(d2)
    
    mask_upper = np.triu(np.ones((n, n), dtype=bool), k=1)
    d_upper = d[mask_upper]
    
    if np.any(d_upper < 1e-9):
        return 1e18, np.zeros_like(pts)
        
    energy = -np.sum(np.log(d_upper))
    
    inv_d2 = 1.0 / d2
    np.fill_diagonal(inv_d2, 0.0)
    
    grad = np.sum(inv_d2[:, :, np.newaxis] * diff, axis=1)
    
    radial = np.sum(grad * pts, axis=1, keepdims=True)
    grad = grad - radial * pts
    
    return energy, grad

def _hessian_tangent(pts):
    """
    Compute the Hessian of the energy restricted to the tangent space.
    For the log energy E = -sum log(d_ij), the Hessian on the sphere is complex.
    We use a finite-difference approach for the Newton step in the tangent space.
    """
    n = len(pts)
    # We'll compute the Hessian via finite differences of the gradient in tangent space
    # This is expensive but gives a Newton step
    # Actually, let's use an approximate Hessian via the formula
    # For log energy, the Hessian on R^3 is:
    # H_ij = sum_{k != i, k != j} ... complex
    # Let's just do a simple 2-point finite difference for the Newton direction
    # Or better: use the fact that for small steps, we can approximate
    
    # Actually, for Newton polish, we need to solve:
    # H * v = -g where H is the Hessian on the tangent space
    # Let's compute H using finite differences of the projected gradient
    
    eps = 1e-5
    dim = 3 * n
    
    # Flatten gradient
    _, g = _energy_and_grad(pts)
    g_flat = g.flatten()
    
    # Build Hessian by finite differences
    # This is O(n^2 * dim) which is expensive
    # For 100 points, dim=300, this would be 300 * (2*100^2) = 6M ops per component
    # Too expensive. Let's use a quasi-Newton or just do a few Newton-like steps
    # with a simplified Hessian approximation.
    
    # Alternative: use the diagonal of the Hessian for a Gauss-Newton step
    # Or just use L-BFGS with tighter tolerances
    
    return None  # We'll skip the full Hessian and use L-BFGS polish instead

def _exp_map(pts, tangent):
    """
    Exponential map on the unit sphere: move points along tangent vectors.
    For a unit vector p and tangent vector v (p·v=0),
    exp_p(v) = p*cos(||v||) + v/sin(||v||)*||v|| = p*cos(r) + v*sin(r)/r
    where r = ||v||.
    """
    norms = np.linalg.norm(tangent, axis=1, keepdims=True)
    # Avoid division by zero
    safe_norms = np.where(norms < 1e-15, 1.0, norms)
    cos_r = np.cos(norms)
    sin_r_over_r = np.sin(safe_norms) / safe_norms
    # Where norms are ~0, sin_r_over_r should be 1
    sin_r_over_r = np.where(norms < 1e-15, 1.0, sin_r_over_r)
    return pts * cos_r + tangent * sin_r_over_r

def _fibonacci_sphere(n):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, seed):
    """Generate random points on unit sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms

def _icosahedron_sphere(n, seed=0):
    """
    Generate points using icosahedron-based subdivision.
    Start with icosahedron, subdivide faces, project to sphere.
    """
    rng = np.random.RandomState(seed)
    
    # Icosahedron vertices
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1,  t,  0], [ 1,  t,  0], [-1, -t,  0], [ 1, -t,  0],
        [ 0, -1,  t], [ 0,  1,  t], [ 0, -1, -t], [ 0,  1, -t],
        [ t,  0, -1], [ t,  0,  1], [-t,  0, -1], [-t,  0,  1],
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    
    # Icosahedron faces (20 triangles)
    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ]
    
    pts = verts.copy()
    
    # Subdivide faces to get more points
    # Each subdivision of a triangle into 4 gives 3 new points
    # Start with 12 points, need 100
    # After 1 subdivision: 12 + 20*3 = 72
    # After 2 subdivisions: 72 + 80*3 = 312 (too many)
    # So do 1 full subdivision, then add more via random farthest-point
    
    # Subdivide each face once
    new_pts = []
    for face in faces:
        v0, v1, v2 = verts[face[0]], verts[face[1]], verts[face[2]]
        m01 = (v0 + v1) / 2.0
        m12 = (v1 + v2) / 2.0
        m20 = (v2 + v0) / 2.0
        new_pts.extend([m01, m12, m20])
    
    new_pts = np.array(new_pts)
    norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
    new_pts = new_pts / norms
    
    pts = np.vstack([pts, new_pts])
    
    # If we still need more points, add via farthest-point sampling
    while len(pts) < n:
        candidates = rng.randn(200, 3)
        candidates = candidates / np.linalg.norm(candidates, axis=1, keepdims=True)
        
        best_idx = 0
        best_min_dist = -1
        for c in candidates:
            dists = np.linalg.norm(pts - c, axis=1)
            min_d = np.min(dists)
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_idx = c
        
        pts = np.vstack([pts, best_idx.reshape(1, 3)])
    
    return pts[:n]

def _project_to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms

def _lbfgs_optimize(pts0, deadline, maxiter=5000, ftol=1e-14, gtol=1e-10):
    """
    Riemannian L-BFGS optimization using the exponential map as retraction.
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
                     'ftol': ftol, 'gtol': gtol, 'm': 20}
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
    Newton polish using finite-difference Hessian in tangent space.
    This is expensive but converges quadratically near the optimum.
    """
    pts = pts0.copy()
    n = len(pts)
    
    for it in range(maxiter):
        if time.time() > deadline - 2.0:
            break
        
        e, g = _energy_and_grad(pts)
        g_norm = np.linalg.norm(g)
        if g_norm < 1e-12:
            break
        
        # Compute Hessian via finite differences in tangent space
        eps = 1e-6
        dim = 3 * n
        H = np.zeros((dim, dim))
        
        g_flat = g.flatten()
        
        # Only compute Hessian for a subset of directions to save time
        # Use a random subset of 50 directions
        rng = np.random.RandomState(it)
        n_dirs = min(50, dim)
        dirs = rng.randn(n_dirs, dim)
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        
        # Ensure directions are in tangent space (approximately)
        # For each direction, compute finite difference gradient
        H_approx = np.zeros((n_dirs, n_dirs))
        
        for i in range(n_dirs):
            d = dirs[i]
            d_pts = d.reshape(-1, 3)
            # Project to tangent space
            radial = np.sum(d_pts * pts, axis=1, keepdims=True)
            d_pts = d_pts - radial * pts
            d_pts /= (np.linalg.norm(d_pts, axis=1, keepdims=True) + 1e-15)
            
            pts_plus = _project_to_sphere(pts + eps * d_pts)
            pts_minus = _project_to_sphere(pts - eps * d_pts)
            
            _, g_plus = _energy_and_grad(pts_plus)
            _, g_minus = _energy_and_grad(pts_minus)
            
            # Finite difference Hessian row
            h_row = (g_plus.flatten() - g_minus.flatten()) / (2 * eps)
            H_approx[i] = np.dot(h_row, dirs)
        
        # Solve for Newton step in the reduced space
        try:
            # Regularize
            H_reg = H_approx + 1e-6 * np.eye(n_dirs)
            g_reduced = np.dot(dirs, g_flat)
            step = np.linalg.solve(H_reg, -g_reduced)
            
            # Map back to full space
            newton_dir = np.dot(step, dirs)
            newton_dir_pts = newton_dir.reshape(-1, 3)
            
            # Project to tangent space
            radial = np.sum(newton_dir_pts * pts, axis=1, keepdims=True)
            newton_dir_pts = newton_dir_pts - radial * pts
            
            # Damp the step
            step_norm = np.linalg.norm(newton_dir_pts)
            if step_norm > 0.1:
                newton_dir_pts *= 0.1 / step_norm
            
            # Take the step using exponential map
            new_pts = _exp_map(pts, newton_dir_pts)
            new_e, _ = _energy_and_grad(new_pts)
            
            if new_e < e:
                pts = new_pts
            else:
                # Try a smaller step
                newton_dir_pts *= 0.5
                new_pts = _exp_map(pts, newton_dir_pts)
                new_e, _ = _energy_and_grad(new_pts)
                if new_e < e:
                    pts = new_pts
                else:
                    break
        except np.linalg.LinAlgError:
            break
    
    final_e, _ = _energy_and_grad(pts)
    return pts, final_e

def _local_search(pts0, deadline, n_restarts=5):
    """
    Try small perturbations of the current best solution to escape local minima.
    """
    best_pts = pts0.copy()
    best_energy = _energy_and_grad(pts0)[0]
    
    rng = np.random.RandomState(42)
    
    for restart in range(n_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Perturb points slightly
        noise = rng.randn(N, 3) * 0.01
        perturbed = _project_to_sphere(pts0 + noise)
        
        # Optimize from perturbed starting point
        pts_opt, e_opt = _lbfgs_optimize(perturbed, deadline, maxiter=2000, ftol=1e-12, gtol=1e-8)
        
        if e_opt < best_energy:
            best_energy = e_opt
            best_pts = pts_opt.copy()
    
    return best_pts, best_energy

def construct():
    """
    Construct 100 points on the unit sphere minimizing logarithmic energy.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s buffer before 150s limit

    best_pts = None
    best_energy = np.inf

    # 1. Diverse Initializations
    initializations = []
    
    # Fibonacci spiral - strong heuristic
    initializations.append(_fibonacci_sphere(N))
    
    # Icosahedron-based
    initializations.append(_icosahedron_sphere(N, seed=1))
    initializations.append(_icosahedron_sphere(N, seed=2))
    
    # Random starts with different seeds
    seeds = [42, 123, 777, 999, 2023, 555, 888, 111, 222, 333]
    for s in seeds:
        initializations.append(_random_sphere(N, s))

    # 2. Coarse Optimization Phase
    for i, pts0 in enumerate(initializations):
        if time.time() > deadline - 40.0:
            break
        
        pts_opt, e_opt = _lbfgs_optimize(pts0, deadline, maxiter=3000, ftol=1e-12, gtol=1e-8)
        
        if e_opt < best_energy:
            best_energy = e_opt
            best_pts = pts_opt.copy()

    if best_pts is None:
        return _fibonacci_sphere(N)

    # 3. Local Search Phase - try to escape local minima
    time_left = deadline - time.time()
    if time_left > 25.0:
        pts_ls, e_ls = _local_search(best_pts, deadline, n_restarts=4)
        if e_ls < best_energy:
            best_energy = e_ls
            best_pts = pts_ls

    # 4. Fine Polishing Phase
    time_left = deadline - time.time()
    if time_left > 20.0:
        pts_polished, e_polished = _lbfgs_optimize(best_pts, deadline, maxiter=5000, ftol=1e-15, gtol=1e-11)
        if e_polished < best_energy:
            best_energy = e_polished
            best_pts = pts_polished

    # 5. Newton Polish
    time_left = deadline - time.time()
    if time_left > 15.0:
        pts_newton, e_newton = _newton_polish(best_pts, deadline, maxiter=50)
        if e_newton < best_energy:
            best_energy = e_newton
            best_pts = pts_newton

    # 6. Ultra-Fine Polish
    time_left = deadline - time.time()
    if time_left > 10.0:
        pts_ultra, e_ultra = _lbfgs_optimize(best_pts, deadline, maxiter=10000, ftol=1e-16, gtol=1e-13)
        if e_ultra < best_energy:
            best_energy = e_ultra
            best_pts = pts_ultra

    # 7. Final Newton Polish
    time_left = deadline - time.time()
    if time_left > 5.0:
        pts_final, e_final = _newton_polish(best_pts, deadline, maxiter=20)
        if e_final < best_energy:
            best_energy = e_final
            best_pts = pts_final

    # 8. Final Check and Return
    best_pts = _project_to_sphere(best_pts)
    
    return best_pts
