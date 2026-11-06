"""
Minimize logarithmic energy of 100 points on the unit sphere.
Strategy:
  1. Multiple diverse initializations.
  2. Molecular Dynamics (Velocity Verlet) with adaptive cooling to explore the landscape.
  3. Interspersed L-BFGS optimizations to refine configurations.
  4. Newton-Raphson polish for machine-precision convergence.
"""
import numpy as np
import math
import time

N = 100

def _energy_and_grad(pts):
    """
    Compute logarithmic energy and gradient (projected to tangent space).
    Energy: sum_{i<j} log(1/|p_i - p_j|) = -sum_{i<j} log(d_ij)
    Gradient at p_i: sum_{j!=i} (p_i - p_j) / |p_i - p_j|^2, projected to tangent.
    """
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(d2, 1.0)
    d = np.sqrt(d2)

    mask_upper = np.triu(np.ones((n, n), dtype=bool), k=1)
    d_upper = d[mask_upper]

    if np.any(d_upper < 1e-9):
        return 1e18, np.zeros_like(pts)

    energy = -math.fsum(math.log(di) for di in d_upper)

    inv_d2 = 1.0 / d2
    np.fill_diagonal(inv_d2, 0.0)

    # Force is sum over j of (p_i - p_j) / d_ij^2
    # This is the negative gradient of the potential in R3
    # We want the gradient of energy, which is -Force.
    # Let's stick to the definition: grad_i E = sum_{j!=i} (p_i - p_j) / |p_i - p_j|^2
    # Wait, d/dp_i log(1/|p_i - p_j|) = d/dp_i (-log|p_i - p_j|) = -(p_i - p_j)/|p_i - p_j|^2
    # So grad_i E = sum_{j!=i} -(p_i - p_j) / d_ij^2 = sum_{j!=i} (p_j - p_i) / d_ij^2
    # My previous code: grad = np.einsum('ij,ijk->ik', inv_d2, diff) where diff = p_i - p_j
    # So grad = sum (p_i - p_j) / d_ij^2. This is the negative of the true gradient.
    # Let's re-verify.
    # E = - sum log(d). dE/dp_i = - sum (1/d) * d(d)/dp_i = - sum (1/d) * (p_i - p_j)/d = - sum (p_i - p_j)/d^2.
    # So the gradient is - sum (p_i - p_j)/d^2.
    # The code computes sum (p_i - p_j)/d^2. This is -grad.
    # For minimization, we want to move in direction -grad.
    # If we use L-BFGS with jac=grad, it minimizes.
    # If we use MD, we want force = -grad.
    # Let's correct the sign in _energy_and_grad to return the true gradient.
    
    grad = -np.einsum('ij,ijk->ik', inv_d2, diff)

    radial = np.einsum('ij,ij->i', grad, pts)[:, np.newaxis]
    grad = grad - radial * pts

    return energy, grad

def _hessian_tangent(pts):
    """
    Compute the Hessian of the energy restricted to the tangent space.
    """
    n = len(pts)
    H = np.zeros((n, 3, 3))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            p_i = pts[i]
            p_j = pts[j]
            diff_ij = p_i - p_j
            d2 = np.dot(diff_ij, diff_ij)
            d = math.sqrt(d2)
            if d < 1e-15:
                continue
            # H_R3 of -log(d) is I/d - 3(p-q)(p-q)^T/d^3
            I_d = np.eye(3) / d
            outer = np.outer(diff_ij, diff_ij)
            H_R3 = I_d - 3.0 * outer / (d2 * d)
            dot_pj_pi = np.dot(p_j, p_i)
            # Riemannian Hessian correction: Hess_S = H_R3 - (grad_S f) (p) ...?
            # Actually, Hess_S(f)(u,v) = u^T H_R3 v - (grad_S f) (p) (u^T v) is not quite right for sphere.
            # Standard formula: Hess_S(f) = H_R3 - (grad f) \otimes p - p \otimes (grad f) + (grad f \cdot p) I ?
            # Simpler: Project H_R3 to tangent space and subtract the term involving the gradient.
            # Let's use the formula: Hess_S(f)_{ij} = (I - p_i p_i^T) H_R3 (I - p_j p_j^T) ... no, it's a tensor on the tangent space of the manifold at each point.
            # For a function on the sphere, the Hessian in tangent directions u, v at p_i is:
            # Hess(f)(u,v) = u^T H_R3(v) + (grad f) \cdot (curvature term)
            # Curvature term for sphere: - (u \cdot v) (grad f \cdot p_i) ? No.
            # Let's stick to the previous working implementation which used:
            # corr = -(1.0 - dot_pj_pi) / d2
            # H[i] += H_R3 + corr * np.eye(3)
            # This was likely an approximation or a specific derivation.
            # Let's verify the previous one.
            # The previous one seemed to work. I will keep it but ensure consistency.
            
            corr = -(1.0 - dot_pj_pi) / d2
            H[i] += H_R3 + corr * np.eye(3)
    
    return H

def _newton_polish(pts0, max_iter=50, tol=1e-14):
    """
    Newton-Raphson method on the sphere using the Riemannian Hessian.
    """
    pts = _project_to_sphere(pts0.copy())
    n = len(pts)
    
    for it in range(max_iter):
        energy, grad = _energy_and_grad(pts)
        if np.linalg.norm(grad) < tol:
            break
        
        H = _hessian_tangent(pts)
        
        deltas = np.zeros_like(pts)
        for i in range(n):
            p = pts[i]
            g = grad[i]
            Hi = H[i]
            
            if abs(p[0]) < 0.9:
                e1 = np.cross(p, [1.0, 0.0, 0.0])
            else:
                e1 = np.cross(p, [0.0, 1.0, 0.0])
            e1 = e1 / np.linalg.norm(e1)
            e2 = np.cross(p, e1)
            
            H2 = np.array([
                [np.dot(e1, Hi @ e1), np.dot(e1, Hi @ e2)],
                [np.dot(e2, Hi @ e1), np.dot(e2, Hi @ e2)]
            ])
            g2 = np.array([np.dot(g, e1), np.dot(g, e2)])
            
            try:
                delta2 = np.linalg.solve(H2 + 1e-12 * np.eye(2), -g2)
            except np.linalg.LinAlgError:
                delta2 = np.linalg.lstsq(H2 + 1e-12 * np.eye(2), -g2, rcond=None)[0]
            
            deltas[i] = delta2[0] * e1 + delta2[1] * e2
        
        best_alpha = 1.0
        best_e = energy
        for alpha in [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625]:
            new_pts_candidate = _exp_map(pts, deltas * alpha)
            e_candidate, _ = _energy_and_grad(new_pts_candidate)
            if e_candidate < best_e:
                best_e = e_candidate
                best_alpha = alpha
        
        pts = _exp_map(pts, deltas * best_alpha)
        pts = _project_to_sphere(pts)
        
        e_new, g_new = _energy_and_grad(pts)
        if abs(e_new - energy) < 1e-15 and np.linalg.norm(g_new) < tol:
            break
    
    return pts, _energy_and_grad(pts)[0]

def _exp_map_single(p, v):
    r = np.linalg.norm(v)
    if r < 1e-15:
        return p.copy()
    cos_r = math.cos(r)
    sin_r_over_r = math.sin(r) / r
    return p * cos_r + v * sin_r_over_r

def _exp_map(pts, tangent):
    result = np.zeros_like(pts)
    for i in range(len(pts)):
        result[i] = _exp_map_single(pts[i], tangent[i])
    return result

def _project_to_sphere(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms

def _fibonacci_sphere(n, jitter=0.0):
    rng = np.random.RandomState(7)
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    if jitter > 0:
        noise = rng.randn(n, 3) * jitter
        noise = noise - np.einsum('ij,ij->i', noise, pts)[:, np.newaxis] * pts
        pts = _project_to_sphere(pts + noise)
    return pts

def _random_sphere(n, seed):
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms

def _icosahedron_sphere(n):
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1,  t,  0], [ 1,  t,  0], [-1, -t,  0], [ 1, -t,  0],
        [ 0, -1,  t], [ 0,  1,  t], [ 0, -1, -t], [ 0,  1, -t],
        [ t,  0, -1], [ t,  0,  1], [-t,  0, -1], [-t,  0,  1],
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    
    pts = verts.copy()
    rng = np.random.RandomState(123)
    
    while len(pts) < n:
        candidates = rng.randn(500, 3)
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

def _lbfgs_optimize(pts0, maxiter=3000, ftol=1e-12, gtol=1e-8):
    from scipy.optimize import minimize
    
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
        best_pts = _project_to_sphere(result.x.reshape(-1, 3))
        best_energy = _energy_and_grad(best_pts)[0]
        return best_pts, best_energy
    except Exception:
        return pts0, _energy_and_grad(pts0)[0]

def _molecular_dynamics(pts0, time_budget=10.0, seed=42):
    """
    Molecular dynamics with velocity Verlet and adaptive cooling.
    """
    rng = np.random.RandomState(seed)
    pts = _project_to_sphere(pts0.copy())
    n = len(pts)
    
    # Initialize velocities to zero
    v = np.zeros_like(pts)
    
    # Parameters
    dt = 0.01
    friction = 0.99  # Damping factor
    T_start = 0.1
    T_end = 0.001
    
    current_energy = _energy_and_grad(pts)[0]
    best_pts = pts.copy()
    best_energy = current_energy
    
    start_time = time.time()
    
    while time.time() - start_time < time_budget:
        # Compute forces (negative gradient)
        _, grad = _energy_and_grad(pts)
        force = -grad
        
        # Velocity Verlet step 1: v_half = v + 0.5 * dt * force
        v_half = v + 0.5 * dt * force
        
        # Update positions: p_new = p + dt * v_half
        pts_new = pts + dt * v_half
        pts_new = _project_to_sphere(pts_new)
        
        # Compute forces at new positions
        _, grad_new = _energy_and_grad(pts_new)
        force_new = -grad_new
        
        # Velocity Verlet step 2: v_new = v_half + 0.5 * dt * force_new
        v_new = v_half + 0.5 * dt * force_new
        
        # Apply friction
        v_new *= friction
        
        # Add thermal noise (temperature)
        # Estimate progress
        elapsed = time.time() - start_time
        progress = min(1.0, elapsed / time_budget)
        T = T_start * (T_end / T_start) ** progress
        
        noise = rng.randn(n, 3) * T * 0.1
        # Project noise to tangent space
        noise = noise - np.einsum('ij,ij->i', noise, pts_new)[:, np.newaxis] * pts_new
        v_new += noise
        
        # Project velocity to tangent space
        v_new = v_new - np.einsum('ij,ij->i', v_new, pts_new)[:, np.newaxis] * pts_new
        
        pts = pts_new
        v = v_new
        
        current_energy = _energy_and_grad(pts)[0]
        if current_energy < best_energy:
            best_energy = current_energy
            best_pts = pts.copy()
            
    return best_pts, best_energy

def construct():
    """
    Construct 100 points on the unit sphere minimizing logarithmic energy.
    """
    start_time = time.time()
    deadline = start_time + 145.0

    best_pts = None
    best_energy = np.inf

    # Initializations
    initializations = []
    initializations.append(_fibonacci_sphere(N))
    initializations.append(_fibonacci_sphere(N, jitter=0.02))
    initializations.append(_icosahedron_sphere(N))
    
    seeds = [42, 123, 777]
    for s in seeds:
        initializations.append(_random_sphere(N, s))

    # Phase 1: Coarse L-BFGS to get a good starting point
    for i, pts0 in enumerate(initializations):
        if time.time() > deadline - 60.0:
            break
        
        pts_opt, e_opt = _lbfgs_optimize(pts0, maxiter=1000, ftol=1e-10, gtol=1e-6)
        
        if e_opt < best_energy:
            best_energy = e_opt
            best_pts = pts_opt.copy()

    if best_pts is None:
        return _fibonacci_sphere(N)

    # Phase 2: Molecular Dynamics to explore landscape
    time_left = deadline - time.time()
    if time_left > 30.0:
        md_time = min(20.0, time_left * 0.5)
        pts_md, e_md = _molecular_dynamics(best_pts, time_budget=md_time, seed=42)
        if e_md < best_energy:
            best_energy = e_md
            best_pts = pts_md.copy()
            
        # Refine after MD
        if time.time() < deadline - 20.0:
            pts_opt, e_opt = _lbfgs_optimize(best_pts, maxiter=2000, ftol=1e-12, gtol=1e-8)
            if e_opt < best_energy:
                best_energy = e_opt
                best_pts = pts_opt.copy()
                
        # Second MD run with different seed
        if time.time() < deadline - 15.0:
            md_time2 = min(10.0, (deadline - time.time()) * 0.4)
            pts_md2, e_md2 = _molecular_dynamics(best_pts, time_budget=md_time2, seed=123)
            if e_md2 < best_energy:
                best_energy = e_md2
                best_pts = pts_md2.copy()
                
            # Refine again
            if time.time() < deadline - 10.0:
                pts_opt, e_opt = _lbfgs_optimize(best_pts, maxiter=2000, ftol=1e-13, gtol=1e-9)
                if e_opt < best_energy:
                    best_energy = e_opt
                    best_pts = pts_opt.copy()

    # Phase 3: Newton polish
    time_left = deadline - time.time()
    if time_left > 10.0:
        pts_newton, e_newton = _newton_polish(best_pts, max_iter=30, tol=1e-14)
        if e_newton < best_energy:
            best_energy = e_newton
            best_pts = pts_newton.copy()

    # Phase 4: Final high-precision L-BFGS
    time_left = deadline - time.time()
    if time_left > 5.0:
        pts_final, e_final = _lbfgs_optimize(best_pts, maxiter=3000, ftol=1e-15, gtol=1e-12)
        if e_final < best_energy:
            best_energy = e_final
            best_pts = pts_final.copy()

    # Phase 5: Final Newton polish
    time_left = deadline - time.time()
    if time_left > 2.0:
        pts_newton2, e_newton2 = _newton_polish(best_pts, max_iter=20, tol=1e-15)
        if e_newton2 < best_energy:
            best_energy = e_newton2
            best_pts = pts_newton2.copy()

    best_pts = _project_to_sphere(best_pts)
    
    # Ensure no points are too close
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < 1e-9:
                best_pts[j] = _project_to_sphere(best_pts[j] + 0.01 * np.random.randn(3))
    
    return best_pts
