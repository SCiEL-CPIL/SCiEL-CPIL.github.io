"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize

def _energy_and_grad(pts):
    """
    Compute logarithmic energy E = sum_{i<j} log(1/|x_i - x_j|)
    and its gradient w.r.t. the Cartesian coordinates.
    E = -sum_{i<j} log(|x_i - x_j|)
    dE/dx_i = - sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    """
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    dist2 = np.sum(diff * diff, axis=2)  # (n, n)
    np.fill_diagonal(dist2, 1.0)
    
    dist = np.sqrt(dist2)
    
    # Energy: sum over i<j of log(1/dist[i,j]) = -sum over i<j log(dist[i,j])
    iu = np.triu_indices(n, k=1)
    energy = -math.fsum(math.log(d) for d in dist[iu])
    
    # Gradient: dE/dx_i = - sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    weight = np.zeros_like(dist2)
    np.fill_diagonal(weight, 0.0)
    weight = 1.0 / dist2
    
    grad = -np.einsum('ij,ijk->ik', weight, diff)
    
    return energy, grad

def _sphere_energy_and_grad(x_flat):
    """
    Objective and gradient for scipy.optimize.minimize.
    x_flat is a 3*N array of coordinates.
    We normalize each point to the sphere inside this function.
    """
    N_local = len(x_flat) // 3
    pts = x_flat.reshape(N_local, 3)
    
    # Normalize to unit sphere
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    pts = pts / norms
    
    energy, grad = _energy_and_grad(pts)
    
    # Project gradient onto tangent space for the effective gradient
    proj = np.einsum('ij,ij->i', grad, pts)
    grad_tangent = grad - proj[:, np.newaxis] * pts
    
    return energy, grad_tangent.flatten()

def _retract(pts, direction, step_size):
    """
    Retraction on the sphere: move along tangent direction and re-normalize.
    """
    new_pts = pts + step_size * direction
    norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
    new_pts = new_pts / np.maximum(norms, 1e-15)
    return new_pts

def _lbfgs_sphere(x0, deadline, maxiter=5000):
    """
    Run L-BFGS-B on the sphere with time guard and tight tolerances.
    """
    best_x = x0.copy()
    best_f, _ = _sphere_energy_and_grad(x0)
    
    def callback(xk):
        nonlocal best_x, best_f
        if time.time() > deadline:
            return True
        f_k, _ = _sphere_energy_and_grad(xk)
        if f_k < best_f:
            best_f = f_k
            best_x = xk.copy()
        return False
    
    try:
        res = minimize(
            _sphere_energy_and_grad,
            x0,
            jac=True,
            method='L-BFGS-B',
            options={
                'maxiter': maxiter,
                'ftol': 1e-16,
                'gtol': 1e-14,
                'maxfun': 15000
            },
            callback=callback
        )
        if res.fun < best_f:
            best_f = res.fun
            best_x = res.x.copy()
    except Exception:
        pass
    
    return best_x, best_f

def _polish(pts, deadline, max_iter=500, lr_init=0.05, lr_max=0.15):
    """
    High-precision polish using gradient descent with backtracking line search
    on the tangent space. Uses adaptive learning rate.
    """
    f, g = _energy_and_grad(pts)
    proj = np.einsum('ij,ij->i', g, pts)
    g_tangent = g - proj[:, np.newaxis] * pts
    
    lr = lr_init
    for it in range(max_iter):
        if time.time() > deadline - 2.0:
            break
        
        new_pts = _retract(pts, -g_tangent, lr)
        new_f, new_g = _energy_and_grad(new_pts)
        new_proj = np.einsum('ij,ij->i', new_g, new_pts)
        new_g_tangent = new_g - new_proj[:, np.newaxis] * new_pts
        
        if new_f < f:
            pts = new_pts
            f = new_f
            g = new_g
            g_tangent = new_g_tangent
            lr = min(lr * 1.15, lr_max)
        else:
            lr *= 0.5
            if lr < 1e-14:
                break
    
    return pts, f

def _ultra_polish(pts, deadline, max_iter=2000):
    """
    Ultra-precise Newton-style polish with very small steps and tight Armijo.
    """
    f, g = _energy_and_grad(pts)
    proj = np.einsum('ij,ij->i', g, pts)
    g_t = g - proj[:, np.newaxis] * pts
    
    # Estimate local curvature scale from gradient magnitude
    g_norm = np.linalg.norm(g_t)
    if g_norm < 1e-15:
        return pts, f
    
    # Start with a small step relative to gradient scale
    alpha = 0.01
    for it in range(max_iter):
        if time.time() > deadline - 1.0:
            break
        
        # Armijo backtracking
        sufficient = 1e-18 * alpha * np.sum(g_t * (-g_t))
        cand = _retract(pts, -g_t, alpha)
        cand_f, cand_g = _energy_and_grad(cand)
        cand_proj = np.einsum('ij,ij->i', cand_g, cand)
        cand_g_t = cand_g - cand_proj[:, np.newaxis] * cand
        
        if cand_f <= f - sufficient:
            pts = cand
            f = cand_f
            g = cand_g
            g_t = cand_g_t
            # Slightly increase step if we made good progress
            if it % 10 == 0:
                alpha = min(alpha * 1.05, 0.05)
        else:
            alpha *= 0.5
            if alpha < 1e-18:
                break
        
        # Check convergence
        if np.linalg.norm(g_t) < 1e-15:
            break
    
    return pts, f

def _lbfgs_sphere_custom(pts, max_iter=500, tol=1e-12, deadline=None, m=25):
    """
    Custom L-BFGS on the sphere using Riemannian gradient and retraction.
    """
    x = pts.copy()
    f, g = _energy_and_grad(x)
    
    proj = np.einsum('ij,ij->i', g, x)
    g_t = g - proj[:, np.newaxis] * x
    
    s_list = []
    y_list = []
    
    for k in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        if np.linalg.norm(g_t) < tol:
            break
        
        if len(s_list) > 0:
            rhos = []
            for s, y in zip(s_list, y_list):
                sy = np.sum(s * y)
                rhos.append(1.0 / sy if abs(sy) > 1e-15 else 0.0)
            
            q = g_t.copy()
            alphas = []
            for i in range(len(s_list) - 1, -1, -1):
                s_i = s_list[i]
                y_i = y_list[i]
                rho_i = rhos[i]
                alpha_i = rho_i * np.sum(s_i * q)
                alphas.append(alpha_i)
                q = q - alpha_i * y_i
            
            gamma = 1.0
            r = -gamma * q
            
            for i in range(len(s_list)):
                s_i = s_list[i]
                y_i = y_list[i]
                rho_i = rhos[i]
                beta_i = rho_i * np.sum(y_i * r)
                r = r + (alphas[i] - beta_i) * s_i
            
            direction = r
        else:
            direction = -g_t
        
        # Line search with backtracking
        step_size = 1.0
        new_x = x
        new_f = f
        new_g = g
        new_g_t = g_t
        for _ in range(50):
            candidate = _retract(x, direction, step_size)
            new_f, new_g = _energy_and_grad(candidate)
            new_proj = np.einsum('ij,ij->i', new_g, candidate)
            new_g_t = new_g - new_proj[:, np.newaxis] * candidate
            
            if new_f <= f + 1e-5 * step_size * np.sum(g_t * direction):
                new_x = candidate
                break
            step_size *= 0.5
        else:
            new_x = candidate
        
        s_k = new_x - x
        y_k = new_g_t - g_t
        sy = np.sum(s_k * y_k)
        if sy > 1e-12:
            s_list.append(s_k)
            y_list.append(y_k)
            if len(s_list) > m:
                s_list.pop(0)
                y_list.pop(0)
        
        x = new_x
        f = new_f
        g = new_g
        proj = np.einsum('ij,ij->i', g, x)
        g_t = g - proj[:, np.newaxis] * x
    
    return x, f

def _rcg_polish(pts, deadline, max_iter=200):
    """
    Riemannian Conjugate Gradient polish for deeper local optimization.
    """
    f, g = _energy_and_grad(pts)
    proj = np.einsum('ij,ij->i', g, pts)
    d = g - proj[:, np.newaxis] * pts  # initial search direction (negative gradient)
    d = -d  # we want to minimize, so go in -gradient direction
    
    # Ensure d is in tangent space
    proj_d = np.einsum('ij,ij->i', d, pts)
    d = d - proj_d[:, np.newaxis] * pts
    
    d_prev = None
    beta = 0.0
    
    for it in range(max_iter):
        if time.time() > deadline - 2.0:
            break
        
        # Line search with backtracking
        lr = 0.1
        for _ls in range(30):
            new_pts = _retract(pts, d, lr)
            new_f, new_g = _energy_and_grad(new_pts)
            new_proj = np.einsum('ij,ij->i', new_g, new_pts)
            new_g_tangent = new_g - new_proj[:, np.newaxis] * new_pts
            
            if new_f < f - 1e-4 * lr * np.sum(d * d):
                break
            lr *= 0.5
        
        if new_f >= f:
            break
        
        # Compute beta (Polak-Ribiere)
        if d_prev is not None:
            inner = np.sum(new_g_tangent * (new_g_tangent - d_prev))
            denom = np.sum(d_prev * d_prev)
            if denom > 1e-15:
                beta = max(0.0, inner / denom)
        
        d_prev = d
        d = -new_g_tangent + beta * d
        
        # Ensure d is in tangent space
        proj_d = np.einsum('ij,ij->i', d, pts)
        d = d - proj_d[:, np.newaxis] * pts
        
        pts = new_pts
        f = new_f
        g = new_g
        g_tangent = new_g_tangent
    
    return pts, f

def _spherical_harmonic_init(L=12, seed=42):
    """
    Generate initial points based on critical points of a random spherical harmonic field.
    This provides a structured initialization with global spectral characteristics.
    """
    rng = np.random.default_rng(seed)
    
    # Generate random spherical harmonic coefficients up to degree L
    # For each degree l, we have (2l+1) coefficients (m from -l to l)
    # We use real spherical harmonics for simplicity
    
    # First, generate points on a fine grid to find critical points
    # Use a Fibonacci spiral for the base grid
    n_grid = 500
    i = np.arange(n_grid) + 0.5
    z = 1.0 - 2.0 * i / n_grid
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    grid_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Compute a random spherical harmonic field on the grid
    # For simplicity, use a sum of random spherical harmonics
    # We'll approximate by using random linear combinations of basis functions
    
    # Generate random coefficients for degrees 1 to L
    field = np.zeros(n_grid)
    for l in range(1, L + 1):
        for m in range(-l, l + 1):
            # Compute |Y_l^m| on the grid (approximation)
            # Use associated Legendre polynomials
            # For simplicity, use a random weighting
            coeff = rng.standard_normal()
            
            # Approximate spherical harmonic using Legendre polynomials
            # This is a rough approximation but provides diversity
            P_lm = np.zeros(n_grid)
            for k in range(n_grid):
                # Simple approximation using Chebyshev-like basis
                P_lm[k] = np.cos(l * np.arccos(np.clip(z[k], -1, 1)) + m * phi[k])
            
            field += coeff * P_lm
    
    # Find local minima of the field (these will be our initial points)
    # Sort by field value and take the N lowest
    sorted_idx = np.argsort(field)
    
    # Select N points, ensuring they are well-distributed
    selected = []
    min_dist = 0.05
    for idx in sorted_idx:
        if len(selected) >= N:
            break
        pt = grid_pts[idx]
        is_valid = True
        for sel_pt in selected:
            if np.linalg.norm(pt - sel_pt) < min_dist:
                is_valid = False
                break
        if is_valid:
            selected.append(pt)
    
    # If we don't have enough points, fill with remaining sorted points
    while len(selected) < N:
        for idx in sorted_idx:
            if grid_pts[idx] not in selected:
                selected.append(grid_pts[idx])
                if len(selected) >= N:
                    break
    
    return np.array(selected[:N])

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Generate multiple diverse initializations
    initializations = []
    
    # 1. Fibonacci spiral (golden angle)
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    initializations.append(fib_pts)
    
    # 2. Random points (seeded for reproducibility)
    for seed_offset in range(5):
        rng_local = np.random.default_rng(42 + seed_offset)
        random_pts = rng_local.standard_normal((N, 3))
        norms = np.linalg.norm(random_pts, axis=1, keepdims=True)
        random_pts = random_pts / np.maximum(norms, 1e-15)
        initializations.append(random_pts)
    
    # 3. Perturbed Fibonacci
    rng_pert = np.random.default_rng(123)
    for pert_scale in [0.01, 0.05, 0.1]:
        perturbed = fib_pts + rng_pert.standard_normal((N, 3)) * pert_scale
        norms = np.linalg.norm(perturbed, axis=1, keepdims=True)
        perturbed = perturbed / np.maximum(norms, 1e-15)
        initializations.append(perturbed)
    
    # 4. Spherical harmonic-based initialization
    try:
        sh_pts = _spherical_harmonic_init(L=12, seed=42)
        initializations.append(sh_pts)
        sh_pts2 = _spherical_harmonic_init(L=10, seed=123)
        initializations.append(sh_pts2)
    except Exception:
        pass
    
    # 5. Regular icosahedral-like base with perturbations
    for angle_shift in [0.1, 0.2, 0.3]:
        i2 = np.arange(N) + 0.5
        z2 = 1.0 - 2.0 * i2 / N
        phi2 = np.pi * (1.0 + np.sqrt(5.0)) * i2 + angle_shift
        r2 = np.sqrt(1.0 - z2 * z2)
        shifted_pts = np.column_stack([r2 * np.cos(phi2), r2 * np.sin(phi2), z2])
        initializations.append(shifted_pts)
    
    # Optimize each initialization and keep the best
    best_x = None
    best_f = float('inf')
    
    # Sort initializations by initial energy (best first) to prioritize good starts
    init_energies = []
    for init_pts in initializations:
        e, _ = _energy_and_grad(init_pts)
        init_energies.append((e, init_pts))
    init_energies.sort(key=lambda x: x[0])
    
    for idx, (init_e, init_pts) in enumerate(init_energies):
        if time.time() > deadline - 10.0:
            break
        
        x0 = init_pts.flatten()
        
        # Quick check: how much time do we have?
        remaining = deadline - time.time()
        if remaining < 10.0:
            break
        
        # Allocate time proportionally: earlier initializations get more time
        maxiter_for_this = min(5000, int(remaining * 20))
        if maxiter_for_this < 100:
            maxiter_for_this = 100
        
        x_opt, f_opt = _lbfgs_sphere(x0, deadline, maxiter=maxiter_for_this)
        
        if f_opt < best_f:
            best_f = f_opt
            best_x = x_opt.copy()
        
        # Early termination: if we've found a very good solution, reduce remaining budget
        if f_opt < -1083.3:
            if idx > 3:
                break
    
    if best_x is None:
        # Fallback to Fibonacci
        best_x = fib_pts.flatten()
        best_f, _ = _sphere_energy_and_grad(best_x)
    
    # Final high-precision polish on the best result
    if time.time() < deadline - 5.0:
        pts = best_x.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / np.maximum(norms, 1e-15)
        
        # First pass: standard gradient polish
        pts, f = _polish(pts, deadline, max_iter=300)
        
        # Second pass: Custom L-BFGS on the sphere
        if time.time() < deadline - 4.0:
            pts, f = _lbfgs_sphere_custom(pts, max_iter=200, tol=1e-13, deadline=deadline - 3.0)
        
        # Third pass: Riemannian Conjugate Gradient for deeper convergence
        if time.time() < deadline - 3.0:
            pts, f = _rcg_polish(pts, deadline, max_iter=150)
        
        # Fourth pass: Ultra-precise polish
        if time.time() < deadline - 2.0:
            pts, f = _ultra_polish(pts, deadline, max_iter=1000)
        
        # Final gradient polish with tight tolerance
        if time.time() < deadline - 1.0:
            pts, f = _polish(pts, deadline, max_iter=200, lr_init=0.02, lr_max=0.05)
        
        best_x = pts.flatten()
        best_f = f
    
    # Ensure points are on the sphere
    pts_opt = best_x.reshape(N, 3)
    norms = np.linalg.norm(pts_opt, axis=1, keepdims=True)
    pts_opt = pts_opt / np.maximum(norms, 1e-15)
    
    return pts_opt
# EVOLVE-BLOCK-END
