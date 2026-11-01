"""Combined program: L-BFGS-B + Riemannian CG + polish for log-energy minimization."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150
DEADLINE = 120.0
SAFETY_MARGIN = 5.0


def _energy(x):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)."""
    n = len(x)
    G = x @ x.T
    idx = np.triu_indices(n, k=1)
    vals = 1.0 - G[idx]
    vals = np.maximum(vals, 1e-15)
    energy = -0.5 * np.sum(np.log(2.0 * vals))
    return energy


def _energy_grad(x):
    """Compute energy and tangent-space gradient on the sphere."""
    n = len(x)
    G = x @ x.T
    idx = np.triu_indices(n, k=1)
    vals = 1.0 - G[idx]
    vals = np.maximum(vals, 1e-15)
    
    energy = -0.5 * np.sum(np.log(2.0 * vals))
    
    # d(energy)/dG[i,j] for i<j:
    # energy = -0.5 * sum_{i<j} log(2*(1-G[i,j]))
    # dE/dG[i,j] = -0.5 * (1/(2*(1-G[i,j]))) * (-2) = 0.5/(1-G[i,j])
    dE_dG = 0.5 / vals
    
    # Build symmetric matrix
    dE_dG_full = np.zeros((n, n))
    dE_dG_full[idx] = dE_dG
    dE_dG_full = dE_dG_full + dE_dG_full.T
    
    # Gradient: grad_x = dE_dG_full @ x
    # This is the Euclidean gradient; project to tangent space
    grad = dE_dG_full @ x
    
    # Project to tangent space: grad_tangent = grad - (grad . x) * x
    dot = np.sum(grad * x, axis=1, keepdims=True)
    grad -= dot * x
    
    return energy, grad


def _project(x):
    """Project points back to the unit sphere."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return x / norms


def _fibonacci_sphere(n):
    """Generate Fibonacci sphere points."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _project(pts)


def _exponential_map(pts, tangent_vecs):
    """Apply exponential map on sphere: move points along tangent vectors."""
    norms = np.linalg.norm(tangent_vecs, axis=1, keepdims=True)
    norms_safe = np.maximum(norms, 1e-15)
    cos_norms = np.cos(norms_safe)
    sin_norms_over_norms = np.sin(norms_safe) / norms_safe
    return cos_norms * pts + sin_norms_over_norms * tangent_vecs


def _riemannian_cg(pts, time_limit=15.0, max_iter=500):
    """Riemannian conjugate gradient on the sphere using exponential maps."""
    start = time.time()
    current = pts.copy()
    energy, grad = _energy_grad(current)
    
    # Initial search direction = negative gradient (already in tangent space)
    d = -grad.copy()
    prev_grad = grad.copy()
    
    for it in range(max_iter):
        if time.time() - start > time_limit - 0.1:
            break
        
        # Backtracking line search
        alpha = 0.1
        best_energy = energy
        best_pts = current.copy()
        best_alpha = 0.0
        best_grad = grad.copy()
        
        for ls in range(40):
            new_pts = _exponential_map(current, alpha * d)
            new_energy, new_grad = _energy_grad(new_pts)
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_pts = new_pts
                best_alpha = alpha
                best_grad = new_grad
            
            if new_energy > energy:
                break
            
            alpha *= 0.8
        
        if best_alpha == 0.0:
            break
        
        # Update
        current = best_pts
        energy = best_energy
        grad = best_grad
        
        # Check convergence
        if np.linalg.norm(grad) < 1e-14:
            break
        
        # Polak-Ribiere beta for conjugate gradient
        d_new = -grad
        beta = 0.0
        if it > 0:
            beta_num = np.sum(grad * (grad - prev_grad))
            beta_den = np.sum(prev_grad * prev_grad)
            if beta_den > 1e-30:
                beta = max(0.0, beta_num / beta_den)
        
        d = d_new + beta * d
        prev_grad = grad.copy()
    
    return current, energy


def _lbfgs_optimize(pts, time_limit=30.0):
    """Optimize using L-BFGS-B with projected gradient."""
    start = time.time()
    x0 = pts.flatten()
    
    def fun_and_grad(x):
        p = x.reshape(N, 3)
        p = _project(p)
        energy, grad = _energy_grad(p)
        return energy, grad.flatten()
    
    def callback(xk):
        return (time.time() - start) > (time_limit - 1.0)
    
    try:
        res = minimize(fun_and_grad, x0, method='L-BFGS-B', jac=True,
                       options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-12,
                                'maxfun': 10000, 'maxls': 50,
                                'callback': callback})
        x_opt = res.x.reshape(N, 3)
        x_opt = _project(x_opt)
        energy = _energy(x_opt)
    except Exception:
        x_opt = pts
        energy = _energy(pts)
    
    return x_opt, energy


def _polish(pts, time_limit=5.0):
    """High-precision polish with very small steps."""
    start = time.time()
    current = pts.copy()
    energy = _energy(current)
    step_size = 1e-4
    
    for it in range(10000):
        if time.time() - start > time_limit - 0.1:
            break
        
        _, grad = _energy_grad(current)
        grad_norm = np.linalg.norm(grad)
        
        if grad_norm < 1e-16:
            break
        
        alpha = step_size
        accepted = False
        for _ in range(30):
            new_pts = current - alpha * grad
            new_pts = _project(new_pts)
            new_energy = _energy(new_pts)
            
            if new_energy < energy - 1e-16:
                accepted = True
                break
            alpha *= 0.5
        
        if accepted:
            current = new_pts
            energy = new_energy
            step_size = min(alpha * 1.02, 0.01)
        else:
            step_size *= 0.5
            if step_size < 1e-18:
                break
    
    return current, energy


def construct():
    t_start = time.time()
    deadline = t_start + DEADLINE - SAFETY_MARGIN
    
    best_x = None
    best_energy = float('inf')
    
    def update_best(x, e):
        nonlocal best_x, best_energy
        if e < best_energy:
            best_energy = e
            best_x = x.copy()
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Fibonacci Sphere
    fib_pts = _fibonacci_sphere(N)
    initializations.append(fib_pts.copy())
    
    # 2. Random uniform on sphere (multiple seeds)
    for seed in range(3):
        rng_local = np.random.default_rng(seed * 123 + 7)
        pts_rand = rng_local.standard_normal((N, 3))
        pts_rand = _project(pts_rand)
        initializations.append(pts_rand)
    
    # 3. Perturbed Fibonacci
    for seed in range(2):
        rng_local = np.random.default_rng(seed * 456 + 13)
        pts_pert = fib_pts.copy() + rng_local.standard_normal((N, 3)) * 0.02
        pts_pert = _project(pts_pert)
        initializations.append(pts_pert)
    
    # 4. Different Fibonacci offset
    i = np.arange(N) + 0.3
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts_offset = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    initializations.append(_project(pts_offset))
    
    # Multi-phase optimization for each initialization
    for idx, init_pts in enumerate(initializations):
        if time.time() > deadline - 15:
            break
        
        current_pts = init_pts.copy()
        
        # Phase 1: L-BFGS-B for coarse optimization
        remaining = deadline - time.time()
        if remaining > 10:
            phase1_time = min(20.0, remaining - 10.0)
            if phase1_time > 2:
                current_pts, e1 = _lbfgs_optimize(current_pts, time_limit=phase1_time)
                update_best(current_pts, e1)
        
        # Phase 2: Riemannian CG for fine optimization
        remaining = deadline - time.time()
        if remaining > 8:
            phase2_time = min(10.0, remaining - 8.0)
            if phase2_time > 2:
                current_pts, e2 = _riemannian_cg(current_pts, time_limit=phase2_time)
                update_best(current_pts, e2)
        
        # Phase 3: Final L-BFGS-B polish
        remaining = deadline - time.time()
        if remaining > 5:
            phase3_time = min(5.0, remaining - 5.0)
            if phase3_time > 1:
                current_pts, e3 = _lbfgs_optimize(current_pts, time_limit=phase3_time)
                update_best(current_pts, e3)
        
        # Phase 4: High-precision polish
        remaining = deadline - time.time()
        if remaining > 3:
            phase4_time = min(3.0, remaining - 3.0)
            if phase4_time > 0.5:
                current_pts, e4 = _polish(current_pts, time_limit=phase4_time)
                update_best(current_pts, e4)
    
    # If we have best_x, use it; otherwise fallback to Fibonacci
    if best_x is None:
        best_x = _fibonacci_sphere(N)
    
    # Final validation
    best_x = _project(best_x)
    
    if not np.all(np.isfinite(best_x)):
        best_x = _fibonacci_sphere(N)
    
    # Check minimum pairwise distance
    G = best_x @ best_x.T
    idx = np.triu_indices(N, k=1)
    min_dist_sq = np.min(2.0 - 2.0 * G[idx])
    if min_dist_sq < (1e-9) ** 2:
        best_x = _fibonacci_sphere(N)
    
    return best_x
