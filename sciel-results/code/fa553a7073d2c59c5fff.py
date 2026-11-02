import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150
DEADLINE_S = 145.0  # Safety margin before 150s limit

def _log_energy_fast(pts):
    """Vectorized log energy computation for speed."""
    n = len(pts)
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diffs * diffs, axis=2)
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(dists_sq[iu])
    d = np.maximum(d, 1e-15)
    return np.sum(np.log(1.0 / d))

def _energy_and_grad(pts):
    """Vectorized energy and gradient (projected to tangent space)."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dists_sq, 1.0)
    inv_d2 = 1.0 / dists_sq
    np.fill_diagonal(inv_d2, 0.0)

    weighted_sum_pts = inv_d2 @ pts
    sum_inv_d2 = np.sum(inv_d2, axis=1)
    grad = -pts * sum_inv_d2[:, np.newaxis] + weighted_sum_pts

    # Project gradient to tangent space: g - (g . p) p
    dot_prod = np.sum(grad * pts, axis=1)
    grad -= dot_prod[:, np.newaxis] * pts

    # Compute energy (upper triangle only)
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(dists_sq[iu])
    d = np.maximum(d, 1e-15)
    energy = np.sum(np.log(1.0 / d))

    return energy, grad

def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _exponential_map(pts, tangent_vecs):
    """Apply exponential map on sphere: move points along tangent vectors."""
    norms = np.linalg.norm(tangent_vecs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    cos_norms = np.cos(norms)
    sin_norms_over_norms = np.sin(norms) / norms
    return cos_norms * pts + sin_norms_over_norms * tangent_vecs

def _riemannian_cg(pts, max_iter=200, tol=1e-10, time_limit=None, start_time=None):
    """Riemannian conjugate gradient on the sphere for log-energy minimization."""
    if start_time is None:
        start_time = time.time()
    pts = pts.copy()
    n = len(pts)
    
    def _time_exceeded():
        if time_limit is None:
            return False
        return time.time() - start_time > time_limit
    
    # Initial gradient
    energy, grad = _energy_and_grad(pts)
    p = -grad  # initial search direction
    
    for it in range(max_iter):
        if _time_exceeded():
            break
        
        # Adaptive Line Search: Backtracking Armijo
        c1 = 1e-4
        alpha = 0.1
        grad_dot_p = np.sum(grad * p)
        if grad_dot_p >= 0:
            # Reset direction if not a descent direction
            p = -grad
            grad_dot_p = np.sum(grad * p)
            if grad_dot_p >= 0:
                break
        
        new_energy = energy
        new_pts = pts
        new_grad = grad
        found = False
        
        for _ in range(20):
            trial_pts = _exponential_map(pts, alpha * p)
            trial_pts = _normalize(trial_pts)
            trial_e, trial_grad = _energy_and_grad(trial_pts)
            
            if trial_e <= energy - c1 * alpha * grad_dot_p:
                new_energy = trial_e
                new_pts = trial_pts
                new_grad = trial_grad
                found = True
                break
            
            alpha *= 0.5
            if _time_exceeded():
                break
        
        if not found:
            break
        
        # Check convergence
        grad_norm = np.linalg.norm(new_grad)
        if grad_norm < tol:
            pts = new_pts
            energy = new_energy
            break
        
        # Conjugate gradient update (Polak-Ribiere)
        beta_num = np.sum(new_grad * (new_grad - grad))
        beta_den = np.sum(grad * grad)
        if beta_den > 1e-15:
            beta = beta_num / beta_den
        else:
            beta = 0.0
        beta = max(0.0, min(beta, 1.0))
        
        p = -new_grad + beta * p
        # Project p to tangent space
        dot_p = np.sum(p * new_pts, axis=1, keepdims=True)
        p -= dot_p * new_pts
        
        pts = new_pts
        energy = new_energy
        grad = new_grad
    
    return pts, _log_energy_fast(pts)

def _lbfgs_sphere(pts, time_limit=10.0, ftol=1e-14, gtol=1e-10):
    """L-BFGS-B optimization on the sphere using ambient coordinates with
    normalization after each step."""
    start = time.time()
    x0 = pts.flatten().copy()

    def obj_and_grad(x):
        if time.time() - start > time_limit:
            return float('inf'), np.zeros_like(x)
        pts_local = x.reshape(N, 3)
        pts_local = _normalize(pts_local)
        energy, grad = _energy_and_grad(pts_local)
        # For L-BFGS-B in ambient space, use the unprojected gradient
        # but since we normalize, the effective gradient is the projected one
        # mapped back. We return the ambient gradient which is consistent with
        # the normalized points if we treat the normalization as a projection.
        # Actually, the standard trick is to return the gradient of the energy
        # w.r.t the unnormalized coordinates.
        # dE/dx = dE/dp * dp/dx. Since p = x/||x||, dp/dx is complicated.
        # However, if we define the objective as E(normalize(x)), the gradient
        # w.r.t x is (I - pp^T) * grad_p / ||x||? No.
        # The simplest and most robust way is to use the tangent gradient
        # and project it back to ambient space? No, L-BFGS-B expects the gradient
        # w.r.t the variables x.
        # Let's use the fact that for small perturbations, moving in the tangent
        # direction is equivalent to moving in the ambient direction projected.
        # The gradient w.r.t x is simply the ambient gradient:
        # g_ambient = g_tangent + (g_tangent . p) p
        # But g_tangent is already orthogonal to p. So g_ambient = g_tangent?
        # No, the chain rule gives:
        # grad_x E(normalize(x)) = (I - pp^T) grad_p E / ||x|| ?
        # Actually, a common and effective approach in these tasks is to just
        # return the tangent gradient flattened. L-BFGS-B will take a step in x-space,
        # which will have a radial component, but the next normalization projects it back.
        # This works surprisingly well.
        return energy, grad.flatten()

    result = minimize(
        obj_and_grad,
        x0,
        method='L-BFGS-B',
        jac=True,
        options={
            'maxiter': 500,
            'ftol': ftol,
            'gtol': gtol,
            'maxfun': 2000,
        }
    )

    if result.success or result.fun < float('inf'):
        best_pts = _normalize(result.x.reshape(N, 3))
        best_energy = _log_energy_fast(best_pts)
    else:
        best_pts = pts.copy()
        best_energy = _log_energy_fast(pts)

    return best_pts, best_energy

def _fibonacci_sphere(n):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

def _halton_sphere(n, base_start=2):
    """Generate Halton sequence points on unit sphere."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r

    pts = []
    for i in range(1, n + 1):
        u1 = halton(i, base_start)
        u2 = halton(i, base_start + 1)
        theta = 2.0 * np.pi * u1
        phi = np.arccos(2.0 * u2 - 1.0)

        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)

        pts.append([x, y, z])

    pts = np.array(pts)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

def _fps_fast(n, seed=42, batch_size=1000):
    """Fast farthest point sampling using vectorized operations."""
    rng = np.random.default_rng(seed)
    pts = [rng.standard_normal(3)]
    pts[0] = pts[0] / np.linalg.norm(pts[0])
    pts_arr = np.array(pts)

    for _ in range(n - 1):
        candidates = rng.standard_normal((batch_size, 3))
        candidates = _normalize(candidates)

        min_dists = np.full(batch_size, np.inf)
        chunk = 50
        for i in range(0, len(pts_arr), chunk):
            chunk_pts = pts_arr[i:i+chunk]
            d = np.linalg.norm(candidates[:, np.newaxis, :] - chunk_pts[np.newaxis, :, :], axis=2)
            min_dists = np.minimum(min_dists, np.min(d, axis=1))

        best_idx = np.argmax(min_dists)
        pts_arr = np.vstack([pts_arr, candidates[best_idx:best_idx+1]])

    return _normalize(pts_arr)

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_S

    # 1. Generate diverse initializations
    inits = []
    inits.append(_fibonacci_sphere(N))
    inits.append(_halton_sphere(N))
    
    # FPS inits
    inits.append(_fps_fast(N, seed=7))
    inits.append(_fps_fast(N, seed=13))
    
    # Random inits
    for s in range(4):
        rng = np.random.default_rng(s * 100 + 1)
        pts = rng.standard_normal((N, 3))
        pts = _normalize(pts)
        inits.append(pts)
        
    # Perturbed Fibonacci
    for s in range(2):
        rng = np.random.default_rng(s * 200 + 50)
        base = _fibonacci_sphere(N)
        pts = base + rng.standard_normal((N, 3)) * 0.1
        pts = _normalize(pts)
        inits.append(pts)

    best_pts = None
    best_energy = float('inf')
    
    # 2. Optimize each initialization with a time budget
    # We will do a quick pass first to find the best candidate
    time_per_init = 10.0 # 10 seconds per init * 10 inits = 100s
    for idx, init_pts in enumerate(inits):
        if time.time() > deadline - 20.0:
            break
            
        # Phase 1: Riemannian CG (fast)
        cg_time = 3.0
        opt_pts, _ = _riemannian_cg(init_pts, max_iter=100, tol=1e-8, 
                                     time_limit=cg_time, start_time=time.time())
        
        # Phase 2: L-BFGS-B (polish)
        lbfgs_time = 5.0
        opt_pts, opt_energy = _lbfgs_sphere(opt_pts, time_limit=lbfgs_time, ftol=1e-12, gtol=1e-10)
        
        if opt_energy < best_energy:
            best_energy = opt_energy
            best_pts = opt_pts.copy()

    if best_pts is None:
        best_pts = _fibonacci_sphere(N)
        best_energy = _log_energy_fast(best_pts)

    # 3. Basin Hopping on the best candidate
    # Use the remaining time for basin hopping
    current = best_pts.copy()
    energy = best_energy
    
    temp = 0.5
    decay = 0.8
    max_restarts = 20
    
    for rh in range(max_restarts):
        if time.time() > deadline - 5.0:
            break
            
        # Adaptive perturbation
        perturb_scale = 0.1 + 0.05 * rh
        
        rng = np.random.default_rng(rh * 7919 + 1)
        perturbations = rng.standard_normal((N, 3))
        dot_prods = np.sum(perturbations * current, axis=1, keepdims=True)
        perturbations -= dot_prods * current
        pert_norms = np.linalg.norm(perturbations, axis=1, keepdims=True)
        pert_norms = np.maximum(pert_norms, 1e-15)
        perturbations = perturbations / pert_norms * perturb_scale

        new_pts = _exponential_map(current, perturbations)
        new_pts = _normalize(new_pts)

        # Optimize the perturbed point
        time_left = deadline - time.time() - 2.0
        if time_left < 4.0:
            break
            
        # Quick CG
        cg_t = min(2.0, time_left * 0.3)
        new_opt_pts, _ = _riemannian_cg(new_pts, max_iter=50, tol=1e-7,
                                          time_limit=cg_t, start_time=time.time())
        # L-BFGS polish
        lbfgs_t = min(4.0, time_left - cg_t - 0.5)
        if lbfgs_t > 1.0:
            new_opt_pts, new_energy = _lbfgs_sphere(new_opt_pts, time_limit=lbfgs_t, ftol=1e-12, gtol=1e-10)
        else:
            new_energy = _log_energy_fast(new_opt_pts)
            
        # Metropolis Criterion
        delta_e = new_energy - energy
        if delta_e < 0 or np.random.rand() < math.exp(-delta_e / max(temp, 1e-10)):
            current = new_opt_pts
            energy = new_energy
            if energy < best_energy:
                best_energy = energy
                best_pts = current.copy()

        temp *= decay

    # 4. Final High-Precision Polish
    remaining = deadline - time.time() - 1.0
    if remaining > 3.0:
        # Use L-BFGS-B with very tight tolerances
        final_pts, final_energy = _lbfgs_sphere(best_pts, time_limit=remaining, ftol=1e-15, gtol=1e-12)
        if final_energy < best_energy:
            best_energy = final_energy
            best_pts = final_pts

    return _normalize(best_pts)
