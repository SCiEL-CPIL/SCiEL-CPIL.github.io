"""Rewritten program for logarithmic energy minimization of 150 points on the sphere."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize


def _fibonacci_sphere(n):
    """Generate n points on the unit sphere using the golden-angle spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_uniform_sphere(n, seed=None):
    """Generate n uniformly random points on the unit sphere."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    x = rng.randn(n, 3)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def _geodesic_dome(n):
    """Generate points on sphere using geodesic dome subdivision (icosahedron)."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    
    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ]
    
    # 2 subdivisions: 20*4^2 = 320 faces, 10*4^2+2 = 162 vertices
    for _ in range(2):
        new_faces = []
        new_verts = []
        vert_to_idx = {}
        
        for v in verts:
            key = (round(v[0], 12), round(v[1], 12), round(v[2], 12))
            vert_to_idx[key] = len(verts) + len(new_verts)
        
        for f in faces:
            a = verts[f[0]]
            b = verts[f[1]]
            c = verts[f[2]]
            
            ab = (a + b) / 2.0
            ab /= np.linalg.norm(ab)
            bc = (b + c) / 2.0
            bc /= np.linalg.norm(bc)
            ca = (c + a) / 2.0
            ca /= np.linalg.norm(ca)
            
            for v in [ab, bc, ca]:
                key = (round(v[0], 12), round(v[1], 12), round(v[2], 12))
                if key not in vert_to_idx:
                    vert_to_idx[key] = len(verts) + len(new_verts)
                    new_verts.append(v)
            
            ia = f[0]
            ib = f[1]
            ic = f[2]
            iab = vert_to_idx[(round(ab[0], 12), round(ab[1], 12), round(ab[2], 12))]
            ibc = vert_to_idx[(round(bc[0], 12), round(bc[1], 12), round(bc[2], 12))]
            ica = vert_to_idx[(round(ca[0], 12), round(ca[1], 12), round(ca[2], 12))]
            
            new_faces.append([ia, iab, ica])
            new_faces.append([iab, ib, ibc])
            new_faces.append([ica, ibc, ic])
            new_faces.append([iab, ibc, ica])
        
        verts = np.vstack([verts, np.array(new_verts)]) if new_verts else verts
        faces = new_faces
    
    if len(verts) >= n:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(verts), size=n, replace=False)
        return verts[indices]
    else:
        remaining = n - len(verts)
        extra = _random_uniform_sphere(remaining, seed=999)
        return np.vstack([verts, extra])


def _project_to_sphere(points):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms


def _energy(x):
    """Compute the logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)."""
    x = x.reshape(N, 3)
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    dists = np.sqrt(dists_sq)
    np.fill_diagonal(dists, 1.0)
    iu = np.triu_indices(N, k=1)
    log_dists = np.log(dists[iu[0], iu[1]])
    return -math.fsum(log_dists.tolist())


def _energy_and_grad(x):
    """Compute energy and gradient.
    x: (N, 3) array of points on sphere.
    Returns (energy, grad) where grad is (N, 3) Euclidean gradient.
    """
    x = x.reshape(N, 3)
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    dists = np.sqrt(dists_sq)
    np.fill_diagonal(dists, 1.0)
    
    iu = np.triu_indices(N, k=1)
    log_dists = np.log(dists[iu[0], iu[1]])
    energy = -math.fsum(log_dists.tolist())
    
    inv_d2 = 1.0 / np.maximum(dists_sq, 1e-30)
    np.fill_diagonal(inv_d2, 0.0)
    grad = np.sum(diff * inv_d2[:, :, np.newaxis], axis=1)
    
    return energy, grad


def _tangent_project(grad, x):
    """Project gradient to tangent space at each point."""
    x = x.reshape(N, 3)
    grad = grad.reshape(N, 3)
    radial = np.sum(grad * x, axis=1, keepdims=True)
    return grad - radial * x


def _exp_map(x, v):
    """Riemannian exponential map on the unit sphere."""
    x = x.reshape(N, 3)
    v = v.reshape(N, 3)
    radial_v = np.sum(v * x, axis=1, keepdims=True)
    v = v - radial_v * x
    theta = np.linalg.norm(v, axis=1, keepdims=True)
    safe_theta = np.where(theta > 1e-15, theta, 1.0)
    sin_theta_over_theta = np.sin(safe_theta) / safe_theta
    cos_theta = np.cos(safe_theta)
    sin_theta_over_theta = np.where(theta > 1e-15, sin_theta_over_theta, 1.0)
    cos_theta = np.where(theta > 1e-15, cos_theta, 1.0)
    result = cos_theta * x + sin_theta_over_theta * v
    result_norms = np.linalg.norm(result, axis=1, keepdims=True)
    result = result / np.maximum(result_norms, 1e-15)
    return result


def _lbfgs_optimize(x0, maxiter=200, ftol=1e-14, gtol=1e-12):
    """Run L-BFGS-B optimization on the sphere using projection."""
    
    def objective_and_grad(flat_x):
        x = _project_to_sphere(flat_x.reshape(N, 3))
        e, g = _energy_and_grad(x)
        grad_t = _tangent_project(g, x)
        return e, grad_t.flatten()
    
    try:
        result = minimize(
            objective_and_grad,
            x0.flatten(),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol, 'maxls': 30}
        )
        x_opt = _project_to_sphere(result.x.reshape(N, 3))
        e_opt = _energy(x_opt)
        return x_opt, e_opt
    except Exception:
        return x0, _energy(x0)


def _adam_optimize(x0, max_iter=3000, deadline=None, lr=0.005, beta1=0.9, beta2=0.999, eps=1e-8):
    """Adam optimizer on the sphere using exponential map."""
    x = _project_to_sphere(x0.copy())
    best_x = x.copy()
    best_e = _energy(x)
    
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        e, g = _energy_and_grad(x)
        grad_t = _tangent_project(g, x)
        
        m = beta1 * m + (1 - beta1) * grad_t
        v = beta2 * v + (1 - beta2) * (grad_t * grad_t)
        m_hat = m / (1 - beta1 ** (it + 1))
        v_hat = v / (1 - beta2 ** (it + 1))
        update = lr * m_hat / (np.sqrt(v_hat) + eps)
        
        radial_update = np.sum(update * x, axis=1, keepdims=True)
        update = update - radial_update * x
        
        x_trial = _exp_map(x, update)
        e_trial = _energy(x_trial)
        
        if e_trial < best_e:
            best_x = x_trial.copy()
            best_e = e_trial
            x = x_trial
        
        grad_norm = np.linalg.norm(grad_t)
        if grad_norm < 1e-14:
            break
    
    return best_x, best_e


def _gradient_descent_polish(x, max_iter=10000, tol=1e-15, deadline=None):
    """Polish with gradient descent on the sphere using adaptive step size."""
    x = _project_to_sphere(x.copy())
    best_x = x.copy()
    best_e = _energy(x)
    
    e, g = _energy_and_grad(x)
    grad_t = _tangent_project(g, x)
    grad_norm = np.linalg.norm(grad_t)
    
    alpha = 0.5 / max(grad_norm, 1e-10)
    alpha = min(alpha, 0.1)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        x_trial = _exp_map(x, alpha * grad_t)
        e_trial, g_trial = _energy_and_grad(x_trial)
        grad_t_trial = _tangent_project(g_trial, x_trial)
        
        if e_trial < best_e - 1e-16:
            x = x_trial
            best_x = x.copy()
            best_e = e_trial
            grad_t = grad_t_trial
            if it % 5 == 0:
                alpha = min(alpha * 1.05, 0.2)
        else:
            alpha *= 0.5
            if alpha < 1e-16:
                break
        
        grad_norm = np.linalg.norm(grad_t)
        if grad_norm < tol:
            break
    
    return best_x, best_e


def _conjugate_gradient(x, max_iter=5000, tol=1e-14, deadline=None):
    """Conjugate gradient method on the sphere."""
    x = _project_to_sphere(x.copy())
    best_x = x.copy()
    best_e = _energy(x)
    
    e, g = _energy_and_grad(x)
    grad_t = _tangent_project(g, x)
    
    d = -grad_t.copy()
    alpha = 0.01
    beta = 0.0
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        accepted = False
        for _ in range(40):
            x_trial = _exp_map(x, alpha * d)
            e_trial, g_trial = _energy_and_grad(x_trial)
            grad_t_trial = _tangent_project(g_trial, x_trial)
            
            if e_trial < best_e - 1e-16:
                x = x_trial
                best_x = x.copy()
                best_e = e_trial
                grad_t = grad_t_trial
                accepted = True
                break
            else:
                alpha *= 0.5
                if alpha < 1e-16:
                    break
        
        if not accepted:
            break
        
        grad_norm_sq = np.sum(grad_t * grad_t)
        if grad_norm_sq < tol * tol:
            break
        
        if beta > 0:
            d = -grad_t + beta * d
        else:
            d = -grad_t
        radial_d = np.sum(d * x, axis=1, keepdims=True)
        d = d - radial_d * x
        
        e_new, g_new = _energy_and_grad(x)
        grad_t_new = _tangent_project(g_new, x)
        dot = np.sum(grad_t_new * grad_t)
        if grad_norm_sq > 1e-30:
            beta = max(0.0, dot / grad_norm_sq)
        else:
            beta = 0.0
        grad_t = grad_t_new
    
    return best_x, best_e


def _high_precision_polish(x, deadline=None, max_iter=15000):
    """Very tight polish with tiny steps for final refinement."""
    x = _project_to_sphere(x.copy())
    best_x = x.copy()
    best_e = _energy(x)
    
    alpha = 0.01
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        e, g = _energy_and_grad(x)
        grad_t = _tangent_project(g, x)
        grad_norm = np.linalg.norm(grad_t)
        
        if grad_norm < 1e-16:
            break
        
        step = alpha * grad_t / max(grad_norm, 1e-10)
        x_trial = _exp_map(x, step)
        e_trial = _energy(x_trial)
        
        if e_trial < best_e:
            x = x_trial
            best_x = x.copy()
            best_e = e_trial
            alpha = min(alpha * 1.1, 0.1)
        else:
            alpha *= 0.5
            if alpha < 1e-18:
                break
    
    return best_x, best_e


def _basin_hopping(x0, n_restarts=8, deadline=None):
    """Basin hopping: random perturbations followed by local minimization."""
    x = _project_to_sphere(x0.copy())
    best_x = x.copy()
    best_e = _energy(x)
    
    rng = np.random.RandomState(42)
    
    for restart in range(n_restarts):
        if deadline is not None and time.time() > deadline:
            break
        
        # Vary perturbation size
        scale = 0.05 + 0.05 * restart
        perturbation = rng.randn(N, 3) * scale
        x_trial = _project_to_sphere(x + perturbation)
        
        x_opt, e_opt = _lbfgs_optimize(x_trial, maxiter=80, ftol=1e-12, gtol=1e-10)
        if e_opt < best_e:
            best_x = x_opt
            best_e = e_opt
    
    return best_x, best_e


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    # Multiple initializations
    initializations = []
    
    # 1. Fibonacci spiral
    x_fib = _fibonacci_sphere(N)
    x_fib = _project_to_sphere(x_fib)
    initializations.append(x_fib)
    
    # 2. Geodesic dome
    x_dome = _geodesic_dome(N)
    x_dome = _project_to_sphere(x_dome)
    initializations.append(x_dome)
    
    # 3. Random with different seeds
    for seed in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        x_rand = _random_uniform_sphere(N, seed=seed)
        initializations.append(x_rand)
    
    best_x = initializations[0].copy()
    best_e = _energy(best_x)
    
    # Optimize each initialization with L-BFGS
    for idx, x_init in enumerate(initializations):
        if time.time() > deadline - 60:
            break
        
        try:
            x_opt, e_opt = _lbfgs_optimize(x_init, maxiter=200, ftol=1e-13, gtol=1e-11)
            if e_opt < best_e:
                best_x = x_opt
                best_e = e_opt
        except Exception:
            pass
    
    # Adam optimization on the best configuration
    if time.time() < deadline - 50:
        try:
            x_adam, e_adam = _adam_optimize(best_x, max_iter=3000, deadline=deadline - 45, lr=0.005)
            if e_adam < best_e:
                best_x = x_adam
                best_e = e_adam
        except Exception:
            pass
    
    # Conjugate gradient polish
    if time.time() < deadline - 40:
        try:
            x_cg, e_cg = _conjugate_gradient(best_x, max_iter=5000, tol=1e-14, deadline=deadline - 35)
            if e_cg < best_e:
                best_x = x_cg
                best_e = e_cg
        except Exception:
            pass
    
    # Basin hopping
    if time.time() < deadline - 30:
        try:
            x_bh, e_bh = _basin_hopping(best_x, n_restarts=6, deadline=deadline - 25)
            if e_bh < best_e:
                best_x = x_bh
                best_e = e_bh
        except Exception:
            pass
    
    # Another L-BFGS pass after basin hopping
    if time.time() < deadline - 20:
        try:
            x_opt, e_opt = _lbfgs_optimize(best_x, maxiter=300, ftol=1e-14, gtol=1e-12)
            if e_opt < best_e:
                best_x = x_opt
                best_e = e_opt
        except Exception:
            pass
    
    # Gradient descent polish
    if time.time() < deadline - 15:
        try:
            x_final, e_final = _gradient_descent_polish(best_x, max_iter=10000, tol=1e-15, deadline=deadline - 10)
            if e_final < best_e:
                best_x = x_final
                best_e = e_final
        except Exception:
            pass
    
    # High precision polish
    if time.time() < deadline - 8:
        try:
            x_hp, e_hp = _high_precision_polish(best_x, deadline=deadline - 4, max_iter=15000)
            if e_hp < best_e:
                best_x = x_hp
                best_e = e_hp
        except Exception:
            pass
    
    # Final CG pass
    if time.time() < deadline - 4:
        try:
            x_cg2, e_cg2 = _conjugate_gradient(best_x, max_iter=5000, tol=1e-15, deadline=deadline - 2)
            if e_cg2 < best_e:
                best_x = x_cg2
                best_e = e_cg2
        except Exception:
            pass
    
    # Final high precision polish
    if time.time() < deadline - 2:
        try:
            x_hp2, e_hp2 = _high_precision_polish(best_x, deadline=deadline - 0.5, max_iter=10000)
            if e_hp2 < best_e:
                best_x = x_hp2
                best_e = e_hp2
        except Exception:
            pass
    
    # Final projection and validation
    best_x = _project_to_sphere(best_x)
    
    # Ensure no points are too close
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_x[i] - best_x[j])
            if d < 1e-9:
                rng = np.random.RandomState(12345 + i * 100 + j)
                best_x[j] += 1e-10 * rng.randn(3)
                best_x = _project_to_sphere(best_x)
    
    # Final check for non-finite values
    if not np.all(np.isfinite(best_x)):
        best_x = _fibonacci_sphere(N)
        best_x = _project_to_sphere(best_x)
    
    return best_x
# EVOLVE-BLOCK-END
