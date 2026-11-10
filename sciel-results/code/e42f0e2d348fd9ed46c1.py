"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START

def _log_energy(pts):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    iu = np.triu_indices(n, k=1)
    d_sq = dist_sq[iu]
    d = np.sqrt(d_sq)
    d = np.maximum(d, 1e-15)
    return math.fsum(math.log(1.0 / di) for di in d)

def _log_energy_grad(pts):
    """Compute logarithmic energy and its gradient w.r.t. free R^3 coordinates.
    
    Energy = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log|d_ij|
    Gradient for point i: -sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    """
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    
    inv_dist_sq = np.zeros_like(dist_sq)
    off_diag = dist_sq > 1e-30
    inv_dist_sq[off_diag] = 1.0 / dist_sq[off_diag]
    
    grad = -np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(dist_sq[iu])
    d = np.maximum(d, 1e-15)
    energy = math.fsum(math.log(1.0 / di) for di in d)
    
    return energy, grad

def _project_to_tangent(pts, grad):
    """Project gradient onto tangent space of unit sphere."""
    return grad - np.sum(grad * pts, axis=1, keepdims=True) * pts

def _make_fibonacci_sphere():
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    return pts

def _random_sphere(n, rng=None):
    """Generate n random points on the unit sphere."""
    if rng is None:
        rng = np.random
    pts = rng.randn(n, 3)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

def _spherical_design(n, rng=None):
    """Generate points using a spherical design approach for better initial coverage."""
    if rng is None:
        rng = np.random
    i = np.arange(n) + 0.5
    pattern = rng.randint(5)
    if pattern == 0:
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    elif pattern == 1:
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(2.0)) * i
    elif pattern == 2:
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(3.0)) * i
    elif pattern == 3:
        z = 1.0 - 2.0 * (i + 0.3) / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i + 0.5
    else:
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(1.618)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    pts = pts / norms
    return pts

def _antipodal_pairs(seed=None):
    """Generate points as antipodal pairs with random orientations."""
    rng = np.random.default_rng(seed)
    n_pairs = N // 2
    pts = rng.normal(size=(n_pairs, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / np.maximum(norms, 1e-15)
    all_pts = np.vstack([pts, -pts])
    return all_pts

def _clustered_start(seed=None):
    """Generate clustered starting points for diversity."""
    rng = np.random.default_rng(seed)
    n_clusters = 10
    pts_per_cluster = N // n_clusters
    cluster_centers = rng.normal(size=(n_clusters, 3))
    norms = np.linalg.norm(cluster_centers, axis=1, keepdims=True)
    cluster_centers = cluster_centers / np.maximum(norms, 1e-15)
    pts = []
    for center in cluster_centers:
        offsets = rng.normal(0, 0.3, (pts_per_cluster, 3))
        cluster_pts = center + offsets
        norms = np.linalg.norm(cluster_pts, axis=1, keepdims=True)
        cluster_pts = cluster_pts / np.maximum(norms, 1e-15)
        pts.append(cluster_pts)
    return np.vstack(pts)

def _lbfgs_optimize(pts0, maxiter=2000, ftol=1e-15, gtol=1e-12, time_budget=None):
    """Optimize using L-BFGS-B with tight tolerances, starting from pts0."""
    pts = pts0.copy()
    best_pts = pts.copy()
    best_energy = _log_energy(pts)
    start_time = time.time()
    
    def objective(x_flat):
        if time_budget is not None and time.time() - start_time > time_budget:
            raise RuntimeError("Time budget exceeded")
        pts_2d = x_flat.reshape(-1, 3)
        norms = np.linalg.norm(pts_2d, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-15)
        pts_norm = pts_2d / norms
        e, _ = _log_energy_grad(pts_norm)
        if e < best_energy:
            nonlocal best_energy, best_pts
            best_energy = e
            best_pts = pts_norm.copy()
        return e
    
    def gradient(x_flat):
        if time_budget is not None and time.time() - start_time > time_budget:
            raise RuntimeError("Time budget exceeded")
        pts_2d = x_flat.reshape(-1, 3)
        norms = np.linalg.norm(pts_2d, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-15)
        pts_norm = pts_2d / norms
        _, g = _log_energy_grad(pts_norm)
        g_tan = _project_to_tangent(pts_norm, g)
        return g_tan.flatten()
    
    x0 = pts.flatten()
    
    try:
        result = minimize(
            objective,
            x0,
            jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'maxfun': maxiter * 5, 'ftol': ftol, 'gtol': gtol}
        )
        
        pts_opt = result.x.reshape(-1, 3)
        norms = np.linalg.norm(pts_opt, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-15)
        pts_opt = pts_opt / norms
        e_opt = _log_energy(pts_opt)
        if e_opt < best_energy:
            best_energy = e_opt
            best_pts = pts_opt
    except RuntimeError:
        pass
    except Exception:
        pass
    
    return best_pts, best_energy

def _conjugate_gradient_optimize(pts0, maxiter=5000, ftol=1e-14, gtol=1e-11, time_budget=None):
    """Optimize using conjugate gradient method on the sphere."""
    pts = pts0.copy()
    best_pts = pts.copy()
    best_energy = _log_energy(pts)
    start_time = time.time()
    
    _, grad0 = _log_energy_grad(pts)
    grad_tan = _project_to_tangent(pts, grad0)
    
    d = -grad_tan
    alpha = 0.1
    
    for it in range(maxiter):
        if time_budget is not None and time.time() - start_time > time_budget:
            break
        
        best_alpha = alpha
        best_e = best_energy
        alpha_ls = alpha
        for _ in range(10):
            pts_trial = pts + alpha_ls * d
            norms = np.linalg.norm(pts_trial, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-15)
            pts_trial = pts_trial / norms
            e_trial = _log_energy(pts_trial)
            if e_trial < best_e:
                best_alpha = alpha_ls
                best_e = e_trial
            else:
                alpha_ls *= 0.5
                if alpha_ls < 1e-12:
                    break
        
        if best_e >= best_energy:
            alpha *= 0.5
            if alpha < 1e-12:
                break
            continue
        
        pts = pts + best_alpha * d
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-15)
        pts = pts / norms
        
        if best_e < best_energy:
            best_energy = best_e
            best_pts = pts.copy()
        
        _, grad_new = _log_energy_grad(pts)
        grad_new_tan = _project_to_tangent(pts, grad_new)
        
        if np.linalg.norm(grad_new_tan) < gtol:
            break
        
        num = np.sum(grad_new_tan * (grad_new_tan - grad_tan))
        den = np.sum(grad_tan * grad_tan) + 1e-15
        beta = max(0, num / den)
        
        d = -grad_new_tan + beta * d
        grad_tan = grad_new_tan
        alpha = best_alpha
    
    return best_pts, best_energy

def _adam_optimize(pts0, time_limit=100.0, lr_init=0.01, no_improve_break=500):
    """Optimize points on the sphere using Adam gradient descent with adaptive learning rate."""
    pts = pts0.copy()
    best_pts = pts.copy()
    best_energy = _log_energy(pts)
    
    lr = lr_init
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    
    m = np.zeros_like(pts)
    v = np.zeros_like(pts)
    t = 0
    
    start_time = time.time()
    no_improve_steps = 0
    max_no_improve = no_improve_break
    
    while time.time() - start_time < time_limit:
        t += 1
        energy, grad = _log_energy_grad(pts)
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
            no_improve_steps = 0
        else:
            no_improve_steps += 1
        
        if no_improve_steps > max_no_improve:
            break
            
        grad_tangent = _project_to_tangent(pts, grad)
        
        m = beta1 * m + (1 - beta1) * grad_tangent
        v = beta2 * v + (1 - beta2) * (grad_tangent * grad_tangent)
        
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        
        step = lr * m_hat / (np.sqrt(v_hat) + epsilon)
        pts_new = pts - step
        
        norms = np.linalg.norm(pts_new, axis=1, keepdims=True)
        pts_new = pts_new / norms
        
        if not np.all(np.isfinite(pts_new)):
            break
            
        pts = pts_new
        
        if t % 50 == 0:
            current_energy = _log_energy(pts)
            if current_energy > best_energy + 1.0:
                lr *= 0.95
                if lr < 1e-7:
                    break

    return best_pts, best_energy

def _refine(pts, time_budget=5.0):
    """High-precision refinement using adaptive line search with warm-started step size."""
    start = time.time()
    pts = pts.copy()
    best_e = _log_energy(pts)
    last_lr = 0.01
    
    for step in range(500):
        if time.time() - start > time_budget:
            break
        
        e, g = _log_energy_grad(pts)
        
        gnorm = np.linalg.norm(g)
        if gnorm < 1e-15:
            break
        
        best_e_local = e
        best_pts_local = pts.copy()
        best_lr = 0.0
        lr = max(last_lr * 2.0, 0.01)
        
        for _ in range(25):
            new_pts = pts - lr * g
            norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-15)
            new_pts = new_pts / norms
            new_e, _ = _log_energy_grad(new_pts)
            
            if new_e < best_e_local:
                best_e_local = new_e
                best_pts_local = new_pts
                best_lr = lr
                last_lr = lr
            else:
                lr *= 0.5
                if lr < 1e-12:
                    break
        
        if np.allclose(best_pts_local, pts, atol=1e-16):
            break
        
        pts = best_pts_local
        best_e = best_e_local
    
    return pts

def _high_precision_polish(pts, time_budget=5.0):
    """Final ultra-precise polish with very tight tolerances."""
    start = time.time()
    pts = pts.copy()
    
    for step in range(100):
        if time.time() - start > time_budget:
            break
        
        e, g = _log_energy_grad(pts)
        
        gnorm = np.linalg.norm(g)
        if gnorm < 1e-18:
            break
        
        best_e_local = e
        best_pts_local = pts.copy()
        lr = 0.001
        
        for _ in range(30):
            new_pts = pts - lr * g
            norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-15)
            new_pts = new_pts / norms
            new_e, _ = _log_energy_grad(new_pts)
            
            if new_e < best_e_local:
                best_e_local = new_e
                best_pts_local = new_pts
            else:
                lr *= 0.5
                if lr < 1e-15:
                    break
        
        if np.allclose(best_pts_local, pts, atol=1e-18):
            break
        
        pts = best_pts_local
    
    return pts

def construct():
    """Construct 100 points on the unit sphere minimizing logarithmic energy."""
    deadline = time.time() + 149.0
    
    inits = []
    inits.append(_make_fibonacci_sphere())
    rng_init = np.random.RandomState(123)
    inits.append(_spherical_design(N, rng_init))
    inits.append(_random_sphere(N, np.random.RandomState(456)))
    inits.append(_random_sphere(N, np.random.RandomState(789)))
    inits.append(_antipodal_pairs(seed=42))
    inits.append(_clustered_start(seed=99))
    
    best_pts = inits[0].copy()
    best_energy = _log_energy(inits[0])
    
    # Phase 1: L-BFGS from each initialization
    for init_pts in inits:
        if time.time() > deadline - 40.0:
            break
        remaining = deadline - time.time() - 35.0
        maxiter = min(3000, max(500, int(remaining * 20)))
        pts_opt, energy_opt = _lbfgs_optimize(init_pts, maxiter=maxiter, ftol=1e-15, gtol=1e-12)
        if energy_opt < best_energy:
            best_energy = energy_opt
            best_pts = pts_opt
    
    # Phase 2: Multi-restart with diverse strategies
    rng = np.random.RandomState(42)
    n_restarts = 40
    perturbation_scales = [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6]
    
    for r in range(n_restarts):
        if time.time() > deadline - 30.0:
            break
        
        strategy = r % 8
        if strategy == 0:
            pts_rand = best_pts + perturbation_scales[r % len(perturbation_scales)] * rng.randn(N, 3)
        elif strategy == 1:
            pts_rand = inits[0] + perturbation_scales[(r + 1) % len(perturbation_scales)] * rng.randn(N, 3)
        elif strategy == 2:
            pts_rand = _random_sphere(N, rng)
        elif strategy == 3:
            pts_rand = _spherical_design(N, rng)
        elif strategy == 4:
            pts_rand = 0.5 * best_pts + 0.5 * _random_sphere(N, rng)
        elif strategy == 5:
            pts_rand = best_pts + 0.5 * rng.randn(N, 3)
        elif strategy == 6:
            pts_rand = inits[1] + perturbation_scales[(r + 2) % len(perturbation_scales)] * rng.randn(N, 3)
        else:
            pts_rand = inits[2] + perturbation_scales[(r + 3) % len(perturbation_scales)] * rng.randn(N, 3)
        
        norms = np.linalg.norm(pts_rand, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-15)
        pts_rand = pts_rand / norms
        
        remaining = deadline - time.time()
        if remaining < 15.0:
            break
        
        maxiter = min(4000, max(800, int(remaining * 20)))
        pts_opt, energy_opt = _lbfgs_optimize(pts_rand, maxiter=maxiter, ftol=1e-14, gtol=1e-11)
        if energy_opt < best_energy:
            best_energy = energy_opt
            best_pts = pts_opt
    
    # Phase 3: Conjugate gradient
    if time.time() < deadline - 20.0:
        remaining = max(deadline - time.time() - 15.0, 3.0)
        maxiter_cg = int(remaining * 40)
        pts_opt, energy_opt = _conjugate_gradient_optimize(best_pts, maxiter=maxiter_cg, ftol=1e-14, gtol=1e-11)
        if energy_opt < best_energy:
            best_energy = energy_opt
            best_pts = pts_opt
    
    # Phase 4: Adam fine-tuning
    if time.time() < deadline - 15.0:
        remaining = max(deadline - time.time() - 10.0, 2.0)
        pts_opt, energy_opt = _adam_optimize(best_pts, time_limit=remaining, lr_init=0.001, no_improve_break=200)
        if energy_opt < best_energy:
            best_energy = energy_opt
            best_pts = pts_opt
    
    # Phase 5: Refinement with adaptive line search
    if time.time() < deadline - 12.0:
        remaining = max(deadline - time.time() - 10.0, 2.0)
        pts_opt = _refine(best_pts, time_budget=remaining)
        e_opt = _log_energy(pts_opt)
        if e_opt < best_energy:
            best_energy = e_opt
            best_pts = pts_opt
    
    # Phase 6: Final L-BFGS polish
    if time.time() < deadline - 8.0:
        pts_opt, energy_opt = _lbfgs_optimize(best_pts, maxiter=8000, ftol=1e-16, gtol=1e-13)
        if energy_opt < best_energy:
            best_energy = energy_opt
            best_pts = pts_opt
    
    # Phase 7: Ultra-tight L-BFGS
    if time.time() < deadline - 5.0:
        pts_opt, energy_opt = _lbfgs_optimize(best_pts, maxiter=15000, ftol=1e-16, gtol=1e-14)
        if energy_opt < best_energy:
            best_energy = energy_opt
            best_pts = pts_opt
    
    # Phase 8: High-precision polish
    if time.time() < deadline - 3.0:
        remaining = max(deadline - time.time() - 2.0, 1.0)
        pts_opt = _high_precision_polish(best_pts, time_budget=remaining)
        e_opt = _log_energy(pts_opt)
        if e_opt < best_energy:
            best_energy = e_opt
            best_pts = pts_opt
    
    # Ensure points are on the unit sphere and valid
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    best_pts = best_pts / norms
    
    # Check for very close points and perturb if necessary
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(N, k=1)
    if np.min(dist[iu]) < 1e-9:
        for _ in range(100):
            min_idx = np.argmin(dist[iu])
            i, j = iu[0][min_idx], iu[1][min_idx]
            rand_dir = np.random.randn(3)
            rand_dir /= np.linalg.norm(rand_dir)
            best_pts[j] = best_pts[j] + 1e-6 * rand_dir
            best_pts[j] /= np.linalg.norm(best_pts[j])
            diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
            dist = np.linalg.norm(diff, axis=2)
            if np.min(dist[iu]) >= 1e-9:
                break
    
    return best_pts

# EVOLVE-BLOCK-END
