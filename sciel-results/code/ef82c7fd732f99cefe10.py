"""Logarithmic energy minimization for N points on the unit sphere.
Combines multi-start BFGS optimization with Riemannian Basin Hopping 
using exponential map perturbations to escape local minima."""
import time
import math
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

N = 100
_DEADLINE = 145.0
_START = time.time()


def _remaining():
    return _DEADLINE - (time.time() - _START)


def _project(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return X / n


def _energy(pts):
    """Fast energy computation using matrix operations."""
    n = pts.shape[0]
    # For unit vectors, dist_sq = 2 - 2*(x_i . x_j)
    gram = pts @ pts.T
    dist_sq = 2.0 - 2.0 * gram
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(np.maximum(dist_sq, 1e-30))
    iu = np.triu_indices(n, k=1)
    return float(-np.sum(np.log(dist[iu])))


def _energy_grad(pts):
    """Compute energy and gradient (projected onto tangent space)."""
    n = pts.shape[0]
    gram = pts @ pts.T
    dist_sq = 2.0 - 2.0 * gram
    np.fill_diagonal(dist_sq, 1.0)
    dist_sq = np.maximum(dist_sq, 1e-30)
    dist = np.sqrt(dist_sq)
    
    iu = np.triu_indices(n, k=1)
    energy = float(-np.sum(np.log(dist[iu])))
    
    # W_ij = 1/dist_sq_ij for i != j
    W = np.where(dist_sq > 1e-30, 1.0 / dist_sq, 0.0)
    np.fill_diagonal(W, 0.0)
    
    # Gradient of sum_{i<j} log(1/|x_i - x_j|) w.r.t. x_k:
    # d/dx_k sum_{i<j} -log(|x_i - x_j|) = sum_{j!=k} (x_k - x_j)/|x_k - x_j|^2
    # = (sum_j W_kj) * x_k - sum_j W_kj * x_j
    sum_W = W.sum(axis=1, keepdims=True)
    Wx = W @ pts
    raw_grad = sum_W * pts - Wx
    
    # Project onto tangent space: grad_tan = grad - (grad . x) x
    dot = np.sum(raw_grad * pts, axis=1, keepdims=True)
    grad = raw_grad - dot * pts
    
    return energy, grad


def _precise_energy(pts):
    """Compute energy using math.fsum for accuracy."""
    n = pts.shape[0]
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-9:
                return float('inf')
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)


def _fibonacci_spiral(n, offset=0.5):
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0, 1.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(rng):
    v = rng.standard_normal((N, 3))
    return _project(v)


def _sobol_sphere(n, seed=0):
    """Generate points on sphere using Sobol sequence."""
    sampler_2d = qmc.Sobol(d=2, scramble=True, seed=seed)
    m = int(2 ** np.ceil(np.log2(n)))
    y = sampler_2d.random(m)
    
    theta = np.arccos(1 - 2 * y[:, 0])
    phi = 2 * np.pi * y[:, 1]
    
    x = np.sin(theta) * np.cos(phi)
    y_coord = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    
    pts = np.column_stack([x, y_coord, z])
    return pts[:n]


def _optimize(pts, deadline, max_iter=500, ftol=1e-12, gtol=1e-10):
    """Run BFGS optimization from a starting point."""
    n = pts.shape[0]
    
    def objective_and_grad(flat):
        p = flat.reshape(n, 3)
        norms = np.linalg.norm(p, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        p = p / norms
        e, g = _energy_grad(p)
        return e, g.reshape(-1)
    
    try:
        x0 = pts.reshape(-1)
        # Use L-BFGS-B for better memory efficiency and robustness
        result = minimize(
            objective_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={
                'maxiter': max_iter,
                'gtol': gtol,
                'ftol': ftol,
                'maxfun': max_iter * 2,
            }
        )
        new_pts = _project(result.x.reshape(n, 3))
        # Verify improvement
        e_new = _energy(new_pts)
        e_old = _energy(pts)
        if e_new < e_old:
            return new_pts
        return pts
    except Exception:
        return pts


def _riemannian_perturb(pts, rng, step_size):
    """
    Perturb points on the sphere using Riemannian exponential map.
    For each point x, pick a random tangent vector v, normalize it to length step_size,
    and move along the geodesic.
    """
    n = pts.shape[0]
    # Generate random vectors
    v = rng.standard_normal((n, 3))
    
    # Project v to tangent space at pts
    # v_tangent = v - (v . x) x
    dot_vx = np.sum(v * pts, axis=1, keepdims=True)
    v_tangent = v - dot_vx * pts
    
    # Normalize tangent vectors to unit length
    v_norms = np.linalg.norm(v_tangent, axis=1, keepdims=True)
    v_norms = np.where(v_norms < 1e-12, 1.0, v_norms)
    v_unit = v_tangent / v_norms
    
    # Move along geodesic by step_size
    # exp_x(v) = cos(||v||) x + sin(||v||) (v/||v||)
    # Here ||v|| = step_size
    cos_s = np.cos(step_size)
    sin_s = np.sin(step_size)
    
    new_pts = cos_s * pts + sin_s * v_unit
    return _project(new_pts)


def _riemannian_perturb_subset(pts, rng, step_size, k):
    """
    Perturb only a random subset of k points using Riemannian exponential map.
    This allows more targeted escapes from local minima.
    """
    n = pts.shape[0]
    new_pts = pts.copy()
    
    # Choose which points to perturb
    idx = rng.choice(n, size=k, replace=False)
    
    # Generate random tangent vectors for selected points
    v = rng.standard_normal((k, 3))
    pts_sel = pts[idx]
    
    # Project v to tangent space
    dot_vx = np.sum(v * pts_sel, axis=1, keepdims=True)
    v_tangent = v - dot_vx * pts_sel
    
    # Normalize
    v_norms = np.linalg.norm(v_tangent, axis=1, keepdims=True)
    v_norms = np.where(v_norms < 1e-12, 1.0, v_norms)
    v_unit = v_tangent / v_norms
    
    # Move along geodesic
    cos_s = np.cos(step_size)
    sin_s = np.sin(step_size)
    new_pts[idx] = cos_s * pts_sel + sin_s * v_unit
    
    return _project(new_pts)


def _valid(X):
    if X.shape != (N, 3) or not np.all(np.isfinite(X)):
        return False
    if np.any(np.linalg.norm(X, axis=1) < 1e-12):
        return False
    d2 = np.einsum('ij,ij->i', X, X)[:, None] + np.einsum('ij,ij->i', X, X)[None, :] \
         - 2.0 * (X @ X.T)
    np.fill_diagonal(d2, np.inf)
    if np.min(d2) < (1e-9) ** 2:
        return False
    return True


def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Sobol sequences
    for s in range(20):
        initializations.append(_sobol_sphere(N, seed=s))
        
    # 2. Fibonacci spirals
    for offset in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        initializations.append(_fibonacci_spiral(N, offset=offset))
        
    # 3. Jittered Fibonacci spirals
    base = _fibonacci_spiral(N, offset=0.5)
    for s in range(12):
        jitter_scale = 0.03 + 0.015 * s
        jittered = base + jitter_scale * rng.standard_normal((N, 3))
        initializations.append(_project(jittered))
        
    # 4. Random points
    for s in range(20):
        initializations.append(_random_sphere(np.random.default_rng(100 + s)))
        
    # 5. Antipodal pairs + jitter (exploits symmetry)
    base_antipodal = np.zeros((N, 3))
    for i in range(N // 2):
        v = rng.standard_normal(3)
        v = v / np.linalg.norm(v)
        base_antipodal[2*i] = v
        base_antipodal[2*i+1] = -v
    if N % 2 == 1:
        v = rng.standard_normal(3)
        v = v / np.linalg.norm(v)
        base_antipodal[N-1] = v
    for s in range(5):
        jitter_scale = 0.05 + 0.02 * s
        jittered = base_antipodal + jitter_scale * rng.standard_normal((N, 3))
        initializations.append(_project(jittered))
        
    best_pts = None
    best_energy = float('inf')
    
    # Score all initializations
    scored_inits = []
    for init_pts in initializations:
        if time.time() > deadline - 30.0:
            break
        e_quick = _energy(init_pts)
        scored_inits.append((e_quick, init_pts))
        
    scored_inits.sort(key=lambda x: x[0])
    
    # Phase 1: Optimize top candidates with L-BFGS-B
    top_candidates = scored_inits[:40]
    
    for e_quick, init_pts in top_candidates:
        if time.time() > deadline - 30.0:
            break
        
        remaining = deadline - 30.0 - time.time()
        if remaining < 1.0:
            break
        max_iter = min(1000, int(remaining * 30))
        max_iter = max(200, max_iter)
        
        optimized = _optimize(init_pts, deadline, max_iter=max_iter, ftol=1e-13, gtol=1e-11)
        
        e_precise = _precise_energy(optimized)
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = optimized.copy()
            
    # Phase 2: Riemannian Basin Hopping with adaptive strategy
    if best_pts is not None and time.time() < deadline - 8.0:
        # Cooling schedule: start large, decrease over time
        # Alternate between full perturbations and subset perturbations
        base_step = 0.5
        n_total_perturb = 0
        max_total_perturb = 150
        stuck_counter = 0
        best_energy_bh = best_energy
        
        while time.time() < deadline - 4.0 and n_total_perturb < max_total_perturb:
            # Adaptive step size: decreases as we go
            progress = n_total_perturb / max_total_perturb
            step_size = base_step * (1.0 - 0.6 * progress) + 0.03
            
            # More diverse perturbation strategies
            strategy = n_total_perturb % 4
            if strategy == 0:
                # Full perturbation with moderate step
                perturbed = _riemannian_perturb(best_pts, rng, step_size)
            elif strategy == 1:
                # Subset perturbation: move ~25% of points
                k_subset = max(5, N // 4)
                perturbed = _riemannian_perturb_subset(best_pts, rng, step_size * 1.3, k_subset)
            elif strategy == 2:
                # Subset perturbation: move ~10% of points
                k_subset = max(3, N // 10)
                perturbed = _riemannian_perturb_subset(best_pts, rng, step_size * 0.8, k_subset)
            else:
                # Small full perturbation for fine-tuning
                perturbed = _riemannian_perturb(best_pts, rng, step_size * 0.5)
            
            # Refine with adaptive iterations
            remaining = deadline - 4.0 - time.time()
            if remaining > 2.0:
                max_iter = min(500, int(remaining * 40))
                max_iter = max(100, max_iter)
            elif remaining > 1.0:
                max_iter = 200
            else:
                max_iter = 80
            
            refined = _optimize(perturbed, deadline, max_iter=max_iter, ftol=1e-13, gtol=1e-11)
            
            e_precise = _precise_energy(refined)
            if e_precise < best_energy:
                best_energy = e_precise
                best_pts = refined.copy()
                stuck_counter = 0
            else:
                stuck_counter += 1
            
            n_total_perturb += 1
            
            # If stuck for 10 iterations, try a larger kick
            if stuck_counter >= 10:
                kick = _riemannian_perturb(best_pts, rng, 0.7)
                refined_kick = _optimize(kick, deadline, max_iter=300, ftol=1e-12, gtol=1e-10)
                e_kick = _precise_energy(refined_kick)
                if e_kick < best_energy:
                    best_energy = e_kick
                    best_pts = refined_kick.copy()
                    stuck_counter = 0
                else:
                    stuck_counter = 0  # Reset after kick attempt
                    
    # Phase 3: Final ultra-precise polish
    if best_pts is not None and time.time() < deadline - 2.0:
        # Multiple polish passes with decreasing tolerance
        polish_tolerances = [(1e-13, 1e-11), (1e-14, 1e-12), (1e-15, 1e-13)]
        for ftol_p, gtol_p in polish_tolerances:
            if time.time() > deadline - 1.0:
                break
            remaining = deadline - 1.0 - time.time()
            if remaining < 0.5:
                break
            max_iter = min(2000, int(remaining * 60))
            max_iter = max(200, max_iter)
            refined = _optimize(best_pts, deadline, max_iter=max_iter, ftol=ftol_p, gtol=gtol_p)
            e_precise = _precise_energy(refined)
            if e_precise < best_energy:
                best_energy = e_precise
                best_pts = refined.copy()
    
    if best_pts is None:
        best_pts = _fibonacci_spiral(N, offset=0.5)
    
    # Final normalization
    best_pts = _project(best_pts)
    
    # Final validation
    final_energy = _precise_energy(best_pts)
    if not np.isfinite(final_energy):
        best_pts = _fibonacci_spiral(N, offset=0.5)
        best_pts = _project(best_pts)
    
    return best_pts
