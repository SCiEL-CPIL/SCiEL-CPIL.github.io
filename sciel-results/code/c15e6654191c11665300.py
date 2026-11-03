"""Logarithmic energy minimization for 100 points on the unit sphere.
Combines multi-start L-BFGS-B with diverse initializations and Newton polish."""
import time
import math
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc
from scipy.sparse import diags

N = 100
DEADLINE = 145.0  # safety margin before 150s kill
_START = time.time()


def _remaining():
    return DEADLINE - (time.time() - _START)


def _project(X):
    """Project points to unit sphere."""
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return X / n


def _energy(pts):
    """Fast energy computation using matrix operations."""
    n = pts.shape[0]
    sq_norms = np.einsum('ij,ij->i', pts, pts)
    dist_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2.0 * (pts @ pts.T)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(np.maximum(dist_sq, 1e-30))
    iu = np.triu_indices(n, k=1)
    return float(-np.sum(np.log(dist[iu])))


def _energy_grad(pts):
    """Compute energy and gradient (projected onto tangent space).
    Returns (energy, gradient) where gradient is the true gradient of the energy."""
    n = pts.shape[0]
    sq_norms = np.einsum('ij,ij->i', pts, pts)
    dist_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2.0 * (pts @ pts.T)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(np.maximum(dist_sq, 1e-30))
    
    iu = np.triu_indices(n, k=1)
    energy = float(-np.sum(np.log(dist[iu])))
    
    # Gradient of E = sum log(1/d) = -sum log(d)
    # dE/dx_i = - sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    W = 1.0 / dist_sq
    np.fill_diagonal(W, 0.0)
    
    # sum_j W_ij * (x_i - x_j) = x_i * sum_j W_ij - sum_j W_ij * x_j
    sum_W = W.sum(axis=1, keepdims=True)
    Wx = W @ pts
    raw_grad = sum_W * pts - Wx
    
    # raw_grad is sum (x_i - x_j)/d^2
    # True gradient is -raw_grad
    grad = -raw_grad
    
    # Project onto tangent space: grad - (grad . x_i) x_i
    dot = np.sum(grad * pts, axis=1, keepdims=True)
    grad = grad - dot * pts
    
    return energy, grad


def _hessian_vector(pts, v):
    """Compute Hessian-vector product for the energy on the sphere.
    Used for Newton-like refinement.
    
    The Hessian of E = -sum_{i<j} log(|x_i - x_j|) with respect to x_i, x_j.
    For each pair (i,j): d^2 E / dx_i dx_j = -delta_ij/d^2 + (x_i-x_j)(x_i-x_j)^T/d^4
    """
    n = pts.shape[0]
    sq_norms = np.einsum('ij,ij->i', pts, pts)
    dist_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2.0 * (pts @ pts.T)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(np.maximum(dist_sq, 1e-30))
    
    # Hessian block (i,j) for i != j:
    # H_ij = (x_i - x_j)(x_i - x_j)^T / d^4
    # H_ii = -sum_{j!=i} [delta_ii/d^2 - (x_i-x_j)(x_i-x_j)^T/d^4]
    #       = -sum_{j!=i} 1/d^2 + sum_{j!=i} (x_i-x_j)(x_i-x_j)^T/d^4
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    diff_sq = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(diff_sq, 1.0)
    
    # Hessian-vector product: (H v)_i = sum_j H_ij v_j
    # H_ij v_j = (x_i - x_j) * ((x_i - x_j) . v_j) / d^4
    #           = (x_i - x_j) * dot_ij / d^4
    # where dot_ij = (x_i - x_j) . v_j
    
    dot = np.einsum('ijk,jk->ij', diff, v)  # (n, n)
    dot /= diff_sq ** 2  # divide by d^4
    np.fill_diagonal(dot, 0.0)
    
    # (H v)_i = sum_j (x_i - x_j) * dot_ij
    #         = x_i * sum_j dot_ij - sum_j dot_ij * x_j
    sum_dot = dot.sum(axis=1, keepdims=True)  # (n, 1)
    Hv = sum_dot * pts - (dot @ pts)
    
    # Diagonal correction: for i=j, we need -sum_{j!=i} 1/d^2 * v_i
    # H_ii v_i = -sum_{j!=i} 1/d_ii^2 * v_i + sum_{j!=i} (x_i-x_j)((x_i-x_j).v_i)/d^4
    # The first term: -sum_j 1/d^2 * v_i
    diag_sum = (1.0 / diff_sq).sum(axis=1)  # sum_j 1/d_ij^2
    Hv -= diag_sum[:, np.newaxis] * v
    
    # Project onto tangent space
    dot_proj = np.sum(Hv * pts, axis=1, keepdims=True)
    Hv = Hv - dot_proj * pts
    
    return Hv


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
    """Golden-angle (Fibonacci) spiral points on the unit sphere."""
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0, 1.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(rng):
    """Generate n random points on unit sphere."""
    v = rng.standard_normal((N, 3))
    return _project(v)


def _sobol_sphere(n, seed=0):
    """Generate points on sphere using Sobol sequence."""
    sampler_2d = qmc.Sobol(d=2, scramble=True, seed=seed)
    m = int(2 ** np.ceil(np.log2(n)))
    y = sampler_2d.random(m)
    
    # Map to sphere using spherical coordinates
    theta = np.arccos(1 - 2 * y[:, 0])
    phi = 2 * np.pi * y[:, 1]
    
    x = np.sin(theta) * np.cos(phi)
    y_coord = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    
    pts = np.column_stack([x, y_coord, z])
    return pts[:n]


def _valid(X):
    """Check if configuration is valid."""
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


def _optimize_lbfgs(pts, max_iter=1500, ftol=1e-16, gtol=1e-13):
    """Run L-BFGS-B optimization from a starting point."""
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
        result = minimize(
            objective_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={
                'maxiter': max_iter,
                'gtol': gtol,
                'ftol': ftol,
                'maxls': 50,
                'maxcor': 30,
            }
        )
        new_pts = _project(result.x.reshape(n, 3))
        return new_pts
    except Exception:
        return pts


def _newton_polish(pts, max_iter=50, tol=1e-15):
    """Newton-like refinement using Hessian-vector products.
    Uses conjugate gradient to solve the Newton system.
    """
    n = pts.shape[0]
    best_e = _energy(pts)
    best_pts = pts.copy()
    
    for it in range(max_iter):
        if _remaining() < 3:
            break
        
        e, g = _energy_grad(pts)
        if e < best_e:
            best_e = e
            best_pts = pts.copy()
        
        # Check convergence
        g_norm = np.linalg.norm(g)
        if g_norm < tol:
            break
        
        # Solve H delta = -g using CG with Hessian-vector products
        # H is the projected Hessian
        def hvp(v):
            return _hessian_vector(pts, v)
        
        # Conjugate gradient on the tangent space
        # Initialize
        r = -g  # residual
        p_vec = r.copy()
        rs_old = np.sum(r * r)
        
        delta = np.zeros_like(pts)
        
        for cg_iter in range(100):
            if _remaining() < 3:
                break
            
            Ap = hvp(p_vec)
            
            # Project Ap onto tangent space
            Ap_dot = np.sum(Ap * pts, axis=1, keepdims=True)
            Ap = Ap - Ap_dot * pts
            
            p_dot = np.sum(p_vec * p_vec)
            if p_dot < 1e-30:
                break
            
            alpha = rs_old / p_dot
            delta += alpha * p_vec
            r -= alpha * Ap
            rs_new = np.sum(r * r)
            
            if np.sqrt(rs_new) < tol:
                break
            
            beta = rs_new / rs_old
            p_vec = r + beta * p_vec
            rs_old = rs_new
        
        # Line search: find step size
        step = delta
        max_step_norm = np.max(np.linalg.norm(step, axis=1))
        if max_step_norm > 0.1:
            step = step * (0.1 / max_step_norm)
        
        new_pts = _project(pts + step)
        new_e = _energy(new_pts)
        
        if new_e < e:
            pts = new_pts
        else:
            # Try smaller step
            for scale in [0.5, 0.25, 0.1]:
                trial_pts = _project(pts + scale * step)
                trial_e = _energy(trial_pts)
                if trial_e < e:
                    pts = trial_pts
                    break
            else:
                break
    
    return best_pts


def _polish(pts, max_iter=3000, ftol=1e-18, gtol=1e-15):
    """High-precision polish pass with tighter tolerances using L-BFGS-B."""
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
        result = minimize(
            objective_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={
                'maxiter': max_iter,
                'gtol': gtol,
                'ftol': ftol,
                'maxls': 100,
                'maxcor': 50,
            }
        )
        new_pts = _project(result.x.reshape(n, 3))
        return new_pts
    except Exception:
        return pts


def _gradient_descent(pts, max_iter=500, lr_init=0.005, lr_decay=0.999):
    """Gradient descent with adaptive learning rate on the sphere."""
    n = pts.shape[0]
    lr = lr_init
    best_e = _energy(pts)
    best_pts = pts.copy()
    
    for it in range(max_iter):
        if _remaining() < 5:
            break
        e, g = _energy_grad(pts)
        if e < best_e:
            best_e = e
            best_pts = pts.copy()
        
        # Adaptive step size
        step = lr * g
        # Ensure step doesn't cause points to collapse
        max_step = np.max(np.linalg.norm(step, axis=1))
        if max_step > 0.1:
            step = step * (0.1 / max_step)
        
        pts = _project(pts + step)
        lr *= lr_decay
        if lr < 1e-8:
            break
    
    return best_pts


def _simulated_annealing(pts, max_iter=200, temp_init=0.01, temp_decay=0.995):
    """Simulated annealing on the sphere to escape local minima."""
    n = pts.shape[0]
    rng = np.random.default_rng(12345)
    temp = temp_init
    best_e = _energy(pts)
    best_pts = pts.copy()
    
    for it in range(max_iter):
        if _remaining() < 5:
            break
        e, g = _energy_grad(pts)
        
        # Propose a move: gradient step + random perturbation
        step = 0.005 * g + temp * rng.standard_normal((n, 3))
        max_step = np.max(np.linalg.norm(step, axis=1))
        if max_step > 0.05:
            step = step * (0.05 / max_step)
        
        new_pts = _project(pts + step)
        new_e = _energy(new_pts)
        
        # Accept if better or with probability exp(-(new_e - e)/temp)
        if new_e < e or (np.random.random() < math.exp(-(new_e - e) / (temp + 1e-15))):
            pts = new_pts
            e = new_e
            if e < best_e:
                best_e = e
                best_pts = pts.copy()
        
        temp *= temp_decay
    
    return best_pts


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Sobol sequences (Low discrepancy) - 10 variants
    for s in range(10):
        if _remaining() < 10:
            break
        initializations.append(_sobol_sphere(N, seed=s))
    
    # 2. Fibonacci spirals with various offsets
    for offset in [0.1, 0.3, 0.5, 0.7, 0.9]:
        if _remaining() < 10:
            break
        initializations.append(_fibonacci_spiral(N, offset=offset))
    
    # 3. Jittered Fibonacci spirals
    base = _fibonacci_spiral(N, offset=0.5)
    for s in range(6):
        if _remaining() < 10:
            break
        jitter_scale = 0.03 + 0.01 * s
        jittered = base + jitter_scale * rng.standard_normal((N, 3))
        initializations.append(_project(jittered))
    
    # 4. Random points with different seeds
    for s in range(10):
        if _remaining() < 10:
            break
        initializations.append(_random_sphere(np.random.default_rng(100 + s)))
    
    best_pts = None
    best_energy = float('inf')
    
    # Score all initializations with fast energy
    scored_inits = []
    for init_pts in initializations:
        if _remaining() < 15:
            break
        e_quick = _energy(init_pts)
        scored_inits.append((e_quick, init_pts))
    
    scored_inits.sort(key=lambda x: x[0])
    
    # Optimize top candidates with L-BFGS-B (faster than BFGS for large n)
    top_candidates = scored_inits[:25]
    
    for e_quick, init_pts in top_candidates:
        if _remaining() < 8:
            break
        
        # L-BFGS-B optimization
        optimized = _optimize_lbfgs(init_pts, max_iter=1000, ftol=1e-15, gtol=1e-12)
        
        e_precise = _precise_energy(optimized)
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = optimized.copy()
    
    # Additional refinement: gradient descent + simulated annealing on best
    if best_pts is not None and _remaining() > 15:
        # Gradient descent polish
        gd_pts = _gradient_descent(best_pts, max_iter=300, lr_init=0.003)
        e_gd = _precise_energy(gd_pts)
        if e_gd < best_energy:
            best_energy = e_gd
            best_pts = gd_pts.copy()
        
        # Simulated annealing to escape local minima
        if _remaining() > 15:
            sa_pts = _simulated_annealing(best_pts, max_iter=150, temp_init=0.008)
            e_sa = _precise_energy(sa_pts)
            if e_sa < best_energy:
                best_energy = e_sa
                best_pts = sa_pts.copy()
    
    # Final polish on the global best with tighter tolerances
    if best_pts is not None and _remaining() > 10:
        refined = _polish(best_pts, max_iter=3000, ftol=1e-18, gtol=1e-15)
        e_precise = _precise_energy(refined)
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = refined
    
    # Newton-like polish for final precision
    if best_pts is not None and _remaining() > 5:
        newton_pts = _newton_polish(best_pts, max_iter=30, tol=1e-16)
        e_newton = _precise_energy(newton_pts)
        if e_newton < best_energy:
            best_energy = e_newton
            best_pts = newton_pts
    
    if best_pts is None:
        best_pts = _fibonacci_spiral(N, offset=0.5)
    
    # Final normalization and validation
    best_pts = _project(best_pts)
    
    # Verify no points too close
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < 1e-9:
                # Perturb slightly
                best_pts[j] += 1e-8 * rng.standard_normal(3)
                best_pts = _project(best_pts)
    
    return best_pts
