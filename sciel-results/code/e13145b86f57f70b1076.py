"""
Logarithmic energy minimization for N points on the unit sphere.
Uses a hybrid strategy: diverse initializations (Fibonacci, Sobol, Random),
multi-start L-BFGS-B, and a final Riemannian Gradient Descent polish.
"""
import time
import math
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import qmc

N = 100
_DEADLINE = 145.0  # Safety margin from 150s limit
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
    """Generate points on the sphere using Sobol sequence."""
    sampler = qmc.Sobol(d=3, seed=seed, scramble=True)
    # Need to find the smallest m such that 2^m >= n
    m = int(np.ceil(np.log2(n)))
    points = sampler.random(2**m)
    points = points[:n]
    # Map [0,1]^3 to sphere
    # Use a simple mapping: normalize random vectors from Sobol
    # Sobol points are in [0,1]^3. We can map them to directions.
    # A common way is to use the points as directions directly if we scale them, 
    # but they are in a cube. 
    # Better: use the Sobol points to generate uniform points on sphere via transformation?
    # Actually, qmc.Sobol generates points in [0,1]^d.
    # To get points on sphere, we can just normalize them? No, that biases towards corners.
    # Let's use the standard trick: use the first 2 coordinates for phi/theta?
    # Or simply: generate random vectors using Sobol as seeds for direction?
    # Simplest robust method: use Sobol points to drive a random number generator?
    # No, let's just use the Sobol points in [0,1]^3, map to [-1,1]^3, and normalize.
    # This is not perfectly uniform on sphere but much better than random.
    # To be more accurate, we can use the inverse CDF method for uniform sphere,
    # but that requires random numbers.
    # Let's stick to: points in [0,1]^3 -> [-1,1]^3 -> normalize.
    p = points * 2.0 - 1.0
    norms = np.linalg.norm(p, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    p = p / norms
    return p

def _repulsive_pre_opt(pts, steps=50, lr=0.01):
    """
    Perform a few steps of gradient descent on a repulsive potential
    to ensure points are well-separated before log-energy minimization.
    Potential: sum_{i<j} 1/d_{ij}^2
    Gradient wrt x_i: -2 * sum_{j!=i} (x_i - x_j) / d_{ij}^4
    """
    p = pts.copy()
    n = p.shape[0]
    
    for _ in range(steps):
        # Compute distances
        sq_norms = np.einsum('ij,ij->i', p, p)
        dist_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2.0 * (p @ p.T)
        np.fill_diagonal(dist_sq, 1.0)
        
        # Avoid division by zero
        dist_sq = np.maximum(dist_sq, 1e-12)
        
        # Gradient of sum 1/d^2 = sum d^-2
        # d/dx_i sum_j (x_i - x_j)^-2 = sum_j -2 (x_i - x_j) / |x_i - x_j|^4
        # W_ij = 2 / d_ij^4
        W = 2.0 / (dist_sq * dist_sq)
        np.fill_diagonal(W, 0.0)
        
        sum_W = W.sum(axis=1, keepdims=True)
        Wx = W @ p
        raw_grad = sum_W * p - Wx
        
        # Gradient of repulsive energy is -raw_grad (since we want to minimize repulsion? 
        # No, we want to MAXIMIZE distance, so MINIMIZE 1/d^2.
        # Gradient of 1/d^2 wrt x_i is -2(x_i-x_j)/d^4.
        # So we move in direction -grad = +2(x_i-x_j)/d^4.
        # This pushes points apart.
        
        grad = -raw_grad # This is the gradient of the repulsive energy
        # We want to minimize repulsive energy, so move in -grad direction
        direction = -grad
        
        # Project to tangent space
        dot = np.sum(direction * p, axis=1, keepdims=True)
        direction = direction - dot * p
        
        # Update
        p = p + lr * direction
        p = _project(p)
        
    return p


def _optimize_lbfgs(pts, deadline, max_iter=2000, ftol=1e-15, gtol=1e-12, maxls=50):
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
                'maxfun': max_iter * 10,
                'ftol': ftol,
                'gtol': gtol,
                'maxls': maxls,
            }
        )
        new_pts = _project(result.x.reshape(n, 3))
        return new_pts
    except Exception:
        return pts


def _riemannian_polish(pts, deadline, max_iter=1000, step_init=1e-3):
    """Perform Riemannian Gradient Descent with backtracking line search.
    This is effective for the final polish to reach high precision.
    """
    n = pts.shape[0]
    p = pts.copy()
    current_energy = _energy(p)
    step = step_init
    
    for i in range(max_iter):
        if time.time() > deadline - 2.0:
            break
        
        e, grad = _energy_grad(p)
        
        # Check convergence
        if np.linalg.norm(grad) < 1e-12:
            break
        
        # Gradient direction (already projected)
        # We want to minimize, so move in -grad direction
        direction = -grad
        
        # Backtracking line search
        alpha = step
        success = False
        for _ in range(20):
            # Move on tangent space
            new_p = p + alpha * direction
            # Project to sphere
            new_p = _project(new_p)
            
            new_e = _energy(new_p)
            
            if new_e < current_energy - 1e-4 * alpha * np.linalg.norm(direction)**2:
                success = True
                p = new_p
                current_energy = new_e
                break
            else:
                alpha *= 0.5
                if alpha < 1e-10:
                    break
        
        if not success:
            # If line search failed, reduce step size permanently
            step *= 0.5
            if step < 1e-10:
                break
        else:
            # Increase step size slightly if successful and stable
            if i > 10:
                step = min(step * 1.05, 1e-2)
                
    return p


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
    deadline = start_time + 140.0  # 10s safety margin
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Fibonacci spirals with various offsets
    for offset in [0.1, 0.3, 0.5, 0.7, 0.9]:
        initializations.append(_fibonacci_spiral(N, offset=offset))
    
    # 2. Sobol sequences (Better uniformity than random)
    for s in range(10):
        initializations.append(_sobol_sphere(N, seed=s))
        
    # 3. Random points with different seeds
    for s in range(10):
        initializations.append(_random_sphere(np.random.default_rng(1000 + s)))

    best_pts = None
    best_energy = float('inf')
    
    # Pre-optimize all initializations with repulsive force to avoid clusters
    pre_optimized_inits = []
    for init_pts in initializations:
        if time.time() > deadline - 60.0:
            break
        # Quick repulsive step to spread points
        pre_pts = _repulsive_pre_opt(init_pts, steps=20, lr=0.05)
        pre_optimized_inits.append(pre_pts)
        
    # Sort initializations by quick energy after pre-optimization
    scored_inits = []
    for init_pts in pre_optimized_inits:
        if time.time() > deadline - 30.0:
            break
        e_quick = _energy(init_pts)
        scored_inits.append((e_quick, init_pts))
    
    scored_inits.sort(key=lambda x: x[0])
    
    # Optimize top candidates
    # We prioritize the best initial points to spend more time on them
    top_candidates = scored_inits[:10] # Reduced count to allow more time per candidate
    
    for idx, (e_quick, init_pts) in enumerate(top_candidates):
        if time.time() > deadline - 10.0:
            break
        
        # Phase 1: Coarse L-BFGS-B optimization
        # Use looser tolerances to quickly converge to a local minimum
        optimized = _optimize_lbfgs(init_pts, deadline, max_iter=500, ftol=1e-8, gtol=1e-5, maxls=30)
        
        # Phase 2: Fine L-BFGS-B optimization
        if time.time() < deadline - 5.0:
            optimized = _optimize_lbfgs(optimized, deadline, max_iter=2000, ftol=1e-15, gtol=1e-12, maxls=50)
            
        # Phase 3: Riemannian Gradient Descent Polish
        # This can often find a slightly lower energy than L-BFGS-B by navigating the manifold directly
        if time.time() < deadline - 2.0:
            polished = _riemannian_polish(optimized, deadline, max_iter=200, step_init=1e-4)
            e_polished = _precise_energy(polished)
            if e_polished < best_energy:
                best_energy = e_polished
                best_pts = polished.copy()
            else:
                e_opt = _precise_energy(optimized)
                if e_opt < best_energy:
                    best_energy = e_opt
                    best_pts = optimized.copy()
        else:
            e_opt = _precise_energy(optimized)
            if e_opt < best_energy:
                best_energy = e_opt
                best_pts = optimized.copy()

    # Final polish on the global best
    if best_pts is not None and time.time() < deadline - 1.0:
        # Try one more very tight L-BFGS pass
        refined = _optimize_lbfgs(best_pts, deadline, max_iter=1000, ftol=1e-16, gtol=1e-14, maxls=100)
        e_refined = _precise_energy(refined)
        if e_refined < best_energy:
            best_energy = e_refined
            best_pts = refined
            
        # And a final Riemannian polish
        if time.time() < deadline - 0.1:
            final_polished = _riemannian_polish(best_pts, deadline, max_iter=100, step_init=1e-5)
            e_final = _precise_energy(final_polished)
            if e_final < best_energy:
                best_energy = e_final
                best_pts = final_polished
    
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
