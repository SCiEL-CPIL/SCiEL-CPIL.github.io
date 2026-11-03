import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100
_DEADLINE = 150.0
_SAFETY = 12.0
_START = time.time()

# Precompute upper triangle indices for N=100
_IU = np.triu_indices(N, k=1)

def _remaining():
    return _DEADLINE - _SAFETY - (time.time() - _START)

def _fast_energy_and_grad(pts):
    """Compute logarithmic energy and gradient using vectorized operations."""
    n = pts.shape[0]
    # Compute squared distances
    # d_ij^2 = 2 - 2 * dot(x_i, x_j)
    dot_products = pts @ pts.T
    dist_sq = 2.0 - 2.0 * dot_products
    np.fill_diagonal(dist_sq, 1.0)
    np.maximum(dist_sq, 1e-15, out=dist_sq)
    
    # Energy: sum_{i<j} log(1/d_ij) = -0.5 * sum_{i,j, i!=j} log(d_ij)
    # Note: log(d_ij) = 0.5 * log(d_ij^2)
    # So sum_{i<j} log(1/d_ij) = -0.5 * sum_{i<j} log(d_ij^2)
    # Sum over i<j is half the sum over i!=j.
    # Wait, sum_{i<j} f(i,j) = 0.5 * (sum_{i,j} f(i,j) - sum_i f(i,i))
    # Here f(i,j) = log(d_ij^2).
    # Energy = -0.5 * sum_{i<j} log(d_ij^2) = -0.25 * sum_{i!=j} log(d_ij^2)
    # Let's stick to the explicit upper triangle to avoid factor errors.
    
    energy = -0.5 * np.sum(np.log(dist_sq[_IU]))
    
    # Gradient
    # E = - sum_{i<j} log(d_ij)
    # dE/dx_k = - sum_{j != k} (1/d_kj) * (x_k - x_j)/d_kj = - sum_{j != k} (x_k - x_j)/d_kj^2
    # d_kj^2 = dist_sq[k,j]
    # Let W[k,j] = 1/dist_sq[k,j] for k!=j, 0 otherwise.
    # Grad_k = - sum_{j != k} W[k,j] (x_k - x_j) = - sum_j W[k,j] x_k + sum_j W[k,j] x_j
    # Grad_k = - x_k * sum_j W[k,j] + (W @ x)_k
    # Note: The gradient on the sphere is the projection of this ambient gradient onto the tangent space.
    
    W = 1.0 / dist_sq
    np.fill_diagonal(W, 0.0)
    
    W_row_sum = W.sum(axis=1, keepdims=True)
    grad = W @ pts - W_row_sum * pts
    
    # Project gradient onto tangent space
    dot = np.sum(grad * pts, axis=1, keepdims=True)
    grad -= dot * pts
    
    return float(energy), grad

def _fast_energy_only(pts):
    """Compute just the logarithmic energy (faster, no gradient)."""
    dot_products = pts @ pts.T
    dist_sq = 2.0 - 2.0 * dot_products
    np.fill_diagonal(dist_sq, 1.0)
    np.maximum(dist_sq, 1e-15, out=dist_sq)
    return float(-0.5 * np.sum(np.log(dist_sq[_IU])))

def _precise_energy(pts):
    """Compute energy using math.fsum for accuracy."""
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, 1.0)
    iu = np.triu_indices(n, k=1)
    d_upper = dists[iu]
    if np.any(d_upper < 1e-9):
        return float('inf')
    terms = [math.log(1.0 / d) for d in d_upper]
    return math.fsum(terms)

def _project(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return X / n

def _fibonacci_spiral(n, offset=0.5):
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0, 1.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(seed):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((N, 3))
    return _project(v)

def _sobol_sphere(seed):
    """Generate points on the sphere using Sobol sequence."""
    from scipy.stats import qmc
    sampler = qmc.Sobol(d=3, scramble=True, seed=seed)
    samples = sampler.random(N)
    samples = samples - 0.5
    return _project(samples)

def _repulsive_spread(pts, n_steps=100, lr_start=0.05, deadline=None):
    """
    Pre-optimization: spread points using repulsive 1/r^2 energy.
    This is equivalent to minimizing the Riesz s-energy with s=2, which is known
    to have better global properties (less prone to local minima clustering) than s=0 (log).
    """
    n = pts.shape[0]
    current = pts.copy()
    lr = lr_start
    
    for step in range(n_steps):
        if deadline is not None and time.time() > deadline:
            break
        
        dot_products = current @ current.T
        dist_sq = 2.0 - 2.0 * dot_products
        np.fill_diagonal(dist_sq, 1.0)
        np.maximum(dist_sq, 1e-10, out=dist_sq)
        
        # Gradient of sum 1/d is -sum (x_i - x_j)/d^3
        # Gradient of sum 1/d^2 (Riesz s=2) is -2 sum (x_i - x_j)/d^4
        # We want to MINIMIZE energy, so we go AGAINST the gradient of repulsion?
        # No, we want points to repel each other.
        # Potential U = sum 1/d. Force is -grad U.
        # We want to move points to increase d.
        # So we perform gradient ASCENT on 1/d potential, or gradient DESCENT on -1/d potential?
        # Actually, standard minimization of log(1/d) = -log(d) works.
        # For spreading, we can minimize sum 1/d^2.
        # Grad of sum_{i<j} 1/d_ij^2 wrt x_k:
        # Term involving x_k: sum_{j!=k} (x_k - x_j) / d_kj^4
        # We want to minimize this energy.
        
        W = 1.0 / (dist_sq * dist_sq) # 1/d^4
        np.fill_diagonal(W, 0.0)
        
        W_row_sum = W.sum(axis=1, keepdims=True)
        # Grad_k = sum_j W[k,j] (x_k - x_j) = W_row_sum * x_k - W @ x_j
        grad = W_row_sum * current - W @ current
        
        # Project gradient onto tangent space
        dot = np.sum(grad * current, axis=1, keepdims=True)
        grad -= dot * current
        
        # Update: Gradient Descent on Repulsive Energy
        current = current - lr * grad
        current = _project(current)
        
        # Anneal learning rate
        lr *= 0.99

    return current

def _optimize_from_start(pts, deadline):
    """Run L-BFGS-B optimization from a starting point, respecting deadline."""
    n = pts.shape[0]
    best_pts = pts.copy()
    best_energy = _precise_energy(pts)
    
    def objective_and_grad(flat):
        p = flat.reshape(n, 3)
        norms = np.linalg.norm(p, axis=1, keepdims=True)
        np.maximum(norms, 1e-12, out=norms)
        p /= norms
        e, g = _fast_energy_and_grad(p)
        return e, g.reshape(-1)
    
    # Phase 1: Coarse optimization
    if time.time() < deadline - 4.0:
        try:
            result = minimize(
                objective_and_grad,
                pts.reshape(-1),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 100, 'maxfun': 500, 'ftol': 1e-6, 'gtol': 1e-4, 'maxls': 20}
            )
            new_pts = _project(result.x.reshape(n, 3))
            e_precise = _precise_energy(new_pts)
            if e_precise < best_energy:
                best_pts = new_pts
                best_energy = e_precise
        except Exception:
            pass
    
    # Phase 2: Medium refinement
    if time.time() < deadline - 3.0:
        try:
            result = minimize(
                objective_and_grad,
                best_pts.reshape(-1),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 200, 'maxfun': 1000, 'ftol': 1e-10, 'gtol': 1e-7, 'maxls': 30}
            )
            new_pts = _project(result.x.reshape(n, 3))
            e_precise = _precise_energy(new_pts)
            if e_precise < best_energy:
                best_pts = new_pts
                best_energy = e_precise
        except Exception:
            pass
    
    # Phase 3: Fine refinement
    if time.time() < deadline - 2.0:
        try:
            result = minimize(
                objective_and_grad,
                best_pts.reshape(-1),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 400, 'maxfun': 2000, 'ftol': 1e-13, 'gtol': 1e-9, 'maxls': 40}
            )
            new_pts = _project(result.x.reshape(n, 3))
            e_precise = _precise_energy(new_pts)
            if e_precise < best_energy:
                best_pts = new_pts
                best_energy = e_precise
        except Exception:
            pass
            
    return best_pts, best_energy

def _final_polish(pts, deadline):
    """Final ultra-precise polish."""
    n = pts.shape[0]
    best_pts = pts.copy()
    best_energy = _precise_energy(pts)
    
    def objective_and_grad(flat):
        p = flat.reshape(n, 3)
        norms = np.linalg.norm(p, axis=1, keepdims=True)
        np.maximum(norms, 1e-12, out=norms)
        p /= norms
        e, g = _fast_energy_and_grad(p)
        return e, g.reshape(-1)
    
    # High precision L-BFGS
    if time.time() < deadline - 1.0:
        try:
            result = minimize(
                objective_and_grad,
                best_pts.reshape(-1),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 1000, 'maxfun': 5000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
            )
            new_pts = _project(result.x.reshape(n, 3))
            e_precise = _precise_energy(new_pts)
            if e_precise < best_energy:
                best_pts = new_pts
                best_energy = e_precise
        except Exception:
            pass
            
    return best_pts, best_energy

def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE - _SAFETY  # ~138s
    
    # Generate diverse initializations
    initializations = []
    
    # Fibonacci spiral with various offsets
    for offset in [0.5, 0.3, 0.7, 0.1, 0.9]:
        initializations.append(_fibonacci_spiral(N, offset=offset))
        
    # Sobol quasi-random sequences
    for s in range(3):
        initializations.append(_sobol_sphere(s))
        
    # Random points
    for s in range(4):
        initializations.append(_random_sphere(100 + s * 13))

    # Apply repulsive pre-spread to all initializations
    # This step is crucial to get out of bad local minima of the log energy
    for i in range(len(initializations)):
        if time.time() > deadline - 5.0:
            break
        # Use a shorter deadline for pre-spread to save time for main optimization
        spread_deadline = deadline - 10.0 - (len(initializations) - i) * 5.0
        if spread_deadline < time.time() + 2.0:
            spread_deadline = time.time() + 2.0
        initializations[i] = _repulsive_spread(initializations[i], n_steps=50, lr_start=0.02, deadline=spread_deadline)
        
    best_pts = None
    best_energy = float('inf')
    
    total_restarts = len(initializations)
    
    # First pass: try all initializations with optimization
    for idx, init_pts in enumerate(initializations):
        if time.time() > deadline - 15.0:
            break
            
        remaining = deadline - time.time()
        if remaining < 10.0:
            break
            
        # Allocate time: give earlier restarts slightly more time if possible, 
        # but ensure enough time for the final polish.
        # We reserve 15s for polish at the end.
        # So we have ~120s for restarts.
        # 12 restarts -> 10s each.
        alloc = min(10.0, remaining - 15.0)
        if alloc < 3.0:
            break
            
        restart_deadline = time.time() + alloc
        
        # Optimize from this start
        optimized, e_precise = _optimize_from_start(init_pts, restart_deadline)
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = optimized

    # Final high-precision polish on the global best
    if best_pts is not None and time.time() < deadline - 5.0:
        polish_deadline = deadline - 2.0
        best_pts, best_energy = _final_polish(best_pts, polish_deadline)
        
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
