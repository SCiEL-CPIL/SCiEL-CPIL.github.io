"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import math
import time
from scipy.optimize import minimize
from scipy.stats import qmc

N = 100

# EVOLVE-BLOCK-START
def _energy_and_grad(pts):
    """
    Compute the logarithmic energy and its gradient for points on the sphere.
    Energy E = sum_{i<j} log(1 / ||x_i - x_j||) = - sum_{i<j} log(||x_i - x_j||)
    
    Parameters:
    pts: (N, 3) array of points.
    
    Returns:
    energy: float
    grad: (N, 3) array, the Riemannian gradient projected to tangent space.
    """
    n = pts.shape[0]
    # Compute pairwise squared distances
    # diff[i, j] = x_i - x_j
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    
    # Energy: -sum_{i<j} log(dist_ij)
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    # Gradient calculation
    # d/dx_i [ -log(||x_i - x_j||) ] = -(x_i - x_j) / ||x_i - x_j||^2
    # Sum over j != i: grad_ambient_i = - sum_{j!=i} (x_i - x_j) / dist_ij^2
    #                   = sum_{j!=i} (x_j - x_i) / dist_ij^2
    
    # W_ij = 1 / dist_ij^2 for i != j, 0 otherwise
    W = 1.0 / dist_sq
    np.fill_diagonal(W, 0.0)
    
    # grad_ambient_i = sum_j W_ij * (x_j - x_i) = sum_j W_ij * x_j - x_i * sum_j W_ij
    # sum_j W_ij * x_j is (W @ pts)_i
    # sum_j W_ij is W.sum(axis=1)
    
    grad_ambient = W @ pts - W.sum(axis=1, keepdims=True) * pts
    
    # Project to tangent space: grad_tangent = grad_ambient - (grad_ambient . x_i) * x_i
    # Since x_i is unit vector, x_i . x_i = 1
    dot = np.sum(grad_ambient * pts, axis=1, keepdims=True)
    grad_tangent = grad_ambient - dot * pts
    
    return energy, grad_tangent

def _fast_energy(pts):
    """Compute energy quickly for initialization sorting."""
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    iu = np.triu_indices(n, k=1)
    return -np.sum(np.log(dist[iu]))

def _precise_energy(pts):
    """Compute energy using math.fsum for final scoring accuracy."""
    n = pts.shape[0]
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-9:
                return float('inf')
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return pts / norms

def _generate_sobol(n, seed=42):
    """Generate points on the sphere using Sobol sequence."""
    sampler = qmc.Sobol(d=3, seed=seed)
    # Use enough samples to pick n good ones, or just take n
    # Sobol sequences are low-discrepancy, good for uniform distribution
    # We generate in [0,1)^3 and map to sphere
    # A better mapping: use spherical coordinates derived from Sobol
    # Or just normalize the Sobol points shifted to center
    samples = sampler.random(n)
    # Shift to center of cube [-0.5, 0.5]
    samples = samples - 0.5
    # Normalize
    return _normalize(samples)

def _generate_random(n, rng):
    pts = rng.standard_normal((n, 3))
    return _normalize(pts)

def _generate_fibonacci(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0, 1.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _gradient_flow(pts, deadline, rng):
    """Gradient flow on the sphere with adaptive step size and annealing."""
    n = pts.shape[0]
    current = pts.copy()
    current_energy = _fast_energy(current)
    best = current.copy()
    best_energy = current_energy
    
    # Adaptive parameters
    base_lr = 0.01
    min_lr = 1e-6
    max_steps = 2000
    patience = 200
    no_improve = 0
    
    for step in range(max_steps):
        if time.time() > deadline - 5.0:
            break
        
        # Compute gradient
        _, grad = _energy_and_grad(current)
        
        # Adaptive learning rate: decay over time
        lr = base_lr * (0.995 ** step)
        lr = max(lr, min_lr)
        
        # Gradient step
        new_pts = current - lr * grad
        
        # Normalize to sphere
        new_pts = _normalize(new_pts)
        
        # Compute new energy
        new_energy = _fast_energy(new_pts)
        
        if new_energy < current_energy:
            current = new_pts
            current_energy = new_energy
            no_improve = 0
            if current_energy < best_energy:
                best = current.copy()
                best_energy = current_energy
        else:
            no_improve += 1
            # Occasionally perturb to escape local minima
            if no_improve > patience and step % 50 == 0:
                perturbation = rng.standard_normal((n, 3)) * 0.01
                perturbed = _normalize(current + perturbation)
                perturbed_energy = _fast_energy(perturbed)
                if perturbed_energy < current_energy:
                    current = perturbed
                    current_energy = perturbed_energy
                    no_improve = 0
        
        # Early termination if stuck for too long
        if no_improve > 500:
            break
    
    return best, best_energy

def _optimize_single(pts, deadline):
    """Run multi-phase optimization from a given start."""
    n = pts.shape[0]
    best_pts = pts.copy()
    best_energy = _precise_energy(pts)
    
    def objective_and_grad(flat):
        p = flat.reshape(n, 3)
        p = _normalize(p)
        e, g = _energy_and_grad(p)
        return e, g.reshape(-1)
    
    x0 = pts.reshape(-1)
    
    # Phase 0: Gradient flow to find a good basin
    if time.time() < deadline - 20.0:
        try:
            gf_pts, gf_energy = _gradient_flow(pts, deadline, np.random.default_rng(123))
            if gf_energy < best_energy:
                best_pts = gf_pts
                best_energy = gf_energy
        except Exception:
            pass
    
    # Phase 1: Standard L-BFGS-B
    try:
        result = minimize(
            objective_and_grad,
            best_pts.reshape(-1),
            method='L-BFGS-B',
            jac=True,
            options={
                'maxiter': 1000,
                'maxfun': 5000,
                'ftol': 1e-10,
                'gtol': 1e-6,
            }
        )
        if result.fun < best_energy:
            new_pts = _normalize(result.x.reshape(n, 3))
            e_precise = _precise_energy(new_pts)
            if e_precise < best_energy:
                best_pts = new_pts
                best_energy = e_precise
    except Exception:
        pass
    
    if time.time() > deadline - 10.0:
        return best_pts, best_energy
    
    # Phase 2: Polish with tighter tolerances
    try:
        x0 = best_pts.reshape(-1)
        result = minimize(
            objective_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={
                'maxiter': 2000,
                'maxfun': 10000,
                'ftol': 1e-14,
                'gtol': 1e-8,
            }
        )
        if result.fun < best_energy:
            new_pts = _normalize(result.x.reshape(n, 3))
            e_precise = _precise_energy(new_pts)
            if e_precise < best_energy:
                best_pts = new_pts
                best_energy = e_precise
    except Exception:
        pass
    
    if time.time() > deadline - 5.0:
        return best_pts, best_energy
    
    # Phase 3: Final high-precision polish
    try:
        x0 = best_pts.reshape(-1)
        result = minimize(
            objective_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={
                'maxiter': 5000,
                'maxfun': 20000,
                'ftol': 1e-18,
                'gtol': 1e-14,
            }
        )
        if result.fun < best_energy:
            new_pts = _normalize(result.x.reshape(n, 3))
            e_precise = _precise_energy(new_pts)
            if e_precise < best_energy:
                best_pts = new_pts
                best_energy = e_precise
    except Exception:
        pass
    
    return best_pts, best_energy

def construct():
    start_time = time.time()
    # Safety margin: 150s limit, aim for 145s
    deadline = start_time + 145.0
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Fibonacci Spiral (often good for spherical distributions)
    initializations.append(_generate_fibonacci(N))
    
    # 2. Sobol Sequence (Quasi-random, very uniform)
    initializations.append(_generate_sobol(N, seed=1))
    initializations.append(_generate_sobol(N, seed=2))
    
    # 3. Random Points
    for _ in range(3):
        initializations.append(_generate_random(N, rng))
        
    # 4. Perturbed Fibonacci
    base = _generate_fibonacci(N)
    for _ in range(2):
        perturbed = base + rng.standard_normal((N, 3)) * 0.05
        initializations.append(_normalize(perturbed))
    
    # 5. Additional diverse random starts
    for _ in range(2):
        initializations.append(_generate_random(N, rng))

    best_pts = None
    best_energy = float('inf')
    
    # Sort initializations by quick energy to prioritize good starts
    # Note: _fast_energy is approximate but fast
    scored_inits = []
    for init in initializations:
        e_quick = _fast_energy(init)
        scored_inits.append((e_quick, init))
    scored_inits.sort(key=lambda x: x[0])
    
    # Optimize each initialization
    for idx, (e_quick, init_pts) in enumerate(scored_inits):
        if time.time() > deadline - 30.0:
            break
            
        # Allocate more time to earlier (better) initializations
        remaining = deadline - 30.0 - time.time()
        # Give the first few initializations more time
        if idx == 0:
            alloc_deadline = time.time() + min(remaining * 0.4, 40.0)
        elif idx == 1:
            alloc_deadline = time.time() + min(remaining * 0.3, 30.0)
        elif idx == 2:
            alloc_deadline = time.time() + min(remaining * 0.2, 20.0)
        else:
            alloc_deadline = time.time() + min(remaining * 0.1, 10.0)
        
        optimized_pts, e_precise = _optimize_single(init_pts, alloc_deadline)
        
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = optimized_pts
    
    if best_pts is None:
        best_pts = _generate_fibonacci(N)
        
    # Final normalization
    best_pts = _normalize(best_pts)
    
    # Final validation
    final_energy = _precise_energy(best_pts)
    if not np.isfinite(final_energy):
        best_pts = _generate_fibonacci(N)
        best_pts = _normalize(best_pts)
        
    return best_pts
# EVOLVE-BLOCK-END
