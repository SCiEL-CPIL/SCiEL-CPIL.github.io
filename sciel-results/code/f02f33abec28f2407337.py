"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import math
import time
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START
def _energy_and_grad(pts):
    """
    Compute the logarithmic energy and its gradient for points on the sphere.
    Energy E = sum_{i<j} log(1 / ||x_i - x_j||) = - sum_{i<j} log(||x_i - x_j||)
    """
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    # W[i,j] = 1/dist_sq[i,j] for i != j, 0 on diagonal
    W = np.zeros_like(dist_sq)
    mask = ~np.eye(n, dtype=bool)
    W[mask] = 1.0 / dist_sq[mask]
    
    # Gradient of E w.r.t. pts[i] in ambient space:
    # dE/dx_i = sum_{j!=i} (x_i - x_j) / dist_sq[i,j]
    #         = (x_i * sum_j 1/d2_ij) - sum_j x_j / d2_ij
    row_sum = W.sum(axis=1, keepdims=True)
    grad_ambient = pts * row_sum - W @ pts
    
    # Project to tangent space: grad_tan = grad_amb - (grad_amb . x_i) x_i
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

def _generate_fibonacci(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0, 1.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _generate_random(n, rng):
    pts = rng.standard_normal((n, 3))
    return _normalize(pts)

def _generate_sobol(n, rng=None):
    """Generate points using a quasi-random (Sobol-like) sequence on the sphere."""
    from scipy.stats import qmc
    sampler = qmc.Sobol(d=3, scramble=True)
    samples = sampler.random(n)
    # Map [0,1]^3 to sphere via normalizing
    pts = samples - 0.5
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    pts = pts / norms
    # Add small perturbation to break symmetry
    if rng is not None:
        pts += rng.standard_normal((n, 3)) * 0.001
        pts = _normalize(pts)
    return pts

def _optimize_single(pts, deadline):
    """Run L-BFGS-B optimization from a given start."""
    n = pts.shape[0]
    best_pts = pts.copy()
    best_energy = _precise_energy(pts)
    
    def objective_and_grad(flat):
        p = flat.reshape(n, 3)
        p = _normalize(p)
        e, g = _energy_and_grad(p)
        return e, g.reshape(-1)
    
    x0 = pts.reshape(-1)
    
    # Phase 1: Standard L-BFGS-B
    try:
        result = minimize(
            objective_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={
                'maxiter': 2000,
                'maxfun': 10000,
                'ftol': 1e-12,
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
    
    if time.time() > deadline:
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
                'maxiter': 5000,
                'maxfun': 20000,
                'ftol': 1e-15,
                'gtol': 1e-10,
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
    deadline = start_time + 145.0
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Fibonacci Spiral
    initializations.append(_generate_fibonacci(N))
    
    # 2. Random Points
    for i in range(10):
        initializations.append(_generate_random(N, rng))
    
    # 3. Sobol quasi-random points (better coverage)
    for i in range(5):
        initializations.append(_generate_sobol(N, rng))
        
    # 4. Perturbed Fibonacci
    base = _generate_fibonacci(N)
    for scale in [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]:
        perturbed = base + rng.standard_normal((N, 3)) * scale
        initializations.append(_normalize(perturbed))
    
    # 5. More structured initializations
    # Golden ratio based with different offsets
    for offset in [0.0, 0.05, 0.1, 0.15, 0.2]:
        i = np.arange(N) + 0.5 + offset
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.clip(1.0 - z * z, 0, 1.0))
        initializations.append(np.column_stack([r * np.cos(phi), r * np.sin(phi), z]))
    
    # 6. Symmetric configurations: two hemispheres with Fibonacci
    for hemi_offset in [0.0, 0.25, 0.5]:
        i_top = np.arange(N // 2) + 0.5
        i_bot = np.arange(N // 2) + 0.5
        z_top = 1.0 - 2.0 * i_top / (N // 2)
        z_top = z_top * 0.5 + 0.5  # Scale to upper hemisphere
        z_bot = 1.0 - 2.0 * i_bot / (N // 2)
        z_bot = z_bot * 0.5 - 0.5  # Scale to lower hemisphere
        phi_top = np.pi * (1.0 + np.sqrt(5.0)) * i_top
        phi_bot = np.pi * (1.0 + np.sqrt(5.0)) * i_bot + np.pi / (N // 2)
        r_top = np.sqrt(np.clip(1.0 - z_top * z_top, 0, 1.0))
        r_bot = np.sqrt(np.clip(1.0 - z_bot * z_bot, 0, 1.0))
        top = np.column_stack([r_top * np.cos(phi_top), r_top * np.sin(phi_top), z_top])
        bot = np.column_stack([r_bot * np.cos(phi_bot), r_bot * np.sin(phi_bot), z_bot])
        init = np.vstack([top, bot])
        if len(init) < N:
            extra = _generate_random(N - len(init), rng)
            init = np.vstack([init, extra])
        elif len(init) > N:
            init = init[:N]
        initializations.append(_normalize(init))
    
    # Sort initializations by quick energy to prioritize good starts
    scored_inits = []
    for init in initializations:
        e_quick = _fast_energy(init)
        scored_inits.append((e_quick, init))
    scored_inits.sort(key=lambda x: x[0])
    
    best_pts = None
    best_energy = float('inf')
    
    for e_quick, init_pts in scored_inits:
        if time.time() > deadline:
            break
            
        optimized_pts, e_precise = _optimize_single(init_pts, deadline)
        
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
        final_energy = _precise_energy(best_pts)
    
    # Final high-precision polish if time permits
    if time.time() < deadline - 5.0 and final_energy < float('inf'):
        try:
            polished_pts, polished_e = _newton_polish(best_pts, max_iter=50, tol=1e-14)
            if polished_e < final_energy:
                best_pts = polished_pts
        except Exception:
            pass
        best_pts = _normalize(best_pts)
        
    return best_pts
# EVOLVE-BLOCK-END
