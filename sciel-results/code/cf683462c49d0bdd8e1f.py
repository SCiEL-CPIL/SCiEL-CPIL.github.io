"""Minimal logarithmic energy of 150 points on the sphere - improved optimizer."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150

def _log_energy_flat(flat_pts):
    """Compute logarithmic energy from flat array using math.fsum for precision."""
    pts = flat_pts.reshape(N, 3)
    n = N
    terms = []
    # Use vectorized distance calculation for speed, then fsum for precision
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    iu = np.triu_indices(n, k=1)
    d_upper = np.sqrt(dists_sq[iu])
    
    # Check for invalid distances
    if np.any(d_upper < 1e-12):
        return float('inf')
        
    terms = [math.log(1.0 / d) for d in d_upper]
    return math.fsum(terms)

def _energy_and_grad_flat(flat_pts):
    """Vectorized energy and gradient from flat array."""
    pts = flat_pts.reshape(N, 3)
    n = N
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dists_sq, 1.0)
    dists = np.sqrt(dists_sq)
    
    iu = np.triu_indices(n, k=1)
    d_upper = dists[iu]
    valid_mask = d_upper > 1e-12
    if not np.all(valid_mask):
        return float('inf'), np.zeros_like(flat_pts)
        
    terms = [math.log(1.0 / d) for d in d_upper]
    energy = math.fsum(terms)
    
    inv_d2 = 1.0 / dists_sq
    np.fill_diagonal(inv_d2, 0.0)
    
    weighted_sum = inv_d2 @ pts
    sum_inv = np.sum(inv_d2, axis=1)
    grad = pts * sum_inv[:, np.newaxis] - weighted_sum
    
    return energy, grad.flatten()


def _halton_sphere(n):
    """Generate n points on sphere using Halton sequence in 3D."""
    def halton(index, base):
        result = 0.0
        f = 1.0
        i = index
        while i > 0:
            f /= base
            result += f * (i % base)
            i //= base
        return result
    
    points = []
    for i in range(1, n + 1):
        x = halton(i, 2)
        y = halton(i, 3)
        z = halton(i, 5)
        p = np.array([x, y, z])
        norm = np.linalg.norm(p)
        if norm > 1e-15:
            points.append(p / norm)
        else:
            points.append(np.array([1.0, 0.0, 0.0]))
    return np.array(points)

def _fibonacci_sphere(n):
    """Generate n points on sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)

def _optimize_sphere(flat_pts, time_limit, tol=1e-12, maxiter=500):
    """
    Optimize points on sphere using L-BFGS-B on flattened coordinates.
    We use a penalty for deviation from unit sphere to handle constraints softly,
    or simply rely on L-BFGS-B bounds if we parameterized differently.
    However, standard L-BFGS-B works on R^3N. We need to keep them on the sphere.
    
    A robust way: use a retraction in the callback or just minimize the energy
    and project periodically? 
    Actually, `minimize` with `method='L-BFGS-B'` doesn't support non-linear constraints easily.
    
    Better approach: Use `scipy.optimize.minimize` with a custom objective that
    projects back to sphere? No, that breaks gradient.
    
    Let's stick to the tangent space logic but implement a simple L-BFGS loop
    using `scipy.optimize._minimize_lbfgsb` if possible? No, that's private.
    
    Alternative: Use `minimize` with `method='BFGS'` or `L-BFGS-B` on the flat vector,
    but add a strong penalty term for (|x|^2 - 1)^2 to the objective and gradient.
    This is a common technique for constrained optimization when solvers don't support it.
    """
    start_time = time.time()
    
    def obj_grad(flat):
        if time.time() - start_time > time_limit:
            return float('inf'), np.zeros_like(flat)
        pts = flat.reshape(N, 3)
        # Project to sphere to ensure we are evaluating on the manifold
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.where(norms < 1e-15, 1.0, norms)
        pts_proj = pts / norms
        
        energy, grad_manifold = _energy_and_grad_flat(pts_proj.flatten())
        
        # Add penalty for deviation from sphere
        # Penalty: 0.5 * sum(|x|^2 - 1)^2
        # Grad penalty: 2 * (|x|^2 - 1) * x
        deviations = np.sum(pts * pts, axis=1, keepdims=True) - 1.0
        penalty = 0.5 * np.sum(deviations * deviations)
        grad_penalty = 2.0 * deviations * pts
        
        total_grad = grad_manifold.reshape(N, 3) + grad_penalty
        total_energy = energy + penalty
        
        # Check for time limit
        if time.time() - start_time > time_limit:
             return total_energy, total_grad.flatten()
             
        return total_energy, total_grad.flatten()

    res = minimize(
        lambda x: obj_grad(x)[0],
        flat_pts,
        jac=lambda x: obj_grad(x)[1],
        method='L-BFGS-B',
        options={'maxiter': maxiter, 'ftol': tol, 'gtol': tol, 'disp': False}
    )
    
    # Project final result to sphere
    final_pts = res.x.reshape(N, 3)
    norms = np.linalg.norm(final_pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    final_pts = final_pts / norms
    
    # Calculate true energy
    energy, _ = _energy_and_grad_flat(final_pts.flatten())
    
    return final_pts, energy


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    # Generate initializations
    inits = []
    
    # 1. Fibonacci spiral (very good starting point)
    inits.append(_fibonacci_sphere(N))
    
    # 2. Halton sequence
    inits.append(_halton_sphere(N))
    
    # 3. Random initializations
    for seed in [42, 123, 456]:
        rng = np.random.default_rng(seed)
        pts_rand = rng.standard_normal((N, 3))
        pts_rand /= np.linalg.norm(pts_rand, axis=1, keepdims=True)
        inits.append(pts_rand)
        
    # 4. Perturbed Fibonacci
    for seed, strength in [(7, 0.05), (99, 0.1)]:
        rng = np.random.default_rng(seed)
        pts_fib = _fibonacci_sphere(N)
        pts_pert = pts_fib + rng.standard_normal((N, 3)) * strength
        pts_pert /= np.linalg.norm(pts_pert, axis=1, keepdims=True)
        inits.append(pts_pert)

    best_pts = None
    best_energy = float('inf')
    
    # Phase 1: Coarse optimization to find best basin
    # We allocate a small amount of time to each to find a good local minimum
    coarse_time_per_init = 10.0
    
    for idx, init_pts in enumerate(inits):
        if time.time() > deadline - 60:  # Leave 60s for deep polish
            break
        
        remaining = deadline - time.time()
        time_alloc = min(coarse_time_per_init, remaining - 60)
        if time_alloc < 1.0:
            break
            
        pts, energy = _optimize_sphere(
            init_pts.flatten(),
            time_limit=time_alloc,
            tol=1e-8,
            maxiter=100
        )
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
            
    # Phase 2: Deep polish on the best result
    if best_pts is not None and time.time() < deadline - 5:
        remaining = deadline - time.time()
        # Use high precision and more iterations
        best_pts, best_energy = _optimize_sphere(
            best_pts.flatten(),
            time_limit=remaining - 2,
            tol=1e-14,
            maxiter=1000
        )
        
        # One more polish if time permits
        if time.time() < deadline - 2:
            remaining = deadline - time.time()
            if remaining > 5:
                best_pts, best_energy = _optimize_sphere(
                    best_pts.flatten(),
                    time_limit=remaining - 1,
                    tol=1e-15,
                    maxiter=500
                )

    # Ensure valid output
    if best_pts is None:
        best_pts = _fibonacci_sphere(N)
    else:
        norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
        norms = np.where(norms < 1e-15, 1.0, norms)
        best_pts = best_pts / norms
    
    # Verify all points are valid and not too close
    valid = True
    for i in range(N):
        if not np.all(np.isfinite(best_pts[i])):
            valid = False
            break
        if np.linalg.norm(best_pts[i]) < 1e-10:
            valid = False
            break
    
    if valid:
        min_dist = float('inf')
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(best_pts[i] - best_pts[j])
                if d < min_dist:
                    min_dist = d
        if min_dist < 1e-9:
            valid = False
    
    if not valid:
        best_pts = _fibonacci_sphere(N)
    
    return best_pts
