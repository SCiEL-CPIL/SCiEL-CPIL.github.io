import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150
_DEADLINE_MARGIN = 5.0  # seconds to leave before hard timeout

def _log_energy(pts):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|) using math.fsum."""
    n = len(pts)
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-12:
                return float('inf')
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)

def _energy_and_grad(pts):
    """Vectorized energy and gradient (projected to tangent space).
    Uses dot products for efficiency and stability.
    """
    n = len(pts)
    # d_ij^2 = |x_i - x_j|^2 = |x_i|^2 + |x_j|^2 - 2 x_i . x_j = 2 - 2 x_i . x_j
    dot_products = pts @ pts.T
    dists_sq = 2.0 - 2.0 * dot_products
    # Numerical safety: ensure diagonal is 0 (dist 0) and off-diagonals are positive
    np.fill_diagonal(dists_sq, 0.0)
    
    # Avoid division by zero for close points
    dists_sq = np.maximum(dists_sq, 1e-15)
    
    inv_d2 = 1.0 / dists_sq
    np.fill_diagonal(inv_d2, 0.0)
    
    # Gradient of E = sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    # dE/dx_i = - sum_{j!=i} (1/d_ij^2) (x_i - x_j)
    # Let W_ij = 1/d_ij^2. Then grad_i = - sum_j W_ij (x_i - x_j) = - x_i sum_j W_ij + sum_j W_ij x_j
    weighted_sum_pts = inv_d2 @ pts
    sum_inv_d2 = np.sum(inv_d2, axis=1)
    grad = -pts * sum_inv_d2[:, np.newaxis] + weighted_sum_pts
    
    # Project gradient to tangent space: grad_tangent = grad - (grad . x) x
    dot_prod = np.sum(grad * pts, axis=1)
    grad -= dot_prod[:, np.newaxis] * pts
    
    # Compute energy (upper triangle only)
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(dists_sq[iu])
    d = np.maximum(d, 1e-15)
    energy = np.sum(np.log(1.0 / d))
    
    return energy, grad

def _log_energy_fast(pts):
    """Vectorized log energy computation for speed."""
    n = len(pts)
    dot_products = pts @ pts.T
    dists_sq = 2.0 - 2.0 * dot_products
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(dists_sq[iu])
    d = np.maximum(d, 1e-15)
    return np.sum(np.log(1.0 / d))

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

def _riemannian_cg(pts, time_limit=15.0, max_iter=300):
    """Riemannian conjugate gradient on the sphere using exponential maps."""
    start = time.time()
    current = pts.copy()
    energy, grad = _energy_and_grad(current)

    d = -grad.copy()
    prev_grad = None

    for it in range(max_iter):
        if time.time() - start > time_limit - 0.1:
            break

        # Backtracking line search with Armijo condition
        alpha = 0.5
        c1 = 1e-4
        best_energy = energy
        best_pts = current.copy()
        best_alpha = 0.0
        best_grad = None
        grad_dot_d = np.sum(grad * d)

        for ls in range(40):
            new_pts = _exponential_map(current, alpha * d)
            new_energy, new_grad = _energy_and_grad(new_pts)

            if new_energy <= energy + c1 * alpha * grad_dot_d:
                best_energy = new_energy
                best_pts = new_pts
                best_alpha = alpha
                best_grad = new_grad
                break

            if new_energy < best_energy:
                best_energy = new_energy
                best_pts = new_pts
                best_alpha = alpha
                best_grad = new_grad

            alpha *= 0.5

        if best_alpha == 0.0:
            # Fallback to steepest descent
            alpha = 1e-3
            new_pts = _exponential_map(current, alpha * (-grad))
            new_energy, new_grad = _energy_and_grad(new_pts)
            if new_energy < energy:
                current = new_pts
                energy = new_energy
                grad = new_grad
                d = -grad.copy()
                prev_grad = None
            else:
                break
            continue

        current = best_pts
        energy = best_energy
        grad = best_grad

        if np.linalg.norm(grad) < 1e-15:
            break

        d_new = -grad
        beta = 0.0
        if prev_grad is not None:
            beta_num = np.sum(grad * (grad - prev_grad))
            beta_den = np.sum(prev_grad * prev_grad)
            if beta_den > 1e-30:
                beta = max(0.0, beta_num / beta_den)

        d = d_new + beta * d
        prev_grad = grad.copy()

    return current, energy

def _lbfgs_scipy(pts, time_limit=20.0, maxiter=2000, gtol=1e-12):
    """Optimize using SciPy's L-BFGS-B with normalization inside the objective."""
    start = time.time()
    x0 = pts.flatten()

    def fun_and_grad(x):
        p = x.reshape(N, 3)
        p = _normalize(p)
        energy, grad = _energy_and_grad(p)
        return energy, grad.flatten()

    try:
        res = minimize(fun_and_grad, x0, method='L-BFGS-B', jac=True,
                       options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': gtol,
                                'maxfun': maxiter * 5, 'maxls': 50, 'm': 20})
        x_opt = res.x.reshape(N, 3)
        x_opt = _normalize(x_opt)
        energy = _log_energy_fast(x_opt)
        return x_opt, energy
    except Exception:
        return pts, _log_energy_fast(pts)

def _dual_optimize(pts, time_limit=30.0):
    """Alternate between SciPy L-BFGS and Riemannian CG for robustness."""
    start = time.time()
    current = pts.copy()
    energy = _log_energy_fast(current)

    # Phase 1: L-BFGS for rapid convergence
    time_left = time_limit - (time.time() - start)
    if time_left > 10:
        current, energy = _lbfgs_scipy(current, time_limit=min(time_left * 0.6, 20.0))

    # Phase 2: Riemannian CG for fine-tuning
    time_left = time_limit - (time.time() - start)
    if time_left > 5:
        current, energy = _riemannian_cg(current, time_limit=min(time_left * 0.8, 15.0))

    # Phase 3: Final L-BFGS polish
    time_left = time_limit - (time.time() - start)
    if time_left > 3:
        current, energy = _lbfgs_scipy(current, time_limit=max(2.0, time_left - 1.0), gtol=1e-14)

    return current, energy

def _basin_hopping(pts, time_limit=30.0, max_restarts=10):
    """Perform local optimization followed by basin hopping."""
    start = time.time()
    current = pts.copy()
    energy = _log_energy_fast(current)
    
    # Initial optimization
    time_left = time_limit - (time.time() - start)
    if time_left > 10:
        current, energy = _dual_optimize(current, time_limit=min(15, time_left - 2))
    
    best_pts = current.copy()
    best_energy = energy
    
    # Basin Hopping Loop
    temp = 0.8
    decay = 0.82
    
    for rh in range(max_restarts):
        time_left = time_limit - (time.time() - start)
        if time_left < 8:
            break
        
        # Perturb with adaptive scale
        rng = np.random.default_rng(rh + 42)
        base_scale = 0.25 * (0.75 ** rh)
        scale = base_scale * (1.0 + 0.3 * math.sin(rh * 1.3))
        perturbations = rng.standard_normal((N, 3)) * scale
        # Project to tangent space
        dot_prods = np.sum(perturbations * current, axis=1, keepdims=True)
        perturbations -= dot_prods * current
        
        new_pts = _exponential_map(current, perturbations)
        new_pts = _normalize(new_pts)
        
        # Optimize the perturbed point using dual optimizer
        opt_time = min(15, time_left - 4)
        if opt_time < 5:
            break
        new_opt_pts, new_energy = _dual_optimize(new_pts, time_limit=opt_time)
        
        # Metropolis Criterion
        delta_e = new_energy - energy
        accept = delta_e < 0
        if not accept and temp > 1e-6:
            try:
                accept = np.random.rand() < math.exp(-delta_e / temp)
            except OverflowError:
                accept = False
        
        if accept:
            current = new_opt_pts
            energy = new_energy
            if energy < best_energy:
                best_energy = energy
                best_pts = current.copy()
        
        temp *= decay
    
    # Final polish within basin hopping
    time_left = time_limit - (time.time() - start)
    if time_left > 3:
        try:
            polished, pol_energy = _dual_optimize(best_pts, time_limit=max(2.0, time_left - 1.0))
            if pol_energy < best_energy:
                best_energy = pol_energy
                best_pts = polished
        except Exception:
            pass
    
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

def construct():
    start_time = time.time()
    deadline = start_time + 150.0 - _DEADLINE_MARGIN
    
    # Initializations
    inits = []
    inits.append(_fibonacci_sphere(N))
    inits.append(_halton_sphere(N))
    
    # Random inits
    for s in range(2):
        rng = np.random.default_rng(s * 7 + 1)
        pts = rng.standard_normal((N, 3))
        pts = _normalize(pts)
        inits.append(pts)
        
    # Perturbed Fibonacci
    rng = np.random.default_rng(200)
    pts = _fibonacci_sphere(N) + rng.standard_normal((N, 3)) * 0.1
    pts = _normalize(pts)
    inits.append(pts)

    best_pts = None
    best_energy = float('inf')
    
    # Process each initialization
    n_inits = len(inits)
    for idx, init_pts in enumerate(inits):
        elapsed = time.time() - start_time
        remaining = deadline - time.time()
        if remaining < 20:
            break
        
        # Allocate time
        time_remaining_for_inits = remaining - 15  # reserve 15s for final polish
        inits_remaining = n_inits - idx
        time_per_restart = time_remaining_for_inits / inits_remaining
        time_per_restart = min(30.0, max(15.0, time_per_restart))
        
        try:
            opt_pts, opt_energy = _basin_hopping(init_pts, time_limit=time_per_restart)
            
            if opt_energy < best_energy:
                best_energy = opt_energy
                best_pts = opt_pts.copy()
                
        except Exception:
            pass
            
    if best_pts is None:
        best_pts = _fibonacci_sphere(N)
        best_energy = _log_energy_fast(best_pts)
        
    # Final Polish - high precision
    time_left = deadline - time.time()
    if time_left > 5:
        polish_time = max(4.0, time_left - 1.0)
        try:
            final_pts, final_energy = _dual_optimize(best_pts, time_limit=polish_time)
            if final_energy < best_energy:
                best_energy = final_energy
                best_pts = final_pts
        except Exception:
            pass
        
    return _normalize(best_pts)
