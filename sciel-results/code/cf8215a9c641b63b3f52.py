import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150

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
    """Vectorized energy and gradient (projected to tangent space)."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dists_sq, 1.0)
    inv_d2 = 1.0 / dists_sq
    np.fill_diagonal(inv_d2, 0.0)
    
    # Gradient of E = sum log(1/d) = - sum log(d)
    # d/dx_i log(1/|x_i - x_j|) = - 1/|x_i - x_j|^2 * (x_i - x_j)
    # Sum over j
    # grad_i = - sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    # Note: The formula in the previous code was:
    # grad = -pts * sum_inv_d2[:, np.newaxis] + weighted_sum_pts
    # Let's verify:
    # weighted_sum_pts = sum_{j != i} inv_d2[i,j] * x_j
    # sum_inv_d2 = sum_{j != i} inv_d2[i,j]
    # grad_i = - x_i * sum_{j} inv_d2[i,j] + sum_{j} inv_d2[i,j] * x_j = sum_{j} inv_d2[i,j] (x_j - x_i)
    # This matches - sum (x_i - x_j) / d^2.
    weighted_sum_pts = inv_d2 @ pts
    sum_inv_d2 = np.sum(inv_d2, axis=1)
    grad = -pts * sum_inv_d2[:, np.newaxis] + weighted_sum_pts
    
    # Project gradient to tangent space: g - (g . p) p
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
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diffs * diffs, axis=2)
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

def _riemannian_cg(pts, time_limit=10.0, max_iter=300):
    """Riemannian conjugate gradient on the sphere using exponential maps."""
    start = time.time()
    current = pts.copy()
    energy, grad = _energy_and_grad(current)
    prev_grad = grad.copy()
    
    # Initial search direction
    d = -grad.copy()
    alpha_init = 0.1
    
    for it in range(max_iter):
        if time.time() - start > time_limit - 0.5:
            break
        
        # Backtracking line search
        alpha = alpha_init
        best_energy = energy
        best_pts = current.copy()
        best_grad = grad
        best_alpha = 0.0
        c1 = 1e-4
        
        for ls in range(50):
            new_pts = _exponential_map(current, alpha * d)
            new_energy, new_grad = _energy_and_grad(new_pts)
            
            # Sufficient decrease (Armijo condition)
            if new_energy <= energy + c1 * alpha * np.sum(grad * d):
                # Accept this step
                best_energy = new_energy
                best_pts = new_pts
                best_grad = new_grad
                best_alpha = alpha
                break
            else:
                alpha *= 0.5
                if alpha < 1e-12:
                    break
        
        if best_alpha == 0.0:
            # No improvement found, try a steepest descent step
            alpha = 0.01
            for ls in range(20):
                new_pts = _exponential_map(current, alpha * (-grad))
                new_energy, new_grad = _energy_and_grad(new_pts)
                if new_energy < energy:
                    best_energy = new_energy
                    best_pts = new_pts
                    best_grad = new_grad
                    best_alpha = alpha
                    break
                alpha *= 0.5
            
            if best_alpha == 0.0:
                break
        
        current = best_pts
        energy = best_energy
        grad = best_grad
        
        if np.linalg.norm(grad) < 1e-14:
            break
        
        # Polak-Ribiere beta with parallel transport approximation
        d_new = -grad
        beta = 0.0
        if it > 0:
            # Transport prev_grad to current point using parallel transport
            # Approximation: project prev_grad onto tangent space of current
            dot_prev = np.sum(prev_grad * current, axis=1, keepdims=True)
            prev_grad_tangent = prev_grad - dot_prev * current
            
            num = np.sum(grad * (grad - prev_grad_tangent))
            den = np.sum(prev_grad_tangent * prev_grad_tangent)
            if den > 1e-30:
                beta = max(0.0, num / den)
                # Cap beta to prevent instability
                beta = min(beta, 1.0)
        else:
            beta = 0.0
            
        d = d_new + beta * d
        # Project d to tangent space
        dot_d = np.sum(d * current, axis=1, keepdims=True)
        d -= dot_d * current
        prev_grad = grad.copy()
        
        # Reset alpha_init if step was too small
        if best_alpha < 1e-6:
            alpha_init = max(best_alpha * 2, 0.001)
        else:
            alpha_init = min(best_alpha * 1.5, 1.0)
        
    return current, energy

def _basin_hopping(pts, time_limit=30.0, max_restarts=10):
    """Perform local optimization followed by basin hopping."""
    start = time.time()
    current = pts.copy()
    energy = _log_energy_fast(current)
    
    # Initial optimization
    time_left_init = time_limit - (time.time() - start) - 5
    if time_left_init > 5:
         current, energy = _riemannian_cg(current, time_limit=min(15, time_left_init))
    
    best_pts = current.copy()
    best_energy = energy
    
    # Basin Hopping Loop
    temp = 1.0
    decay = 0.95
    
    for rh in range(max_restarts):
        if time.time() - start > time_limit - 2:
            break
            
        # Adaptive perturbation scale
        perturb_scale = 0.1 * temp
        
        # Generate random tangent vectors with varying magnitudes
        rng = np.random.default_rng()
        # Mix of small and large perturbations
        if rng.random() < 0.3:
            # Large perturbation to escape local minima
            perturbations = rng.standard_normal((N, 3)) * perturb_scale * 3
        else:
            perturbations = rng.standard_normal((N, 3)) * perturb_scale
        
        # Project to tangent space
        dot_prods = np.sum(perturbations * current, axis=1, keepdims=True)
        perturbations -= dot_prods * current
        
        new_pts = _exponential_map(current, perturbations)
        new_pts = _normalize(new_pts)
        
        # Optimize the perturbed point
        time_left = time_limit - (time.time() - start) - 1
        if time_left < 5:
             break
             
        new_opt_pts, new_energy = _riemannian_cg(new_pts, time_limit=min(10, time_left))
        new_energy = _log_energy_fast(new_opt_pts)  # Recompute to be safe
        
        # Metropolis Criterion with adaptive acceptance
        delta_e = new_energy - energy
        # Increase acceptance probability for larger temperature
        acceptance_prob = math.exp(-delta_e / max(temp, 1e-6))
        
        if delta_e < 0 or rng.random() < acceptance_prob:
            current = new_opt_pts
            energy = new_energy
            if energy < best_energy:
                best_energy = energy
                best_pts = current.copy()
        else:
            # If rejected, occasionally try a smaller perturbation
            if rng.random() < 0.2:
                small_perturbations = rng.standard_normal((N, 3)) * perturb_scale * 0.1
                dot_prods = np.sum(small_perturbations * current, axis=1, keepdims=True)
                small_perturbations -= dot_prods * current
                small_new_pts = _exponential_map(current, small_perturbations)
                small_new_pts = _normalize(small_new_pts)
                
                if time_left > 3:
                    small_opt_pts, small_energy = _riemannian_cg(small_new_pts, time_limit=min(5, time_left - 1))
                    small_energy = _log_energy_fast(small_opt_pts)
                    if small_energy < energy:
                        current = small_opt_pts
                        energy = small_energy
                        if energy < best_energy:
                            best_energy = energy
                            best_pts = current.copy()
            
        temp *= decay
        
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
        # Map to sphere
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
    deadline = start_time + 145.0  # 5s safety margin
    
    # Initializations
    inits = []
    inits.append(_fibonacci_sphere(N))
    inits.append(_halton_sphere(N))
    
    # Random inits
    for s in range(5):
        rng = np.random.default_rng(s)
        pts = rng.standard_normal((N, 3))
        pts = _normalize(pts)
        inits.append(pts)
        
    # Perturbed Fibonacci
    for s in range(5):
        rng = np.random.default_rng(s + 100)
        pts = _fibonacci_sphere(N) + rng.standard_normal((N, 3)) * 0.1
        pts = _normalize(pts)
        inits.append(pts)
        
    best_pts = None
    best_energy = float('inf')
    
    # Process each initialization
    for idx, init_pts in enumerate(inits):
        if time.time() > deadline - 30:
            break
            
        # Determine time budget for this restart
        time_per_restart = 18.0
        if idx == 0:
             time_per_restart = 25.0 # Give first one (Fib) more time
        
        try:
            opt_pts, opt_energy = _basin_hopping(init_pts, time_limit=time_per_restart)
            opt_energy = _log_energy_fast(opt_pts)  # Recompute to be safe
            
            if opt_energy < best_energy:
                best_energy = opt_energy
                best_pts = opt_pts.copy()
                
        except Exception as e:
            pass
            
    # If no valid points found, fallback to Fibonacci
    if best_pts is None:
        best_pts = _fibonacci_sphere(N)
        best_energy = _log_energy_fast(best_pts)
        
    # Final Polish - multiple passes with tighter tolerances
    if time.time() < deadline - 10:
        polish_time = deadline - time.time() - 5
        try:
            final_pts, final_energy = _riemannian_cg(best_pts, time_limit=polish_time, max_iter=500)
            final_energy = _log_energy_fast(final_pts)
            if final_energy < best_energy:
                best_energy = final_energy
                best_pts = final_pts
        except Exception:
            pass
            
    # One more polish if time allows
    if time.time() < deadline - 5:
        polish_time = deadline - time.time() - 2
        try:
            final_pts, final_energy = _riemannian_cg(best_pts, time_limit=polish_time, max_iter=300)
            final_energy = _log_energy_fast(final_pts)
            if final_energy < best_energy:
                best_energy = final_energy
                best_pts = final_pts
        except Exception:
            pass
    
    # Ultra-precise final polish with very small steps
    if time.time() < deadline - 3:
        polish_time = deadline - time.time() - 1
        try:
            # Use even smaller initial step size for precision
            current = best_pts.copy()
            energy, grad = _energy_and_grad(current)
            prev_grad = grad.copy()
            d = -grad.copy()
            alpha_init = 0.01  # Smaller initial step
            
            for it in range(200):
                if time.time() - start > deadline - 1:
                    break
                
                alpha = alpha_init
                best_energy = energy
                best_pts_local = current.copy()
                best_grad = grad
                best_alpha = 0.0
                c1 = 1e-6  # Tighter tolerance
                
                for ls in range(30):
                    new_pts = _exponential_map(current, alpha * d)
                    new_energy, new_grad = _energy_and_grad(new_pts)
                    
                    if new_energy <= energy + c1 * alpha * np.sum(grad * d):
                        best_energy = new_energy
                        best_pts_local = new_pts
                        best_grad = new_grad
                        best_alpha = alpha
                        break
                    else:
                        alpha *= 0.5
                        if alpha < 1e-15:
                            break
                
                if best_alpha == 0.0:
                    break
                
                current = best_pts_local
                energy = best_energy
                grad = best_grad
                
                if np.linalg.norm(grad) < 1e-16:
                    break
                
                d_new = -grad
                beta = 0.0
                if it > 0:
                    dot_prev = np.sum(prev_grad * current, axis=1, keepdims=True)
                    prev_grad_tangent = prev_grad - dot_prev * current
                    num = np.sum(grad * (grad - prev_grad_tangent))
                    den = np.sum(prev_grad_tangent * prev_grad_tangent)
                    if den > 1e-30:
                        beta = max(0.0, num / den)
                        beta = min(beta, 0.5)
                
                d = d_new + beta * d
                dot_d = np.sum(d * current, axis=1, keepdims=True)
                d -= dot_d * current
                prev_grad = grad.copy()
                
                if best_alpha < 1e-8:
                    alpha_init = max(best_alpha * 2, 1e-10)
                else:
                    alpha_init = min(best_alpha * 1.2, 0.05)
            
            final_energy = _log_energy_fast(current)
            if final_energy < best_energy:
                best_energy = final_energy
                best_pts = current
        except Exception:
            pass
            
    return _normalize(best_pts)
