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
    
    for it in range(max_iter):
        if time.time() - start > time_limit - 0.5:
            break
        
        # Backtracking line search
        alpha = 0.5
        best_energy = energy
        best_pts = current.copy()
        best_grad = grad
        best_alpha = 0.0
        
        c1 = 1e-4
        
        for ls in range(30):
            new_pts = _exponential_map(current, alpha * d)
            new_energy, new_grad = _energy_and_grad(new_pts)
            
            # Armijo condition
            if new_energy > energy + c1 * alpha * np.sum(grad * d):
                break
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_pts = new_pts
                best_grad = new_grad
                best_alpha = alpha
            
            alpha *= 0.7
            
            if alpha < 1e-12:
                break
                
        if best_alpha == 0.0:
            break
            
        current = best_pts
        energy = best_energy
        grad = best_grad
        
        if np.linalg.norm(grad) < 1e-14:
            break
        
        # Polak-Ribiere beta with non-negative clipping
        d_new = -grad
        beta = 0.0
        if it > 0:
            num = np.sum(grad * (grad - prev_grad))
            den = np.sum(prev_grad * prev_grad)
            if den > 1e-30:
                beta = max(0.0, num / den)
        else:
            beta = 0.0
            
        d = d_new + beta * d
        prev_grad = grad.copy()
        
    return current, energy

def _basin_hopping(pts, time_limit=30.0, max_restarts=15):
    """Perform local optimization followed by basin hopping with multi-scale perturbations."""
    start = time.time()
    current = pts.copy()
    energy = _log_energy_fast(current)
    
    # Initial optimization
    time_left_init = time_limit - (time.time() - start) - 3
    if time_left_init > 3:
         current, energy = _riemannian_cg(current, time_limit=min(12, time_left_init))
    
    best_pts = current.copy()
    best_energy = energy
    
    # Basin Hopping Loop
    temp = 0.5
    decay = 0.90
    rng = np.random.default_rng(42)
    
    for rh in range(max_restarts):
        if time.time() - start > time_limit - 2:
            break
            
        # Try multiple perturbation scales for better exploration
        best_candidate = current.copy()
        best_candidate_energy = energy
        
        # Use 3 different perturbation scales
        for scale_factor in [0.5, 1.0, 2.0]:
            if time.time() - start > time_limit - 3:
                break
            
            perturbations = rng.standard_normal((N, 3))
            # Project to tangent space
            dot_prods = np.sum(perturbations * current, axis=1, keepdims=True)
            perturbations -= dot_prods * current
            
            # Scale by temperature
            perturb_scale = 0.12 * math.sqrt(temp) * scale_factor
            new_pts = _exponential_map(current, perturbations * perturb_scale)
            new_pts = _normalize(new_pts)
            
            # Optimize the perturbed point
            time_left = time_limit - (time.time() - start) - 2
            if time_left < 3:
                break
                
            opt_limit = min(6, time_left)
            new_opt_pts, new_energy = _riemannian_cg(new_pts, time_limit=opt_limit)
            new_energy = _log_energy_fast(new_opt_pts)
            
            if new_energy < best_candidate_energy:
                best_candidate = new_opt_pts
                best_candidate_energy = new_energy
        
        # Metropolis Criterion
        delta_e = best_candidate_energy - energy
        accept = delta_e < 0 or (temp > 0.01 and rng.random() < math.exp(-delta_e / temp))
        
        if accept:
            current = best_candidate
            energy = best_candidate_energy
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

def _lbfgs_sphere(pts, time_limit=10.0, max_iter=500, m=20):
    """L-BFGS quasi-Newton on the sphere with limited memory and projection line search."""
    start = time.time()
    current = pts.copy()
    energy, grad = _energy_and_grad(current)
    
    n = len(pts)
    # L-BFGS history: store last m pairs of (s, y)
    s_hist = []
    y_hist = []
    rho_hist = []
    
    for it in range(max_iter):
        if time.time() - start > time_limit - 0.5:
            break
        
        # Compute search direction using L-BFGS two-loop recursion
        g_flat = grad.flatten()
        r = g_flat.copy()
        
        # Two-loop recursion
        alphas = []
        qs = []
        for i in range(len(s_hist) - 1, -1, -1):
            s_i = s_hist[i]
            y_i = y_hist[i]
            rho_i = rho_hist[i]
            alpha_i = rho_i * np.dot(s_i, r)
            alphas.append(alpha_i)
            r -= alpha_i * y_i
            qs.append(r.copy())
        
        # Initial Hessian approximation: use identity (or scale by gamma)
        # gamma = dot(s_{k-1}, y_{k-1}) / dot(y_{k-1}, y_{k-1}) if history available
        if len(s_hist) > 0:
            s_last = s_hist[-1]
            y_last = y_hist[-1]
            gamma = np.dot(s_last, y_last) / max(np.dot(y_last, y_last), 1e-30)
            gamma = np.clip(gamma, 1e-10, 1e10)
        else:
            gamma = 1.0
        
        p_flat = gamma * r
        
        # Second loop
        for i in range(len(s_hist)):
            s_i = s_hist[i]
            y_i = y_hist[i]
            rho_i = rho_hist[i]
            beta_i = rho_i * np.dot(y_i, qs[i])
            p_flat += (alphas[i] - beta_i) * s_i
        
        p = -p_flat.reshape(n, 3)
        
        # Project to tangent space
        dot_prod = np.sum(p * current, axis=1, keepdims=True)
        p -= dot_prod * current
        
        # Line search with sphere projection
        alpha = 0.5
        best_energy = energy
        best_pts = current.copy()
        best_grad = grad
        best_alpha = 0.0
        
        c1 = 1e-4
        for ls in range(30):
            # Apply step and project to sphere
            new_pts = _exponential_map(current, alpha * p)
            new_pts = _normalize(new_pts)
            new_energy, new_grad = _energy_and_grad(new_pts)
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_pts = new_pts
                best_grad = new_grad
                best_alpha = alpha
            
            # Armijo condition
            if new_energy > energy + c1 * alpha * np.sum(grad * p):
                break
            
            alpha *= 0.6
            if alpha < 1e-14:
                break
        
        if best_alpha == 0.0:
            break
        
        # Update L-BFGS history
        s = (best_pts - current).flatten()
        y = (best_grad - grad).flatten()
        sy = np.dot(s, y)
        
        if sy > 1e-12:
            s_hist.append(s)
            y_hist.append(y)
            rho_hist.append(1.0 / sy)
            
            # Keep only last m pairs
            if len(s_hist) > m:
                s_hist.pop(0)
                y_hist.pop(0)
                rho_hist.pop(0)
        
        current = best_pts
        energy = best_energy
        grad = best_grad
        
        if np.linalg.norm(grad) < 1e-14:
            break
    
    return current, energy

def construct():
    start_time = time.time()
    deadline = start_time + 148.0  # 2s safety margin
    
    # Initializations - more diverse
    inits = []
    inits.append(_fibonacci_sphere(N))
    inits.append(_halton_sphere(N, base_start=2))
    inits.append(_halton_sphere(N, base_start=3))
    
    # Random inits with different seeds
    for s in range(6):
        rng = np.random.default_rng(s)
        pts = rng.standard_normal((N, 3))
        pts = _normalize(pts)
        inits.append(pts)
        
    # Perturbed Fibonacci with varying scales
    for s, scale in [(100, 0.05), (101, 0.1), (102, 0.2), (103, 0.3)]:
        rng = np.random.default_rng(s)
        pts = _fibonacci_sphere(N) + rng.standard_normal((N, 3)) * scale
        pts = _normalize(pts)
        inits.append(pts)
        
    # Perturbed Halton
    for s, scale in [(200, 0.1), (201, 0.2)]:
        rng = np.random.default_rng(s)
        pts = _halton_sphere(N, base_start=2) + rng.standard_normal((N, 3)) * scale
        pts = _normalize(pts)
        inits.append(pts)
        
    best_pts = None
    best_energy = float('inf')
    
    # Process each initialization with adaptive time allocation
    total_restarts = len(inits)
    for idx, init_pts in enumerate(inits):
        remaining = deadline - time.time()
        if remaining < 30:  # Need at least 30s for final polish
            break
            
        # Allocate time: more for earlier restarts, ensure enough for polish
        time_per_restart = min(18.0, (remaining - 25.0) / max(1, total_restarts - idx))
        
        try:
            opt_pts, opt_energy = _basin_hopping(init_pts, time_limit=time_per_restart)
            opt_energy = _log_energy_fast(opt_pts)
            
            if opt_energy < best_energy:
                best_energy = opt_energy
                best_pts = opt_pts.copy()
                
        except Exception as e:
            pass
            
    # If no valid points found, fallback to Fibonacci
    if best_pts is None:
        best_pts = _fibonacci_sphere(N)
        best_energy = _log_energy_fast(best_pts)
        
    # Final Polish - L-BFGS for faster convergence
    if time.time() < deadline - 20:
        polish_time = deadline - time.time() - 12
        try:
            final_pts, final_energy = _lbfgs_sphere(best_pts, time_limit=polish_time, max_iter=500, m=25)
            final_energy = _log_energy_fast(final_pts)
            if final_energy < best_energy:
                best_energy = final_energy
                best_pts = final_pts
        except Exception:
            pass
            
    # Second polish with CG for refinement
    if time.time() < deadline - 12:
        polish_time = deadline - time.time() - 8
        try:
            final_pts, final_energy = _riemannian_cg(best_pts, time_limit=polish_time, max_iter=800)
            final_energy = _log_energy_fast(final_pts)
            if final_energy < best_energy:
                best_energy = final_energy
                best_pts = final_pts
        except Exception:
            pass
            
    # Third polish with L-BFGS at tighter tolerance if time allows
    if time.time() < deadline - 8:
        polish_time = deadline - time.time() - 4
        try:
            final_pts, final_energy = _lbfgs_sphere(best_pts, time_limit=polish_time, max_iter=400, m=30)
            final_energy = _log_energy_fast(final_pts)
            if final_energy < best_energy:
                best_energy = final_energy
                best_pts = final_pts
        except Exception:
            pass
            
    # Final high-precision CG polish
    if time.time() < deadline - 5:
        polish_time = deadline - time.time() - 2
        try:
            final_pts, final_energy = _riemannian_cg(best_pts, time_limit=polish_time, max_iter=1000)
            final_energy = _log_energy_fast(final_pts)
            if final_energy < best_energy:
                best_energy = final_energy
                best_pts = final_pts
        except Exception:
            pass
            
    return _normalize(best_pts)
