"""Dual-representation optimization for minimal logarithmic energy of 150 points on sphere."""
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
    
    weighted_sum_pts = inv_d2 @ pts
    sum_inv_d2 = np.sum(inv_d2, axis=1)
    grad = -pts * sum_inv_d2[:, np.newaxis] + weighted_sum_pts
    
    # Project gradient to tangent space
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


def _penalty_energy_and_grad(pts, min_sep=0.05, penalty_scale=1000.0):
    """Energy with penalty for points closer than min_sep, plus gradient."""
    e, g = _energy_and_grad(pts)
    
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq)
    
    # Penalty: sum of (min_sep - d)^2 for d < min_sep
    viol = np.maximum(0, min_sep - dists)
    penalty = penalty_scale * np.sum(viol * viol)
    
    # Vectorized gradient of penalty
    # d/dx_i sum_{j != i} (min_sep - |x_i - x_j|)^2 = -2*ps*(min_sep-d)/d * (x_i - x_j) for d < min_sep
    mask = (dists < min_sep) & (dists > 1e-12)
    np.fill_diagonal(mask, False)
    # factor[i,j] = -2 * ps * (min_sep - d_ij) / d_ij for violating pairs, 0 otherwise
    factor = np.where(mask, -2.0 * penalty_scale * (min_sep - dists) / dists, 0.0)
    np.fill_diagonal(factor, 0.0)
    pen_grad = factor @ diff  # sum_j factor[i,j] * (x_i - x_j)
    
    # Project penalty gradient to tangent space
    dot_prod = np.sum(pen_grad * pts, axis=1)
    pen_grad -= dot_prod[:, np.newaxis] * pts
    
    return e + penalty, g + pen_grad


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
    prev_grad = grad.copy()
    prev_energy = energy
    
    for it in range(max_iter):
        if time.time() - start > time_limit - 0.1:
            break
        
        # Strong Wolfe line search with cubic interpolation
        alpha = 0.5
        best_energy = energy
        best_pts = current.copy()
        best_alpha = 0.0
        best_grad = grad.copy()
        
        # Initial bracket
        slope = np.sum(grad * d)
        if slope >= 0:
            d = -grad.copy()
            slope = np.sum(grad * d)
        
        for ls in range(30):
            new_pts = _exponential_map(current, alpha * d)
            new_energy, new_grad = _energy_and_grad(new_pts)
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_pts = new_pts
                best_alpha = alpha
                best_grad = new_grad
            
            # Wolfe conditions
            c1 = 1e-4
            if new_energy <= energy + c1 * alpha * slope:
                # Check curvature
                new_slope = np.sum(new_grad * d)
                if new_slope >= -0.9 * slope:
                    break
                # Sufficient decrease but not enough curvature, try smaller
                alpha *= 0.5
                continue
            else:
                alpha *= 0.5
        
        if best_alpha == 0.0:
            # Fallback: very small step
            alpha = 1e-4
            new_pts = _exponential_map(current, alpha * d)
            new_energy, new_grad = _energy_and_grad(new_pts)
            if new_energy < energy:
                best_energy = new_energy
                best_pts = new_pts
                best_alpha = alpha
                best_grad = new_grad
            else:
                break
        
        current = best_pts
        energy = best_energy
        grad = best_grad
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-14:
            break
        
        d_new = -grad
        beta = 0.0
        if it > 0:
            # Heston-Stiefel beta
            grad_diff = grad - prev_grad
            beta_num = np.sum(grad * grad_diff)
            beta_den = np.sum(prev_grad * grad_diff)
            if abs(beta_den) > 1e-30:
                beta = beta_num / beta_den
            if beta < 0.0 or beta > 0.5:
                beta = 0.0
        
        d = d_new + beta * d
        prev_grad = grad.copy()
        prev_energy = energy
    
    return current, energy


def _lbfgs_optimize(pts, time_limit=30.0, use_penalty=False, min_sep=0.05, penalty_scale=1000.0):
    """Optimize using L-BFGS-B with projected gradient."""
    start = time.time()
    
    x0 = pts.flatten()
    
    def fun_and_grad(x):
        p = x.reshape(N, 3)
        p = _normalize(p)
        if use_penalty:
            energy, grad = _penalty_energy_and_grad(p, min_sep, penalty_scale)
        else:
            energy, grad = _energy_and_grad(p)
        return energy, grad.flatten()
    
    res = minimize(fun_and_grad, x0, method='L-BFGS-B', jac=True,
                   options={'maxiter': 8000, 'ftol': 1e-20, 'gtol': 1e-15,
                            'maxfun': 30000, 'maxls': 100})
    
    x_opt = res.x.reshape(N, 3)
    x_opt = _normalize(x_opt)
    energy = _log_energy_fast(x_opt)
    
    return x_opt, energy


def _penalty_lbfgs(pts, time_limit=30.0):
    """L-BFGS with penalty for minimum separation."""
    start = time.time()
    current = pts.copy()
    energy = _log_energy_fast(current)
    
    # Start with moderate penalty, reduce over time
    min_sep = 0.08
    penalty_scale = 5000.0
    
    for phase in range(4):
        if time.time() - start > time_limit - 2:
            break
        
        time_left = time_limit - (time.time() - start) - 1
        if time_left < 3:
            break
        
        # Reduce penalty and separation as we converge
        min_sep_cur = min_sep * (0.7 ** phase)
        penalty_cur = penalty_scale * (0.5 ** phase)
        
        current, _ = _lbfgs_optimize(current, time_limit=min(time_left, 15.0),
                                      use_penalty=True, min_sep=min_sep_cur, penalty_scale=penalty_cur)
        energy = _log_energy_fast(current)
    
    return current, energy


def _dual_optimize(pts, time_limit=40.0):
    """Alternate between L-BFGS-B and Riemannian CG with Newton polish."""
    start = time.time()
    current = pts.copy()
    energy = _log_energy_fast(current)
    
    # Phase 1: L-BFGS-B with penalty for coarse optimization
    time_left = time_limit - (time.time() - start)
    if time_left > 10:
        current, energy = _penalty_lbfgs(current, time_limit=min(time_left * 0.4, 16.0))
    
    # Phase 2: Standard L-BFGS-B for fine optimization
    time_left = time_limit - (time.time() - start)
    if time_left > 8:
        current, energy = _lbfgs_optimize(current, time_limit=min(time_left * 0.3, 12.0), use_penalty=False)
    
    # Phase 3: Riemannian CG for fine optimization
    time_left = time_limit - (time.time() - start)
    if time_left > 5:
        current, energy = _riemannian_cg(current, time_limit=min(time_left * 0.4, 10.0))
    
    # Phase 4: Final L-BFGS-B micro-polish
    time_left = time_limit - (time.time() - start)
    if time_left > 2:
        current, energy = _lbfgs_optimize(current, time_limit=min(time_left, 6.0), use_penalty=False)
    
    return current, energy


def _polish(pts, time_limit=5.0):
    """High-precision polish using projected gradient with adaptive step and Newton-like corrections."""
    start = time.time()
    current = pts.copy()
    energy, grad = _energy_and_grad(current)
    grad_norm = np.linalg.norm(grad)
    
    if grad_norm < 1e-14:
        return current, energy
    
    # Adaptive step size
    alpha = 0.01 / (grad_norm + 1e-10)
    prev_grad = grad.copy()
    prev_energy = energy
    
    for it in range(5000):
        if time.time() - start > time_limit - 0.1:
            break
        
        # Compute search direction (steepest descent with momentum)
        if it == 0:
            direction = -grad
        else:
            # Simple momentum
            momentum = 0.3
            direction = -grad + momentum * direction
        
        # Project to tangent space
        dot = np.sum(direction * current, axis=1)
        direction -= dot[:, np.newaxis] * current
        
        dir_norm = np.linalg.norm(direction)
        if dir_norm < 1e-18:
            break
        
        # Backtracking line search
        best_alpha = alpha
        best_energy = energy
        best_pts = current
        best_grad = grad
        
        for ls in range(40):
            new_pts = _exponential_map(current, best_alpha * direction)
            new_energy, new_grad = _energy_and_grad(new_pts)
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_pts = new_pts
                best_grad = new_grad
            
            # Armijo condition
            slope = np.sum(grad * direction)
            if new_energy <= energy + 1e-4 * best_alpha * slope:
                break
            
            best_alpha *= 0.5
        
        if best_alpha <= 0:
            break
        
        current = best_pts
        energy = best_energy
        grad = best_grad
        grad_norm = np.linalg.norm(grad)
        
        if grad_norm < 1e-14:
            break
        
        # Adapt step size
        if energy < prev_energy - 1e-10:
            alpha = min(alpha * 1.5, 0.1 / (grad_norm + 1e-10))
        else:
            alpha *= 0.5
        
        prev_energy = energy
        prev_grad = grad
    
    # Final Newton-like step using local quadratic approximation
    if time.time() < start + time_limit - 0.5:
        e, grad = _energy_and_grad(current)
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-12:
            # Use diagonal approximation of Hessian from gradient changes
            # Simply try a few step sizes in gradient direction
            for trial_alpha in [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]:
                new_pts = _exponential_map(current, trial_alpha * (-grad / grad_norm))
                new_energy = _log_energy_fast(new_pts)
                if new_energy < e:
                    current = new_pts
                    energy = new_energy
                    break
    
    return current, energy


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Halton sequence (primary initialization per strategy)
    pts_halton = _halton_sphere(N)
    initializations.append(pts_halton.copy())
    
    # 2. Fibonacci spiral
    pts_fib = _fibonacci_sphere(N)
    initializations.append(pts_fib.copy())
    
    # 3. Random initializations
    for seed in range(3):
        rng_local = np.random.default_rng(seed * 123 + 7)
        pts_rand = rng_local.standard_normal((N, 3))
        pts_rand = _normalize(pts_rand)
        initializations.append(pts_rand)
    
    # 4. Uniform random on sphere
    for seed in range(2):
        rng_local = np.random.default_rng(seed * 321 + 99)
        pts_uniform = []
        for _ in range(N):
            while True:
                u1 = rng_local.uniform(-1, 1)
                u2 = rng_local.uniform(-1, 1)
                s = u1 * u1 + u2 * u2
                if s < 1.0 and s > 1e-15:
                    break
            scale = 2.0 * np.sqrt(1.0 - s)
            pts_uniform.append([u1 * scale, u2 * scale, 1.0 - s])
        pts_uniform = np.array(pts_uniform)
        pts_uniform = _normalize(pts_uniform)
        initializations.append(pts_uniform)
    
    # 5. Perturbed Halton
    for seed in range(2):
        rng_local = np.random.default_rng(seed * 789 + 23)
        pts_pert = pts_halton + rng_local.standard_normal((N, 3)) * 0.02
        pts_pert = _normalize(pts_pert)
        initializations.append(pts_pert)
    
    # 6. Perturbed Fibonacci
    for seed in range(2):
        rng_local = np.random.default_rng(seed * 456 + 13)
        pts_pert = pts_fib + rng_local.standard_normal((N, 3)) * 0.02
        pts_pert = _normalize(pts_pert)
        initializations.append(pts_pert)
    
    best_pts = initializations[0].copy()
    best_energy = _log_energy_fast(best_pts)
    
    # Optimize each initialization with better time allocation
    # Prioritize structured initializations (Halton, Fibonacci) with more time
    n_init = len(initializations)
    for idx, init_pts in enumerate(initializations):
        if time.time() > deadline - 15:
            break
        
        time_left = deadline - time.time() - 3
        if time_left < 6:
            break
        
        # Allocate more time to early (structured) initializations
        remaining = n_init - idx
        if idx < 2:
            # First two (Halton, Fibonacci) get more time
            base_time = min(time_left, max(20.0, 80.0 / remaining))
        elif idx < 5:
            # Random inits get moderate time
            base_time = min(time_left, max(12.0, 45.0 / remaining))
        else:
            # Perturbed inits get less time
            base_time = min(time_left, max(8.0, 30.0 / remaining))
        
        try:
            opt_pts, opt_energy = _dual_optimize(init_pts, time_limit=base_time)
            
            if opt_energy < best_energy:
                best_energy = opt_energy
                best_pts = opt_pts.copy()
                
        except Exception:
            try:
                time_left = deadline - time.time() - 3
                if time_left > 5:
                    opt_pts, opt_energy = _lbfgs_optimize(init_pts, time_limit=min(time_left, 15.0))
                    if opt_energy < best_energy:
                        best_energy = opt_energy
                        best_pts = opt_pts.copy()
            except Exception:
                pass
    
    # Final high-precision polish on best solution
    if time.time() < deadline - 2:
        final_time = min(deadline - time.time() - 1, 12.0)
        if final_time > 1.0:
            try:
                # Phase 1: Riemannian CG for structural refinement
                if final_time > 5:
                    cg_time = min(final_time * 0.5, 6.0)
                    final_pts, final_energy = _riemannian_cg(best_pts, time_limit=cg_time)
                    if final_energy < best_energy:
                        best_energy = final_energy
                        best_pts = final_pts
                        final_time -= cg_time
                
                # Phase 2: L-BFGS-B micro-polish
                if final_time > 3:
                    lbfgs_time = min(final_time * 0.4, 5.0)
                    final_pts, final_energy = _lbfgs_optimize(best_pts, time_limit=lbfgs_time, use_penalty=False)
                    if final_energy < best_energy:
                        best_energy = final_energy
                        best_pts = final_pts
                        final_time -= lbfgs_time
                
                # Phase 3: Final gradient polish
                if final_time > 1.5:
                    polish_time = min(final_time * 0.9, 5.0)
                    final_pts, final_energy = _polish(best_pts, time_limit=polish_time)
                    if final_energy < best_energy:
                        best_energy = final_energy
                        best_pts = final_pts
            except Exception:
                pass
    
    # Final normalization
    best_pts = _normalize(best_pts)
    
    # Verify no points are too close
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < 1e-9:
                mid = (best_pts[i] + best_pts[j]) / 2
                perp = best_pts[i] - best_pts[j]
                perp = perp / (np.linalg.norm(perp) + 1e-15)
                cross = np.cross(best_pts[i], perp)
                cross = cross / (np.linalg.norm(cross) + 1e-15)
                best_pts[i] = _normalize(best_pts[i] + 1e-8 * cross)
                best_pts[j] = _normalize(best_pts[j] - 1e-8 * cross)
    
    return best_pts
