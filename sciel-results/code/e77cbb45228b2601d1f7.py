"""Dual-representation optimization for minimal logarithmic energy of 150 points on sphere."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150


def _log_energy(pts):
    """Compute logarithmic energy using math.fsum for the evaluator."""
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
    return float(np.sum(np.log(1.0 / d)))


def _log_energy_grad(pts):
    """Compute gradient of logarithmic energy w.r.t. points on sphere (projected)."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dists_sq, 1.0)
    
    inv_d2 = 1.0 / dists_sq
    np.fill_diagonal(inv_d2, 0.0)
    
    weighted_sum_pts = inv_d2 @ pts
    sum_inv_d2 = np.sum(inv_d2, axis=1)
    grad = -pts * sum_inv_d2[:, np.newaxis] + weighted_sum_pts
    
    dot_prod = np.sum(grad * pts, axis=1)
    grad -= dot_prod[:, np.newaxis] * pts
    
    return grad


def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _exponential_map(pts, tangent_vecs):
    """Apply exponential map on sphere: move points along tangent vectors."""
    norms = np.linalg.norm(tangent_vecs, axis=1, keepdims=True)
    norms_safe = np.maximum(norms, 1e-15)
    cos_norms = np.cos(norms)
    sin_norms_over_norms = np.sin(norms) / norms_safe
    return cos_norms * pts + sin_norms_over_norms * tangent_vecs


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
    
    pts = np.zeros((n, 3))
    for i in range(n):
        u1 = halton(i + 1, base_start)
        u2 = halton(i + 1, base_start + 1)
        theta = 2.0 * np.pi * u1
        phi = np.arccos(2.0 * u2 - 1.0)
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        pts[i] = [x, y, z]
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts


def _icosahedral_init(n):
    """Generate points using icosahedral symmetry as starting point."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    base = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    base = _normalize(base)
    
    pts_list = [base.copy()]
    rng = np.random.RandomState(42)
    for _ in range(3):
        perturbed = base + rng.randn(12, 3) * 0.1
        pts_list.append(_normalize(perturbed))
    
    all_pts = np.vstack(pts_list)
    indices = np.linspace(0, len(all_pts) - 1, n, dtype=int)
    selected = all_pts[indices]
    return _normalize(selected)


def _riemannian_cg(pts, time_limit=15.0, max_iter=200):
    """Riemannian conjugate gradient on the sphere using exponential maps with Armijo."""
    start = time.time()
    current = _normalize(pts.copy())
    energy, grad = _energy_and_grad(current)
    
    d = -grad.copy()
    prev_grad = grad.copy()
    
    for it in range(max_iter):
        if time.time() - start > time_limit - 0.1:
            break
        
        # Armijo backtracking line search
        alpha = 0.5
        best_energy = energy
        best_pts = current.copy()
        best_alpha = 0.0
        best_grad = grad.copy()
        
        c1 = 1e-4
        for ls in range(50):
            new_pts = _exponential_map(current, alpha * d)
            new_energy, new_grad = _energy_and_grad(new_pts)
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_pts = new_pts
                best_alpha = alpha
                best_grad = new_grad
            
            # Armijo condition
            if new_energy <= energy + c1 * alpha * np.sum(grad * d):
                break
            
            alpha *= 0.7
        
        if best_alpha == 0.0:
            # Try very small step
            alpha = 1e-4
            for ls in range(30):
                new_pts = _exponential_map(current, alpha * d)
                new_energy, new_grad = _energy_and_grad(new_pts)
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_pts = new_pts
                    best_alpha = alpha
                    best_grad = new_grad
                    break
                alpha *= 0.5
            if best_alpha == 0.0:
                break
        
        current = best_pts
        energy = best_energy
        grad = best_grad
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-14:
            break
        
        # Polak-Ribiere beta
        d_new = -grad
        beta = 0.0
        if it > 0:
            beta_num = np.sum(grad * (grad - prev_grad))
            beta_den = np.sum(prev_grad * prev_grad)
            if beta_den > 1e-30:
                beta = max(0.0, beta_num / beta_den)
            if beta > 1.0:
                beta = 0.0
        
        d = d_new + beta * d
        prev_grad = grad.copy()
    
    return current, energy


def _lbfgs_optimize(pts, time_limit=20.0, maxiter=3000):
    """Optimize using L-BFGS-B with projected gradient."""
    start = time.time()
    x0 = _normalize(pts.copy()).flatten()
    
    def fun_and_grad(x):
        p = x.reshape(N, 3)
        p = _normalize(p)
        energy, grad = _energy_and_grad(p)
        return energy, grad.flatten()
    
    elapsed = time.time() - start
    remaining = time_limit - elapsed
    if remaining < 1.0:
        p = _normalize(pts)
        return p, _log_energy_fast(p)
    
    res = minimize(fun_and_grad, x0, method='L-BFGS-B', jac=True,
                   options={'maxiter': maxiter, 'ftol': 1e-16, 'gtol': 1e-13,
                            'maxfun': 10000, 'maxls': 80})
    
    x_opt = res.x.reshape(N, 3)
    x_opt = _normalize(x_opt)
    energy = _log_energy_fast(x_opt)
    
    return x_opt, energy


def _polish(pts, time_limit=5.0):
    """High-precision polish with adaptive step size gradient descent."""
    start = time.time()
    current = _normalize(pts.copy())
    energy = _log_energy_fast(current)
    step_size = 1e-2
    
    for it in range(10000):
        if time.time() - start > time_limit - 0.1:
            break
        
        grad = _log_energy_grad(current)
        grad_norm = np.linalg.norm(grad)
        
        if grad_norm < 1e-16:
            break
        
        alpha = step_size
        accepted = False
        for _ in range(50):
            new_pts = _exponential_map(current, -alpha * grad)
            new_energy = _log_energy_fast(new_pts)
            
            if new_energy < energy - 1e-16:
                accepted = True
                break
            alpha *= 0.5
        
        if accepted:
            current = new_pts
            energy = new_energy
            step_size = min(alpha * 1.1, 0.05)
        else:
            step_size *= 0.5
            if step_size < 1e-20:
                break
    
    return current, energy


def _newton_polish(pts, time_limit=8.0):
    """Newton-type polish using Riemannian Hessian approximation via finite differences."""
    start = time.time()
    current = _normalize(pts.copy())
    energy = _log_energy_fast(current)
    grad = _log_energy_grad(current)
    
    # Use a simple damped Newton approach with line search
    # Approximate Hessian via BFGS update
    m = min(20, N * 3)  # memory for L-BFGS
    s_list = []
    y_list = []
    
    for it in range(200):
        if time.time() - start > time_limit - 0.1:
            break
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-15:
            break
        
        # Compute search direction using L-BFGS two-loop recursion
        # Simplified: use steepest descent direction scaled
        d = -grad.copy()
        
        # Scale direction
        d_norm = np.linalg.norm(d)
        if d_norm > 1e-15:
            d = d / d_norm * min(0.1, 1.0 / d_norm)
        
        # Line search
        alpha = 1.0
        best_energy = energy
        best_pts = current.copy()
        best_alpha = 0.0
        best_grad = grad.copy()
        
        for ls in range(40):
            new_pts = _exponential_map(current, alpha * d)
            new_energy = _log_energy_fast(new_pts)
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_pts = new_pts
                best_alpha = alpha
                best_grad = _log_energy_grad(new_pts)
            
            if new_energy <= energy + 1e-4 * alpha * np.sum(grad * d):
                break
            
            alpha *= 0.7
        
        if best_alpha == 0.0:
            break
        
        # Update BFGS history
        s = best_pts - current
        y = best_grad - grad
        sy = np.sum(s * y)
        
        if sy > 1e-10:
            s_list.append(s)
            y_list.append(y)
            if len(s_list) > m:
                s_list.pop(0)
                y_list.pop(0)
        
        current = best_pts
        energy = best_energy
        grad = best_grad
    
    return current, energy


def construct():
    """Main function to construct optimal point configuration"""
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s limit
    
    best_pts = None
    best_energy = float('inf')
    
    # Generate initial configurations
    initializations = []
    
    # 1. Fibonacci Sphere
    fib_pts = _fibonacci_sphere(N)
    initializations.append(("fib", fib_pts.copy()))
    
    # 2. Icosahedral-based initialization
    ico_pts = _icosahedral_init(N)
    initializations.append(("ico", ico_pts.copy()))
    
    # 3. Halton sequence
    halton_pts = _halton_sphere(N)
    initializations.append(("halton", halton_pts.copy()))
    
    # 4. Random uniform on sphere
    for seed in range(3):
        rng_local = np.random.default_rng(seed * 123 + 7)
        pts_rand = rng_local.standard_normal((N, 3))
        pts_rand = _normalize(pts_rand)
        initializations.append((f"rand{seed}", pts_rand.copy()))
    
    # 5. Perturbed Fibonacci
    for seed in range(2):
        rng_local = np.random.default_rng(seed * 456 + 13)
        pts_pert = fib_pts.copy() + rng_local.standard_normal((N, 3)) * 0.02
        pts_pert = _normalize(pts_pert)
        initializations.append((f"fib_pert{seed}", pts_pert.copy()))
    
    # Evaluate each initialization with time allocation
    # Allocate more time to structured initializations
    time_alloc = {
        "fib": 22, "ico": 18, "halton": 18,
        "rand0": 15, "rand1": 15, "rand2": 15,
        "fib_pert0": 15, "fib_pert1": 15
    }
    
    for name, init_pts in initializations:
        if time.time() > deadline - 10:
            break
        
        alloc = time_alloc.get(name, 12)
        current_pts = init_pts.copy()
        current_energy = _log_energy_fast(current_pts)
        
        # Phase 1: L-BFGS-B for coarse optimization
        try:
            if time.time() < deadline - 8:
                lbfgs_time = min(alloc * 0.5, 15.0)
                current_pts, current_energy = _lbfgs_optimize(
                    current_pts, time_limit=lbfgs_time, maxiter=2000)
        except Exception:
            pass
        
        # Phase 2: Riemannian CG for fine optimization
        try:
            if time.time() < deadline - 5:
                cg_time = min(alloc * 0.3, 10.0)
                current_pts, current_energy = _riemannian_cg(
                    current_pts, time_limit=cg_time, max_iter=150)
        except Exception:
            pass
        
        # Phase 3: Newton-type polish
        try:
            if time.time() < deadline - 3:
                newton_time = min(alloc * 0.2, 6.0)
                current_pts, current_energy = _newton_polish(
                    current_pts, time_limit=newton_time)
        except Exception:
            pass
        
        # Phase 4: Final gradient polish
        try:
            if time.time() < deadline - 1:
                polish_time = min(2.0, alloc * 0.1)
                current_pts, current_energy = _polish(
                    current_pts, time_limit=polish_time)
        except Exception:
            pass
        
        if current_energy < best_energy:
            best_energy = current_energy
            best_pts = current_pts.copy()
    
    # Final high-precision polish on best result
    if best_pts is not None and time.time() < deadline - 1:
        try:
            polish_time = min(3.0, deadline - 1 - time.time())
            if polish_time > 0.5:
                best_pts, best_energy = _polish(best_pts, time_limit=polish_time)
        except Exception:
            pass
    
    # Ensure valid output
    if best_pts is None:
        best_pts = fib_pts
    
    best_pts = _normalize(best_pts)
    
    # Final validation
    if not np.all(np.isfinite(best_pts)):
        return fib_pts
    
    return best_pts
