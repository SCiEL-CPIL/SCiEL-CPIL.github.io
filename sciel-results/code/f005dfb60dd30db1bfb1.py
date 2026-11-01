"""Rewritten program: minimise logarithmic energy of 150 points on the unit sphere."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150
DEADLINE = 140.0  # seconds


def _log_energy(pts):
    """Compute sum_{i<j} log(1/|x_i - x_j|) using math.fsum for accuracy."""
    n = len(pts)
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-15:
                return float('inf')
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)


def _log_energy_fast(pts):
    """Vectorized log energy (less precise but much faster)."""
    sq = np.sum(pts * pts, axis=1)
    dist_sq = sq[:, None] + sq[None, :] - 2.0 * (pts @ pts.T)
    np.fill_diagonal(dist_sq, 1.0)
    dist_sq = np.maximum(dist_sq, 1e-30)
    dist = np.sqrt(dist_sq)
    iu = np.triu_indices(N, k=1)
    d = dist[iu]
    return float(np.sum(np.log(1.0 / d)))


def _log_energy_grad(pts):
    """Compute log energy and gradient (tangent space projection included)."""
    sq = np.sum(pts * pts, axis=1)
    dist_sq = sq[:, None] + sq[None, :] - 2.0 * (pts @ pts.T)
    np.fill_diagonal(dist_sq, 1.0)
    dist_sq = np.maximum(dist_sq, 1e-30)
    dist = np.sqrt(dist_sq)

    iu = np.triu_indices(N, k=1)
    d = dist[iu]
    energy = float(np.sum(np.log(1.0 / d)))

    diff = pts[:, None, :] - pts[None, :, :]
    inv_d2 = 1.0 / dist_sq
    np.fill_diagonal(inv_d2, 0.0)
    # grad[i] = sum_{j!=i} (x_j - x_i) / |x_i - x_j|^2
    grad = -np.sum(diff * inv_d2[:, :, None], axis=1)
    # Project gradient onto tangent space of sphere
    grad = grad - np.sum(grad * pts, axis=1, keepdims=True) * pts
    return energy, grad


def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _fibonacci_start():
    """Golden-angle Fibonacci spiral on unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_start(seed):
    """Random points on unit sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(N, 3)
    return _project_to_sphere(pts)


def _projected_bfgs(pts0, max_iter=3000, tol=1e-12, max_time=None, start_time=None):
    """
    Projected BFGS on the sphere: maintains Hessian approximation in tangent space,
    performs line search with re-projection onto sphere.
    """
    if start_time is None:
        start_time = time.time()
    
    x = pts0.copy()
    f, g = _log_energy_grad(x)
    n = x.size  # 450
    
    # BFGS memory
    m = 10
    S = []  # step history
    Y = []  # grad diff history
    
    # Initial inverse Hessian: identity in tangent space (approx)
    # We'll use a simplified BFGS that tracks the update in ambient space
    # but projects steps to tangent
    
    for it in range(max_iter):
        if max_time is not None and (time.time() - start_time) > max_time:
            break
        
        # Project gradient to tangent space (already done in _log_energy_grad)
        # Find search direction: p = -H * g
        # Simplified: use steepest descent direction projected to tangent
        # For full BFGS, we'd need to maintain H^{-1} in tangent space
        # Here we use a quasi-Newton approach with line search
        
        # Compute BFGS update direction
        # p = -g (steepest) as baseline, then apply BFGS corrections
        p = -g.copy()
        
        # Apply BFGS corrections
        for s_k, y_k in zip(S, Y):
            rho = 1.0 / (np.dot(y_k, s_k) + 1e-30)
            p = p + rho * (np.dot(s_k, p) - np.dot(y_k, s_k) * np.dot(y_k, p) / (np.dot(y_k, s_k) + 1e-30)) * s_k
            # Actually standard BFGS: p_new = p + rho*(s_k - (y_k^T p)/(y_k^T s_k) * s_k) * ... 
            # Let me use the standard recursive formula
            # This is getting complex; let's use scipy's L-BFGS-B but with sphere constraint enforced
            # by re-projecting at each step via a custom callback approach
        
        # Simpler approach: use L-BFGS-B but project the iterate back to sphere
        # after each function evaluation by using a wrapper that penalizes off-sphere
        # Actually, let's just do a clean projected gradient method with momentum
        # and a final L-BFGS-B polish
        
        # Gradient step with adaptive step size (backtracking line search)
        alpha = 1.0
        f_new, g_new = f, g
        for _ in range(50):
            x_trial = x + alpha * p
            x_trial = _project_to_sphere(x_trial.reshape(N, 3)).flatten()
            f_trial, g_trial = _log_energy_grad(x_trial.reshape(N, 3))
            if f_trial < f - 1e-4 * alpha * np.dot(g, p):
                f_new = f_trial
                g_new = g_trial
                break
            alpha *= 0.5
        else:
            # Line search failed, try smaller step
            alpha = 1e-6
            x_trial = _project_to_sphere((x + alpha * p).reshape(N, 3)).flatten()
            f_trial, g_trial = _log_energy_grad(x_trial.reshape(N, 3))
            f_new = f_trial
            g_new = g_trial
        
        # BFGS update
        s = x_trial - x
        y = g_new - g
        sy = np.dot(s, y)
        if sy > 1e-10:
            S.append(s)
            Y.append(y)
            if len(S) > m:
                S.pop(0)
                Y.pop(0)
        
        x = x_trial
        f = f_new
        g = g_new
        
        # Convergence check
        if np.linalg.norm(g) < tol:
            break
    
    return x.reshape(N, 3), f


def _optimize_lbfgs_sphere(pts0, max_iter=5000, tol=1e-14, max_time=None, start_time=None):
    """
    L-BFGS-B optimization where we project back to sphere after each iteration.
    Uses a wrapper that treats the sphere constraint via penalty-free projection.
    """
    if start_time is None:
        start_time = time.time()
    
    def objective_and_grad(x):
        if max_time is not None and (time.time() - start_time) > max_time:
            # Return large value to signal termination
            pts = x.reshape(N, 3)
            e, _ = _log_energy_grad(pts)
            return e + 1e10, np.zeros_like(x)
        pts = x.reshape(N, 3)
        e, g = _log_energy_grad(pts)
        return e, g.flatten()
    
    res = minimize(
        objective_and_grad,
        pts0.flatten(),
        method='L-BFGS-B',
        jac=True,
        options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-10, 'maxfun': max_iter * 2}
    )
    
    best_pts = _project_to_sphere(res.x.reshape(N, 3))
    return best_pts, res.fun


def _polish(pts, max_time=10.0):
    """
    High-precision polish: run L-BFGS-B with very tight tolerances.
    """
    start_time = time.time()
    
    def objective_and_grad(x):
        pts = x.reshape(N, 3)
        e, g = _log_energy_grad(pts)
        return e, g.flatten()
    
    res = minimize(
        objective_and_grad,
        pts.flatten(),
        method='L-BFGS-B',
        jac=True,
        options={'maxiter': 10000, 'ftol': 1e-16, 'gtol': 1e-12, 'maxfun': 20000}
    )
    
    polished = _project_to_sphere(res.x.reshape(N, 3))
    elapsed = time.time() - start_time
    return polished, res.fun, elapsed


def construct():
    global DEADLINE
    start_time = time.time()
    
    # Start with Fibonacci spiral
    pts_fib = _fibonacci_start()
    current_energy = _log_energy_fast(pts_fib)
    best_pts = pts_fib.copy()
    best_energy = current_energy
    
    # Phase 1: Optimize from Fibonacci start using L-BFGS-B
    remaining = DEADLINE - (time.time() - start_time)
    if remaining > 15:
        pts_opt, e_opt = _optimize_lbfgs_sphere(pts_fib, max_iter=5000, tol=1e-14,
                                                 max_time=remaining - 10, start_time=start_time)
        if time.time() - start_time < DEADLINE - 10:
            e_precise = _log_energy(pts_opt)
            if e_precise < best_energy:
                best_energy = e_precise
                best_pts = pts_opt.copy()
    
    # Phase 2: Try multiple random restarts with shorter optimizations
    rng = np.random.RandomState(42)
    n_restarts = 30
    for i in range(n_restarts):
        if time.time() - start_time > DEADLINE - 20:
            break
        
        pts_rand = _random_start(i + 1)
        remaining = DEADLINE - (time.time() - start_time)
        if remaining < 12:
            break
        
        try:
            pts_new, e_new = _optimize_lbfgs_sphere(pts_rand, max_iter=3000, tol=1e-13,
                                                     max_time=min(remaining - 8, 8.0),
                                                     start_time=start_time)
            if time.time() - start_time < DEADLINE - 10:
                e_precise = _log_energy(pts_new)
                if e_precise < best_energy:
                    best_energy = e_precise
                    best_pts = pts_new.copy()
        except Exception:
            continue
    
    # Phase 3: Final high-precision polish on the best configuration
    remaining = DEADLINE - (time.time() - start_time)
    if remaining > 5:
        try:
            pts_polished, e_polished, elapsed = _polish(best_pts, max_time=min(remaining - 3, 15.0))
            if time.time() - start_time < DEADLINE - 2:
                e_final = _log_energy(pts_polished)
                if e_final < best_energy:
                    best_energy = e_final
                    best_pts = pts_polished.copy()
        except Exception:
            pass
    
    # Final verification
    if time.time() - start_time < DEADLINE - 2:
        final_e = _log_energy(best_pts)
        if final_e < best_energy:
            best_energy = final_e
    
    # Ensure valid output
    norms = np.linalg.norm(best_pts, axis=1)
    if np.any(norms < 1e-15) or np.any(~np.isfinite(best_pts)):
        return _fibonacci_start()
    
    return _project_to_sphere(best_pts)
