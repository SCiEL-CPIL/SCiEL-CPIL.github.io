import numpy as np
import math
import time
from scipy.optimize import minimize

N = 100
TIME_LIMIT = 145.0  # Safety margin below 150s evaluator limit


def _log_energy_fsum(pts):
    """Exact scoring energy using math.fsum."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    iu = np.triu_indices(N, k=1)
    d = dists[iu]
    d = np.maximum(d, 1e-15)
    return math.fsum(np.log(1.0 / d).tolist())


def _energy_grad(pts):
    """Vectorized energy and ambient gradient, projected to tangent space."""
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists2, 1.0)
    
    iu = np.triu_indices(n, k=1)
    energy = -0.5 * np.sum(np.log(dists2[iu]))
    
    inv_dists2 = 1.0 / dists2
    np.fill_diagonal(inv_dists2, 0.0)
    s = np.einsum('ij,ijk->ik', inv_dists2, diff)
    grad = -s  # ambient gradient
    
    # Project to tangent space
    coeffs = np.sum(grad * pts, axis=1, keepdims=True)
    grad_t = grad - coeffs * pts
    
    return energy, grad_t


def _project(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _exp_map(pts, vecs):
    """Riemannian exponential map: move pts along tangent vectors vecs."""
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    angles = norms.squeeze()
    # Rodrigues / spherical rotation
    sin_a = np.sin(angles) / angles
    cos_a = np.cos(angles)
    sin_a = np.where(np.abs(angles) < 1e-12, 1.0 - angles**2/6.0, sin_a)
    cos_a = np.where(np.abs(angles) < 1e-12, 1.0 - angles**2/2.0, cos_a)
    new_pts = pts * cos_a[:, np.newaxis] + vecs * sin_a[:, np.newaxis]
    return _project(new_pts)


def _riemannian_flow(pts, time_budget, initial_lr=0.02, min_lr=1e-10):
    """Riemannian gradient descent with backtracking line search on the sphere."""
    pts = _project(pts.copy())
    current_energy = _energy_grad(pts)[0]
    lr = initial_lr
    no_improve = 0
    start = time.time()
    
    while time.time() - start < time_budget:
        e, g_t = _energy_grad(pts)
        grad_norm = np.linalg.norm(g_t)
        if grad_norm < 1e-14:
            break
        
        alpha = lr
        found = False
        for _ in range(40):
            new_pts = _exp_map(pts, -alpha * g_t)
            new_e = _energy_grad(new_pts)[0]
            if new_e < current_energy - 1e-16 * alpha * grad_norm:
                pts = new_pts
                current_energy = new_e
                found = True
                break
            alpha *= 0.5
        
        if found:
            no_improve = 0
            lr = min(alpha * 1.3, 0.5)
        else:
            lr = max(alpha, min_lr)
            no_improve += 1
            if no_improve > 100 or lr < min_lr:
                break
    
    return pts


def _riemannian_fw(pts, time_budget, max_iter=2000):
    """Riemannian Frank-Wolfe (conditional gradient) method.
    
    At each iteration:
    1. Compute gradient g_t at current point.
    2. Linear minimization over tangent space: s = argmin <g_t, v> s.t. ||v|| <= 1
       This gives s = -g_t / ||g_t|| (the direction of steepest descent on the unit ball).
    3. Step size via backtracking: gamma_t = argmin E(exp_{x_t}(gamma_t * s))
    4. Update: x_{t+1} = exp_{x_t}(gamma_t * s)
    """
    pts = _project(pts.copy())
    current_energy = _energy_grad(pts)[0]
    start = time.time()
    
    for _ in range(max_iter):
        if time.time() - start > time_budget:
            break
        
        e, g_t = _energy_grad(pts)
        grad_norm = np.linalg.norm(g_t)
        if grad_norm < 1e-14:
            break
        
        # Linear minimization: s = -g_t / ||g_t|| (unit vector in direction of steepest descent)
        s = -g_t / grad_norm
        
        # Backtracking for step size
        gamma = 1.0
        found = False
        for _ in range(60):
            new_pts = _exp_map(pts, gamma * s)
            new_e = _energy_grad(new_pts)[0]
            if new_e < current_energy - 1e-16 * gamma * grad_norm:
                pts = new_pts
                current_energy = new_e
                found = True
                break
            gamma *= 0.5
        
        if not found:
            # Try a smaller fixed step
            gamma = 0.01
            new_pts = _exp_map(pts, gamma * s)
            new_e = _energy_grad(new_pts)[0]
            if new_e < current_energy:
                pts = new_pts
                current_energy = new_e
            else:
                break
    
    return pts


def _optimize_lbfgs(pts, time_budget):
    """L-BFGS-B with tangent-projected gradients."""
    pts = _project(pts.copy())
    x0 = pts.flatten()
    
    def func(x):
        p = _project(x.reshape(-1, 3))
        return _energy_grad(p)[0]
    
    def jac(x):
        p = _project(x.reshape(-1, 3))
        _, g_t = _energy_grad(p)
        return g_t.flatten()
    
    options = {
        'maxiter': 3000,
        'gtol': 1e-12,
        'ftol': 1e-16,
        'disp': False,
        'maxfun': 15000,
    }
    
    try:
        result = minimize(func, x0, jac=jac, method='L-BFGS-B', options=options)
        if result.fun < 1e17:
            return _project(result.x.reshape(-1, 3))
    except Exception:
        pass
    
    return pts


def _fibonacci_spiral(n=N):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return _project(np.column_stack([r * np.cos(phi), r * np.sin(phi), z]))


def _random_sphere(n=N, seed=None):
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((n, 3))
    return _project(pts)


def _simulated_annealing(pts, time_budget, seed=None):
    """SA with small random rotations to escape local minima."""
    rng = np.random.default_rng(seed)
    pts = _project(pts.copy())
    current_energy = _energy_grad(pts)[0]
    best_pts = pts.copy()
    best_energy = current_energy
    
    T = 0.03
    min_T = 1e-7
    start = time.time()
    
    while time.time() - start < time_budget:
        if T < min_T:
            break
        
        n_moves = int(rng.integers(1, 4))
        indices = rng.choice(N, size=n_moves, replace=False)
        
        new_pts = pts.copy()
        for idx in indices:
            angle = rng.uniform(-np.pi, np.pi) * T
            axis = rng.standard_normal(3)
            axis_norm = np.linalg.norm(axis)
            if axis_norm < 1e-10:
                continue
            axis = axis / axis_norm
            
            c = math.cos(angle)
            s = math.sin(angle)
            k = axis
            K = np.array([[0, -k[2], k[1]],
                          [k[2], 0, -k[0]],
                          [-k[1], k[0], 0]])
            R = c * np.eye(3) + s * K + (1 - c) * np.outer(k, k)
            new_pts[idx] = R @ new_pts[idx]
        
        new_pts = _project(new_pts)
        new_energy = _energy_grad(new_pts)[0]
        delta_e = new_energy - current_energy
        
        if delta_e < 0 or rng.random() < math.exp(-delta_e / max(T, 1e-10)):
            pts = new_pts
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = pts.copy()
        
        T *= 0.9997
    
    return best_pts


def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    def _time_left():
        return max(0.0, deadline - time.time())
    
    best_pts = None
    best_energy = float('inf')
    
    # Diverse starting configurations
    rng = np.random.default_rng(42)
    starts = [
        _fibonacci_spiral(N),
        _project(_fibonacci_spiral(N) + rng.normal(0, 0.08, (N, 3))),
        _project(_fibonacci_spiral(N) + rng.normal(0, 0.2, (N, 3))),
        _random_sphere(N, seed=1),
        _random_sphere(N, seed=2),
        _random_sphere(N, seed=7),
    ]
    
    # Phase 1: Riemannian gradient flow from multiple starts (40% of time)
    opt_deadline = start_time + (deadline - start_time) * 0.35
    
    for i, start in enumerate(starts):
        if time.time() > opt_deadline:
            break
        
        budget = min(8.0, (opt_deadline - time.time()) * 0.3)
        if budget < 0.5:
            break
        
        refined = _riemannian_flow(start, budget, initial_lr=0.03)
        e = _energy_grad(refined)[0]
        
        if e < best_energy:
            best_energy = e
            best_pts = refined.copy()
    
    if best_pts is None:
        best_pts = _fibonacci_spiral(N)
        best_energy = _energy_grad(best_pts)[0]
    
    # Phase 2: Riemannian Frank-Wolfe on the best (25% of time)
    fw_deadline = start_time + (deadline - start_time) * 0.60
    if time.time() < fw_deadline:
        fw_budget = min(10.0, fw_deadline - time.time())
        if fw_budget > 1.0:
            fw_result = _riemannian_fw(best_pts, fw_budget)
            e = _energy_grad(fw_result)[0]
            if e < best_energy:
                best_energy = e
                best_pts = fw_result.copy()
    
    # Phase 3: L-BFGS-B polish (20% of time)
    lbfgs_deadline = start_time + (deadline - start_time) * 0.80
    if time.time() < lbfgs_deadline:
        lbfgs_budget = min(15.0, lbfgs_deadline - time.time())
        if lbfgs_budget > 1.0:
            polished = _optimize_lbfgs(best_pts, lbfgs_budget)
            e = _energy_grad(polished)[0]
            if e < best_energy:
                best_energy = e
                best_pts = polished.copy()
    
    # Phase 4: SA escape + final Riemannian flow polish (remaining time)
    sa_deadline = start_time + (deadline - start_time) * 0.92
    if time.time() < sa_deadline:
        sa_budget = min(8.0, sa_deadline - time.time())
        if sa_budget > 1.0:
            sa_result = _simulated_annealing(best_pts, sa_budget, seed=999)
            e = _energy_grad(sa_result)[0]
            if e < best_energy:
                best_energy = e
                best_pts = sa_result.copy()
    
    # Final high-precision Riemannian flow polish
    final_deadline = deadline - 3.0
    if time.time() < final_deadline:
        final_budget = final_deadline - time.time()
        if final_budget > 0.5:
            final = _riemannian_flow(best_pts, final_budget, initial_lr=0.01, min_lr=1e-12)
            e = _energy_grad(final)[0]
            if e < best_energy:
                best_energy = e
                best_pts = final.copy()
    
    best_pts = _project(best_pts)
    
    # Validity check
    if not np.all(np.isfinite(best_pts)):
        return _fibonacci_spiral(N)
    
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_d = np.min(dists)
    if min_d < 1e-9:
        return _fibonacci_spiral(N)
    
    return best_pts
