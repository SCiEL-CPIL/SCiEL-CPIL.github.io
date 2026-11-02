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
    dot_products = pts @ pts.T
    dists_sq = 2.0 - 2.0 * dot_products
    np.fill_diagonal(dists_sq, 1.0)
    inv_d2 = 1.0 / dists_sq
    np.fill_diagonal(inv_d2, 0.0)

    weighted_sum_pts = inv_d2 @ pts
    sum_inv_d2 = np.sum(inv_d2, axis=1)
    grad = weighted_sum_pts - pts * sum_inv_d2[:, np.newaxis]

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
    dot_products = pts @ pts.T
    dists_sq = 2.0 - 2.0 * dot_products
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(dists_sq[iu])
    d = np.maximum(d, 1e-15)
    return np.sum(np.log(1.0 / d))


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


def _riemannian_adam(pts, time_limit=20.0, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
    """Riemannian Adam optimizer on the sphere."""
    start = time.time()
    current = pts.copy()
    energy, grad = _energy_and_grad(current)
    
    # Initialize moments
    m = np.zeros_like(current)
    v = np.zeros_like(current)
    
    t = 0
    while time.time() - start < time_limit:
        t += 1
        # Update moments
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        
        # Update step
        step = lr * m_hat / (np.sqrt(v_hat) + eps)
        
        # Project step to tangent space
        step -= np.sum(step * current, axis=1, keepdims=True) * current
        
        # Line search (simple backtracking)
        alpha = 1.0
        c1 = 1e-4
        grad_dot_step = np.sum(grad * step)
        
        while alpha > 1e-10:
            new_pts = _exponential_map(current, alpha * step)
            new_energy, _ = _energy_and_grad(new_pts)
            
            if new_energy <= energy + c1 * alpha * grad_dot_step:
                break
            alpha *= 0.5
            
        if alpha <= 1e-10:
            break
            
        current = new_pts
        energy = new_energy
        _, grad = _energy_and_grad(current)
        
        if np.linalg.norm(grad) < 1e-10:
            break

    return current, energy


def _basin_hopping(pts, time_limit=30.0, kick_scale=0.1, T0=1.0, T_final=1e-5, rng=None):
    """Basin Hopping optimization using Riemannian Adam as local minimizer."""
    if rng is None:
        rng = np.random.default_rng()
        
    start = time.time()
    current = pts.copy()
    energy, _ = _energy_and_grad(current)
    
    best_pts = current.copy()
    best_energy = energy
    
    T = T0
    cooling_rate = (T_final / T0) ** (1.0 / max(1, int(time_limit / 0.5))) # Approx steps
    
    while time.time() - start < time_limit - 2.0:
        # 1. Local Minimization
        step_time = min(5.0, time_limit - (time.time() - start) - 2.0)
        if step_time < 1.0:
            break
        current, energy = _riemannian_adam(current, time_limit=step_time, lr=0.01)
        
        if energy < best_energy:
            best_energy = energy
            best_pts = current.copy()
            
        # 2. Kick
        kick = rng.standard_normal((N, 3)) * kick_scale
        kick -= np.sum(kick * current, axis=1, keepdims=True) * current # Project to tangent
        kicked_pts = _exponential_map(current, kick)
        
        # 3. Local Minimization of kicked point
        step_time2 = min(5.0, time_limit - (time.time() - start) - 1.0)
        if step_time2 < 0.5:
            break
        kicked_pts, kicked_energy = _riemannian_adam(kicked_pts, time_limit=step_time2, lr=0.01)
        
        # 4. Acceptance
        delta_E = kicked_energy - energy
        if delta_E < 0 or rng.random() < math.exp(-delta_E / T):
            current = kicked_pts
            energy = kicked_energy
        
        # Cooling
        T *= cooling_rate
        if T < T_final:
            T = T_final
            
    return best_pts, best_energy


def _polish(pts, time_limit=5.0):
    """High-precision polish using Riemannian Adam with small LR."""
    start = time.time()
    current = pts.copy()
    energy = _log_energy_fast(current)

    try:
        polished, polished_energy = _riemannian_adam(current, time_limit=time_limit, lr=0.001)
        if polished_energy < energy:
            current = polished
            energy = polished_energy
    except Exception:
        pass

    return current, energy


def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    initializations = []

    pts_fib = _fibonacci_sphere(N)
    initializations.append(pts_fib.copy())

    pts_halton = _halton_sphere(N)
    initializations.append(pts_halton.copy())

    for seed in range(4):
        rng_local = np.random.default_rng(seed * 123 + 7)
        pts_rand = rng_local.standard_normal((N, 3))
        pts_rand = _normalize(pts_rand)
        initializations.append(pts_rand)

    best_pts = initializations[0].copy()
    best_energy = _log_energy_fast(best_pts)

    # Run Basin Hopping from multiple starts
    num_bh_runs = 5
    bh_time_each = (deadline - start_time - 10.0) / num_bh_runs
    
    for idx in range(num_bh_runs):
        if time.time() > deadline - 15.0:
            break
            
        init_pts = initializations[idx % len(initializations)].copy()
        
        # Add small noise to avoid identical starts if reusing
        if idx >= len(initializations):
            rng_noise = np.random.default_rng(idx)
            noise = rng_noise.standard_normal((N, 3)) * 0.01
            init_pts = _normalize(init_pts + noise)
            
        try:
            rng_bh = np.random.default_rng(42 + idx)
            opt_pts, opt_energy = _basin_hopping(init_pts, time_limit=min(bh_time_each, 25.0), rng=rng_bh)
            
            if opt_energy < best_energy:
                best_energy = opt_energy
                best_pts = opt_pts.copy()
                
        except Exception:
            pass

    # Final polish
    if time.time() < deadline - 5.0:
        try:
            final_pts, final_energy = _polish(best_pts, time_limit=min(deadline - time.time() - 2.0, 10.0))
            if final_energy < best_energy:
                best_energy = final_energy
                best_pts = final_pts
        except Exception:
            pass

    best_pts = _normalize(best_pts)
    return best_pts
