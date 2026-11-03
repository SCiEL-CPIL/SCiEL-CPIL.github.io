"""Logarithmic energy minimization for 100 points on the unit sphere."""
import numpy as np
import math
import time
from scipy.optimize import minimize

N = 100
_START = time.time()
_DEADLINE = _START + 140.0  # safety margin before 150s kill


def _remaining():
    return _DEADLINE - time.time()


def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms


def _energy_fast(pts):
    """Fast energy using dot-product distance computation."""
    n = pts.shape[0]
    sq_norms = np.einsum('ij,ij->i', pts, pts)
    dist_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2.0 * (pts @ pts.T)
    np.fill_diagonal(dist_sq, 1.0)
    dist_sq = np.maximum(dist_sq, 1e-16)
    dist = np.sqrt(dist_sq)
    iu = np.triu_indices(n, k=1)
    return float(-np.sum(np.log(dist[iu])))


def _energy_grad(pts):
    """Compute energy and tangent-space gradient using dot-product distances."""
    n = pts.shape[0]
    # pts is assumed normalized
    sq_norms = np.einsum('ij,ij->i', pts, pts)
    dist_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2.0 * (pts @ pts.T)
    np.fill_diagonal(dist_sq, 1.0)
    dist_sq = np.maximum(dist_sq, 1e-16)
    dist = np.sqrt(dist_sq)

    iu = np.triu_indices(n, k=1)
    energy = float(-np.sum(np.log(dist[iu])))

    W = 1.0 / dist_sq
    np.fill_diagonal(W, 0.0)

    # grad_i = -sum_j W_ij * x_i + sum_j W_ij * x_j
    row_sums = W.sum(axis=1, keepdims=True)
    grad = -row_sums * pts + (W @ pts)

    # Project onto tangent space
    dot = np.sum(grad * pts, axis=1, keepdims=True)
    grad = grad - dot * pts

    return energy, grad


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


def _fibonacci_spiral(n, offset=0.5):
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0, 1.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, rng):
    pts = rng.standard_normal((n, 3))
    return _normalize(pts)


def _exp_map(pts, tangent):
    """Exponential map on the unit sphere: move points along tangent vectors."""
    norms = np.linalg.norm(tangent, axis=1, keepdims=True)
    norms_safe = np.where(norms < 1e-15, 1.0, norms)
    mask_small = norms < 1e-8
    theta = norms_safe
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    v_normalized = tangent / norms_safe
    new_pts = pts * cos_theta + v_normalized * sin_theta
    if np.any(mask_small):
        new_pts[mask_small] = pts[mask_small] + tangent[mask_small]
    return _normalize(new_pts)


def _riemannian_polish(pts, deadline, max_iter=5000, lr_init=1e-3):
    """
    Riemannian gradient descent with backtracking line search on the sphere.
    Uses exponential map for retraction. Optimized for speed by using fast energy
    for step decisions.
    """
    n = pts.shape[0]
    best_pts = _normalize(pts.copy())
    best_energy_fast = _energy_fast(best_pts)
    
    lr = lr_init
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        e, g = _energy_grad(best_pts)
        g_norm = np.linalg.norm(g)
        
        if g_norm < 1e-12:
            break
        
        # Backtracking line search using fast energy
        alpha = lr
        accepted = False
        for _ in range(20):
            new_pts = _exp_map(best_pts, -alpha * g)
            new_e = _energy_fast(new_pts)
            if new_e < e - 1e-4 * alpha * g_norm * g_norm:
                accepted = True
                break
            alpha *= 0.5
        
        if accepted:
            best_pts = new_pts
            best_energy_fast = new_e
            # Increase learning rate slightly if progress is good
            lr = min(lr * 1.1, 0.05)
        else:
            lr *= 0.7
            if lr < 1e-8:
                break
            
    return best_pts


def _optimize_from_start(pts, deadline):
    """
    Optimize from a single starting configuration using multi-phase L-BFGS-B
    followed by Riemannian manifold polish.
    Returns (best_pts, best_precise_energy).
    """
    n = pts.shape[0]
    best_pts = _normalize(pts.copy())
    best_energy = _precise_energy(best_pts)

    def objective_and_grad(flat):
        p = flat.reshape(n, 3)
        p = _normalize(p)
        e, g = _energy_grad(p)
        return e, g.reshape(-1)

    # Phase 1: Coarse L-BFGS-B
    if time.time() < deadline:
        try:
            x0 = best_pts.reshape(-1)
            result = minimize(
                objective_and_grad, x0,
                method='L-BFGS-B', jac=True,
                options={'maxiter': 2000, 'maxfun': 10000, 'ftol': 1e-10, 'gtol': 1e-6}
            )
            if result.fun < best_energy:
                new_pts = _normalize(result.x.reshape(n, 3))
                e_p = _precise_energy(new_pts)
                if e_p < best_energy:
                    best_pts = new_pts
                    best_energy = e_p
        except Exception:
            pass

    if time.time() > deadline:
        return best_pts, best_energy

    # Phase 2: Medium refinement
    if time.time() < deadline:
        try:
            x0 = best_pts.reshape(-1)
            result = minimize(
                objective_and_grad, x0,
                method='L-BFGS-B', jac=True,
                options={'maxiter': 3000, 'maxfun': 15000, 'ftol': 1e-13, 'gtol': 1e-9}
            )
            if result.fun < best_energy:
                new_pts = _normalize(result.x.reshape(n, 3))
                e_p = _precise_energy(new_pts)
                if e_p < best_energy:
                    best_pts = new_pts
                    best_energy = e_p
        except Exception:
            pass

    if time.time() > deadline:
        return best_pts, best_energy

    # Phase 3: Riemannian manifold polish (now faster)
    if time.time() < deadline - 5:
        try:
            r_pts = _riemannian_polish(best_pts, deadline - 5, max_iter=3000, lr_init=1e-3)
            r_e = _precise_energy(r_pts)
            if r_e < best_energy:
                best_pts = r_pts
                best_energy = r_e
        except Exception:
            pass

    if time.time() > deadline:
        return best_pts, best_energy

    # Phase 4: Final L-BFGS-B polish
    if time.time() < deadline - 2:
        try:
            x0 = best_pts.reshape(-1)
            result = minimize(
                objective_and_grad, x0,
                method='L-BFGS-B', jac=True,
                options={'maxiter': 5000, 'maxfun': 25000, 'ftol': 1e-15, 'gtol': 1e-11}
            )
            if result.fun < best_energy:
                new_pts = _normalize(result.x.reshape(n, 3))
                e_p = _precise_energy(new_pts)
                if e_p < best_energy:
                    best_pts = new_pts
                    best_energy = e_p
        except Exception:
            pass

    return best_pts, best_energy


def _spectral_init(n, rng, l_max=8):
    """
    Generate a smooth initial configuration by sampling a random band-limited
    spherical function.
    """
    theta = rng.uniform(0, 2 * np.pi, n)
    cos_phi = 2.0 * rng.uniform(0, 1, n) - 1.0
    sin_phi = np.sqrt(1.0 - cos_phi ** 2)

    pts = np.column_stack([
        sin_phi * np.cos(theta),
        sin_phi * np.sin(theta),
        cos_phi
    ])

    # Add smooth low-frequency perturbation
    for l in range(1, l_max + 1):
        for m in range(-l, l + 1):
            amp = rng.normal(0, 0.01 / (l + 1))
            phase = rng.uniform(0, 2 * np.pi)
            t = np.arctan2(pts[:, 1], pts[:, 0])
            c = pts[:, 2]
            mod = amp * np.cos(m * t + phase) * (c ** abs(m))
            pts[:, 2] += mod * 0.1
            pts[:, 0] += amp * 0.05 * np.sin(m * t + phase) * sin_phi

    return _normalize(pts)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    rng = np.random.default_rng(12345)

    # Generate diverse initializations
    initializations = []

    # 1. Fibonacci spiral (best structured start)
    initializations.append(_fibonacci_spiral(N))

    # 2. Spectral/smooth initializations
    for seed in range(5):
        r = np.random.default_rng(seed * 100 + 7)
        initializations.append(_spectral_init(N, r, l_max=6))

    # 3. Random initializations
    for i in range(10):
        initializations.append(_random_sphere(N, rng))

    # 4. Perturbed Fibonacci
    base = _fibonacci_spiral(N)
    for scale in [0.005, 0.01, 0.02, 0.05]:
        perturbed = base + rng.standard_normal((N, 3)) * scale
        initializations.append(_normalize(perturbed))

    # 5. Additional structured: shifted golden spiral
    for offset in [0.0, 0.15, 0.3, 0.5]:
        initializations.append(_fibonacci_spiral(N, offset))

    # Sort by quick energy to prioritize good starts
    scored_inits = []
    for init in initializations:
        e_quick = _energy_fast(init)
        scored_inits.append((e_quick, init))
    scored_inits.sort(key=lambda x: x[0])

    best_pts = None
    best_energy = float('inf')

    for e_quick, init_pts in scored_inits:
        if time.time() > deadline - 10:
            break

        optimized_pts, e_precise = _optimize_from_start(init_pts, deadline - 10)

        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = optimized_pts

        # If we've found something very good, we can stop early if time is tight
        if best_energy < -1083.37 and time.time() > deadline - 30:
            break

    if best_pts is None:
        best_pts = _fibonacci_spiral(N)

    # Final high-precision polish on the best configuration
    if time.time() < deadline - 5:
        try:
            final_pts, final_e = _optimize_from_start(best_pts, deadline - 5)
            if final_e < best_energy:
                best_pts = final_pts
                best_energy = final_e
        except Exception:
            pass

    # Final normalization and validation
    best_pts = _normalize(best_pts)
    final_energy = _precise_energy(best_pts)
    if not np.isfinite(final_energy):
        best_pts = _normalize(_fibonacci_spiral(N))

    return best_pts
