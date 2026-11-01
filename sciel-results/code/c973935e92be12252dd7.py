"""Combined: multi-restart L-BFGS-B + Riemannian CG with Armijo line search + Newton polish."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150


def _log_energy_fast(pts):
    """Vectorized log energy computation."""
    n = len(pts)
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diffs * diffs, axis=2)
    iu = np.triu_indices(n, k=1)
    d_sq = dists_sq[iu]
    d = np.sqrt(np.maximum(d_sq, 1e-30))
    return np.sum(np.log(1.0 / d))


def _energy_and_grad(pts):
    """Vectorized energy and projected gradient."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dists_sq, 1.0)
    inv_d2 = 1.0 / dists_sq
    np.fill_diagonal(inv_d2, 0.0)

    weighted_sum_pts = inv_d2 @ pts
    sum_inv_d2 = np.sum(inv_d2, axis=1)
    grad = -pts * sum_inv_d2[:, np.newaxis] + weighted_sum_pts

    # Project to tangent space
    dot_prod = np.sum(grad * pts, axis=1)
    grad -= dot_prod[:, np.newaxis] * pts

    iu = np.triu_indices(n, k=1)
    d = np.sqrt(np.maximum(dists_sq[iu], 1e-30))
    energy = np.sum(np.log(1.0 / d))

    return energy, grad


def _fibonacci_sphere(n):
    """Fibonacci spiral on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)


def _halton_sphere(n):
    """Halton sequence mapped to sphere."""
    def halton(index, base):
        f, r, i = 1.0, 0.0, index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r

    pts = []
    for i in range(1, n + 1):
        u1 = halton(i, 2)
        u2 = halton(i, 3)
        theta = 2.0 * np.pi * u1
        phi = np.arccos(2.0 * u2 - 1.0)
        pts.append([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)])
    pts = np.array(pts)
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)


def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _exponential_map(pts, tangent_vecs):
    """Exponential map on unit sphere."""
    norms = np.linalg.norm(tangent_vecs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    cos_n = np.cos(norms)
    sin_n_over_n = np.sin(norms) / norms
    return cos_n * pts + sin_n_over_n * tangent_vecs


def _riemannian_cg(pts, time_limit=15.0, max_iter=300):
    """Riemannian CG with Armijo line search and Polak-Ribiere update."""
    start = time.time()
    current = pts.copy()
    energy, grad = _energy_and_grad(current)

    d = -grad.copy()
    prev_grad = grad.copy()

    for it in range(max_iter):
        if time.time() - start > time_limit - 0.1:
            break

        # Armijo backtracking line search
        alpha = 0.5
        c1 = 1e-4
        grad_dot_d = np.sum(grad * d)
        best_energy = energy
        best_pts = current.copy()
        best_alpha = 0.0
        best_grad = grad.copy()

        for ls in range(50):
            new_pts = _exponential_map(current, alpha * d)
            new_energy, new_grad = _energy_and_grad(new_pts)

            if new_energy < best_energy:
                best_energy = new_energy
                best_pts = new_pts
                best_alpha = alpha
                best_grad = new_grad

            # Armijo condition
            if new_energy <= energy + c1 * alpha * grad_dot_d:
                break

            alpha *= 0.7

        if best_alpha == 0.0:
            # Fallback: try very small steps
            alpha = 1e-4
            for ls in range(20):
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
        if grad_norm < 1e-13:
            break

        # Polak-Ribiere with reset
        beta = 0.0
        if it > 0:
            beta_num = np.sum(grad * (grad - prev_grad))
            beta_den = np.sum(prev_grad * prev_grad)
            if beta_den > 1e-30:
                beta = max(0.0, beta_num / beta_den)
            if beta > 2.0:
                beta = 0.0

        d = -grad + beta * d
        prev_grad = grad.copy()

    return current, energy


def _newton_polish(pts, time_limit=8.0, max_iter=200):
    """Newton-style polish using BFGS Hessian approximation in tangent space."""
    start = time.time()
    current = pts.copy()
    energy, grad = _energy_and_grad(current)

    # BFGS: maintain inverse Hessian approximation in flattened tangent space
    dim = N * 3
    H_inv = np.eye(dim)  # start with identity

    for it in range(max_iter):
        if time.time() - start > time_limit - 0.1:
            break

        grad_flat = grad.flatten()
        search_dir = H_inv @ grad_flat
        search_dir = search_dir.reshape(N, 3)

        # Project to tangent space
        dot_prod = np.sum(search_dir * current, axis=1)
        search_dir -= dot_prod[:, np.newaxis] * current
        search_dir = -search_dir  # descent direction

        # Line search
        alpha = 0.1
        c1 = 1e-4
        grad_dot_d = np.sum(grad * search_dir)
        if grad_dot_d >= 0:
            search_dir = -grad
            grad_dot_d = np.sum(grad * search_dir)

        best_energy = energy
        best_pts = current.copy()
        best_alpha = 0.0
        best_grad = grad.copy()

        for ls in range(30):
            new_pts = _exponential_map(current, alpha * search_dir)
            new_energy, new_grad = _energy_and_grad(new_pts)

            if new_energy < best_energy:
                best_energy = new_energy
                best_pts = new_pts
                best_alpha = alpha
                best_grad = new_grad

            if new_energy <= energy + c1 * alpha * grad_dot_d:
                break

            alpha *= 0.7

        if best_alpha == 0.0:
            break

        # BFGS update
        s = (best_pts - current).flatten()
        y = (best_grad - grad).flatten()
        sy = np.sum(s * y)
        if sy > 1e-12:
            rho = 1.0 / sy
            # BFGS update for inverse Hessian
            H_inv = (np.eye(dim) - rho * np.outer(s, y)) @ H_inv @ (np.eye(dim) - rho * np.outer(y, s)) + rho * np.outer(s, s)

        current = best_pts
        energy = best_energy
        grad = best_grad

        if np.linalg.norm(grad) < 1e-13:
            break

    return current, energy


def _lbfgs_optimize(pts, time_limit=25.0):
    """L-BFGS-B on 3D coordinates with normalization."""
    x0 = pts.flatten()

    def fun_and_grad(x):
        p = x.reshape(N, 3)
        p = _normalize(p)
        energy, grad = _energy_and_grad(p)
        return energy, grad.flatten()

    res = minimize(fun_and_grad, x0, method='L-BFGS-B', jac=True,
                   options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-12,
                            'maxfun': 12000, 'maxls': 50})

    x_opt = res.x.reshape(N, 3)
    x_opt = _normalize(x_opt)
    energy = _log_energy_fast(x_opt)
    return x_opt, energy


def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    best_energy = float('inf')
    best_pts = None

    # Generate diverse initializations
    initializations = []

    # 1. Fibonacci sphere
    fib_pts = _fibonacci_sphere(N)
    initializations.append(fib_pts.copy())

    # 2. Halton sphere
    hal_pts = _halton_sphere(N)
    initializations.append(hal_pts.copy())

    # 3. Random uniform (several seeds)
    for seed in range(5):
        rng = np.random.default_rng(seed * 123 + 7)
        pts = rng.standard_normal((N, 3))
        initializations.append(_normalize(pts))

    # 4. Perturbed Fibonacci
    for seed in range(3):
        rng = np.random.default_rng(seed * 456 + 13)
        pts = fib_pts.copy() + rng.standard_normal((N, 3)) * 0.03
        initializations.append(_normalize(pts))

    # 5. Perturbed Halton
    for seed in range(2):
        rng = np.random.default_rng(seed * 789 + 23)
        pts = hal_pts.copy() + rng.standard_normal((N, 3)) * 0.03
        initializations.append(_normalize(pts))

    # 6. Different Fibonacci offsets
    for offset in [0.3, 0.7]:
        i = np.arange(N) + offset
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0, 1.0 - z * z))
        pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        initializations.append(_normalize(pts))

    for idx, init_pts in enumerate(initializations):
        elapsed = time.time() - start_time
        if elapsed > deadline - 22:
            break

        try:
            # Phase 1: L-BFGS-B coarse optimization
            t_left = deadline - time.time() - 18
            if t_left < 5:
                break
            current_pts, current_energy = _lbfgs_optimize(init_pts, time_limit=min(t_left, 25.0))

            # Phase 2: Riemannian CG fine optimization
            t_left = deadline - time.time() - 10
            if t_left > 2:
                current_pts, current_energy = _riemannian_cg(current_pts, time_limit=min(t_left, 12.0))

            # Phase 3: Newton polish
            t_left = deadline - time.time() - 4
            if t_left > 1:
                current_pts, current_energy = _newton_polish(current_pts, time_limit=min(t_left, 6.0))

            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = current_pts.copy()

        except Exception:
            continue

    # Final polish on best solution
    if best_pts is not None and time.time() < deadline - 2:
        final_time = min(deadline - time.time() - 1, 5.0)
        if final_time > 0.5:
            try:
                final_pts, final_energy = _newton_polish(best_pts, time_limit=final_time)
                if final_energy < best_energy:
                    best_energy = final_energy
                    best_pts = final_pts
            except Exception:
                pass

    if best_pts is not None:
        best_pts = _normalize(best_pts)
    else:
        best_pts = _normalize(fib_pts)

    return best_pts
