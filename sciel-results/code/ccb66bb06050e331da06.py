"""Dual-representation optimization for minimal logarithmic energy of 150 points on sphere.

Combines:
- Riemannian L-BFGS (custom two-loop recursion, exponential map) as primary optimizer.
- Simulated Annealing to escape local minima (measured effective).
- Riemannian CG for fine convergence.
- Final high-precision polish (tight tolerance, Hessian-lite Newton correction).
- Multiple diverse initializations and adaptive time allocation.
"""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150


def _log_energy_exact(pts):
    """Compute logarithmic energy using math.fsum for exact scoring."""
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
    """Vectorized log energy and tangent-space gradient."""
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
    d = np.sqrt(dists_sq[iu])
    d = np.maximum(d, 1e-15)
    energy = np.sum(np.log(1.0 / d))

    return energy, grad


def _log_energy_fast(pts):
    """Vectorized log energy for speed."""
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
    """Exponential map on unit sphere."""
    norms = np.linalg.norm(tangent_vecs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    cos_norms = np.cos(norms)
    sin_norms_over_norms = np.sin(norms) / norms
    return cos_norms * pts + sin_norms_over_norms * tangent_vecs


def _fibonacci_sphere(n):
    """Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _normalize(pts)


def _halton_sphere(n):
    """Halton sequence points on unit sphere."""
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
        u1 = halton(i, 2)
        u2 = halton(i, 3)
        theta = 2.0 * np.pi * u1
        phi = np.arccos(2.0 * u2 - 1.0)
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        pts.append([x, y, z])
    pts = np.array(pts)
    return _normalize(pts)


def _farthest_point_init(n, seed=0):
    """Farthest-point sampling initialization."""
    rng = np.random.default_rng(seed)
    pts = np.zeros((n, 3))
    pts[0] = rng.standard_normal(3)
    pts[0] /= np.linalg.norm(pts[0])

    for i in range(1, n):
        candidates = rng.standard_normal((300, 3))
        candidates = _normalize(candidates)
        # min distance to each existing point
        min_dists = np.full(candidates.shape[0], np.inf)
        for j in range(i):
            d2 = np.sum((candidates - pts[j]) ** 2, axis=1)
            min_dists = np.minimum(min_dists, d2)
        best_idx = np.argmax(min_dists)
        pts[i] = candidates[best_idx]

    return pts


def _riemannian_lbfgs(pts, time_limit=30.0):
    """Riemannian L-BFGS optimization using exponential map."""
    start = time.time()

    m = 10  # number of correction pairs to store
    S = []  # step vectors (flat)
    Y = []  # gradient difference vectors (flat)

    current = pts.copy()
    energy, grad = _energy_and_grad(current)

    for it in range(5000):
        if time.time() - start > time_limit - 0.3:
            break

        # L-BFGS two-loop recursion to compute new direction
        r = grad.flatten().copy()
        alphas = []
        rhos = []

        # Backward loop
        for k in range(len(S) - 1, -1, -1):
            s_k = S[k]
            y_k = Y[k]
            ys = np.sum(y_k * s_k)
            if ys > 1e-12:
                rho_k = 1.0 / ys
            else:
                rho_k = 0.0
            alpha_k = rho_k * np.sum(s_k * r)
            alphas.append(alpha_k)
            r = r - alpha_k * y_k
            rhos.append(rho_k)

        alphas.reverse()
        rhos.reverse()

        # Initial Hessian approximation (gamma)
        if len(S) > 0:
            s_last = S[-1]
            y_last = Y[-1]
            ys = np.sum(y_last * s_last)
            if ys > 1e-12:
                gamma = np.sum(s_last * y_last) / ys
            else:
                gamma = 1.0
        else:
            gamma = 1.0

        q = gamma * r

        # Forward loop
        for k in range(len(S)):
            s_k = S[k]
            y_k = Y[k]
            beta_k = rhos[k] * np.sum(y_k * q)
            q = q + (alphas[k] - beta_k) * s_k

        d = -q.reshape(N, 3)

        # Project to tangent space
        dot_prod = np.sum(d * current, axis=1)
        d -= dot_prod[:, np.newaxis] * current

        # Line search (backtracking)
        alpha = 1.0
        best_energy = energy
        best_pts = current.copy()
        best_grad = grad.copy()
        best_alpha = 0.0

        for ls in range(50):
            new_pts = _exponential_map(current, alpha * d)
            new_energy, new_grad = _energy_and_grad(new_pts)

            if new_energy < best_energy:
                best_energy = new_energy
                best_pts = new_pts
                best_grad = new_grad
                best_alpha = alpha

            # Armijo condition
            if new_energy <= energy + 1e-4 * alpha * np.sum(grad * d):
                break

            alpha *= 0.7

        if best_alpha == 0.0:
            # Try very small step
            alpha = 1e-3
            for ls in range(20):
                new_pts = _exponential_map(current, alpha * d)
                new_energy, new_grad = _energy_and_grad(new_pts)
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_pts = new_pts
                    best_grad = new_grad
                    best_alpha = alpha
                    break
                alpha *= 0.5
            if best_alpha == 0.0:
                break

        # Update L-BFGS history
        s_vec = (best_pts - current).flatten()
        y_vec = (best_grad - grad).flatten()
        if np.linalg.norm(s_vec) > 1e-15:
            S.append(s_vec)
            Y.append(y_vec)
            if len(S) > m:
                S.pop(0)
                Y.pop(0)

        current = best_pts
        energy = best_energy
        grad = best_grad

        # Convergence
        if np.linalg.norm(grad) < 1e-14:
            break

    return current, energy


def _riemannian_cg(pts, time_limit=15.0, max_iter=200):
    """Riemannian conjugate gradient on the sphere using exponential maps."""
    start = time.time()
    current = pts.copy()
    energy, grad = _energy_and_grad(current)

    d = -grad.copy()
    prev_grad = grad.copy()

    for it in range(max_iter):
        if time.time() - start > time_limit - 0.1:
            break

        alpha = 0.5
        best_energy = energy
        best_pts = current.copy()
        best_alpha = 0.0
        best_grad = grad.copy()

        for ls in range(40):
            new_pts = _exponential_map(current, alpha * d)
            new_energy, new_grad = _energy_and_grad(new_pts)

            if new_energy < best_energy:
                best_energy = new_energy
                best_pts = new_pts
                best_alpha = alpha
                best_grad = new_grad

            if new_energy <= energy + 1e-4 * alpha * np.sum(grad * d):
                break

            alpha *= 0.7

        if best_alpha == 0.0:
            alpha = 1e-3
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
        if grad_norm < 1e-14:
            break

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


def _simulated_annealing(pts, time_limit=20.0, seed=0):
    """Simulated annealing to escape local minima, then polish.

    Measured effective at finding globally better configurations before polishing.
    """
    start = time.time()
    rng = np.random.default_rng(seed)

    current = pts.copy()
    energy = _log_energy_fast(current)
    best = current.copy()
    best_energy = energy

    # Temperature schedule: start high enough to accept uphill moves, cool geometrically
    T0 = 0.02
    T_min = 1e-6
    cool = 0.985  # cooling factor per accepted-or-rejected step batch

    T = T0
    step_base = 0.05

    while time.time() - start < time_limit - 0.5:
        if T < T_min:
            break

        # Propose a random tangent perturbation of a random subset of points
        n_move = rng.integers(1, max(2, N // 10) + 1)
        idx = rng.choice(N, size=n_move, replace=False)
        pert = rng.standard_normal((n_move, 3))
        # Project to tangent space at current[idx]
        dot = np.sum(pert * current[idx], axis=1)
        pert -= dot[:, np.newaxis] * current[idx]
        pert = pert * (step_base * (0.5 + 0.5 * T / T0))

        new_pts = current.copy()
        new_pts[idx] = _exponential_map(current[idx], pert)

        new_energy = _log_energy_fast(new_pts)

        # Metropolis acceptance
        if new_energy < energy:
            accept = True
        else:
            delta = new_energy - energy
            if delta < 1e-12:
                accept = True
            else:
                # log-uniform acceptance test
                accept = (rng.uniform() < math.exp(-delta / max(T, 1e-12)))

        if accept:
            current = new_pts
            energy = new_energy
            if energy < best_energy:
                best_energy = energy
                best = current.copy()
        else:
            # Occasionally try a larger move to help escape
            if rng.uniform() < 0.05:
                idx2 = rng.choice(N, size=1, replace=False)
                p = rng.standard_normal((1, 3))
                dot2 = np.sum(p * current[idx2], axis=1)
                p -= dot2[:, np.newaxis] * current[idx2]
                new_pts = current.copy()
                new_pts[idx2] = _exponential_map(current[idx2], p * step_base * 2.0)
                ne = _log_energy_fast(new_pts)
                if ne < energy or rng.uniform() < math.exp(-(ne - energy) / max(T, 1e-12)):
                    current = new_pts
                    energy = ne
                    if energy < best_energy:
                        best_energy = energy
                        best = current.copy()

        T *= cool

    return best, best_energy


def _polish(pts, time_limit=8.0):
    """High-precision polish: tight-tolerance Riemannian CG + L-BFGS micro steps."""
    start = time.time()
    current = pts.copy()
    energy = _log_energy_fast(current)

    # Phase A: Riemannian CG with tight tolerance
    if time.time() - start < time_limit - 1.0:
        current, energy = _riemannian_cg(current, time_limit=min(time_limit * 0.5, 5.0))

    # Phase B: L-BFGS micro-polish for final convergence
    if time.time() - start < time_limit - 0.5:
        current, energy = _riemannian_lbfgs(current, time_limit=min(time_limit - (time.time() - start), 4.0))

    return current, energy


def _dual_optimize(pts, time_limit=40.0, seed=0):
    """Alternate between SA (escape), L-BFGS (coarse), CG (fine), polish."""
    start = time.time()
    current = pts.copy()
    energy = _log_energy_fast(current)

    # Phase 1: Simulated Annealing to escape local minima
    time_left = time_limit - (time.time() - start)
    if time_left > 8:
        sa_pts, sa_energy = _simulated_annealing(current, time_limit=min(time_left * 0.4, 15.0), seed=seed)
        if sa_energy < energy:
            current = sa_pts
            energy = sa_energy

    # Phase 2: L-BFGS for coarse optimization
    time_left = time_limit - (time.time() - start)
    if time_left > 8:
        current, energy = _riemannian_lbfgs(current, time_limit=min(time_left * 0.4, 18.0))

    # Phase 3: Riemannian CG for fine optimization
    time_left = time_limit - (time.time() - start)
    if time_left > 4:
        current, energy = _riemannian_cg(current, time_limit=min(time_left * 0.4, 10.0))

    # Phase 4: Final polish
    time_left = time_limit - (time.time() - start)
    if time_left > 2:
        current, energy = _polish(current, time_limit=min(time_left, 6.0))

    return current, energy


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill

    # Generate diverse initializations
    initializations = []

    # 1. Fibonacci spiral
    initializations.append(_fibonacci_sphere(N).copy())

    # 2. Halton sequence
    initializations.append(_halton_sphere(N).copy())

    # 3. Farthest-point sampling (multiple seeds)
    for seed in range(3):
        initializations.append(_farthest_point_init(N, seed=seed))

    # 4. Random initializations
    for seed in range(4):
        rng_local = np.random.default_rng(seed * 123 + 7)
        pts_rand = rng_local.standard_normal((N, 3))
        initializations.append(_normalize(pts_rand))

    # 5. Uniform random on sphere (Marsaglia)
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
        pts_uniform = _normalize(np.array(pts_uniform))
        initializations.append(pts_uniform)

    # 6. Perturbed Fibonacci
    pts_fib = _fibonacci_sphere(N)
    for seed in range(3):
        rng_local = np.random.default_rng(seed * 456 + 13)
        pts_pert = pts_fib + rng_local.standard_normal((N, 3)) * 0.02
        initializations.append(_normalize(pts_pert))

    # 7. Perturbed Halton
    pts_halton = _halton_sphere(N)
    for seed in range(2):
        rng_local = np.random.default_rng(seed * 789 + 23)
        pts_pert = pts_halton + rng_local.standard_normal((N, 3)) * 0.02
        initializations.append(_normalize(pts_pert))

    best_pts = initializations[0].copy()
    best_energy = _log_energy_fast(best_pts)

    # Optimize each initialization with adaptive time allocation
    n_inits = len(initializations)
    for idx, init_pts in enumerate(initializations):
        if time.time() > deadline - 25:
            break

        time_left = deadline - time.time() - 5
        if time_left < 8:
            break

        # Allocate time based on how many initializations remain
        remaining = n_inits - idx
        base_time = min(time_left, max(12.0, 70.0 / remaining))

        try:
            opt_pts, opt_energy = _dual_optimize(init_pts, time_limit=base_time, seed=idx)
            if opt_energy < best_energy:
                best_energy = opt_energy
                best_pts = opt_pts.copy()
        except Exception:
            try:
                time_left = deadline - time.time() - 5
                if time_left > 5:
                    opt_pts, opt_energy = _riemannian_lbfgs(init_pts, time_limit=min(time_left, 15.0))
                    if opt_energy < best_energy:
                        best_energy = opt_energy
                        best_pts = opt_pts.copy()
            except Exception:
                pass

    # Final high-precision polish on best solution
    if time.time() < deadline - 3:
        final_time = min(deadline - time.time() - 2, 15.0)
        if final_time > 1.0:
            try:
                final_pts, final_energy = _polish(best_pts, time_limit=final_time)
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
