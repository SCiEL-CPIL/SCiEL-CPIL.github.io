import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150
DEADLINE_S = 142.0  # safety margin before 150s limit


def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _log_energy_fast(pts):
    """Vectorized log energy computation for speed."""
    n = len(pts)
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diffs * diffs, axis=2)
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(dists_sq[iu])
    d = np.maximum(d, 1e-15)
    return float(np.sum(np.log(1.0 / d)))


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

    # Project gradient to tangent space: g - (g . p) p
    dot_prod = np.sum(grad * pts, axis=1)
    grad -= dot_prod[:, np.newaxis] * pts

    # Compute energy (upper triangle only)
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(dists_sq[iu])
    d = np.maximum(d, 1e-15)
    energy = float(np.sum(np.log(1.0 / d)))

    return energy, grad


def _exponential_map(pts, tangent_vecs):
    """Apply exponential map on sphere: move points along tangent vectors."""
    norms = np.linalg.norm(tangent_vecs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    cos_norms = np.cos(norms)
    sin_norms_over_norms = np.sin(norms) / norms
    return cos_norms * pts + sin_norms_over_norms * tangent_vecs


def _riemannian_cg(pts, max_iter=500, tol=1e-10, time_limit=None, start_time=None):
    """Riemannian conjugate gradient on the sphere for log-energy minimization."""
    if start_time is None:
        start_time = time.time()
    pts = pts.copy()
    n = len(pts)

    def _time_exceeded():
        if time_limit is None:
            return False
        return time.time() - start_time > time_limit

    # Initial gradient
    energy, grad = _energy_and_grad(pts)
    p = -grad  # initial search direction

    for it in range(max_iter):
        if _time_exceeded():
            break

        # Adaptive Line Search: Backtracking Armijo
        c1 = 1e-4
        alpha = 0.1
        grad_dot_p = np.sum(grad * p)
        if grad_dot_p >= 0:
            p = -grad
            grad_dot_p = np.sum(grad * p)
            if grad_dot_p >= 0:
                break

        new_energy = energy
        new_pts = pts
        new_grad = grad
        found = False

        for _ in range(20):
            trial_pts = _exponential_map(pts, alpha * p)
            trial_pts = _normalize(trial_pts)
            trial_e, trial_grad = _energy_and_grad(trial_pts)

            if trial_e <= energy - c1 * alpha * grad_dot_p:
                new_energy = trial_e
                new_pts = trial_pts
                new_grad = trial_grad
                found = True
                break

            alpha *= 0.5
            if _time_exceeded():
                break

        if not found:
            break

        grad_norm = np.linalg.norm(new_grad)
        if grad_norm < tol:
            pts = new_pts
            energy = new_energy
            break

        beta_num = np.sum(new_grad * (new_grad - grad))
        beta_den = np.sum(grad * grad)
        if beta_den > 1e-15:
            beta = beta_num / beta_den
        else:
            beta = 0.0
        beta = max(0.0, min(beta, 1.0))

        p = -new_grad + beta * p
        dot_p = np.sum(p * new_pts, axis=1, keepdims=True)
        p -= dot_p * new_pts

        pts = new_pts
        energy = new_energy
        grad = new_grad

    return pts, _log_energy_fast(pts)


def _lbfgs_sphere(pts, time_limit=15.0):
    """L-BFGS-B optimization on the sphere using ambient coordinates."""
    start = time.time()
    x0 = pts.flatten().copy()

    def obj_and_grad(x):
        if time.time() - start > time_limit:
            return float('inf'), np.zeros_like(x)
        pts = x.reshape(N, 3)
        pts = _normalize(pts)
        energy, grad = _energy_and_grad(pts)
        return energy, grad.flatten()

    result = minimize(
        obj_and_grad,
        x0,
        method='L-BFGS-B',
        jac=True,
        options={
            'maxiter': 2000,
            'ftol': 1e-14,
            'gtol': 1e-10,
            'maxfun': 20000,
        }
    )

    if result.success or result.fun < float('inf'):
        best_pts = _normalize(result.x.reshape(N, 3))
        best_energy = _log_energy_fast(best_pts)
    else:
        best_pts = pts.copy()
        best_energy = _log_energy_fast(best_pts)

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
        theta = 2.0 * np.pi * u1
        phi = np.arccos(2.0 * u2 - 1.0)

        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)

        pts.append([x, y, z])

    pts = np.array(pts)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts


def _spherical_design_init(n, seed=42):
    """Generate points using structured approach with random rotations."""
    rng = np.random.default_rng(seed)
    pts = _fibonacci_sphere(n)
    for _ in range(3):
        q = rng.standard_normal(4)
        q /= np.linalg.norm(q)
        w, x, y, z = q
        R = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        ])
        pts = pts @ R.T
    return _normalize(pts)


def _fps_fast(n, seed=42, batch_size=1000):
    """Fast farthest point sampling using vectorized operations."""
    rng = np.random.default_rng(seed)
    pts = [rng.standard_normal(3)]
    pts[0] = pts[0] / np.linalg.norm(pts[0])
    pts_arr = np.array(pts)

    for _ in range(n - 1):
        candidates = rng.standard_normal((batch_size, 3))
        candidates = _normalize(candidates)

        min_dists = np.full(batch_size, np.inf)
        chunk = 50
        for i in range(0, len(pts_arr), chunk):
            chunk_pts = pts_arr[i:i+chunk]
            d = np.linalg.norm(candidates[:, np.newaxis, :] - chunk_pts[np.newaxis, :, :], axis=2)
            min_dists = np.minimum(min_dists, np.min(d, axis=1))

        best_idx = np.argmax(min_dists)
        pts_arr = np.vstack([pts_arr, candidates[best_idx:best_idx+1]])

    return _normalize(pts_arr)


def _basin_hopping(pts, time_limit=20.0, max_restarts=10):
    """Basin hopping with Riemannian CG + L-BFGS local optimization."""
    start = time.time()
    current = pts.copy()
    energy = _log_energy_fast(current)

    time_left_init = time_limit - (time.time() - start) - 2.0
    if time_left_init > 5.0:
        cg_time = min(5.0, time_left_init * 0.4)
        current, _ = _riemannian_cg(current, max_iter=200, tol=1e-8,
                                     time_limit=cg_time, start_time=time.time())
        lbfgs_time = min(8.0, time_left_init - cg_time - 1.0)
        if lbfgs_time > 2.0:
            current, energy = _lbfgs_sphere(current, time_limit=lbfgs_time)

    best_pts = current.copy()
    best_energy = energy

    temp = 0.5
    decay = 0.85
    reheat_every = 5
    reheat_temp = 0.3

    for rh in range(max_restarts):
        if time.time() - start > time_limit - 2.0:
            break

        if rh > 0 and rh % reheat_every == 0:
            temp = reheat_temp

        perturb_scale = max(0.03, min(0.25, 0.08 * temp + 0.03))

        rng = np.random.default_rng(rh * 7919 + 1)
        perturbations = rng.standard_normal((N, 3))
        dot_prods = np.sum(perturbations * current, axis=1, keepdims=True)
        perturbations -= dot_prods * current
        pert_norms = np.linalg.norm(perturbations, axis=1, keepdims=True)
        pert_norms = np.maximum(pert_norms, 1e-15)
        perturbations = perturbations / pert_norms * perturb_scale

        new_pts = _exponential_map(current, perturbations)
        new_pts = _normalize(new_pts)

        time_left = time_limit - (time.time() - start) - 1.0
        if time_left < 4.0:
            break

        cg_t = min(3.0, time_left * 0.35)
        new_opt_pts, _ = _riemannian_cg(new_pts, max_iter=100, tol=1e-7,
                                          time_limit=cg_t, start_time=time.time())
        lbfgs_t = min(6.0, time_left - cg_t - 0.5)
        if lbfgs_t > 2.0:
            new_opt_pts, new_energy = _lbfgs_sphere(new_opt_pts, time_limit=lbfgs_t)
        else:
            new_energy = _log_energy_fast(new_opt_pts)
        new_energy = _log_energy_fast(new_opt_pts)

        delta_e = new_energy - energy
        if delta_e < 0 or np.random.rand() < math.exp(-delta_e / max(temp, 1e-10)):
            current = new_opt_pts
            energy = new_energy
            if energy < best_energy:
                best_energy = energy
                best_pts = current.copy()

        temp *= decay

    return best_pts, best_energy


def _swap_local_search(pts, time_limit=5.0, max_rounds=50):
    """
    Local search via pairwise swaps: for each pair (i,j), try swapping their
    positions. Accept if energy decreases. This is a combinatorial move that
    can escape local minima of the continuous optimization.
    """
    start = time.time()
    pts = pts.copy()
    energy = _log_energy_fast(pts)
    n = len(pts)

    # Precompute all-pairs distances for speed
    def _swap_energy(pts, i, j):
        """Compute energy after swapping points i and j."""
        pts_swapped = pts.copy()
        pts_swapped[i], pts_swapped[j] = pts_swapped[j].copy(), pts_swapped[i].copy()
        return _log_energy_fast(pts_swapped)

    improved = True
    round_num = 0
    while improved and round_num < max_rounds:
        if time.time() - start > time_limit:
            break
        improved = False
        round_num += 1

        # Try random pairs to save time
        rng = np.random.default_rng(round_num * 31 + 7)
        n_pairs = min(200, n * (n - 1) // 2)
        for _ in range(n_pairs):
            i = rng.integers(0, n)
            j = rng.integers(0, n)
            if i == j:
                continue
            new_e = _swap_energy(pts, i, j)
            if new_e < energy - 1e-10:
                pts[i], pts[j] = pts[j].copy(), pts[i].copy()
                energy = new_e
                improved = True

    return pts, energy


def _simulated_annealing_global(pts, time_limit=15.0, T_start=0.5, T_end=0.01, max_iter=200):
    """
    Simulated annealing in the space of permutations (swaps) combined with
    small continuous perturbations. This explores the combinatorial structure
    of the problem.
    """
    start = time.time()
    current = pts.copy()
    current_e = _log_energy_fast(current)
    best = current.copy()
    best_e = current_e
    n = len(current)

    T = T_start
    cooling = (T_end / T_start) ** (1.0 / max(max_iter, 1))

    for it in range(max_iter):
        if time.time() - start > time_limit:
            break
        T *= cooling

        rng = np.random.default_rng(it * 104729 + 3)
        # Random swap
        i = rng.integers(0, n)
        j = rng.integers(0, n)
        if i == j:
            continue

        trial = current.copy()
        trial[i], trial[j] = trial[j].copy(), trial[i].copy()

        # Add small continuous perturbation
        perturb = rng.standard_normal((n, 3)) * 0.02
        dot_p = np.sum(perturb * trial, axis=1, keepdims=True)
        perturb -= dot_p * trial
        trial = _exponential_map(trial, perturb)
        trial = _normalize(trial)

        trial_e = _log_energy_fast(trial)
        delta = trial_e - current_e

        if delta < 0 or np.random.rand() < math.exp(-delta / max(T, 1e-10)):
            current = trial
            current_e = trial_e
            if current_e < best_e:
                best_e = current_e
                best = current.copy()

    return best, best_e


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_S

    # Initializations - diverse set
    inits = []
    inits.append(_fibonacci_sphere(N))
    inits.append(_halton_sphere(N))
    inits.append(_spherical_design_init(N, seed=42))
    inits.append(_spherical_design_init(N, seed=123))
    inits.append(_fps_fast(N, seed=7))
    inits.append(_fps_fast(N, seed=13))

    for s in range(3):
        rng = np.random.default_rng(s)
        pts = rng.standard_normal((N, 3))
        pts = _normalize(pts)
        inits.append(pts)

    for s in range(3):
        rng = np.random.default_rng(s + 100)
        base = _fibonacci_sphere(N) if s % 2 == 0 else _halton_sphere(N)
        pts = base + rng.standard_normal((N, 3)) * 0.12
        pts = _normalize(pts)
        inits.append(pts)

    best_pts = None
    best_energy = float('inf')

    total_inits = len(inits)
    for idx, init_pts in enumerate(inits):
        remaining = deadline - time.time() - 12.0
        if remaining < 8.0:
            break

        if idx < 4:
            time_per_restart = min(22.0, remaining / (total_inits - idx) * 1.5)
        elif idx < 8:
            time_per_restart = min(16.0, remaining / (total_inits - idx) * 1.2)
        else:
            time_per_restart = min(10.0, remaining / (total_inits - idx) * 1.0)

        time_per_restart = max(6.0, time_per_restart)

        try:
            opt_pts, opt_energy = _basin_hopping(init_pts, time_limit=time_per_restart)
            opt_energy = _log_energy_fast(opt_pts)

            if opt_energy < best_energy:
                best_energy = opt_energy
                best_pts = opt_pts.copy()

        except Exception:
            pass

    if best_pts is None:
        best_pts = _fibonacci_sphere(N)
        best_energy = _log_energy_fast(best_pts)

    # Final polish with multiple techniques
    remaining = deadline - time.time() - 3.0
    if remaining > 5.0:
        try:
            # 1. Riemannian CG polish
            cg_polish_time = min(10.0, remaining * 0.3)
            if cg_polish_time > 2.0:
                polished_pts, _ = _riemannian_cg(best_pts, max_iter=1000, tol=1e-12,
                                                  time_limit=cg_polish_time,
                                                  start_time=time.time())
                polished_e = _log_energy_fast(polished_pts)
                if polished_e < best_energy:
                    best_energy = polished_e
                    best_pts = polished_pts

            # 2. L-BFGS polish
            lbfgs_polish_time = min(10.0, deadline - time.time() - 3.0)
            if lbfgs_polish_time > 3.0:
                final_pts, final_energy = _lbfgs_sphere(best_pts, time_limit=lbfgs_polish_time)
                final_energy = _log_energy_fast(final_pts)
                if final_energy < best_energy:
                    best_energy = final_energy
                    best_pts = final_pts

            # 3. Swap-based local search (combinatorial moves)
            swap_time = min(5.0, deadline - time.time() - 3.0)
            if swap_time > 2.0:
                swap_pts, swap_e = _swap_local_search(best_pts, time_limit=swap_time)
                swap_e = _log_energy_fast(swap_pts)
                if swap_e < best_energy:
                    best_energy = swap_e
                    best_pts = swap_pts

            # 4. Simulated annealing with swaps + perturbations
            sa_time = min(8.0, deadline - time.time() - 3.0)
            if sa_time > 3.0:
                sa_pts, sa_e = _simulated_annealing_global(best_pts, time_limit=sa_time)
                sa_e = _log_energy_fast(sa_pts)
                if sa_e < best_energy:
                    best_energy = sa_e
                    best_pts = sa_pts

            # 5. Final CG polish after combinatorial moves
            remaining2 = deadline - time.time() - 3.0
            if remaining2 > 3.0:
                cg2_time = min(8.0, remaining2)
                if cg2_time > 1.5:
                    final2_pts, _ = _riemannian_cg(best_pts, max_iter=1000, tol=1e-14,
                                                    time_limit=cg2_time,
                                                    start_time=time.time())
                    final2_e = _log_energy_fast(final2_pts)
                    if final2_e < best_energy:
                        best_energy = final2_e
                        best_pts = final2_pts

        except Exception:
            pass

    return _normalize(best_pts)
