import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150


def _log_energy_fast(pts):
    """Vectorized log energy: sum_{i<j} log(1/|x_i - x_j|)."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(dists_sq[iu])
    d = np.maximum(d, 1e-15)
    return float(np.sum(np.log(1.0 / d)))


def _energy_and_grad(pts):
    """Vectorized energy + tangent-projected gradient (computed together)."""
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

    # Energy (upper triangle)
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(dists_sq[iu])
    d = np.maximum(d, 1e-15)
    energy = float(np.sum(np.log(1.0 / d)))

    return energy, grad


def _fibonacci_sphere(n):
    """Fibonacci spiral points on the unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _normalize(pts)


def _halton_sphere(n, base=2, base2=3):
    """Halton sequence points on the unit sphere."""
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
        u = halton(i, base)
        v = halton(i, base2)
        z = 2.0 * u - 1.0
        r = np.sqrt(max(0.0, 1.0 - z * z))
        theta = 2.0 * np.pi * v
        pts.append([r * np.cos(theta), r * np.sin(theta), z])
    pts = np.array(pts)
    return _normalize(pts)


def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _exponential_map(pts, tangent_vecs):
    """Exponential map on the unit sphere."""
    norms = np.linalg.norm(tangent_vecs, axis=1, keepdims=True)
    norms_safe = np.maximum(norms, 1e-15)
    cos_norms = np.cos(norms)
    sin_norms_over_norms = np.sin(norms) / norms_safe
    mask = norms < 1e-15
    sin_norms_over_norms[mask] = 1.0
    return cos_norms * pts + sin_norms_over_norms * tangent_vecs


def _riemannian_cg(pts, time_limit=15.0, max_iter=400):
    """Riemannian conjugate gradient with Armijo line search + beta cap."""
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
        gd = np.sum(grad * d)

        for ls in range(40):
            new_pts = _exponential_map(current, alpha * d)
            new_energy, new_grad = _energy_and_grad(new_pts)

            if new_energy < best_energy:
                best_energy = new_energy
                best_pts = new_pts
                best_alpha = alpha
                best_grad = new_grad

            # Armijo sufficient-decrease condition
            if new_energy <= energy + 1e-4 * alpha * gd:
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

        if np.linalg.norm(grad) < 1e-14:
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


def _lbfgs_optimize(pts, time_limit=30.0):
    """L-BFGS-B with projected gradient (tight tolerances)."""
    start = time.time()
    x0 = pts.flatten()

    def fun_and_grad(x):
        if time.time() - start > time_limit - 0.5:
            raise ValueError("Time limit exceeded")
        p = _normalize(x.reshape(N, 3))
        energy, grad = _energy_and_grad(p)
        return energy, grad.flatten()

    try:
        res = minimize(fun_and_grad, x0, method='L-BFGS-B', jac=True,
                       options={'maxiter': 4000, 'ftol': 1e-18, 'gtol': 1e-14,
                                'maxfun': 16000, 'maxls': 100})
        x_opt = _normalize(res.x.reshape(N, 3))
        energy = _log_energy_fast(x_opt)
        return x_opt, energy
    except Exception:
        return pts.copy(), _log_energy_fast(pts)


def _newton_polish(pts, time_limit=8.0, max_iter=10000):
    """Exact Newton-type local solve in 3D (unconstrained) coordinates."""
    start = time.time()
    x = pts.flatten().copy()
    n = N

    # Precompute identity
    I_n = np.eye(n)
    I_3 = np.eye(3)
    I_full = np.kron(I_n, I_3)

    for it in range(max_iter):
        if time.time() - start > time_limit - 0.1:
            break

        p = _normalize(x.reshape(n, 3))
        _, g = _energy_and_grad(p)
        gf = g.flatten()
        gn = np.linalg.norm(gf)
        if gn < 1e-15:
            break

        # Hessian of the smooth 3D objective at the normalized point
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        d2 = np.einsum('ijk,ijk->ij', diff, diff)
        np.fill_diagonal(d2, 1.0)
        inv_d2 = 1.0 / d2
        np.fill_diagonal(inv_d2, 0.0)

        row_sum = np.sum(inv_d2, axis=1)
        A = np.diag(row_sum) - inv_d2
        H = np.kron(A, I_3) - I_full
        H = 0.5 * (H + H.T)

        try:
            # Newton direction with damped least-squares fallback
            p_step, res = np.linalg.lstsq(H, -gf, rcond=None)
            if not np.all(np.isfinite(p_step)):
                raise np.linalg.LinAlgError("non-finite step")
        except Exception:
            p_step = -gf

        # Backtracking line search
        alpha = 1.0
        accepted = False
        e_cur = _log_energy_fast(p)
        for _ in range(40):
            x_try = x + alpha * p_step
            if not np.all(np.isfinite(x_try)):
                alpha *= 0.5
                continue
            p_try = _normalize(x_try.reshape(n, 3))
            e_try = _log_energy_fast(p_try)
            if e_try < e_cur - 1e-16 * (1.0 + abs(e_try)) or alpha < 1e-12:
                x = x_try
                accepted = True
                break
            alpha *= 0.5

        if not accepted:
            break

    return _normalize(x.reshape(n, 3)), _log_energy_fast(_normalize(x.reshape(n, 3)))


def _simulated_annealing(pts, time_limit=20.0, max_iter=12000):
    """Simulated annealing to escape local minima (single-point moves) with O(N) delta."""
    start = time.time()
    n = N
    current = pts.copy()
    current_energy = _log_energy_fast(current)
    best = current.copy()
    best_energy = current_energy

    T = 0.1
    T_min = 1e-8
    cooling = 0.999

    for _ in range(max_iter):
        if time.time() - start > time_limit - 0.5:
            break

        idx = np.random.randint(n)
        random_dir = np.random.randn(3)
        tangent = random_dir - np.sum(random_dir * current[idx]) * current[idx]
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm < 1e-15:
            continue
        tangent /= tangent_norm

        step = T * 0.5
        new_pt = current[idx] + step * tangent
        new_pt = new_pt / np.linalg.norm(new_pt)

        # Compute energy delta in O(N)
        old_dists = np.linalg.norm(current - current[idx], axis=1)
        old_dists[idx] = 1.0
        new_dists = np.linalg.norm(current - new_pt, axis=1)
        new_dists[idx] = 1.0

        old_pair = np.sum(np.log(1.0 / np.maximum(old_dists, 1e-15)))
        new_pair = np.sum(np.log(1.0 / np.maximum(new_dists, 1e-15)))
        delta_e = new_pair - old_pair

        if delta_e < 0 or np.random.rand() < np.exp(-delta_e / T):
            current = current.copy()
            current[idx] = new_pt
            current_energy = current_energy + delta_e
            if current_energy < best_energy:
                best_energy = current_energy
                best = current.copy()

        T *= cooling
        if T < T_min:
            T = T_min

    return best, best_energy


def _refine_pipeline(pts, deadline):
    """Run the full refinement pipeline on one start, respecting deadline."""
    current = pts.copy()
    energy = _log_energy_fast(current)

    if time.time() < deadline - 20:
        current, energy = _lbfgs_optimize(current, time_limit=15.0)

    if time.time() < deadline - 12:
        current, energy = _riemannian_cg(current, time_limit=10.0)

    if time.time() < deadline - 10:
        current, energy = _lbfgs_optimize(current, time_limit=4.0)

    # Simulated annealing to escape local minima, then re-polish
    if time.time() < deadline - 14:
        sa_pts, sa_energy = _simulated_annealing(current, time_limit=10.0)
        if sa_energy < energy:
            current, energy = sa_pts, sa_energy
            if time.time() < deadline - 8:
                current, energy = _lbfgs_optimize(current, time_limit=4.0)
            if time.time() < deadline - 5:
                current, energy = _riemannian_cg(current, time_limit=3.0)

    # Newton-type exact local solve
    if time.time() < deadline - 4:
        current, energy = _newton_polish(current, time_limit=5.0)

    return current, energy


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    best_energy = float('inf')
    best_pts = None

    # ---- Build a diverse set of initializations ----
    initializations = []

    fib_pts = _fibonacci_sphere(N)
    halton_pts = _halton_sphere(N)
    initializations.append(fib_pts.copy())
    initializations.append(halton_pts.copy())

    # Random uniform on sphere
    for seed in range(6):
        rng_local = np.random.default_rng(seed * 123 + 7)
        pts_rand = rng_local.standard_normal((N, 3))
        initializations.append(_normalize(pts_rand))

    # Perturbed Fibonacci
    for seed in range(4):
        rng_local = np.random.default_rng(seed * 456 + 13)
        pts_pert = fib_pts.copy() + rng_local.standard_normal((N, 3)) * 0.03
        initializations.append(_normalize(pts_pert))

    # Perturbed Halton
    for seed in range(3):
        rng_local = np.random.default_rng(seed * 789 + 23)
        pts_pert = halton_pts.copy() + rng_local.standard_normal((N, 3)) * 0.03
        initializations.append(_normalize(pts_pert))

    # Fibonacci offset variants
    for offset in [0.3, 0.7, 0.15, 0.85]:
        i = np.arange(N) + offset
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0, 1.0 - z * z))
        pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        initializations.append(_normalize(pts))

    # Spherical spiral
    i = np.arange(N)
    z = 1.0 - 2.0 * i / (N - 1)
    phi = np.linspace(0, 100 * np.pi, N)
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    initializations.append(_normalize(pts))

    # Large-perturbation random restarts (stronger basin hopping)
    for seed in range(4):
        rng_local = np.random.default_rng(seed * 999 + 5)
        pts_pert = fib_pts.copy() + rng_local.standard_normal((N, 3)) * 0.15
        initializations.append(_normalize(pts_pert))

    # Sort by initial energy (best first)
    init_energies = [_log_energy_fast(p) for p in initializations]
    order = np.argsort(init_energies)
    initializations = [initializations[i] for i in order]

    # ---- Refine each start, keep the best ----
    for init_pts in initializations:
        if time.time() > deadline - 30:
            break
        try:
            cur_pts, cur_energy = _refine_pipeline(init_pts, deadline)
            if cur_energy < best_energy:
                best_energy = cur_energy
                best_pts = cur_pts.copy()
        except Exception:
            pass

    if best_pts is None:
        best_pts = initializations[0].copy()
        best_energy = _log_energy_fast(best_pts)

    # ---- Final high-precision polish on the global best ----
    if time.time() < deadline - 8:
        polished, pe = _newton_polish(best_pts, time_limit=8.0)
        if pe < best_energy:
            best_pts, best_energy = polished, pe

    if time.time() < deadline - 4:
        polished, pe = _riemannian_cg(best_pts, time_limit=3.0)
        if pe < best_energy:
            best_pts, best_energy = polished, pe

    if time.time() < deadline - 2:
        polished, pe = _lbfgs_optimize(best_pts, time_limit=2.0)
        if pe < best_energy:
            best_pts, best_energy = polished, pe

    if time.time() < deadline - 1:
        polished, pe = _newton_polish(best_pts, time_limit=2.0)
        if pe < best_energy:
            best_pts, best_energy = polished, pe

    # Final basin-hopping: small perturbations + quick polish
    if time.time() < deadline - 3:
        bh_deadline = deadline - 2.0
        for bh_try in range(20):
            if time.time() > bh_deadline:
                break
            rng_bh = np.random.default_rng(bh_try * 31 + 999)
            perturb = best_pts.copy() + rng_bh.standard_normal((N, 3)) * 0.02
            perturb = _normalize(perturb)
            if time.time() < bh_deadline - 5:
                polished, pe = _riemannian_cg(perturb, time_limit=3.0)
            else:
                polished, pe = _normalize(perturb), _log_energy_fast(perturb)
            if pe < best_energy:
                best_pts, best_energy = polished, pe
                if time.time() < bh_deadline - 2:
                    polished2, pe2 = _newton_polish(best_pts, time_limit=2.0)
                    if pe2 < best_energy:
                        best_pts, best_energy = polished2, pe2

    best_pts = _normalize(best_pts)
    return best_pts
