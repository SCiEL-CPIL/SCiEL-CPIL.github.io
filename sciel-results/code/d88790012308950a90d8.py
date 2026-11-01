"""Dual-representation optimizer for minimal logarithmic energy of 150 points on sphere."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150


def _log_energy(pts):
    """Compute logarithmic energy using math.fsum for precision."""
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
    dists = np.sqrt(dists_sq)

    iu = np.triu_indices(n, k=1)
    d_upper = dists[iu]
    valid_mask = d_upper > 1e-12
    if not np.all(valid_mask):
        return float('inf'), np.zeros_like(pts)

    terms = [math.log(1.0 / d) for d in d_upper]
    energy = math.fsum(terms)

    inv_d2 = 1.0 / dists_sq
    np.fill_diagonal(inv_d2, 0.0)

    weighted_sum = inv_d2 @ pts
    sum_inv = np.sum(inv_d2, axis=1)
    grad = pts * sum_inv[:, np.newaxis] - weighted_sum

    # Project to tangent space
    dot_prod = np.sum(grad * pts, axis=1)
    grad -= dot_prod[:, np.newaxis] * pts

    return energy, grad


def _fibonacci_sphere(n):
    """Generate n points on sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)


def _halton_sphere(n):
    """Generate n points on sphere using Halton sequence in 3D."""
    def halton(index, base):
        result = 0.0
        f = 1.0
        i = index
        while i > 0:
            f /= base
            result += f * (i % base)
            i //= base
        return result

    points = []
    for i in range(1, n + 1):
        x = halton(i, 2)
        y = halton(i, 3)
        z = halton(i, 5)
        p = np.array([x, y, z])
        norm = np.linalg.norm(p)
        if norm > 1e-15:
            points.append(p / norm)
        else:
            points.append(np.array([1.0, 0.0, 0.0]))
    return np.array(points)


def _project_to_sphere(pts):
    """Project points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return pts / norms


def _exp_map(pts, v):
    """Exponential map on sphere: move points along tangent vectors."""
    norms_v = np.linalg.norm(v, axis=1)
    safe_norms = np.where(norms_v < 1e-15, 1.0, norms_v)
    cos_t = np.cos(norms_v)
    sin_t = np.sin(norms_v)
    v_safe = v / safe_norms[:, np.newaxis]
    return pts * cos_t[:, np.newaxis] + v_safe * sin_t[:, np.newaxis]


def _retract(pts, v):
    """Retraction: project pts + v to sphere."""
    return _project_to_sphere(pts + v)


def _lbgfs_sphere(pts0, max_iter=500, m=10, time_limit=None, start_time=None, tol=1e-12):
    """L-BFGS optimization on the sphere with projected gradients and Wolfe line search."""
    if start_time is None:
        start_time = time.time()

    pts = pts0.copy()
    energy, grad = _energy_and_grad(pts)

    s_list = []
    y_list = []

    step_size = 0.5
    min_step = 1e-14
    c1 = 1e-4
    c2 = 0.9

    for it in range(max_iter):
        if time_limit is not None and (time.time() - start_time) > time_limit:
            break

        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            break

        # L-BFGS two-loop recursion
        if len(s_list) == 0:
            direction = -grad
        else:
            q = grad.copy()
            rhos = [0.0] * len(s_list)
            alphas = [0.0] * len(s_list)

            # Backward loop
            for i in range(len(s_list) - 1, -1, -1):
                si = s_list[i]
                yi = y_list[i]
                si_yi = np.sum(si * yi)
                if abs(si_yi) < 1e-16:
                    rhos[i] = 0.0
                else:
                    rhos[i] = 1.0 / si_yi
                alphas[i] = np.sum(si * q) * rhos[i]
                q -= alphas[i] * yi

            # Initial Hessian approximation
            if len(s_list) > 0:
                last_s = s_list[-1]
                last_y = y_list[-1]
                sy = np.sum(last_s * last_y)
                if abs(sy) > 1e-16:
                    gamma = np.sum(last_s * last_s) / sy
                    if gamma < 1e-10 or gamma > 1e10:
                        gamma = 1.0
                else:
                    gamma = 1.0
            else:
                gamma = 1.0

            r = q * gamma

            # Forward loop
            for i in range(len(s_list)):
                si = s_list[i]
                yi = y_list[i]
                beta = rhos[i] * np.sum(yi * r)
                r += si * (alphas[i] - beta)

            direction = -r

        # Ensure direction is on tangent space
        dot_dir = np.sum(direction * pts, axis=1)
        direction -= dot_dir[:, np.newaxis] * pts

        # Wolfe line search
        alpha = step_size
        accepted = False
        grad_dir = np.sum(grad * direction)

        if grad_dir >= 0:
            direction = -grad
            dot_dir = np.sum(direction * pts, axis=1)
            direction -= dot_dir[:, np.newaxis] * pts
            grad_dir = np.sum(grad * direction)

        for ls_iter in range(40):
            new_pts = _retract(pts, alpha * direction)
            new_energy, new_grad = _energy_and_grad(new_pts)

            if not np.isfinite(new_energy):
                alpha *= 0.5
                continue

            armijo = new_energy <= energy + c1 * alpha * grad_dir
            curvature = np.sum(new_grad * direction) >= c2 * grad_dir

            if armijo:
                accepted = True
                break
            else:
                alpha *= 0.5

        if not accepted:
            step_size *= 0.5
            if step_size < min_step:
                break

        # Update L-BFGS memory
        s = alpha * direction
        y = new_grad - grad
        sy = np.sum(s * y)

        if sy > 1e-14:
            if len(s_list) >= m:
                s_list.pop(0)
                y_list.pop(0)
            s_list.append(s)
            y_list.append(y)

        pts = new_pts
        grad = new_grad
        energy = new_energy
        step_size = min(alpha * 1.5, 2.0)

    return pts, energy


def _cg_sphere(pts0, max_iter=500, time_limit=None, start_time=None, tol=1e-12):
    """Conjugate gradient on the sphere using exponential maps."""
    if start_time is None:
        start_time = time.time()

    pts = pts0.copy()
    energy, grad = _energy_and_grad(pts)

    p = -grad
    p = p - np.sum(p * pts, axis=1, keepdims=True) * pts

    for it in range(max_iter):
        if time_limit is not None and (time.time() - start_time) > time_limit:
            break

        if np.linalg.norm(grad) < tol:
            break

        alpha = 0.5
        accepted = False
        for _ in range(50):
            new_pts = _exp_map(pts, alpha * p)
            new_energy, new_grad = _energy_and_grad(new_pts)

            if new_energy < energy + 1e-4 * alpha * np.sum(grad * p):
                accepted = True
                break
            alpha *= 0.5

        if not accepted:
            alpha = 0.01
            new_pts = _exp_map(pts, alpha * p)
            new_energy, new_grad = _energy_and_grad(new_pts)
            if new_energy < energy:
                accepted = True
            else:
                break

        if not accepted:
            break

        # Polak-Ribiere beta
        y_diff = new_grad - grad
        beta = np.sum(new_grad * y_diff) / max(np.sum(grad * grad), 1e-16)
        if beta < 0:
            beta = 0.0

        p_new = -new_grad + beta * p
        p_new = p_new - np.sum(p_new * new_pts, axis=1, keepdims=True) * new_pts

        # Restart if not descent
        if np.sum(new_grad * p_new) >= 0:
            p_new = -new_grad
            p_new = p_new - np.sum(p_new * new_pts, axis=1, keepdims=True) * new_pts

        pts = new_pts
        grad = new_grad
        energy = new_energy
        p = p_new

    return pts, energy


def _dual_optimize(pts0, max_iter=1000, time_limit=None, start_time=None):
    """Alternate between L-BFGS and CG phases."""
    if start_time is None:
        start_time = time.time()

    pts = pts0.copy()
    energy, grad = _energy_and_grad(pts)

    # Phase 1: L-BFGS
    remaining = time_limit - (time.time() - start_time) if time_limit else max_iter * 0.05
    phase1_time = min(remaining * 0.4, 20.0)
    pts, energy = _lbgfs_sphere(
        pts, max_iter=300, m=10,
        time_limit=phase1_time, start_time=time.time()
    )

    # Phase 2: CG with exponential maps
    remaining = time_limit - (time.time() - start_time) if time_limit else max_iter * 0.05
    phase2_time = min(remaining * 0.35, 15.0)
    pts, energy = _cg_sphere(
        pts, max_iter=300,
        time_limit=phase2_time, start_time=time.time()
    )

    # Phase 3: Final L-BFGS polish
    remaining = time_limit - (time.time() - start_time) if time_limit else max_iter * 0.05
    phase3_time = remaining * 0.9
    pts, energy = _lbgfs_sphere(
        pts, max_iter=500, m=20,
        time_limit=phase3_time, start_time=time.time()
    )

    return pts, energy


def _scipy_lbgfs(pts0, time_limit=None, start_time=None):
    """Use scipy's L-BFGS-B as a cross-check/refinement."""
    if start_time is None:
        start_time = time.time()

    def objective_and_grad(flat_pts):
        pts = flat_pts.reshape(N, 3)
        pts = _project_to_sphere(pts)
        energy, grad = _energy_and_grad(pts)
        # Flatten gradient
        return energy, grad.ravel()

    x0 = pts0.ravel()
    
    if time_limit is not None:
        remaining = time_limit - (time.time() - start_time)
        if remaining < 1.0:
            return pts0, _log_energy(pts0)

    result = minimize(
        objective_and_grad,
        x0,
        jac=True,
        method='L-BFGS-B',
        options={'maxiter': 500, 'gtol': 1e-10, 'maxls': 50}
    )

    pts = result.x.reshape(N, 3)
    pts = _project_to_sphere(pts)
    energy = _log_energy(pts)

    return pts, energy


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    # Multiple diverse initializations
    inits = []

    # 1. Fibonacci spiral
    inits.append(_fibonacci_sphere(N))

    # 2. Halton sequence
    inits.append(_halton_sphere(N))

    # 3. Random initializations
    for seed in [42, 123, 456, 789, 1000]:
        rng = np.random.default_rng(seed)
        pts_rand = rng.standard_normal((N, 3))
        pts_rand /= np.linalg.norm(pts_rand, axis=1, keepdims=True)
        inits.append(pts_rand)

    # 4. Perturbed Fibonacci
    for seed, scale in [(7, 0.05), (99, 0.1), (200, 0.02)]:
        rng = np.random.default_rng(seed)
        pts_fib = _fibonacci_sphere(N)
        pts_pert = pts_fib + rng.standard_normal((N, 3)) * scale
        pts_pert /= np.linalg.norm(pts_pert, axis=1, keepdims=True)
        inits.append(pts_pert)

    # 5. Perturbed Halton
    for seed, scale in [(5, 0.03), (77, 0.08)]:
        rng = np.random.default_rng(seed)
        pts_hal = _halton_sphere(N)
        pts_pert = pts_hal + rng.standard_normal((N, 3)) * scale
        pts_pert /= np.linalg.norm(pts_pert, axis=1, keepdims=True)
        inits.append(pts_pert)

    best_pts = None
    best_energy = float('inf')

    # Phase 1: Run dual optimization on all initializations
    for idx, init_pts in enumerate(inits):
        if time.time() > deadline - 40:
            break

        remaining = deadline - time.time()
        time_per_init = min(remaining / max(1, len(inits) - idx), 15.0)

        pts, energy = _dual_optimize(
            init_pts,
            max_iter=800,
            time_limit=time_per_init,
            start_time=time.time()
        )

        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()

    # Phase 2: Refine the best result with our L-BFGS
    if best_pts is not None and time.time() < deadline - 20:
        remaining = deadline - time.time()
        best_pts, best_energy = _lbgfs_sphere(
            best_pts,
            max_iter=500, m=20,
            time_limit=remaining - 10,
            start_time=time.time()
        )

    # Phase 3: Cross-check with scipy L-BFGS-B
    if best_pts is not None and time.time() < deadline - 10:
        remaining = deadline - time.time()
        sc_pts, sc_energy = _scipy_lbgfs(
            best_pts,
            time_limit=remaining - 5,
            start_time=time.time()
        )
        if sc_energy < best_energy:
            best_pts = sc_pts
            best_energy = sc_energy

    # Phase 4: Final CG polish
    if best_pts is not None and time.time() < deadline - 5:
        remaining = deadline - time.time()
        best_pts, best_energy = _cg_sphere(
            best_pts,
            max_iter=300,
            time_limit=remaining - 2,
            start_time=time.time()
        )

    # Ensure valid output
    best_pts = best_pts.copy()
    best_pts /= np.linalg.norm(best_pts, axis=1, keepdims=True)

    valid = True
    for i in range(N):
        if not np.all(np.isfinite(best_pts[i])):
            valid = False
            break
        if np.linalg.norm(best_pts[i]) < 1e-10:
            valid = False
            break

    if valid:
        min_dist = float('inf')
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(best_pts[i] - best_pts[j])
                if d < min_dist:
                    min_dist = d
        if min_dist < 1e-9:
            valid = False

    if not valid:
        best_pts = _fibonacci_sphere(N)

    return best_pts
