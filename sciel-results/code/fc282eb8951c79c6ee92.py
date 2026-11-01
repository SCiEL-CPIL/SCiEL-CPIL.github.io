"""Dual-representation optimizer: Halton init + alternating projected gradient / Riemannian CG."""
import time
import math
import numpy as np

N = 150


def _energy_and_grad(pts):
    """Vectorized energy and tangent-projected gradient."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dists_sq, 1.0)
    dists = np.sqrt(dists_sq)

    iu = np.triu_indices(n, k=1)
    d_upper = dists[iu]
    energy = np.sum(np.log(1.0 / d_upper))

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


def _halton_sphere(n, base_seed=0):
    """Generate n points on sphere using 3D Halton sequence, then map to sphere."""
    def _halton_1d(i, base):
        f = 1.0
        r = 0.0
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r

    # Use first 3 primes as bases
    primes = [2, 3, 5]
    # Offset by seed to get different sequences
    offset = base_seed * 1000 + 1

    pts = np.zeros((n, 3))
    for d in range(3):
        for i in range(n):
            idx = i + offset
            pts[i, d] = _halton_1d(idx, primes[d])

    # Map [0,1]^3 to sphere: use spherical coordinates with uniform distribution
    # u in [0,1] -> theta = 2*pi*u
    # v in [0,1] -> phi = arccos(2*v - 1) for uniform on sphere
    # But we have 3 coords; use first two for angles, ignore third or use it for jitter
    theta = 2.0 * np.pi * pts[:, 0]
    phi = np.arccos(1.0 - 2.0 * pts[:, 1])
    r = np.ones(n)

    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)

    pts_sphere = np.column_stack([x, y, z])
    norms = np.linalg.norm(pts_sphere, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return pts_sphere / norms


def _exp_map(pts, v):
    """Exponential map on S^2: move point x along tangent vector v."""
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    v_hat = v / norms
    theta = norms  # step size
    # x_new = cos(theta)*x + sin(theta)*v_hat
    c = np.cos(theta)
    s = np.sin(theta)
    new_pts = c * pts + s * v_hat
    # Re-normalize for numerical stability
    norms_new = np.linalg.norm(new_pts, axis=1, keepdims=True)
    norms_new = np.where(norms_new < 1e-15, 1.0, norms_new)
    return new_pts / norms_new


def _retract_to_sphere(pts):
    """Simple retraction: normalize to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return pts / norms


def _project_to_tangent(pts, vec):
    """Project vector onto tangent space at each point."""
    dot_prod = np.sum(vec * pts, axis=1)
    return vec - dot_prod[:, np.newaxis] * pts


def _dual_optimize(pts0, max_iter=500, time_limit=None, start_time=None, tol=1e-12):
    """
    Dual-representation optimization:
    - Alternate between projected gradient steps (Cartesian retraction) 
    - and Riemannian CG steps (exponential map)
    Halton initialization provides global dispersion; CG escapes shallow minima.
    """
    if start_time is None:
        start_time = time.time()

    pts = pts0.copy()
    energy, grad = _energy_and_grad(pts)

    # CG state
    d = -grad.copy()  # search direction
    d = _project_to_tangent(pts, d)
    old_grad_norm_sq = np.sum(grad * grad)

    step_size = 0.3
    min_step = 1e-14
    use_cg = True  # alternate flag

    for it in range(max_iter):
        if time_limit is not None and (time.time() - start_time) > time_limit:
            break

        if np.linalg.norm(grad) < tol:
            break

        if use_cg:
            # Riemannian Conjugate Gradient step
            # Fletcher-Reeves beta
            new_grad_norm_sq = np.sum(grad * grad)
            if old_grad_norm_sq > 1e-30:
                beta = new_grad_norm_sq / old_grad_norm_sq
            else:
                beta = 0.0

            d_new = -grad + beta * d
            d_new = _project_to_tangent(pts, d_new)

            # Ensure descent direction
            if np.sum(d_new * grad) > 0:
                d_new = -grad
                d_new = _project_to_tangent(pts, d_new)

            d = d_new
            old_grad_norm_sq = new_grad_norm_sq
            use_cg = False
        else:
            # Projected gradient step (steepest descent in Cartesian, retract)
            d = -grad
            d = _project_to_tangent(pts, d)
            use_cg = True

        # Line search
        alpha = step_size
        accepted = False
        for _ in range(30):
            if use_cg:
                # Exponential map for Riemannian CG
                new_pts = _exp_map(pts, alpha * d)
            else:
                # Cartesian step + retraction
                new_pts = pts + alpha * d
                new_pts = _retract_to_sphere(new_pts)

            new_energy, new_grad = _energy_and_grad(new_pts)

            # Armijo condition
            if new_energy < energy + 1e-4 * alpha * np.sum(grad * d):
                accepted = True
                break
            elif new_energy < energy - 1e-12:
                accepted = True
                break
            alpha *= 0.5

        if not accepted:
            alpha = max(step_size * 0.01, min_step)
            if use_cg:
                new_pts = _exp_map(pts, alpha * d)
            else:
                new_pts = pts + alpha * d
                new_pts = _retract_to_sphere(new_pts)
            new_energy, new_grad = _energy_and_grad(new_pts)

            if new_energy < energy:
                accepted = True
            else:
                step_size *= 0.5
                if step_size < min_step:
                    break

        if not accepted:
            break

        pts = new_pts
        grad = new_grad
        energy = new_energy
        step_size = min(alpha * 1.5, 2.0)

    return pts, energy


def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    # Multiple diverse initializations: Halton + Fibonacci + random
    inits = []

    # 1. Halton sequences with different seeds (low-discrepancy, global dispersion)
    for seed in [0, 1, 2, 3, 4]:
        inits.append(_halton_sphere(N, base_seed=seed))

    # 2. Fibonacci spiral
    inits.append(_fibonacci_sphere(N))

    # 3. Random initializations
    for seed in [42, 123, 456]:
        rng = np.random.default_rng(seed)
        pts_rand = rng.standard_normal((N, 3))
        pts_rand /= np.linalg.norm(pts_rand, axis=1, keepdims=True)
        inits.append(pts_rand)

    # 4. Perturbed Halton
    for seed, strength in [(7, 0.05), (99, 0.1)]:
        rng = np.random.default_rng(seed)
        pts_h = _halton_sphere(N, base_seed=seed)
        pts_pert = pts_h + rng.standard_normal((N, 3)) * strength
        pts_pert /= np.linalg.norm(pts_pert, axis=1, keepdims=True)
        inits.append(pts_pert)

    best_pts = None
    best_energy = float('inf')

    # Phase 1: Coarse optimization with multiple restarts using dual optimizer
    for idx, init_pts in enumerate(inits):
        if time.time() > deadline - 40:
            break

        remaining = deadline - time.time()
        time_alloc = min(remaining * 0.12, 12.0)
        time_alloc = max(time_alloc, 2.0)

        pts, energy = _dual_optimize(
            init_pts,
            max_iter=400,
            time_limit=time_alloc,
            start_time=time.time(),
            tol=1e-10
        )

        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()

    # Phase 2: Fine-tune with tighter tolerance
    if best_pts is not None and time.time() < deadline - 15:
        remaining = deadline - time.time()
        best_pts, best_energy = _dual_optimize(
            best_pts,
            max_iter=800,
            time_limit=remaining - 5,
            start_time=time.time(),
            tol=1e-13
        )

    # Phase 3: Final high-precision polish
    if best_pts is not None and time.time() < deadline - 5:
        remaining = deadline - time.time()
        if remaining > 8:
            best_pts, best_energy = _dual_optimize(
                best_pts,
                max_iter=500,
                time_limit=remaining - 2,
                start_time=time.time(),
                tol=1e-15
            )

    # Ensure valid output
    if best_pts is None:
        best_pts = _fibonacci_sphere(N)

    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    best_pts = best_pts / norms

    # Verify validity
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
