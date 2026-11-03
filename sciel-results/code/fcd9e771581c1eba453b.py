"""Improved logarithmic energy minimizer for 100 points on the sphere."""
import numpy as np
import math
import time

N = 100


def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return pts / norms


def _energy_and_grad(pts):
    """Vectorized energy and tangent gradient."""
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)

    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu]))

    W = 1.0 / dist_sq
    np.fill_diagonal(W, 0.0)

    # dE/dx_i = sum_j W_ij * (x_i - x_j) = W @ pts - (W.sum(axis=1, keepdims=True)) * pts
    grad_ambient = W @ pts - (W.sum(axis=1, keepdims=True)) * pts
    dot = np.sum(grad_ambient * pts, axis=1, keepdims=True)
    grad_tangent = grad_ambient - dot * pts

    return energy, grad_tangent


def _fast_energy(pts):
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    iu = np.triu_indices(n, k=1)
    return -np.sum(np.log(dist[iu]))


def _precise_energy(pts):
    n = pts.shape[0]
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-9:
                return float('inf')
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)


def _generate_fibonacci(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0, 1.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _generate_random(n, rng):
    pts = rng.standard_normal((n, 3))
    return _normalize(pts)


def _generate_sobol(n, seed=123):
    from scipy.stats import qmc
    sampler = qmc.Sobol(d=3, scramble=True, seed=seed)
    raw = sampler.random(n)
    z = 2.0 * raw[:, 0] - 1.0
    theta = 2.0 * np.pi * raw[:, 1]
    r = np.sqrt(np.clip(1.0 - z * z, 0, 1.0))
    pts = np.column_stack([r * np.cos(theta), r * np.sin(theta), z])
    return _normalize(pts)


def _coordinate_descent(pts, deadline, max_sweeps=2000, tol=1e-14):
    """
    Fast coordinate-descent polish. For each point i, given all other points fixed,
    the energy contribution is sum_{j != i} log(1/||x_i - x_j||), which is minimized
    on the sphere at x_i = normalize(sum_j x_j / ||x_i - x_j||).
    This is a 1D subproblem solved in O(N) per point, O(N^2) per sweep.
    """
    n = pts.shape[0]
    current = pts.copy()
    current_e = _fast_energy(current)
    best = current.copy()
    best_e = current_e

    for sweep in range(max_sweeps):
        if time.time() > deadline:
            break

        improved = 0.0
        for i in range(n):
            # Compute optimal position for point i given others
            others = np.delete(current, i, axis=0)  # (n-1, 3)
            diff = others - current[i]  # (n-1, 3)
            dists = np.linalg.norm(diff, axis=1)
            dists = np.clip(dists, 1e-12, None)
            weights = 1.0 / dists
            # Optimal x_i on sphere: normalize(sum_j w_j * x_j)
            new_pos = np.sum(others * weights[:, np.newaxis], axis=0)
            new_norm = np.linalg.norm(new_pos)
            if new_norm < 1e-12:
                continue
            new_pos = new_pos / new_norm

            # Check if this improves energy
            old_dist_sq = np.sum((others - current[i]) ** 2, axis=1)
            new_dist_sq = np.sum((others - new_pos) ** 2, axis=1)
            old_contrib = -np.sum(np.log(np.sqrt(old_dist_sq)))
            new_contrib = -np.sum(np.log(np.sqrt(new_dist_sq)))

            if new_contrib < old_contrib - 1e-16:
                current[i] = new_pos
                improved += (old_contrib - new_contrib)

        # Check convergence
        new_e = _fast_energy(current)
        if new_e < best_e:
            best_e = new_e
            best = current.copy()

        if improved < tol:
            break

    return best, best_e


def _gradient_flow(pts, rng, n_steps, lr_start, lr_end, deadline):
    n = pts.shape[0]
    current = pts.copy()
    best = pts.copy()
    best_energy = _fast_energy(pts)

    for step in range(n_steps):
        if time.time() > deadline:
            break

        progress = step / max(n_steps - 1, 1)
        lr = lr_start * (lr_end / lr_start) ** progress

        e, g = _energy_and_grad(current)

        noise_scale = 0.005 * (1.0 - progress)
        noise = rng.standard_normal((n, 3)) * noise_scale

        current = current - lr * g + noise
        current = _normalize(current)

        e_new = _fast_energy(current)
        if e_new < best_energy:
            best_energy = e_new
            best = current.copy()

    return best, best_energy


def _simulated_annealing(pts, rng, deadline, max_iter=3000):
    n = pts.shape[0]
    current = pts.copy()
    current_e = _fast_energy(current)
    best = current.copy()
    best_e = current_e

    T_start = 0.3
    T_end = 0.001
    T = T_start

    for it in range(max_iter):
        if time.time() > deadline:
            break

        progress = it / max(max_iter - 1, 1)
        T = T_start * (T_end / T_start) ** progress

        scale = 0.04 * (1.0 - 0.5 * progress)
        perturbed = current + rng.standard_normal((n, 3)) * scale
        perturbed = _normalize(perturbed)

        new_e = _fast_energy(perturbed)

        if new_e < current_e:
            current = perturbed
            current_e = new_e
            if current_e < best_e:
                best_e = current_e
                best = current.copy()
        else:
            delta = new_e - current_e
            if np.random.random() < math.exp(-delta / max(T, 1e-10)):
                current = perturbed
                current_e = new_e

    return best, best_e


def _optimize_single(pts, deadline):
    """Coordinate descent + gradient flow hybrid for a single start."""
    n = pts.shape[0]
    best_pts = pts.copy()
    best_energy = _fast_energy(pts)

    # Phase 1: Coordinate descent (fast, exact 1D solves)
    cd_deadline = min(deadline, time.time() + 20.0)
    cd_pts, cd_e = _coordinate_descent(pts, cd_deadline, max_sweeps=500)
    if cd_e < best_energy:
        best_energy = cd_e
        best_pts = cd_pts

    if time.time() > deadline:
        return best_pts, best_energy

    # Phase 2: Gradient flow to escape any remaining local structure
    gf_deadline = min(deadline, time.time() + 15.0)
    rng = np.random.default_rng(hash(tuple(map(float, best_pts[0]))) % (2**32))
    gf_pts, gf_e = _gradient_flow(best_pts, rng, 300, 0.02, 0.0005, gf_deadline)
    if gf_e < best_energy:
        best_energy = gf_e
        best_pts = gf_pts

    if time.time() > deadline:
        return best_pts, best_energy

    # Phase 3: Final coordinate descent polish
    cd2_pts, cd2_e = _coordinate_descent(best_pts, deadline, max_sweeps=1000, tol=1e-15)
    if cd2_e < best_energy:
        best_energy = cd2_e
        best_pts = cd2_pts

    return best_pts, best_energy


def _final_polish(pts, deadline):
    """
    Ultra-precise final polish: many coordinate descent sweeps with tiny tolerance,
    interleaved with tiny random perturbations to escape micro-local-minima.
    """
    n = pts.shape[0]
    best_pts = pts.copy()
    best_energy = _fast_energy(pts)
    rng = np.random.default_rng(777)

    # First: long CD run
    cd_pts, cd_e = _coordinate_descent(pts, deadline, max_sweeps=5000, tol=1e-16)
    if cd_e < best_energy:
        best_energy = cd_e
        best_pts = cd_pts

    # Then: try tiny perturbations + CD
    for scale in [1e-5, 5e-6, 1e-6, 5e-7, 1e-7, 5e-8]:
        if time.time() > deadline:
            break
        for _ in range(2):
            if time.time() > deadline:
                break
            perturbed = _normalize(best_pts + rng.standard_normal((n, 3)) * scale)
            cd_pts, cd_e = _coordinate_descent(perturbed, deadline, max_sweeps=3000, tol=1e-16)
            if cd_e < best_energy:
                best_energy = cd_e
                best_pts = cd_pts

    # One more long CD from the best
    if time.time() < deadline - 2.0:
        cd_pts, cd_e = _coordinate_descent(best_pts, deadline, max_sweeps=3000, tol=1e-16)
        if cd_e < best_energy:
            best_energy = cd_e
            best_pts = cd_pts

    return best_pts, best_energy


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    rng = np.random.default_rng(42)

    # Generate diverse initializations
    initializations = []

    # Fibonacci spiral
    initializations.append(_generate_fibonacci(N))

    # Sobol sequences
    for seed in [123, 456, 789, 321, 654]:
        initializations.append(_generate_sobol(N, seed=seed))

    # Random points
    for i in range(8):
        initializations.append(_generate_random(N, rng))

    # Perturbed Fibonacci
    base = _generate_fibonacci(N)
    for scale in [0.01, 0.05, 0.1, 0.2, 0.3]:
        perturbed = base + rng.standard_normal((N, 3)) * scale
        initializations.append(_normalize(perturbed))

    # Golden ratio variations
    for offset in [0.0, 0.1, 0.2, 0.3, 0.4]:
        i = np.arange(N) + 0.5 + offset
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.clip(1.0 - z * z, 0, 1.0))
        initializations.append(np.column_stack([r * np.cos(phi), r * np.sin(phi), z]))

    # Sort by quick energy
    scored_inits = []
    for init in initializations:
        e_quick = _fast_energy(init)
        scored_inits.append((e_quick, init))
    scored_inits.sort(key=lambda x: x[0])

    best_pts = None
    best_energy = float('inf')

    # Phase A: Quick gradient flow on top initializations
    flow_deadline = start_time + 25.0
    for e_quick, init_pts in scored_inits[:8]:
        if time.time() > flow_deadline:
            break
        flowed_pts, e_precise = _gradient_flow(init_pts, rng, 300, 0.04, 0.001, flow_deadline)
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = flowed_pts

    # Phase B: Simulated annealing
    sa_deadline = start_time + 45.0
    if best_pts is not None:
        sa_pts, sa_e = _simulated_annealing(best_pts, rng, sa_deadline)
        if sa_e < best_energy:
            best_energy = sa_e
            best_pts = sa_pts
    else:
        best_pts = scored_inits[0][1]
        best_energy = scored_inits[0][0]

    # Phase C: Coordinate-descent + gradient hybrid on top starts
    if best_pts is not None:
        scored_inits.insert(0, (best_energy, best_pts))

    opt_deadline = deadline - 35.0  # Reserve 35s for final polish
    for e_quick, init_pts in scored_inits[:10]:
        if time.time() > opt_deadline:
            break
        optimized_pts, e_precise = _optimize_single(init_pts, opt_deadline)
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = optimized_pts

    if best_pts is None:
        best_pts = _generate_fibonacci(N)

    # Phase D: Final ultra-precise polish
    polish_deadline = deadline - 2.0
    best_pts, best_energy = _final_polish(best_pts, polish_deadline)

    # Final normalization and validation
    best_pts = _normalize(best_pts)
    final_energy = _precise_energy(best_pts)
    if not np.isfinite(final_energy):
        best_pts = _normalize(_generate_fibonacci(N))

    return best_pts
