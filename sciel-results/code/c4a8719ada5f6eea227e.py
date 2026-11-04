"""Rewritten program for logarithmic energy minimization of 150 points on the sphere."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 150


def _log_energy(points):
    """Compute log(1/d) energy for points on unit sphere using math.fsum."""
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dists_sq, 1.0)
    iu = np.triu_indices(n, k=1)
    log_dists = 0.5 * np.log(dists_sq[iu])
    return -math.fsum(log_dists.tolist())


def _fibonacci_sphere(n):
    """Generate Fibonacci sphere points."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_uniform_sphere(n, seed=42):
    """Generate random uniform points on sphere."""
    rng = np.random.RandomState(seed)
    points = rng.randn(n, 3)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    return points / norms


def _project_to_sphere(points):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms


def _energy_and_grad(x):
    """Energy and gradient for L-BFGS-B. x is flat array of shape (N*3,)."""
    points = x.reshape(N, 3)
    points = _project_to_sphere(points)

    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dists_sq, 1.0)

    # Fast energy for optimizer (numpy sum, not math.fsum)
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(0.5 * np.log(dists_sq[iu]))

    # Compute gradient: dE/dx_i = sum_{j!=i} (x_j - x_i) / d_ij^2
    weights = np.zeros_like(dists_sq)
    mask = ~np.eye(n, dtype=bool)
    weights[mask] = 1.0 / dists_sq[mask]

    sum_weights = np.sum(weights, axis=1)
    weighted_sum = weights @ points

    grads = weighted_sum - points * sum_weights[:, np.newaxis]

    # Project gradient to tangent space
    dot_products = np.sum(grads * points, axis=1, keepdims=True)
    grads_tangent = grads - dot_products * points

    return energy, grads_tangent.flatten()


def _geodesic_dome_init(n):
    """Generate initial points using a geodesic dome subdivision."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)

    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ]

    def subdivide_face(v0, v1, v2, subdiv=2):
        if subdiv == 1:
            return [(v0, v1, v2)]

        m01 = (v0 + v1) / 2.0
        m12 = (v1 + v2) / 2.0
        m20 = (v2 + v0) / 2.0
        m01 = m01 / np.linalg.norm(m01)
        m12 = m12 / np.linalg.norm(m12)
        m20 = m20 / np.linalg.norm(m20)

        return (subdivide_face(v0, m01, m20, subdiv - 1) +
                subdivide_face(m01, v1, m12, subdiv - 1) +
                subdivide_face(m20, m12, v2, subdiv - 1) +
                subdivide_face(m01, m12, m20, subdiv - 1))

    all_verts = set()
    for f in faces:
        v0, v1, v2 = verts[f[0]], verts[f[1]], verts[f[2]]
        sub_faces = subdivide_face(v0, v1, v2, subdiv=2)
        for sf in sub_faces:
            for v in sf:
                all_verts.add(tuple(np.round(v, 12)))

    result = np.array(list(all_verts))
    result = result / np.linalg.norm(result, axis=1, keepdims=True)

    if len(result) > n:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(result), size=n, replace=False)
        result = result[idx]
    elif len(result) < n:
        extra = _random_uniform_sphere(n - len(result), seed=99)
        result = np.vstack([result, extra])

    return result


def _basin_hopping_step(points, rng, step_size=0.1):
    """Perturb points on sphere for basin hopping."""
    n = len(points)
    perturbed = points.copy()
    for i in range(n):
        rand_dir = rng.randn(3)
        tangent = rand_dir - np.dot(rand_dir, points[i]) * points[i]
        norm = np.linalg.norm(tangent)
        if norm > 1e-15:
            tangent /= norm
            perturbed[i] = points[i] + step_size * tangent
    return _project_to_sphere(perturbed)


def _lbfgs_polish(points, maxiter=5000, ftol=1e-15, gtol=1e-12):
    """Local refinement using L-BFGS-B with tight tolerances."""
    x0 = points.flatten()
    try:
        res = minimize(
            _energy_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': maxiter, 'maxfun': maxiter * 10,
                     'ftol': ftol, 'gtol': gtol, 'maxls': 100}
        )
        p_res = _project_to_sphere(res.x.reshape(N, 3))
        return p_res
    except Exception:
        return points


def _bb_step_size(prev_grad, curr_grad, prev_step, prev_x, curr_x):
    """Barzilai-Borwein step size selection."""
    s = curr_x - prev_x
    y = curr_grad - prev_grad
    s_norm_sq = np.dot(s, s)
    y_norm_sq = np.dot(y, y)
    if s_norm_sq < 1e-30 or y_norm_sq < 1e-30:
        return prev_step
    alpha_bb = s_norm_sq / max(np.dot(s, y), 1e-30)
    beta_bb = np.dot(s, y) / max(y_norm_sq, 1e-30)
    # Use a blend for stability
    step = 0.5 * (alpha_bb + beta_bb)
    step = np.clip(step, 1e-10, 10.0)
    return step


def _gradient_descent_sphere(points, max_iter=2000, deadline=None):
    """Gradient descent on the sphere with Barzilai-Borwein step sizes."""
    current = points.copy()
    prev_grad = None
    prev_x = None
    step_size = 0.01
    best_energy = _log_energy(current)
    best_points = current.copy()

    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break

        # Compute gradient
        x_flat = current.flatten()
        _, grad_flat = _energy_and_grad(x_flat)
        grad = grad_flat.reshape(N, 3)

        # Barzilai-Borwein step size
        if prev_grad is not None and prev_x is not None:
            step_size = _bb_step_size(prev_grad.flatten(), grad.flatten(),
                                      step_size, prev_x, current)

        # Update
        new_points = current - step_size * grad
        new_points = _project_to_sphere(new_points)

        # Check energy
        new_energy = _log_energy(new_points)
        if new_energy < best_energy:
            best_energy = new_energy
            best_points = new_points.copy()

        prev_grad = grad
        prev_x = current
        current = new_points

        # Adaptive step decay if not improving
        if new_energy > best_energy + 1e-10:
            step_size *= 0.9

    return best_points, best_energy


def _simulated_annealing(points, max_iter=5000, deadline=None):
    """Simulated annealing on the sphere."""
    current = points.copy()
    current_energy = _log_energy(current)
    best_points = current.copy()
    best_energy = current_energy

    T = 0.1
    T_min = 1e-6
    cooling = 0.999

    rng = np.random.RandomState(123)

    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break

        # Propose perturbation
        perturbed = _basin_hopping_step(current, rng, step_size=0.05)
        perturbed_energy = _log_energy(perturbed)

        # Accept or reject
        delta = perturbed_energy - current_energy
        if delta < 0 or rng.rand() < math.exp(-delta / max(T, 1e-10)):
            current = perturbed
            current_energy = perturbed_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_points = current.copy()

        T = max(T * cooling, T_min)

    return best_points, best_energy


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin

    rng = np.random.RandomState(2024)

    # Multiple initializations
    inits = [
        _geodesic_dome_init(N),
        _fibonacci_sphere(N),
        _random_uniform_sphere(N, seed=1),
        _random_uniform_sphere(N, seed=2),
        _random_uniform_sphere(N, seed=3),
        _random_uniform_sphere(N, seed=4),
        _random_uniform_sphere(N, seed=5),
    ]

    best_points = inits[0].copy()
    best_energy = _log_energy(best_points)

    # Phase 1: Local optimization from multiple starts
    for idx, init in enumerate(inits):
        if time.time() > deadline - 30.0:
            break

        x0 = init.flatten()
        try:
            res = minimize(
                _energy_and_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 2000, 'maxfun': 20000,
                         'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 100}
            )
            p_res = _project_to_sphere(res.x.reshape(N, 3))
            e_check = _log_energy(p_res)
            if e_check < best_energy:
                best_energy = e_check
                best_points = p_res.copy()
        except Exception:
            pass

    # Phase 2: Basin hopping with local refinement
    bh_iterations = 0
    while time.time() < deadline - 15.0 and bh_iterations < 15:
        bh_iterations += 1
        step = 0.10 * (1.0 - bh_iterations / 20.0)
        if step < 0.03:
            step = 0.03
        perturbed = _basin_hopping_step(best_points, rng, step_size=step)

        # L-BFGS polish
        refined = _lbfgs_polish(perturbed, maxiter=1500, ftol=1e-13, gtol=1e-11)
        e_check = _log_energy(refined)
        if e_check < best_energy:
            best_energy = e_check
            best_points = refined.copy()

    # Phase 3: Gradient descent with BB step sizes for fine-tuning
    if time.time() < deadline - 10.0:
        gd_result, gd_energy = _gradient_descent_sphere(
            best_points, max_iter=1000, deadline=deadline - 8.0
        )
        if gd_energy < best_energy:
            best_energy = gd_energy
            best_points = gd_result.copy()

    # Phase 4: Simulated annealing for additional exploration
    if time.time() < deadline - 8.0:
        sa_result, sa_energy = _simulated_annealing(
            best_points, max_iter=2000, deadline=deadline - 5.0
        )
        if sa_energy < best_energy:
            best_energy = sa_energy
            best_points = sa_result.copy()

    # Phase 5: Final high-precision polish
    if time.time() < deadline - 3.0:
        polished = _lbfgs_polish(best_points, maxiter=8000, ftol=1e-16, gtol=1e-13)
        e_final = _log_energy(polished)
        if e_final < best_energy:
            best_energy = e_final
            best_points = polished.copy()

    # Final ultra-tight polish
    if time.time() < deadline - 1.0:
        polished = _lbfgs_polish(best_points, maxiter=5000, ftol=1e-16, gtol=1e-14)
        e_final = _log_energy(polished)
        if e_final < best_energy:
            best_energy = e_final
            best_points = polished.copy()

    best_points = _project_to_sphere(best_points)

    if not np.all(np.isfinite(best_points)):
        return _fibonacci_sphere(N)

    # Check for points too close
    diff = best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    iu = np.triu_indices(N, k=1)
    if np.min(dists_sq[iu]) < (1e-9) ** 2:
        return _fibonacci_sphere(N)

    return best_points
