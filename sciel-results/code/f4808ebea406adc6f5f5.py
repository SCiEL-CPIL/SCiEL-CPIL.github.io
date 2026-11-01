import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150
DEADLINE_S = 142.0  # safety margin before 150s limit


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
    energy = np.sum(np.log(1.0 / d))

    return energy, grad


def _log_energy_fast(pts):
    """Vectorized log energy computation for speed."""
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
    """Apply exponential map on sphere: move points along tangent vectors."""
    norms = np.linalg.norm(tangent_vecs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    cos_norms = np.cos(norms)
    sin_norms_over_norms = np.sin(norms) / norms
    return cos_norms * pts + sin_norms_over_norms * tangent_vecs


def _riemannian_cg(pts, max_iter=300, tol=1e-12, time_limit=None, start_time=None):
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

        # Adaptive line search
        best_alpha = 0.0
        best_e = energy
        alphas = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]
        for alpha in alphas:
            new_pts = _exponential_map(pts, alpha * p)
            new_pts = _normalize(new_pts)
            new_e, _ = _energy_and_grad(new_pts)
            if new_e < best_e:
                best_e = new_e
                best_alpha = alpha
            if _time_exceeded():
                break

        if best_alpha == 0.0:
            break

        new_pts = _exponential_map(pts, best_alpha * p)
        new_pts = _normalize(new_pts)
        new_energy, new_grad = _energy_and_grad(new_pts)

        if new_energy >= energy:
            # Try smaller steps
            found = False
            for alpha in [best_alpha * 0.5, best_alpha * 0.25, best_alpha * 0.1, best_alpha * 0.05]:
                trial_pts = _exponential_map(pts, alpha * p)
                trial_pts = _normalize(trial_pts)
                trial_e, _ = _energy_and_grad(trial_pts)
                if trial_e < energy:
                    new_pts = trial_pts
                    new_energy, new_grad = _energy_and_grad(trial_pts)
                    found = True
                    break
            if not found:
                break

        # Conjugate gradient update (Polak-Ribiere)
        grad_diff = new_grad - grad
        beta = np.sum(new_grad * grad_diff) / (np.sum(grad * grad) + 1e-15)
        beta = max(0.0, min(beta, 1.0))

        # Check convergence
        grad_norm = np.linalg.norm(new_grad)
        if grad_norm < tol:
            pts = new_pts
            break

        p = -new_grad + beta * p
        # Project p to tangent space
        dot_p = np.sum(p * new_pts, axis=1, keepdims=True)
        p -= dot_p * new_pts

        pts = new_pts
        energy = new_energy
        grad = new_grad

    return pts, _log_energy_fast(pts)


def _lbfgs_sphere(pts, time_limit=15.0):
    """L-BFGS-B optimization on the sphere using ambient coordinates with
    normalization after each step."""
    start = time.time()
    x0 = pts.flatten().copy()

    def obj_and_grad(x):
        if time.time() - start > time_limit:
            return float('inf'), np.zeros_like(x)
        pts_local = x.reshape(N, 3)
        pts_local = _normalize(pts_local)
        energy, grad = _energy_and_grad(pts_local)
        return energy, grad.flatten()

    result = minimize(
        obj_and_grad,
        x0,
        method='L-BFGS-B',
        jac=True,
        options={
            'maxiter': 1500,
            'ftol': 1e-15,
            'gtol': 1e-12,
            'maxfun': 15000,
        }
    )

    if result.success or result.fun < float('inf'):
        best_pts = _normalize(result.x.reshape(N, 3))
        best_energy = _log_energy_fast(best_pts)
    else:
        best_pts = pts.copy()
        best_energy = _log_energy_fast(pts)

    return best_pts, best_energy


def _fps_fast(n, seed=42, batch_size=2000):
    """Fast farthest point sampling using vectorized operations."""
    rng = np.random.default_rng(seed)
    pts = [rng.standard_normal(3)]
    pts[0] = pts[0] / np.linalg.norm(pts[0])
    pts_arr = np.array(pts)

    for _ in range(n - 1):
        candidates = rng.standard_normal((batch_size, 3))
        candidates = _normalize(candidates)

        # Compute min distance from each candidate to existing points
        min_dists = np.full(batch_size, np.inf)
        chunk = 50
        for i in range(0, len(pts_arr), chunk):
            chunk_pts = pts_arr[i:i+chunk]
            d = np.linalg.norm(candidates[:, np.newaxis, :] - chunk_pts[np.newaxis, :, :], axis=2)
            min_dists = np.minimum(min_dists, np.min(d, axis=1))

        best_idx = np.argmax(min_dists)
        pts_arr = np.vstack([pts_arr, candidates[best_idx:best_idx+1]])

    return _normalize(pts_arr)


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


def _basin_hopping(pts, time_limit=20.0, max_restarts=10):
    """Basin hopping with Riemannian CG local optimization and adaptive perturbation."""
    start = time.time()
    current = pts.copy()
    energy = _log_energy_fast(current)

    # Initial optimization with Riemannian CG
    time_left_init = time_limit - (time.time() - start) - 2.0
    if time_left_init > 5.0:
        current, energy = _riemannian_cg(current, time_limit=min(10.0, time_left_init))

    best_pts = current.copy()
    best_energy = energy

    # Basin Hopping Loop with reheat
    temp = 0.4
    decay = 0.8
    reheat_every = 4
    reheat_temp = 0.25

    for rh in range(max_restarts):
        if time.time() - start > time_limit - 2.0:
            break

        # Reheat
        if rh > 0 and rh % reheat_every == 0:
            temp = reheat_temp

        # Adaptive perturbation
        perturb_scale = max(0.02, min(0.2, 0.06 * temp + 0.02))

        # Generate random tangent vectors
        rng = np.random.default_rng(rh * 7919 + 1)
        perturbations = rng.standard_normal((N, 3))
        dot_prods = np.sum(perturbations * current, axis=1, keepdims=True)
        perturbations -= dot_prods * current
        pert_norms = np.linalg.norm(perturbations, axis=1, keepdims=True)
        pert_norms = np.maximum(pert_norms, 1e-15)
        perturbations = perturbations / pert_norms * perturb_scale

        new_pts = _exponential_map(current, perturbations)
        new_pts = _normalize(new_pts)

        # Optimize the perturbed point with Riemannian CG
        time_left = time_limit - (time.time() - start) - 1.0
        if time_left < 3.0:
            break

        new_opt_pts, new_energy = _riemannian_cg(new_pts, time_limit=min(6.0, time_left))
        new_energy = _log_energy_fast(new_opt_pts)

        # Metropolis Criterion
        delta_e = new_energy - energy
        if delta_e < 0 or np.random.rand() < math.exp(-delta_e / max(temp, 1e-10)):
            current = new_opt_pts
            energy = new_energy
            if energy < best_energy:
                best_energy = energy
                best_pts = current.copy()

        temp *= decay

    return best_pts, best_energy


def _final_polish(pts, time_limit=10.0):
    """Final high-precision polish: alternate Riemannian CG and L-BFGS-B with tight tolerances."""
    start = time.time()
    current = pts.copy()
    energy = _log_energy_fast(current)

    # First pass: Riemannian CG with very tight tolerance
    time_left = time_limit - (time.time() - start)
    if time_left > 3.0:
        current, energy = _riemannian_cg(current, max_iter=200, tol=1e-14, time_limit=min(5.0, time_left))
        energy = _log_energy_fast(current)

    # Second pass: L-BFGS-B with very tight tolerance
    time_left = time_limit - (time.time() - start)
    if time_left > 3.0:
        current, energy = _lbfgs_sphere(current, time_limit=min(5.0, time_left))
        energy = _log_energy_fast(current)

    # Third pass: Riemannian CG again for fine-tuning
    time_left = time_limit - (time.time() - start)
    if time_left > 2.0:
        current, energy = _riemannian_cg(current, max_iter=100, tol=1e-15, time_limit=min(3.0, time_left))
        energy = _log_energy_fast(current)

    return current, energy


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_S

    # Initializations - diverse set
    inits = []
    inits.append(_fibonacci_sphere(N))
    inits.append(_halton_sphere(N))
    inits.append(_spherical_design_init(N, seed=42))
    inits.append(_spherical_design_init(N, seed=123))

    # FPS-based inits
    inits.append(_fps_fast(N, seed=7))
    inits.append(_fps_fast(N, seed=13))

    # Random inits
    for s in range(3):
        rng = np.random.default_rng(s)
        pts = rng.standard_normal((N, 3))
        pts = _normalize(pts)
        inits.append(pts)

    # Perturbed structured inits
    for s in range(3):
        rng = np.random.default_rng(s + 100)
        base = _fibonacci_sphere(N) if s % 2 == 0 else _halton_sphere(N)
        pts = base + rng.standard_normal((N, 3)) * 0.12
        pts = _normalize(pts)
        inits.append(pts)

    best_pts = None
    best_energy = float('inf')

    # Process each initialization with adaptive time budget
    total_inits = len(inits)
    for idx, init_pts in enumerate(inits):
        remaining = deadline - time.time() - 15.0  # reserve 15s for final polish
        if remaining < 8.0:
            break

        # Adaptive time per restart: more time for early inits, less for later
        if idx < 5:
            time_per_restart = min(16.0, remaining / (total_inits - idx) * 1.3)
        else:
            time_per_restart = min(10.0, remaining / (total_inits - idx) * 1.0)

        time_per_restart = max(5.0, time_per_restart)

        try:
            opt_pts, opt_energy = _basin_hopping(init_pts, time_limit=time_per_restart)
            opt_energy = _log_energy_fast(opt_pts)

            if opt_energy < best_energy:
                best_energy = opt_energy
                best_pts = opt_pts.copy()

        except Exception:
            pass

    # If no valid points found, fallback to Fibonacci
    if best_pts is None:
        best_pts = _fibonacci_sphere(N)
        best_energy = _log_energy_fast(best_pts)

    # Final high-precision polish
    remaining = deadline - time.time() - 2.0
    if remaining > 5.0:
        try:
            polish_time = min(12.0, remaining)
            final_pts, final_energy = _final_polish(best_pts, time_limit=polish_time)
            final_energy = _log_energy_fast(final_pts)
            if final_energy < best_energy:
                best_energy = final_energy
                best_pts = final_pts
        except Exception:
            pass

    return _normalize(best_pts)
