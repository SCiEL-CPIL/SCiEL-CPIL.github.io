"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import scipy.optimize as optimize
from scipy.spatial import ConvexHull
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

N = 13

# EVOLVE-BLOCK-START

# Precompute triple indices globally
_TRIPLE_IDX = np.array([
    (i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)
], dtype=np.int32)

def _triangle_areas(pts):
    pi = pts[_TRIPLE_IDX[:, 0]]
    pj = pts[_TRIPLE_IDX[:, 1]]
    pk = pts[_TRIPLE_IDX[:, 2]]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)

def _hull_area(pts):
    try:
        if len(pts) < 3:
            return 0.0
        hull = ConvexHull(pts)
        return float(hull.volume)
    except Exception:
        return 0.0

def _score(pts):
    if len(pts) < 3:
        return 0.0
    hull_a = _hull_area(pts)
    if hull_a < 1e-12:
        return 0.0
    areas = _triangle_areas(pts)
    min_a = float(np.min(areas))
    if min_a < 1e-15:
        return 0.0
    return min_a / hull_a

def _soft_min(values, temp):
    if temp <= 1e-12:
        return float(np.min(values))
    mv = np.min(values)
    shifted = values - mv
    log_sum = np.log(np.sum(np.exp(-shifted / temp)))
    return mv - temp * log_sum

def _soft_score(pts, temp):
    if len(pts) < 3:
        return 0.0
    hull_a = _hull_area(pts)
    if hull_a < 1e-12:
        return 0.0
    areas = _triangle_areas(pts)
    if np.any(areas < 1e-15):
        return 0.0
    sm = _soft_min(areas, temp)
    if sm < 1e-15:
        return 0.0
    return sm / hull_a

def _neg_soft_score(pts, temp):
    s = _soft_score(pts, temp)
    if s <= 0:
        return 1e10
    return -s

# Fourier parameterization: r(theta) = 1 + sum_{k=1}^{K} (a_k cos(k*theta) + b_k sin(k*theta))
# Parameters: [a1, b1, a2, b2, ..., aK, bK] -> 2K dimensions
# We use K=5 -> 10 params, plus optionally a global scale -> 11 params

def _fourier_to_points(params, K=5):
    """Convert Fourier coefficients to N points on a closed curve."""
    # params: array of length 2*K
    # Generate N angles
    thetas = np.linspace(0, 2*np.pi, N, endpoint=False)
    r = np.ones(N)
    for k in range(1, K+1):
        a_k = params[2*(k-1)]
        b_k = params[2*(k-1)+1]
        r += a_k * np.cos(k * thetas) + b_k * np.sin(k * thetas)
    # Ensure r > 0 (clamp to avoid negative radii)
    r = np.maximum(r, 0.1)
    pts = np.column_stack([r * np.cos(thetas), r * np.sin(thetas)])
    return pts

def _neg_soft_score_fourier(params, temp, K=5):
    pts = _fourier_to_points(params, K)
    return _neg_soft_score(pts, temp)

def _neg_score_fourier(params, K=5):
    pts = _fourier_to_points(params, K)
    s = _score(pts)
    if s <= 0:
        return 1e10
    return -s

def _penalty_objective_fourier(params, mu, K=5):
    """Penalty objective in Fourier space: minimize -t/hull + mu * sum(max(0, t-area_i)^2)
    We fix t implicitly as the minimum area, so we maximize soft_min(areas)/hull_area."""
    pts = _fourier_to_points(params, K)
    hull_a = _hull_area(pts)
    if hull_a < 1e-12:
        return 1e10
    areas = _triangle_areas(pts)
    min_a = float(np.min(areas))
    if min_a < 1e-15:
        return 1e10
    # Penalty: encourage all areas to be close to min_a (balanced)
    violations = np.maximum(0, min_a - areas)
    penalty = np.sum(violations ** 2)
    return -min_a / hull_a + mu * penalty

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    best_x = None
    best_score = -np.inf

    def try_update(x_candidate):
        nonlocal best_x, best_score
        s = _score(x_candidate)
        if s > best_score:
            best_score = s
            best_x = x_candidate.copy()
        return s

    # Phase 0: Regular polygon baseline
    angles_reg = np.linspace(0, 2*np.pi, N, endpoint=False)
    x_reg = np.column_stack([np.cos(angles_reg), np.sin(angles_reg)])
    try_update(x_reg)

    # Phase 1: Fourier-space multi-start optimization
    K = 5  # number of harmonics -> 10 parameters
    n_params = 2 * K

    # Generate diverse Fourier starting points
    fourier_starts = []

    # All zeros (regular polygon)
    fourier_starts.append(np.zeros(n_params))

    # Small random perturbations
    rng = np.random.RandomState(42)
    for scale in [0.01, 0.02, 0.03, 0.05, 0.08, 0.12]:
        fourier_starts.append(rng.randn(n_params) * scale)

    # Structured perturbations: emphasize certain harmonics
    for k_emph in range(1, K+1):
        for amp in [0.03, 0.05, 0.08]:
            p = np.zeros(n_params)
            p[2*(k_emph-1)] = amp
            p[2*(k_emph-1)+1] = amp * 0.5
            fourier_starts.append(p + rng.randn(n_params) * 0.01)

    # Two-harmonic combinations
    for k1 in range(1, K+1):
        for k2 in range(k1+1, K+1):
            for amp in [0.03, 0.05]:
                p = np.zeros(n_params)
                p[2*(k1-1)] = amp
                p[2*(k2-1)+1] = -amp * 0.7
                fourier_starts.append(p + rng.randn(n_params) * 0.005)

    # Optimize each Fourier start with annealed soft-min
    for i, f_start in enumerate(fourier_starts):
        if time.time() > deadline - 30:
            break

        # Anneal temperature in Fourier space
        temps = np.exp(np.linspace(np.log(0.05), np.log(1e-4), 8))
        f_curr = f_start.copy()
        for temp in temps:
            if time.time() > deadline - 25:
                break
            try:
                res = optimize.minimize(
                    lambda p: _neg_soft_score_fourier(p, temp, K),
                    f_curr,
                    method='Nelder-Mead',
                    options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
                )
                f_curr = res.x
            except Exception:
                pass

        # Convert to Cartesian and polish
        x_cand = _fourier_to_points(f_curr, K)
        s = try_update(x_cand)

        # Quick L-BFGS-B polish in Cartesian
        if time.time() < deadline - 15:
            try:
                res = optimize.minimize(
                    lambda x: _neg_soft_score(x.reshape(N, 2), 1e-4),
                    x_cand.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                try_update(res.x.reshape(N, 2))
            except Exception:
                pass

    # Phase 2: Penalty-based refinement in Cartesian space
    if best_x is not None and time.time() < deadline - 25:
        x_curr = best_x.copy()
        mus = np.geomspace(1.0, 1e4, 6)
        for mu in mus:
            if time.time() > deadline - 20:
                break
            try:
                def penalty_obj(x_flat):
                    pts = x_flat.reshape(N, 2)
                    hull_a = _hull_area(pts)
                    if hull_a < 1e-12:
                        return 1e10
                    areas = _triangle_areas(pts)
                    min_a = float(np.min(areas))
                    if min_a < 1e-15:
                        return 1e10
                    violations = np.maximum(0, min_a - areas)
                    penalty = np.sum(violations ** 2)
                    return -min_a / hull_a + mu * penalty

                res = optimize.minimize(
                    penalty_obj, x_curr.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                cand = res.x.reshape(N, 2)
                s = try_update(cand)
                if s > best_score:
                    x_curr = cand
            except Exception:
                pass

    # Phase 3: Hard-objective Nelder-Mead polish
    if best_x is not None and time.time() < deadline - 15:
        try:
            res = optimize.minimize(
                lambda x: -_score(x.reshape(N, 2)),
                best_x.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            try_update(res.x.reshape(N, 2))
        except Exception:
            pass

    # Phase 4: L-BFGS-B on soft-min with very low temperature
    if best_x is not None and time.time() < deadline - 10:
        try:
            res = optimize.minimize(
                lambda x: _neg_soft_score(x.reshape(N, 2), 1e-5),
                best_x.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            try_update(res.x.reshape(N, 2))
        except Exception:
            pass

    # Phase 5: More Fourier restarts from best solution (project back to Fourier)
    if best_x is not None and time.time() < deadline - 10:
        # Project best Cartesian to Fourier coefficients
        thetas = np.linspace(0, 2*np.pi, N, endpoint=False)
        r_best = np.linalg.norm(best_x, axis=1)
        # Fit Fourier coefficients via least squares
        A = np.zeros((N, 2*K))
        for k in range(1, K+1):
            A[:, 2*(k-1)] = np.cos(k * thetas)
            A[:, 2*(k-1)+1] = np.sin(k * thetas)
        # r = 1 + A @ coeffs
        coeffs, _, _, _ = np.linalg.lstsq(A, r_best - 1.0, rcond=None)

        # Perturb and re-optimize
        for perturb_scale in [0.01, 0.02, 0.03]:
            if time.time() > deadline - 8:
                break
            f_start = coeffs + rng.randn(n_params) * perturb_scale
            temps = np.exp(np.linspace(np.log(0.03), np.log(1e-4), 6))
            f_curr = f_start.copy()
            for temp in temps:
                if time.time() > deadline - 6:
                    break
                try:
                    res = optimize.minimize(
                        lambda p: _neg_soft_score_fourier(p, temp, K),
                        f_curr,
                        method='Nelder-Mead',
                        options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
                    )
                    f_curr = res.x
                except Exception:
                    pass
            x_cand = _fourier_to_points(f_curr, K)
            try_update(x_cand)
            if time.time() < deadline - 5:
                try:
                    res = optimize.minimize(
                        lambda x: -_score(x.reshape(N, 2)),
                        x_cand.flatten(),
                        method='Nelder-Mead',
                        options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
                    )
                    try_update(res.x.reshape(N, 2))
                except Exception:
                    pass

    # Phase 6: Final polish
    if best_x is not None and time.time() < deadline - 5:
        try:
            res = optimize.minimize(
                lambda x: -_score(x.reshape(N, 2)),
                best_x.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-11, 'fatol': 1e-11, 'adaptive': True}
            )
            try_update(res.x.reshape(N, 2))
        except Exception:
            pass

    if best_x is None:
        best_x = x_reg

    return best_x.reshape(-1, 2)

# EVOLVE-BLOCK-END
