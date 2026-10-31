import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations

N = 11
H = np.sqrt(3) / 2

# Triangle vertices in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, H])

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)


def calc_areas_bary(bary):
    """
    Calculate areas of all triangles formed by N points given in barycentric coords (u, v).
    Returns array of shape (165,) containing the area ratios.
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v

    ui = u[tri_idx[:, 0]]
    vi = v[tri_idx[:, 0]]
    wi = w[tri_idx[:, 0]]

    uj = u[tri_idx[:, 1]]
    vj = v[tri_idx[:, 1]]
    wj = w[tri_idx[:, 1]]

    uk = u[tri_idx[:, 2]]
    vk = v[tri_idx[:, 2]]
    wk = w[tri_idx[:, 2]]

    det = ui * (vj * wk - vk * wj) - vi * (uj * wk - uk * wj) + wi * (uj * vk - uk * vj)

    return np.abs(det)


def clamp_bary(bary, eps=1e-7):
    """Clamp barycentric coordinates to the valid triangle region."""
    u = bary[:, 0].copy()
    v = bary[:, 1].copy()

    u = np.clip(u, eps, 1.0 - eps)
    v = np.clip(v, eps, 1.0 - eps)

    sums = u + v
    invalid = sums > (1.0 - 2 * eps)
    if np.any(invalid):
        scale = (1.0 - 2 * eps) / np.where(sums > 1e-12, sums, 1.0)
        u[invalid] *= scale[invalid]
        v[invalid] *= scale[invalid]

    u = np.clip(u, eps, 1.0 - eps)
    v = np.clip(v, eps, 1.0 - eps)

    return np.column_stack([u, v])


def bary_to_cart(bary):
    """Convert barycentric coords (u, v) to Cartesian (x, y)."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = w * H
    return np.column_stack([x, y])


def soft_min_obj(bary_flat, beta=100.0):
    """Objective: negative soft-min of areas."""
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)

    scaled = -beta * areas
    max_scaled = np.max(scaled)
    if max_scaled > 700:
        return -np.min(areas)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min = -log_sum / beta
    return -soft_min


def hard_min_obj(bary_flat):
    """Objective: negative hard min of areas."""
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return -np.min(areas)


def min_area_bary(bary):
    """Minimum area among all triangles."""
    bary = clamp_bary(bary)
    return float(np.min(calc_areas_bary(bary)))


def generate_hexagonal_init():
    """Generate hexagonal lattice-like initial configurations in barycentric coords."""
    best_bary = None
    best_val = -np.inf

    for s in np.linspace(0.08, 0.45, 25):
        points = []
        for i in range(8):
            for j in range(8):
                u = i * s
                v = j * s
                if u >= 0 and v >= 0 and u + v <= 1.0 and len(points) < N:
                    points.append([u, v])
        if len(points) == N:
            bary = np.array(points)
            bary = clamp_bary(bary)
            val = min_area_bary(bary)
            if val > best_val:
                best_val = val
                best_bary = bary.copy()

    if best_bary is None:
        rng = np.random.default_rng(42)
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        best_bary = np.column_stack([u, v])
    return best_bary


def generate_symmetric_init():
    """3-fold symmetric structured initialization in barycentric coords."""
    # Center
    center = np.array([1.0 / 3, 1.0 / 3])
    pts = [center]

    # Inner ring (3 points, 120° apart)
    r1 = 0.18
    for angle in [0, 2 * np.pi / 3, 4 * np.pi / 3]:
        # In barycentric, rotate around center
        u = center[0] + r1 * np.cos(angle)
        v = center[1] + r1 * np.sin(angle)
        pts.append([u, v])

    # Outer ring (6 points)
    r2 = 0.30
    for angle in [np.pi / 6, np.pi / 6 + 2 * np.pi / 6, np.pi / 6 + 4 * np.pi / 6,
                  np.pi / 6 + 6 * np.pi / 6, np.pi / 6 + 8 * np.pi / 6, np.pi / 6 + 10 * np.pi / 6]:
        u = center[0] + r2 * np.cos(angle)
        v = center[1] + r2 * np.sin(angle)
        pts.append([u, v])

    bary = np.array(pts[:N])
    return clamp_bary(bary)


def generate_random_init(seed):
    """Generate random valid barycentric coordinates."""
    rng = np.random.default_rng(seed)
    u = rng.random(N)
    v = rng.random(N) * (1 - u)
    return np.column_stack([u, v])


def try_two_stage_optimize(bary_init, deadline, maxiter1=300, maxiter2=400, beta1=50.0, beta2=250.0):
    """Two-stage soft-min optimization: low beta then high beta."""
    if time.time() > deadline - 8:
        return bary_init, min_area_bary(bary_init)

    try:
        bounds = [(0, 1)] * (N * 2)

        # Stage 1: Exploration with smooth landscape
        res1 = minimize(
            soft_min_obj, bary_init.flatten(), args=(beta1,),
            method='L-BFGS-B', bounds=bounds,
            options={'maxiter': maxiter1, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        bary1 = clamp_bary(res1.x.reshape(N, 2))

        if time.time() > deadline - 5:
            return bary1, min_area_bary(bary1)

        # Stage 2: Exploitation with sharper landscape
        res2 = minimize(
            soft_min_obj, bary1.flatten(), args=(beta2,),
            method='L-BFGS-B', bounds=bounds,
            options={'maxiter': maxiter2, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        bary2 = clamp_bary(res2.x.reshape(N, 2))
        return bary2, min_area_bary(bary2)
    except Exception:
        return bary_init, min_area_bary(bary_init)


def nm_polish_bary(bary, deadline, maxiter=3000):
    """Nelder-Mead polish in barycentric space."""
    if time.time() > deadline - 5:
        return bary, min_area_bary(bary)
    try:
        res = minimize(
            hard_min_obj, bary.flatten(),
            method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
        )
        opt = clamp_bary(res.x.reshape(N, 2))
        return opt, min_area_bary(opt)
    except Exception:
        return bary, min_area_bary(bary)


def coord_descent_bary(bary, deadline, max_rounds=150):
    """Coordinate descent in barycentric space with adaptive step refinement."""
    current = bary.copy()
    score = min_area_bary(current)
    steps = [0.03, 0.015, 0.008, 0.004, 0.002, 0.001, 0.0005, 0.0002, 0.0001]

    for step in steps:
        rounds_per_step = max(1, max_rounds // len(steps))
        for _ in range(rounds_per_step):
            if time.time() > deadline:
                return current, score
            improved = False
            for i in range(N):
                base_u, base_v = current[i]
                for du in [-step, 0, step]:
                    for dv in [-step, 0, step]:
                        if du == 0 and dv == 0:
                            continue
                        trial = current.copy()
                        trial[i] = [base_u + du, base_v + dv]
                        trial = clamp_bary(trial)
                        s = min_area_bary(trial)
                        if s > score + 1e-15:
                            current = trial
                            score = s
                            improved = True
            if not improved:
                break
    return current, score


def construct():
    t0 = time.time()
    deadline = t0 + 145.0

    def tleft():
        return deadline - time.time()

    best_bary = None
    best_score = -1.0

    def update_best(bary):
        nonlocal best_bary, best_score
        bary = clamp_bary(bary)
        sc = min_area_bary(bary)
        if sc > best_score:
            best_score = sc
            best_bary = bary.copy()
            return True
        return False

    # Generate diverse initializations
    inits = [generate_hexagonal_init(), generate_symmetric_init()]
    for s in [1, 7, 13, 29, 42, 100, 200, 300, 500, 700, 900, 1200, 1500, 2000, 3000, 4000]:
        inits.append(generate_random_init(s))

    # Phase 1: Multi-start two-stage L-BFGS
    for init in inits:
        if tleft() < 12:
            break
        bary, sc = try_two_stage_optimize(init, deadline, maxiter1=250, maxiter2=350, beta1=50.0, beta2=200.0)
        update_best(bary)

    if best_bary is None:
        best_bary = generate_hexagonal_init()
        best_score = min_area_bary(best_bary)

    # Phase 2: Nelder-Mead polish on best
    if tleft() > 15:
        bary, sc = nm_polish_bary(best_bary, deadline, maxiter=2500)
        update_best(bary)

    # Phase 3: Perturbation multi-start around best
    rng = np.random.default_rng(99999)
    for trial in range(25):
        if tleft() < 10:
            break
        scale = 0.004 + 0.002 * trial
        pert = best_bary + rng.normal(0, scale, size=(N, 2))
        pert = clamp_bary(pert)

        bary, sc = try_two_stage_optimize(pert, deadline, maxiter1=200, maxiter2=300, beta1=80.0, beta2=300.0)
        update_best(bary)

        if tleft() > 8:
            bary, sc = nm_polish_bary(best_bary, deadline, maxiter=1500)
            update_best(bary)

    # Phase 4: Coordinate descent refinement
    if tleft() > 10:
        bary, sc = coord_descent_bary(best_bary, deadline, max_rounds=120)
        update_best(bary)

    # Phase 5: High-precision final polish
    if tleft() > 15:
        bary, sc = try_two_stage_optimize(best_bary, deadline, maxiter1=400, maxiter2=500, beta1=100.0, beta2=500.0)
        update_best(bary)

        if tleft() > 10:
            try:
                res = minimize(
                    hard_min_obj, best_bary.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
                )
                opt = clamp_bary(res.x.reshape(N, 2))
                sc = min_area_bary(opt)
                if sc > best_score:
                    best_score = sc
                    best_bary = opt.copy()
            except Exception:
                pass

        if tleft() > 8:
            bary, sc = coord_descent_bary(best_bary, deadline, max_rounds=80)
            update_best(bary)

    # Convert to Cartesian and return
    cart = bary_to_cart(best_bary)
    return cart
