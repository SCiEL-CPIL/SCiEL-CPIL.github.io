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

def clamp_bary(bary, eps=1e-6):
    """
    Clamp barycentric coordinates to the strictly valid triangle region.
    u >= eps, v >= eps, u+v <= 1 - eps
    """
    u = bary[:, 0].copy()
    v = bary[:, 1].copy()

    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)

    sums = u + v
    invalid = sums > (1 - eps)
    if np.any(invalid):
        scale = (1 - eps) / np.where(sums > 0, sums, 1.0)
        u[invalid] *= scale[invalid]
        v[invalid] *= scale[invalid]

    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)

    return np.column_stack([u, v])

def soft_min_obj(bary_flat, beta=100.0):
    """
    Objective function for optimization.
    Minimizes the negative soft-min of the areas.
    """
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
    """
    Objective function using hard min.
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return -np.min(areas)

def epigraph_obj(params, beta=200.0):
    """
    Epigraph formulation: minimize -t + penalty for areas < t.
    params: [t, u1, v1, ..., u11, v11]
    """
    t = params[0]
    bary = params[1:].reshape(N, 2)
    bary = clamp_bary(bary)

    areas = calc_areas_bary(bary)

    # Penalty for areas below t: sum of (max(0, t - area))^2
    deficit = np.maximum(0, t - areas)
    penalty = np.sum(deficit ** 2)

    return -t + 10.0 * penalty

def bary_to_cart(bary):
    """
    Convert barycentric coords (u, v) to Cartesian (x, y).
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = w * H
    return np.column_stack([x, y])

def generate_hexagonal_init():
    """
    Generate a hexagonal lattice-like initial configuration.
    """
    best_bary = None
    best_val = -np.inf

    for s in np.linspace(0.1, 0.5, 20):
        points = []
        for i in range(5):
            for j in range(5):
                u = i * s
                v = j * s * np.sqrt(3) / 2
                if u + v <= 1 and len(points) < N:
                    points.append([u, v])

        if len(points) == N:
            bary = np.array(points)
            val = -hard_min_obj(bary.flatten())
            if val > best_val:
                best_val = val
                best_bary = bary.copy()

    if best_bary is None:
        rng = np.random.default_rng(42)
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        best_bary = np.column_stack([u, v])

    return best_bary

def generate_random_init(seed=None):
    """Generate random valid barycentric coordinates."""
    rng = np.random.default_rng(seed)
    u = rng.random(N)
    v = rng.random(N) * (1 - u)
    return np.column_stack([u, v])

def generate_perturbed_init(base, noise_scale=0.05, seed=None):
    """Generate perturbed version of a base configuration."""
    rng = np.random.default_rng(seed)
    bary = base.copy()
    bary += rng.normal(0, noise_scale, bary.shape)
    return clamp_bary(bary)

def generate_structured_init(seed=None):
    """
    Generate structured initial configurations based on known patterns.
    """
    rng = np.random.default_rng(seed)
    inits = []

    # 1. Hexagonal with jitter
    hex_init = generate_hexagonal_init()
    inits.append(hex_init)
    inits.append(generate_perturbed_init(hex_init, 0.03, seed=seed))
    inits.append(generate_perturbed_init(hex_init, 0.08, seed=seed+1))

    # 2. Grid-like
    for s in [0.2, 0.25, 0.3]:
        pts = []
        for i in range(4):
            for j in range(4):
                u = i * s
                v = j * s
                if u + v <= 1.0 - 0.01 and len(pts) < N:
                    pts.append([u, v])
        if len(pts) == N:
            inits.append(np.array(pts))

    # 3. Concentric
    for r_scale in [0.3, 0.5]:
        pts = [[0.5, H/3]]  # center-ish
        for angle in np.linspace(0, 2*np.pi, 5, endpoint=False):
            u = 0.5 + r_scale * 0.4 * np.cos(angle)
            v = H/3 + r_scale * 0.4 * np.sin(angle) * H
            if u >= 0.01 and v >= 0.01 and u + v <= 0.99:
                pts.append([u, v])
        if len(pts) == N:
            inits.append(np.array(pts))

    return inits

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    best_bary = None
    best_val = -np.inf

    def update_best(bary):
        nonlocal best_bary, best_val
        bary = clamp_bary(bary)
        val = -hard_min_obj(bary.flatten())
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False

    def try_optimize_lbfgs(bary_init, beta1=50.0, beta2=200.0, maxiter1=200, maxiter2=300):
        """
        Two-stage L-BFGS-B optimization with soft-min.
        """
        if time.time() > deadline - 10:
            return

        try:
            def obj1(x):
                return soft_min_obj(x, beta=beta1)

            bounds = [(0, 1)] * (N * 2)

            result1 = minimize(
                obj1,
                bary_init.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': maxiter1, 'ftol': 1e-12, 'gtol': 1e-8}
            )

            if time.time() > deadline - 5:
                update_best(result1.x.reshape(N, 2))
                return

            def obj2(x):
                return soft_min_obj(x, beta=beta2)

            result2 = minimize(
                obj2,
                result1.x,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': maxiter2, 'ftol': 1e-14, 'gtol': 1e-10}
            )

            update_best(result2.x.reshape(N, 2))

        except Exception:
            pass

    def try_optimize_epigraph(bary_init, beta=200.0, maxiter=500):
        """
        Epigraph formulation optimization.
        """
        if time.time() > deadline - 10:
            return

        try:
            t0 = -hard_min_obj(bary_init.flatten())
            params0 = np.concatenate([[t0], bary_init.flatten()])

            bounds = [(0, 0.1)] + [(0, 1)] * (N * 2)

            result = minimize(
                epigraph_obj,
                params0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10}
            )

            if result.success or result.fun < epigraph_obj(params0, beta):
                bary_result = result.x[1:].reshape(N, 2)
                update_best(bary_result)

        except Exception:
            pass

    def try_slsqp_polish(bary_init, maxiter=500):
        """
        SLSQP polish on hard min.
        """
        if time.time() > deadline - 10:
            return

        try:
            def obj(x):
                bary = x.reshape(N, 2)
                bary = clamp_bary(bary)
                areas = calc_areas_bary(bary)
                return -np.min(areas)

            # Bounds for SLSQP
            bounds = [(1e-6, 1 - 1e-6)] * (N * 2)

            result = minimize(
                obj,
                bary_init.flatten(),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': maxiter, 'ftol': 1e-14}
            )

            update_best(result.x.reshape(N, 2))

        except Exception:
            pass

    def try_nelder_mead_polish(bary_init, maxiter=2000):
        """
        Nelder-Mead polish.
        """
        if time.time() > deadline - 10:
            return

        try:
            result = minimize(
                hard_min_obj,
                bary_init.flatten(),
                method='Nelder-Mead',
                options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            update_best(result.x.reshape(N, 2))

        except Exception:
            pass

    # Phase 1: Multiple diverse initializations
    inits = generate_structured_init(seed=42)

    # Add random inits
    for i in range(5):
        inits.append(generate_random_init(seed=100 + i))

    for init in inits:
        if time.time() > deadline - 30:
            break
        try_optimize_lbfgs(init, beta1=50.0, beta2=200.0, maxiter1=200, maxiter2=300)

    # Phase 2: Epigraph refinement on best
    if best_bary is not None:
        if time.time() < deadline - 25:
            try_optimize_epigraph(best_bary, beta=200.0, maxiter=500)

        # Perturb and re-optimize
        for i in range(4):
            if time.time() > deadline - 20:
                break
            perturbed = generate_perturbed_init(best_bary, noise_scale=0.02, seed=200 + i)
            try_optimize_lbfgs(perturbed, beta1=100.0, beta2=300.0, maxiter1=150, maxiter2=200)

    # Phase 3: Polish stages
    if best_bary is not None:
        if time.time() < deadline - 15:
            try_slsqp_polish(best_bary, maxiter=300)

        if time.time() < deadline - 10:
            try_nelder_mead_polish(best_bary, maxiter=2000)

        # Final SLSQP after Nelder-Mead
        if time.time() < deadline - 5:
            try_slsqp_polish(best_bary, maxiter=200)

    # Return best
    if best_bary is not None:
        return bary_to_cart(best_bary)
    else:
        rng = np.random.default_rng(0)
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        bary = np.column_stack([u, v])
        return bary_to_cart(bary)
