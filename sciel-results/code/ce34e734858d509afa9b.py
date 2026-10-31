import numpy as np
from scipy.optimize import minimize
from itertools import combinations
import time

N = 11
TIME_LIMIT = 145.0  # Safety margin for 150s limit

tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)
N_TRI = len(tri_indices)

H = np.sqrt(3) / 2

def bary_to_cart(bary):
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = w * H
    return np.column_stack([x, y])

def calc_areas_bary(bary):
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    ui = u[tri_idx[:, 0]]; vi = v[tri_idx[:, 0]]; wi = w[tri_idx[:, 0]]
    uj = u[tri_idx[:, 1]]; vj = v[tri_idx[:, 1]]; wj = w[tri_idx[:, 1]]
    uk = u[tri_idx[:, 2]]; vk = v[tri_idx[:, 2]]; wk = w[tri_idx[:, 2]]
    det = ui * (vj * wk - vk * wj) - vi * (uj * wk - uk * wj) + wi * (uj * vk - uk * vj)
    return np.abs(det)

def clamp_bary(bary, eps=1e-10):
    u = bary[:, 0].copy()
    v = bary[:, 1].copy()
    u = np.clip(u, eps, 1.0 - eps)
    v = np.clip(v, eps, 1.0 - eps)
    s = u + v
    bad = s > 1.0 - eps
    if np.any(bad):
        sc = (1.0 - eps) / np.maximum(s, eps)
        u[bad] *= sc[bad]
        v[bad] *= sc[bad]
    u = np.clip(u, eps, 1.0 - eps)
    v = np.clip(v, eps, 1.0 - eps)
    return np.column_stack([u, v])

def hard_min_value(bary):
    return float(np.min(calc_areas_bary(bary)))

# --- Explicit gradient for epigraph objective E = -t + sum(max(0, t-area_k)^2) ---
def epigraph_value_and_grad(x):
    """
    x = [u0,v0,...,u10,v10, t]
    E = -t + sum_k max(0, t - area_k)^2
    Returns (E, grad) where grad has shape (2N+1,).
    """
    bary = x[:2 * N].reshape(N, 2)
    t = x[2 * N]
    bary = clamp_bary(bary)
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    areas = calc_areas_bary(bary)

    viol = np.maximum(0.0, t - areas)
    n_viol = np.sum(viol > 0)
    E = -t + np.sum(viol ** 2)

    grad_t = -1.0 + 2.0 * np.sum(viol)
    grad_bary = np.zeros((N, 2))

    if n_viol > 0:
        # For each violating triangle, d(area)/d(u_i) and d(area)/d(v_i)
        # area = u_i*(v_j*w_k - v_k*w_j) - v_i*(u_j*w_k - u_k*w_j) + w_i*(u_j*v_k - u_k*v_j)
        # with w_i = 1-u_i-v_i
        # d(area)/du_i = (v_j*w_k - v_k*w_j) - (u_j*v_k - u_k*v_j)
        # d(area)/dv_i = -(u_j*w_k - u_k*w_j) - (u_j*v_k - u_k*v_j)
        for k in range(N_TRI):
            if viol[k] <= 0:
                continue
            i, j, kk = tri_idx[k]
            coeff = 2.0 * viol[k]

            # For vertex i
            A = v[j] * w[kk] - v[kk] * w[j]
            B = u[j] * w[kk] - u[kk] * w[j]
            C = u[j] * v[kk] - u[kk] * v[j]
            grad_bary[i, 0] -= coeff * (A - C)
            grad_bary[i, 1] -= coeff * (B + C)

            # For vertex j
            A2 = v[i] * w[kk] - v[kk] * w[i]
            B2 = u[i] * w[kk] - u[kk] * w[i]
            C2 = u[i] * v[kk] - u[kk] * v[i]
            grad_bary[j, 0] -= coeff * (A2 - C2)
            grad_bary[j, 1] -= coeff * (B2 + C2)

            # For vertex kk
            A3 = v[i] * w[j] - v[j] * w[i]
            B3 = u[i] * w[j] - u[j] * w[i]
            C3 = u[i] * v[j] - u[j] * v[i]
            grad_bary[kk, 0] -= coeff * (A3 - C3)
            grad_bary[kk, 1] -= coeff * (B3 + C3)

    grad = np.zeros(2 * N + 1)
    grad[:2 * N] = grad_bary.flatten()
    grad[2 * N] = grad_t
    return E, grad

def epigraph_obj(x):
    return epigraph_value_and_grad(x)[0]

def epigraph_obj_grad(x):
    return epigraph_value_and_grad(x)[1]

# --- Energy field objective: E = sum(1/areas^2), gradient descent with momentum ---
def energy_value_and_grad(bary):
    """
    E = sum_k 1/area_k^2
    grad_bary[i,0] = -2 * sum_{k: i in tri_k} 1/area_k^3 * d(area_k)/du_i
    """
    bary = clamp_bary(bary)
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    areas = calc_areas_bary(bary)
    # Guard against zero areas
    areas_safe = np.maximum(areas, 1e-12)
    inv3 = 1.0 / (areas_safe ** 3)

    E = np.sum(1.0 / (areas_safe ** 2))
    grad_bary = np.zeros((N, 2))

    for k in range(N_TRI):
        i, j, kk = tri_idx[k]
        coeff = -2.0 * inv3[k]

        A = v[j] * w[kk] - v[kk] * w[j]
        B = u[j] * w[kk] - u[kk] * w[j]
        C = u[j] * v[kk] - u[kk] * v[j]
        grad_bary[i, 0] += coeff * (A - C)
        grad_bary[i, 1] += coeff * (B + C)

        A2 = v[i] * w[kk] - v[kk] * w[i]
        B2 = u[i] * w[kk] - u[kk] * w[i]
        C2 = u[i] * v[kk] - u[kk] * v[i]
        grad_bary[j, 0] += coeff * (A2 - C2)
        grad_bary[j, 1] += coeff * (B2 + C2)

        A3 = v[i] * w[j] - v[j] * w[i]
        B3 = u[i] * w[j] - u[j] * w[i]
        C3 = u[i] * v[j] - u[j] * v[i]
        grad_bary[kk, 0] += coeff * (A3 - C3)
        grad_bary[kk, 1] += coeff * (B3 + C3)

    return E, grad_bary

def energy_gradient_descent(bary_init, time_limit=5.0):
    """
    Gradient descent with adaptive step and momentum on the energy field.
    Returns refined bary.
    """
    start = time.time()
    bary = bary_init.copy()
    momentum = np.zeros_like(bary)
    lr = 0.005
    beta = 0.9
    best_val = hard_min_value(bary)
    best_bary = bary.copy()
    stale = 0

    while time.time() - start < time_limit:
        E, grad = energy_value_and_grad(bary)
        # Adaptive step: scale by inverse norm
        gnorm = np.linalg.norm(grad)
        if gnorm < 1e-15:
            break
        step = lr / (gnorm + 1e-8)
        momentum = beta * momentum + (1 - beta) * grad
        bary = bary - step * momentum
        bary = clamp_bary(bary)

        val = hard_min_value(bary)
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            stale = 0
        else:
            stale += 1
            if stale > 30:
                lr *= 0.5
                if lr < 1e-8:
                    break
                stale = 0

    return best_bary

# --- Soft-min objective (from current program, kept for L-BFGS-B stages) ---
def soft_min_obj(bary_flat, beta=50.0):
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    if np.any(areas < 1e-15):
        return 1e10
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    if max_scaled > 700:
        return -np.min(areas)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min = -log_sum / beta
    return -soft_min

def hard_min_obj(bary_flat):
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return -np.min(areas)

# --- Initialization strategies ---
def generate_voronoi_init():
    """
    Structured init: approximate 3-fold symmetric placement.
    1 center + 3 inner + 3 mid + 3 outer + 1 extra = 11
    """
    center = np.array([1.0 / 3, 1.0 / 3])
    pts = [center]

    # Inner ring: 3 points at radius ~0.12 from center
    for k in range(3):
        ang = 2 * np.pi * k / 3 - np.pi / 2
        # In barycentric, moving from center in Cartesian direction
        # Approximate: offset in barycentric
        du = 0.12 * np.cos(ang) * 0.8
        dv = 0.12 * np.sin(ang) * 0.6
        pts.append([1/3 + du, 1/3 + dv])

    # Mid ring: 3 points at radius ~0.22
    for k in range(3):
        ang = 2 * np.pi * k / 3 - np.pi / 2 + np.pi / 3
        du = 0.22 * np.cos(ang) * 0.8
        dv = 0.22 * np.sin(ang) * 0.6
        pts.append([1/3 + du, 1/3 + dv])

    # Outer ring: 3 points near edges
    for k in range(3):
        ang = 2 * np.pi * k / 3 - np.pi / 2
        du = 0.32 * np.cos(ang) * 0.8
        dv = 0.32 * np.sin(ang) * 0.6
        pts.append([1/3 + du, 1/3 + dv])

    # 1 extra point
    pts.append([0.5, 0.15])

    bary = np.array(pts)
    bary = clamp_bary(bary)
    return bary

def generate_lattice_init():
    """K=3 triangular lattice (10 pts) + 1 perturbed center."""
    pts = []
    for i in range(4):
        for j in range(4 - i):
            pts.append([i / 3.0, j / 3.0])
    # 10 points; perturb the center (1/3, 1/3)
    # Find center index
    for idx, p in enumerate(pts):
        if abs(p[0] - 1/3) < 0.01 and abs(p[1] - 1/3) < 0.01:
            pts[idx] = [1/3.0 + 0.04, 1/3.0 + 0.02]
            break
    pts.append([0.5, 0.1])
    return clamp_bary(np.array(pts))

def generate_random_init(seed=None):
    rng = np.random.default_rng(seed)
    u = rng.random(N)
    v = rng.random(N) * (1 - u)
    return np.column_stack([u, v])

def generate_perturbed_init(base_bary, sigma=0.05, seed=None):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma, base_bary.shape)
    return clamp_bary(base_bary + noise)

# --- Refinement: explicit-gradient epigraph (structural change) ---
def refine_epigraph_grad(bary_init, time_limit=8.0):
    """
    Refine using explicit-gradient epigraph with L-BFGS-B.
    This is a different optimization family: penalty epigraph with analytic gradient,
    vs. the current program's log-sum-exp soft-min.
    """
    start = time.time()
    bary = clamp_bary(bary_init)
    t0 = hard_min_value(bary) * 0.95
    x0 = np.concatenate([bary.flatten(), [t0]])
    bounds = [(0, 1)] * (2 * N) + [(0, 0.05)]

    best_bary = bary.copy()
    best_val = hard_min_value(bary)

    # Stage 1: L-BFGS-B with explicit gradient
    try:
        res = minimize(
            epigraph_obj,
            x0,
            jac=epigraph_obj_grad,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 5000}
        )
        if res.x is not None:
            new_bary = clamp_bary(res.x[:2 * N].reshape(N, 2))
            new_val = hard_min_value(new_bary)
            if new_val > best_val:
                best_val = new_val
                best_bary = new_bary
    except Exception:
        pass

    if time.time() - start > time_limit * 0.5:
        return best_bary

    # Stage 2: SLSQP on epigraph (different method, same formulation)
    try:
        x0_2 = np.concatenate([best_bary.flatten(), [hard_min_value(best_bary) * 0.98]])
        res2 = minimize(
            epigraph_obj,
            x0_2,
            jac=epigraph_obj_grad,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-15, 'iprint': 0}
        )
        if res2.x is not None:
            new_bary = clamp_bary(res2.x[:2 * N].reshape(N, 2))
            new_val = hard_min_value(new_bary)
            if new_val > best_val:
                best_val = new_val
                best_bary = new_bary
    except Exception:
        pass

    return best_bary

# --- Refinement: multi-stage soft-min + NM (from current program) ---
def refine_local(bary_init, time_limit=5.0):
    start_time = time.time()
    bounds = [(0, 1)] * (N * 2)
    bary_current = bary_init.copy()

    # Stage 1: Coarse soft-min
    try:
        result1 = minimize(
            soft_min_obj, bary_current.flatten(), args=(20.0,),
            method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if result1.success or result1.fun < -hard_min_obj(bary_current.flatten()):
            bary_current = clamp_bary(result1.x.reshape(N, 2))
    except Exception:
        pass

    if time.time() - start_time > time_limit * 0.25:
        return bary_current

    # Stage 2: Medium soft-min
    try:
        result2 = minimize(
            soft_min_obj, bary_current.flatten(), args=(50.0,),
            method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-13, 'gtol': 1e-9}
        )
        if result2.success or result2.fun < -hard_min_obj(bary_current.flatten()):
            bary_current = clamp_bary(result2.x.reshape(N, 2))
    except Exception:
        pass

    if time.time() - start_time > time_limit * 0.45:
        return bary_current

    # Stage 3: Sharp soft-min
    try:
        result3 = minimize(
            soft_min_obj, bary_current.flatten(), args=(200.0,),
            method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        if result3.success or result3.fun < -hard_min_obj(bary_current.flatten()):
            bary_current = clamp_bary(result3.x.reshape(N, 2))
    except Exception:
        pass

    if time.time() - start_time > time_limit * 0.65:
        return bary_current

    # Stage 4: Nelder-Mead on hard min
    try:
        result4 = minimize(
            hard_min_obj, bary_current.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 1500, 'xatol': 1e-13, 'fatol': 1e-15, 'adaptive': True}
        )
        bary_current = clamp_bary(result4.x.reshape(N, 2))
    except Exception:
        pass

    if time.time() - start_time > time_limit * 0.85:
        return bary_current

    # Stage 5: Final NM polish
    try:
        result5 = minimize(
            hard_min_obj, bary_current.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-15, 'fatol': 1e-17, 'adaptive': True}
        )
        bary_current = clamp_bary(result5.x.reshape(N, 2))
    except Exception:
        pass

    return bary_current

# --- Main construct ---
def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT

    best_bary = None
    best_val = -np.inf

    def update_best(bary):
        nonlocal best_bary, best_val
        bary = clamp_bary(bary)
        val = hard_min_value(bary)
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False

    # 1. Voronoi init + explicit-gradient epigraph (structural change)
    vor_init = generate_voronoi_init()
    refined_vor_ep = refine_epigraph_grad(vor_init, time_limit=10.0)
    update_best(refined_vor_ep)

    # 2. Voronoi init + local multi-stage refinement
    if time.time() < deadline - 30:
        refined_vor_local = refine_local(vor_init, time_limit=8.0)
        update_best(refined_vor_local)

    # 3. Lattice init + explicit-gradient epigraph
    lat_init = generate_lattice_init()
    refined_lat_ep = refine_epigraph_grad(lat_init, time_limit=8.0)
    update_best(refined_lat_ep)

    # 4. Energy-field gradient descent on voronoi init (new global search mechanism)
    if time.time() < deadline - 25:
        refined_energy = energy_gradient_descent(vor_init, time_limit=6.0)
        update_best(refined_energy)
        # Then polish with local refinement
        if time.time() < deadline - 15:
            refined_energy_pol = refine_local(refined_energy, time_limit=5.0)
            update_best(refined_energy_pol)

    # 5. Multi-start: random + perturbed with local refinement
    rng = np.random.default_rng(123)
    restart_count = 0
    while time.time() < deadline - 15:
        restart_count += 1
        if best_bary is not None and restart_count % 2 == 0:
            init = generate_perturbed_init(best_bary, sigma=0.04, seed=2000 + restart_count)
        else:
            init = generate_random_init(seed=3000 + restart_count)

        # Alternate between epigraph-grad and local refinement
        if restart_count % 3 == 0:
            refined = refine_epigraph_grad(init, time_limit=4.0)
        else:
            refined = refine_local(init, time_limit=3.0)
        update_best(refined)

    # 6. Final high-precision polish on best
    if best_bary is not None and time.time() < deadline - 5:
        # Epigraph-grad polish
        refined_final_ep = refine_epigraph_grad(best_bary, time_limit=4.0)
        update_best(refined_final_ep)

        if time.time() < deadline - 3:
            try:
                bounds = [(0, 1)] * (N * 2)
                result_slsqp = minimize(
                    hard_min_obj, best_bary.flatten(),
                    method='SLSQP', bounds=bounds,
                    options={'maxiter': 1000, 'ftol': 1e-15}
                )
                update_best(clamp_bary(result_slsqp.x.reshape(N, 2)))
            except Exception:
                pass

        if time.time() < deadline - 2:
            try:
                result_nm = minimize(
                    hard_min_obj, best_bary.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-14, 'fatol': 1e-18, 'adaptive': True}
                )
                update_best(clamp_bary(result_nm.x.reshape(N, 2)))
            except Exception:
                pass

    # Convert to cartesian
    cart = bary_to_cart(best_bary)
    return cart
