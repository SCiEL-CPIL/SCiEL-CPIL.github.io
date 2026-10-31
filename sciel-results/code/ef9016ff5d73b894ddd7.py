import numpy as np
from scipy.optimize import minimize, differential_evolution
from itertools import combinations
import time

N = 11
TIME_LIMIT = 145.0  # Safety margin for 150s limit

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)

# Triangle vertices in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2

def bary_to_cart(bary):
    """Convert barycentric coords (u, v) to Cartesian (x, y)."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = w * H
    return np.column_stack([x, y])

def calc_areas_bary(bary):
    """Calculate areas of all triangles formed by N points in barycentric coords."""
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

def clamp_bary(bary, eps=1e-9):
    """Clamp barycentric coordinates to the valid triangle region."""
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

def hard_min_obj(bary_flat):
    """Objective using hard min."""
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return -np.min(areas)

def epigraph_obj(x, beta=100.0):
    """
    Epigraph formulation with smooth log-sum-exp surrogate.
    x = [bary (N*2), t]
    Minimize t subject to: logsumexp(-beta*areas) - beta*t <= 0
    We minimize t + penalty * max(0, logsumexp(-beta*areas) - beta*t)
    """
    bary = x[:N*2].reshape(N, 2)
    t = x[N*2]
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)

    if np.any(areas < 1e-15):
        return 1e10 + t

    scaled = -beta * areas
    max_scaled = np.max(scaled)
    if max_scaled > 700:
        # Fallback to hard min
        return -np.min(areas) + t

    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    constraint_violation = log_sum - beta * t

    if constraint_violation > 0:
        return t + 1000.0 * constraint_violation
    else:
        return t

def epigraph_grad(x, beta=100.0):
    """
    Gradient of the epigraph objective for L-BFGS-B.
    """
    bary = x[:N*2].reshape(N, 2)
    t = x[N*2]
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)

    if np.any(areas < 1e-15):
        return np.zeros(N*2 + 1)

    scaled = -beta * areas
    max_scaled = np.max(scaled)
    if max_scaled > 700:
        return np.zeros(N*2 + 1)

    exp_scaled = np.exp(scaled - max_scaled)
    sum_exp = np.sum(exp_scaled)
    log_sum = max_scaled + np.log(sum_exp)
    constraint_violation = log_sum - beta * t

    # Gradient w.r.t. t
    grad_t = 1.0
    if constraint_violation > 0:
        grad_t += 1000.0 * (-beta)

    # Gradient w.r.t. bary via chain rule
    # d(logsumexp)/d(area_i) = -beta * exp(scaled_i) / sum_exp
    d_logsum_d_areas = -beta * exp_scaled / sum_exp

    # d(area_i)/d(bary) - we need the Jacobian of areas w.r.t. bary
    # area_i = |det_i|, where det_i is the determinant for triangle i
    # We compute the gradient numerically for simplicity but analytically it's:
    # For triangle (i,j,k): area = 0.5 * |det|
    # d(area)/d(u_a) for point a in the triangle

    # Compute gradient of each area w.r.t. bary coordinates
    grad_areas = np.zeros((N*2, len(tri_idx)))

    for idx in range(len(tri_idx)):
        i, j, k = tri_idx[idx]
        # Barycentric coords
        ui, vi = bary[i]
        uj, vj = bary[j]
        uk, vk = bary[k]
        wi = 1 - ui - vi
        wj = 1 - uj - vj
        wk = 1 - uk - vk

        # det = ui*(vj*wk - vk*wj) - vi*(uj*wk - uk*wj) + wi*(uj*vk - uk*vj)
        # d(det)/dui = vj*wk - vk*wj
        # d(det)/dvi = -(uj*wk - uk*wj)
        # d(det)/duj = -vi*wk + wi*vk
        # d(det)/dvj = vi*wk - wi*uk
        # d(det)/duk = vi*wj - wi*vj
        # d(det)/dvk = -vi*wj + wi*vj

        d_det_d_ui = vj * wk - vk * wj
        d_det_d_vi = -(uj * wk - uk * wj)
        d_det_d_uj = -vi * wk + wi * vk
        d_det_d_vj = vi * wk - wi * uk
        d_det_d_uk = vi * wj - wi * vj
        d_det_d_vk = -vi * wj + wi * vj

        sign = np.sign(ui * (vj * wk - vk * wj) - vi * (uj * wk - uk * wj) + wi * (uj * vk - uk * vj))
        if sign == 0:
            sign = 1.0

        grad_areas[2*i, idx] = sign * d_det_d_ui
        grad_areas[2*i+1, idx] = sign * d_det_d_vi
        grad_areas[2*j, idx] = sign * d_det_d_uj
        grad_areas[2*j+1, idx] = sign * d_det_d_vj
        grad_areas[2*k, idx] = sign * d_det_d_uk
        grad_areas[2*k+1, idx] = sign * d_det_d_vk

    # d(obj)/d(bary) = d_logsum_d_areas @ grad_areas
    grad_bary = d_logsum_d_areas @ grad_areas

    if constraint_violation > 0:
        grad_bary *= 1000.0

    return np.concatenate([grad_bary, [grad_t]])

def generate_hexagonal_init():
    """
    Generate a hexagonal lattice-like initial configuration.
    """
    best_bary = None
    best_val = -np.inf

    for scale in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]:
        pts = []
        max_k = int(1.0 / scale) + 1
        for k in range(max_k + 1):
            w_val = k * scale
            if w_val > 1.0:
                break
            num_pts_in_row = int((1 - w_val) / scale) + 1
            for j in range(num_pts_in_row):
                u_val = j * scale
                v_val = (1 - w_val) - u_val
                if u_val >= -1e-9 and v_val >= -1e-9 and u_val + v_val <= 1 + 1e-9:
                    pts.append([u_val, v_val])

        if len(pts) >= N:
            rng = np.random.default_rng(42)
            for _ in range(5):
                if len(pts) == N:
                    bary = np.array(pts)
                else:
                    idx = rng.choice(len(pts), N, replace=False)
                    bary = np.array(pts)[idx]
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

def generate_vertex_edge_center_init():
    """
    Place points at vertices, edge midpoints, and center with slight perturbations.
    """
    pts = [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.0],
        [0.5, 0.5],
        [0.0, 0.5],
        [1/3, 1/3],
        [0.25, 0.25],
        [0.5, 0.25],
        [0.25, 0.5],
        [0.75, 0.1],
    ]

    if len(pts) > N:
        pts = pts[:N]
    else:
        while len(pts) < N:
            u = np.random.random()
            v = np.random.random() * (1 - u)
            pts.append([u, v])

    return np.array(pts)

def generate_voronoi_init():
    """
    Generate initial points using a Voronoi-like approach:
    start with random points, compute Voronoi cells, and move each point
    to the centroid of its cell (Lloyd's algorithm).
    """
    rng = np.random.default_rng(123)
    u = rng.random(N)
    v = rng.random(N) * (1 - u)
    bary = np.column_stack([u, v])

    # Simple Lloyd's algorithm iterations
    for _ in range(10):
        # For each point, compute the centroid of its Voronoi cell
        # Approximate by averaging nearby points weighted by inverse distance
        new_bary = bary.copy()
        for i in range(N):
            # Weighted average of all points, weight = 1/(dist^2 + eps)
            du = bary[:, 0] - bary[i, 0]
            dv = bary[:, 1] - bary[i, 1]
            dist2 = du**2 + dv**2
            weights = 1.0 / (dist2 + 1e-6)
            weights[i] = 0.0  # Exclude self
            if np.sum(weights) > 0:
                new_u = np.sum(weights * bary[:, 0]) / np.sum(weights)
                new_v = np.sum(weights * bary[:, 1]) / np.sum(weights)
                new_bary[i] = [new_u, new_v]
        bary = clamp_bary(new_bary)

    return bary

def generate_random_init(seed=None):
    """Generate random valid barycentric coordinates."""
    rng = np.random.default_rng(seed)
    u = rng.random(N)
    v = rng.random(N) * (1 - u)
    return np.column_stack([u, v])

def generate_perturbed_init(base_bary, sigma=0.05, seed=None):
    """Generate a perturbed version of a base configuration."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma, base_bary.shape)
    return clamp_bary(base_bary + noise)

def coordinate_descent_refine(bary_init, time_limit=5.0, max_iter=200):
    """
    Specialized coordinate descent refinement.
    """
    start_time = time.time()
    bary = bary_init.copy()
    current_min_area = -hard_min_obj(bary.flatten())

    step_sizes = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002]

    for step_size in step_sizes:
        if time.time() - start_time > time_limit:
            break

        for _ in range(max_iter):
            if time.time() - start_time > time_limit:
                break

            improved = False
            indices = np.random.permutation(N)

            for idx in indices:
                u, v = bary[idx]
                best_u, best_v = u, v
                best_local_min = current_min_area

                directions = [
                    (step_size, 0), (-step_size, 0),
                    (0, step_size), (0, -step_size),
                    (step_size, step_size), (-step_size, -step_size),
                    (step_size, -step_size), (-step_size, step_size)
                ]

                for du, dv in directions:
                    new_u = u + du
                    new_v = v + dv

                    if new_u < 0 or new_v < 0 or new_u + new_v > 1:
                        continue

                    old_u, old_v = bary[idx, 0], bary[idx, 1]
                    bary[idx, 0] = new_u
                    bary[idx, 1] = new_v

                    areas = calc_areas_bary(bary)
                    new_min_area = np.min(areas)

                    if new_min_area > best_local_min:
                        best_local_min = new_min_area
                        best_u, best_v = new_u, new_v
                    else:
                        bary[idx, 0] = old_u
                        bary[idx, 1] = old_v

                if best_u != u or best_v != v:
                    bary[idx, 0] = best_u
                    bary[idx, 1] = best_v
                    improved = True
                    current_min_area = best_local_min
            if not improved:
                break

    return clamp_bary(bary)

def solve_epigraph(bary_init, time_limit=8.0, beta_start=50.0, beta_end=500.0):
    """
    Solve the epigraph formulation with increasing beta (annealing).
    """
    start_time = time.time()
    bary = bary_init.copy()

    # Anneal beta from low to high for better convergence
    betas = [beta_start, beta_start * 2, beta_start * 4, beta_start * 8,
             beta_start * 16, beta_end]

    for beta in betas:
        if time.time() - start_time > time_limit:
            break

        # Initial t estimate: current min area
        t0 = -hard_min_obj(bary.flatten())

        x0 = np.concatenate([bary.flatten(), [t0]])

        bounds = [(0, 1)] * (N * 2) + [(0, 1)]

        try:
            result = minimize(
                epigraph_obj,
                x0,
                args=(beta,),
                method='L-BFGS-B',
                jac=epigraph_grad,
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-12}
            )
            if result.x[N*2] > 0:  # t should be positive (min area)
                bary_new = clamp_bary(result.x[:N*2].reshape(N, 2))
                val_new = -hard_min_obj(bary_new.flatten())
                val_old = -hard_min_obj(bary.flatten())
                if val_new >= val_old - 1e-12:
                    bary = bary_new
        except Exception:
            pass

    return bary

def refine_local(bary_init, time_limit=8.0):
    """
    Refine a configuration using multiple optimization stages.
    """
    start_time = time.time()
    bary_current = bary_init.copy()

    # Stage 1: Epigraph formulation (main workhorse)
    try:
        bary_current = solve_epigraph(bary_current, time_limit=time_limit * 0.4)
    except Exception:
        pass

    if time.time() - start_time > time_limit * 0.5:
        return bary_current

    # Stage 2: Coordinate Descent (structural change)
    try:
        bary_current = coordinate_descent_refine(bary_current, time_limit=time_limit * 0.2)
    except Exception:
        pass

    if time.time() - start_time > time_limit * 0.7:
        return bary_current

    # Stage 3: Hard min refinement with Nelder-Mead
    try:
        result = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-13, 'fatol': 1e-17, 'adaptive': True}
        )
        bary_current = clamp_bary(result.x.reshape(N, 2))
    except Exception:
        pass

    if time.time() - start_time > time_limit * 0.85:
        return bary_current

    # Stage 4: SLSQP polish
    try:
        bounds = [(0, 1)] * (N * 2)
        result = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-15}
        )
        bary_current = clamp_bary(result.x.reshape(N, 2))
    except Exception:
        pass

    return bary_current

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT

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

    # 1. Try structured initializations
    inits = [
        generate_hexagonal_init(),
        generate_vertex_edge_center_init(),
        generate_voronoi_init(),
    ]

    for init in inits:
        if time.time() > deadline - 25:
            break
        refined = refine_local(init, time_limit=10.0)
        update_best(refined)

    # 2. Global Search with Differential Evolution (limited)
    if time.time() < deadline - 50:
        try:
            def de_obj(x):
                bary = x.reshape(N, 2)
                bary = clamp_bary(bary)
                areas = calc_areas_bary(bary)
                return -np.min(areas)

            bounds = [(0, 1)] * (N * 2)
            result_de = differential_evolution(
                de_obj,
                bounds,
                seed=42,
                maxiter=25,
                popsize=12,
                tol=1e-6,
                polish=False
            )

            de_bary = clamp_bary(result_de.x.reshape(N, 2))
            refined_de = refine_local(de_bary, time_limit=10.0)
            update_best(refined_de)
        except Exception:
            pass

    # 3. Multi-start: random + perturbed restarts with local refinement
    restart_count = 0
    while time.time() < deadline - 20:
        restart_count += 1
        if best_bary is not None and restart_count % 2 == 0:
            sigma = 0.03 + 0.01 * (restart_count % 5)
            init = generate_perturbed_init(best_bary, sigma=sigma, seed=2000 + restart_count)
        else:
            init = generate_random_init(seed=3000 + restart_count)

        refined = refine_local(init, time_limit=5.0)
        update_best(refined)

    # 4. Final high-precision polish
    if best_bary is not None and time.time() < deadline - 15:
        try:
            # High-precision epigraph polish
            bary_polished = solve_epigraph(best_bary, time_limit=8.0, beta_start=200.0, beta_end=1000.0)
            update_best(bary_polished)
        except Exception:
            pass

        if best_bary is not None and time.time() < deadline - 8:
            try:
                result_slsqp = minimize(
                    hard_min_obj,
                    best_bary.flatten(),
                    method='SLSQP',
                    bounds=[(0, 1)] * (N * 2),
                    options={'maxiter': 3000, 'ftol': 1e-17}
                )
                update_best(clamp_bary(result_slsqp.x.reshape(N, 2)))
            except Exception:
                pass

        if best_bary is not None and time.time() < deadline - 4:
            try:
                result_nm = minimize(
                    hard_min_obj,
                    best_bary.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 8000, 'xatol': 1e-15, 'fatol': 1e-19, 'adaptive': True}
                )
                update_best(clamp_bary(result_nm.x.reshape(N, 2)))
            except Exception:
                pass

        # One more SLSQP pass
        if best_bary is not None and time.time() < deadline - 2:
            try:
                result_final = minimize(
                    hard_min_obj,
                    best_bary.flatten(),
                    method='SLSQP',
                    bounds=[(0, 1)] * (N * 2),
                    options={'maxiter': 4000, 'ftol': 1e-18}
                )
                update_best(clamp_bary(result_final.x.reshape(N, 2)))
            except Exception:
                pass

    # Convert to cartesian
    cart = bary_to_cart(best_bary)

    return cart
