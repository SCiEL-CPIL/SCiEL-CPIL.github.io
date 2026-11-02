import numpy as np
import scipy.spatial
import time
from scipy.optimize import minimize

N = 50
TIME_LIMIT = 140.0  # safety margin before 150s kill


def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _compute_facet_data(points):
    """
    Compute convex hull and per-facet circumcenter and angular circumradius.
    Returns (hull, centers, radii) or (None, None, None) on failure.
    """
    points = _normalize(points)
    try:
        hull = scipy.spatial.ConvexHull(points)
    except Exception:
        return None, None, None

    facets = hull.simplices
    n_facets = len(facets)
    if n_facets == 0:
        return hull, np.zeros((0, 3)), np.zeros(0)

    a = points[facets[:, 0]]
    b = points[facets[:, 1]]
    c = points[facets[:, 2]]

    v1 = a - b
    v2 = b - c
    n = np.cross(v1, v2)
    norm_n = np.linalg.norm(n, axis=1)

    valid = norm_n > 1e-12
    centers = np.zeros((n_facets, 3))
    radii = np.zeros(n_facets)

    if np.any(valid):
        cc = n[valid] / norm_n[valid, np.newaxis]
        dots = np.sum(cc * a[valid], axis=1)
        cc[dots < 0] *= -1
        centers[valid] = cc
        cos_r = np.clip(np.sum(cc * a[valid], axis=1), -1.0, 1.0)
        radii[valid] = np.arccos(cos_r)

    return hull, centers, radii


def _compute_covering_radius(points):
    hull, centers, radii = _compute_facet_data(points)
    if hull is None:
        return np.pi
    if len(radii) == 0:
        return 0.0
    return float(np.max(radii))


def _smooth_max_radius_and_grad(points, beta=80.0):
    """
    Smooth max of facet circumradii with gradient.
    Uses log-sum-exp for stability.
    """
    hull, centers, radii = _compute_facet_data(points)
    if hull is None:
        return np.pi, np.zeros_like(points)

    n_facets = len(radii)
    if n_facets == 0:
        return 0.0, np.zeros_like(points)

    max_r = np.max(radii)
    shifted = beta * (radii - max_r)
    max_shifted = np.max(shifted)
    exp_terms = np.exp(shifted - max_shifted)
    log_sum = max_shifted + np.log(np.sum(exp_terms))
    smooth_r = max_r + log_sum / beta
    weights = exp_terms / np.sum(exp_terms)

    grads = np.zeros_like(points)
    facets = hull.simplices

    for i in range(n_facets):
        w = weights[i]
        if w < 1e-14:
            continue
        r = radii[i]
        if r < 1e-12:
            continue
        C = centers[i]
        sin_r = np.sin(r)
        if sin_r < 1e-10:
            continue

        simplex = facets[i]
        p_pts = points[simplex]
        # d arccos(C . p) / d p = -C / sin(r)
        grad_contrib = -C / sin_r
        dots = np.sum(p_pts * grad_contrib, axis=1)
        radial_comps = dots[:, np.newaxis] * p_pts
        tangential_grads = grad_contrib - radial_comps

        for k in range(3):
            grads[simplex[k]] += w * tangential_grads[k]

    return smooth_r, grads


def _pairwise_repulsion_and_grad(points, k=12):
    """
    Smooth k-nearest-neighbor repulsion. Encourages minimum separation.
    Uses soft-min of distances to k nearest neighbors.
    """
    # Compute all pairwise dot products
    dots = points @ points.T
    np.fill_diagonal(dots, 2.0)  # self-distance = 0 (max dot)
    # Angular distance = arccos(dot), but we work with dot for smoothness
    # We want to maximize the k-th smallest distance = minimize k-th largest dot
    # Use soft-min over the k largest dots (closest neighbors)

    # For each point, find the k largest dots (excluding self)
    # Sort dots in descending order, take first k
    sorted_dots = np.sort(dots, axis=1)[:, ::-1]  # descending
    k_dots = sorted_dots[:, :k]  # k closest neighbors

    # Soft max of k_dots for each point (smooth maximum of closest neighbors)
    beta_rep = 30.0
    max_kd = np.max(k_dots, axis=1)
    shifted_kd = beta_rep * (k_dots - max_kd[:, np.newaxis])
    max_shifted = np.max(shifted_kd, axis=1)
    exp_kd = np.exp(shifted_kd - max_shifted[:, np.newaxis])
    log_sum_kd = max_shifted + np.log(np.sum(exp_kd, axis=1))
    soft_max_kd = max_kd + log_sum_kd / beta_rep

    # We want to MINIMIZE the soft max of closest-neighbor dots
    # (i.e., push the closest neighbors further away)
    # Objective: sum_i soft_max_kd[i]
    obj = np.sum(soft_max_kd)

    # Gradient: for each point i, gradient w.r.t. p_i
    # d soft_max_kd[i] / d p_i = sum_{j in kNN(i)} w_{ij} * d dot(p_i, p_j) / d p_i
    # = sum_j w_{ij} * p_j
    # where w_{ij} are the softmax weights over the k nearest neighbors of i

    grads = np.zeros_like(points)
    for i in range(N):
        w_i = exp_kd[i] / np.sum(exp_kd[i])
        # Find which indices are the k nearest neighbors
        # The k largest dots for point i
        row = dots[i].copy()
        row[i] = -2.0  # exclude self
        k_indices = np.argsort(row)[::-1][:k]
        for j, idx in enumerate(k_indices):
            grads[i] += w_i[j] * points[idx]

    # Symmetrize: also add contributions from other points' kNN lists
    # For each point j, if i is in j's kNN, add contribution
    # This is O(N*k) additional work
    for j in range(N):
        row = dots[j].copy()
        row[j] = -2.0
        k_indices = np.argsort(row)[::-1][:k]
        w_j = exp_kd[j] / np.sum(exp_kd[j])
        for m, idx in enumerate(k_indices):
            grads[idx] += w_j[m] * points[j]

    return obj, grads


def _combined_objective_and_grad(points, alpha_rep=0.05, beta_smooth=80.0):
    """
    Combined objective: smooth_max_radius + alpha_rep * repulsion_penalty
    """
    smooth_r, grad_r = _smooth_max_radius_and_grad(points, beta=beta_smooth)
    rep_obj, grad_rep = _pairwise_repulsion_and_grad(points, k=12)

    obj = smooth_r + alpha_rep * rep_obj
    grad = grad_r + alpha_rep * grad_rep
    return obj, grad


def _lbfgs_optimize(points, time_budget, alpha_rep=0.05, beta_schedule=None):
    """
    L-BFGS-B optimization with combined objective.
    """
    if beta_schedule is None:
        beta_schedule = [40.0, 80.0, 150.0, 300.0]

    start_time = time.time()
    current_pts = points.copy()
    current_radius = _compute_covering_radius(current_pts)
    best_pts = current_pts.copy()
    best_radius = current_radius

    for beta in beta_schedule:
        if time.time() - start_time > time_budget:
            break

        def objective_and_grad(x, b=beta):
            if time.time() - start_time > time_budget:
                raise Exception("timeout")
            pts = _normalize(x.reshape(N, 3))
            obj, grads = _combined_objective_and_grad(pts, alpha_rep=alpha_rep, beta_smooth=b)
            return obj, grads.reshape(-1)

        try:
            x0 = current_pts.reshape(-1)
            elapsed = time.time() - start_time
            remaining = time_budget - elapsed
            max_iter = int(60 + 100 * (remaining / time_budget))

            result = minimize(
                objective_and_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': max_iter, 'maxfun': max_iter * 2,
                         'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 100}
            )

            new_pts = _normalize(result.x.reshape(N, 3))
            new_rad = _compute_covering_radius(new_pts)

            if new_rad < current_radius:
                current_pts = new_pts
                current_radius = new_rad

            if current_radius < best_radius:
                best_radius = current_radius
                best_pts = current_pts.copy()

        except Exception:
            pass

    return best_pts


def _differential_evolution(points_pool, time_budget, pop_size=15, max_generations=30,
                            F=0.8, CR=0.7, alpha_rep=0.05):
    """
    Differential evolution on the sphere for global search.
    """
    start_time = time.time()
    rng = np.random.default_rng(12345)

    # Initialize population
    pop = [points_pool.copy()]
    for _ in range(pop_size - 1):
        base = points_pool.copy()
        # Add random perturbation
        perturbation = rng.normal(scale=0.3, size=(N, 3))
        new_pts = _normalize(base + perturbation)
        pop.append(new_pts)

    # Evaluate initial population
    pop_radii = [_compute_covering_radius(p) for p in pop]

    best_idx = np.argmin(pop_radii)
    best_pts = pop[best_idx].copy()
    best_radius = pop_radii[best_idx]

    for gen in range(max_generations):
        if time.time() - start_time > time_budget:
            break

        new_pop = [pop[best_idx].copy()]  # keep best
        new_radii = [pop_radii[best_idx]]

        for i in range(1, pop_size):
            if time.time() - start_time > time_budget:
                break

            # Select three distinct random individuals
            a, b, c = rng.choice(pop_size, size=3, replace=False)
            while a == b or a == c or b == c:
                a, b, c = rng.choice(pop_size, size=3, replace=False)

            # DE/rand/1
            v = pop[a] + F * (pop[b] - pop[c])
            v = _normalize(v)

            # Crossover
            trial = v.copy()
            for d in range(N * 3):
                if rng.random() < CR:
                    trial[d] = v[d]
                else:
                    trial[d] = pop[i][d]
            trial = _normalize(trial.reshape(N, 3))

            trial_rad = _compute_covering_radius(trial)
            if trial_rad <= pop_radii[i]:
                new_pop.append(trial)
                new_radii.append(trial_rad)
            else:
                new_pop.append(pop[i])
                new_radii.append(pop_radii[i])

        pop = new_pop
        pop_radii = new_radii
        best_idx = np.argmin(pop_radii)
        if pop_radii[best_idx] < best_radius:
            best_radius = pop_radii[best_idx]
            best_pts = pop[best_idx].copy()

    return best_pts


def _facet_equalization_optimize(points, time_budget, max_iter=200):
    """
    Directly equalize facet radii by moving vertices of the worst facet.
    Uses a simple projected gradient step.
    """
    start_time = time.time()
    current_pts = points.copy()
    best_radius = _compute_covering_radius(current_pts)
    best_pts = current_pts.copy()

    step_size = 0.03
    min_step = 1e-12
    decay = 0.9

    for _ in range(max_iter):
        if time.time() - start_time > time_budget:
            break

        hull, centers, radii = _compute_facet_data(current_pts)
        if hull is None:
            break

        if len(radii) == 0:
            break

        # Find the worst facet
        worst_idx = np.argmax(radii)
        r_worst = radii[worst_idx]
        if r_worst < 1e-12:
            break

        C = centers[worst_idx]
        sin_r = np.sin(r_worst)
        if sin_r < 1e-10:
            break

        simplex = hull.simplices[worst_idx]
        p_pts = current_pts[simplex]

        # Gradient direction: move vertices away from circumcenter
        grad_contrib = -C / sin_r
        dots = np.sum(p_pts * grad_contrib, axis=1)
        radial_comps = dots[:, np.newaxis] * p_pts
        tangential_grads = grad_contrib - radial_comps

        new_pts = current_pts.copy()
        for k in range(3):
            new_pts[simplex[k]] = current_pts[simplex[k]] + step_size * tangential_grads[k]
        new_pts = _normalize(new_pts)

        new_radius = _compute_covering_radius(new_pts)

        if new_radius < best_radius - 1e-14:
            best_radius = new_radius
            best_pts = new_pts.copy()
            current_pts = new_pts
            step_size = min(step_size * 1.03, 0.08)
        else:
            step_size *= decay
            if step_size < min_step:
                break

    return best_pts


def _generate_seeds():
    seeds = []

    # 1. Fibonacci sphere
    j = np.arange(N) + 0.5
    z = 1.0 - 2.0 * j / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * j
    r = np.sqrt(1.0 - z * z)
    fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    seeds.append(fib)

    # 2. Random starts with various seeds
    for seed_val in [42, 1, 2, 3, 7, 13, 99, 123, 555, 777, 1000, 2000]:
        rng = np.random.default_rng(seed_val)
        pts = rng.normal(size=(N, 3))
        pts = _normalize(pts)
        seeds.append(pts)

    # 3. Greedy farthest-point initialization
    def greedy_init(seed_val):
        rng = np.random.default_rng(seed_val)
        pts = [rng.normal(size=3)]
        pts[0] = pts[0] / np.linalg.norm(pts[0])
        for _ in range(N - 1):
            candidates = rng.normal(size=(3000, 3))
            candidates = _normalize(candidates)
            dots = candidates @ np.array(pts).T
            min_dots = np.max(dots, axis=1)
            idx = np.argmin(min_dots)
            pts.append(candidates[idx])
        return np.array(pts)

    seeds.append(greedy_init(101))
    seeds.append(greedy_init(202))
    seeds.append(greedy_init(303))
    seeds.append(greedy_init(404))

    # 4. Icosahedron-based
    phi_golden = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0],
        [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1],
        [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1]
    ], dtype=float)
    ico = _normalize(ico)
    ico_extended = [ico[i] for i in range(12)]
    rng = np.random.default_rng(42)
    while len(ico_extended) < N:
        base = ico_extended[len(ico_extended) % 12]
        perturbation = rng.normal(scale=0.15, size=3)
        new_pt = base + perturbation
        ico_extended.append(new_pt / np.linalg.norm(new_pt))
    seeds.append(np.array(ico_extended))

    return seeds


def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT

    best_points = None
    best_radius = np.inf

    seeds = _generate_seeds()
    n_seeds = len(seeds)

    # Phase 1: Quick L-BFGS on all seeds
    results = []
    for i, seed_pts in enumerate(seeds):
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining < 40:
            break

        budget = min(4.0, remaining / (n_seeds - i + 1))
        if budget < 1.5:
            break

        pts = _lbfgs_optimize(seed_pts, budget, alpha_rep=0.03,
                              beta_schedule=[30.0, 60.0, 120.0])
        rad = _compute_covering_radius(pts)
        results.append((rad, pts, i))

        if rad < best_radius:
            best_radius = rad
            best_points = pts.copy()

    results.sort(key=lambda x: x[0])

    # Phase 2: DE global search on top candidates
    top_candidates = results[:6]
    elapsed = time.time() - start_time
    remaining_for_de = deadline - elapsed - 25.0

    if remaining_for_de > 15:
        de_time_per = remaining_for_de / len(top_candidates)
        for rad, pts, idx in top_candidates:
            if time.time() > deadline - 25:
                break

            de_pts = _differential_evolution(pts, de_time_per * 0.7,
                                             pop_size=12, max_generations=20,
                                             F=0.8, CR=0.7)
            de_rad = _compute_covering_radius(de_pts)

            if de_rad < best_radius:
                best_radius = de_rad
                best_points = de_pts.copy()

            # Polish with L-BFGS
            if time.time() < deadline - 25:
                polished = _lbfgs_optimize(de_pts, de_time_per * 0.3,
                                           alpha_rep=0.02,
                                           beta_schedule=[100.0, 200.0])
                pol_rad = _compute_covering_radius(polished)
                if pol_rad < best_radius:
                    best_radius = pol_rad
                    best_points = polished.copy()

    # Phase 3: Facet equalization on the best
    if best_points is not None:
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining > 8:
            budget = min(12.0, remaining - 3)
            if budget > 2:
                eq_pts = _facet_equalization_optimize(best_points, budget * 0.6, max_iter=300)
                eq_rad = _compute_covering_radius(eq_pts)
                if eq_rad < best_radius:
                    best_radius = eq_rad
                    best_points = eq_pts.copy()

                # Final L-BFGS polish
                if time.time() < deadline - 3:
                    final_pts = _lbfgs_optimize(best_points, budget * 0.4,
                                                alpha_rep=0.01,
                                                beta_schedule=[200.0, 400.0])
                    final_rad = _compute_covering_radius(final_pts)
                    if final_rad < best_radius:
                        best_radius = final_rad
                        best_points = final_pts.copy()

    if best_points is None:
        best_points = seeds[0]

    best_points = _normalize(best_points)
    return best_points
