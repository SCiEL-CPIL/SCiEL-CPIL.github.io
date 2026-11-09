# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex 20-gon maximizing the minimum vertex equidistance score.
    
    Strategy:
      - Parameterize vertices by radial distance r(theta) = 1 + sum c_k * cos(k*theta)
        evaluated at 20 equally-spaced angles. This directly controls chord lengths
        and guarantees convexity for small coefficients.
      - Vectorized score computation for speed.
      - Differential evolution global search with soft-min surrogate.
      - Nelder-Mead local polish with tight tolerances.
      - Deadline guard: return best valid polygon before 625s.
    """
    start_time = time.time()
    deadline = 625.0

    n = 20
    base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    
    # Precompute cos(k*theta) for k=1..M
    M = 4
    n_params = M  # only cosine coefficients (symmetric)
    
    cos_terms = np.zeros((n, M))
    for m in range(1, M + 1):
        cos_terms[:, m - 1] = np.cos(m * base_angles)

    def params_to_polygon(params):
        """Convert radial coefficients to (20, 2) polygon."""
        r = 1.0 + cos_terms @ params
        x = r * np.cos(base_angles)
        y = r * np.sin(base_angles)
        return np.stack([x, y], axis=1)

    def is_convex(polygon):
        """Check strict convexity: all cross products same sign, nonzero."""
        n_pts = len(polygon)
        cross_vals = np.zeros(n_pts)
        for i in range(n_pts):
            p0 = polygon[i]
            p1 = polygon[(i + 1) % n_pts]
            p2 = polygon[(i + 2) % n_pts]
            v1 = p1 - p0
            v2 = p2 - p1
            cross_vals[i] = v1[0] * v2[1] - v1[1] * v2[0]
        if np.any(np.abs(cross_vals) < 1e-12):
            return False
        if not (np.all(cross_vals > 0) or np.all(cross_vals < 0)):
            return False
        return True

    def compute_scores(polygon):
        """Vectorized per-vertex score computation. Returns array of 20 scores."""
        centroid = np.mean(polygon, axis=0)
        centered = polygon - centroid
        dists_c = np.linalg.norm(centered, axis=1)
        mean_dist = np.mean(dists_c)
        if mean_dist <= 0.01:
            return np.zeros(n)
        normalized = centered / mean_dist

        if not is_convex(normalized):
            return np.zeros(n)

        # Check no repeated points
        for i in range(n):
            for j in range(i + 1, n):
                if np.linalg.norm(normalized[i] - normalized[j]) < 1e-10:
                    return np.zeros(n)

        scores = np.ones(n)
        for v_idx in range(n):
            v = normalized[v_idx]
            d = np.linalg.norm(normalized - v, axis=1)
            d[v_idx] = np.inf
            d_sorted = np.sort(d)

            if d_sorted[0] < 0.05:
                scores[v_idx] = 0.0
                continue

            gaps = np.diff(d_sorted)
            k = int(np.argmin(gaps))
            D = (d_sorted[k] + d_sorted[k + 1]) / 2.0
            if D < 0.1:
                scores[v_idx] = 0.0
                continue

            if k == 0:
                selected = d_sorted[0:4]
            elif k == 17:
                selected = d_sorted[15:19]
            else:
                selected = d_sorted[k - 1:k + 3]

            ratios = np.maximum(selected / D, D / selected)
            raw = np.mean(ratios)
            scores[v_idx] = min(1.0, 1.0 / raw) if raw > 0 else 0.0

        return scores

    def compute_min_score(polygon):
        """Return the minimum vertex score (the metric)."""
        scores = compute_scores(polygon)
        return float(np.min(scores))

    def soft_min(scores, beta=80.0):
        """Soft minimum via log-sum-exp for smooth surrogate."""
        scores = np.asarray(scores)
        if np.any(scores <= 0):
            return 0.0
        shifted = -beta * scores
        max_shifted = np.max(shifted)
        log_sum = max_shifted + np.log(np.mean(np.exp(shifted - max_shifted)))
        return -log_sum / beta

    def objective(params):
        """Negative score for minimization. Returns large value on failure."""
        if time.time() - start_time > deadline:
            return 10.0
        try:
            poly = params_to_polygon(params)
            sc = compute_min_score(poly)
            return -sc if sc > 0 else 5.0
        except Exception:
            return 5.0

    def soft_objective(params):
        """Smooth surrogate using soft-min."""
        if time.time() - start_time > deadline:
            return 10.0
        try:
            poly = params_to_polygon(params)
            centroid = np.mean(poly, axis=0)
            centered = poly - centroid
            dists_c = np.linalg.norm(centered, axis=1)
            mean_dist = np.mean(dists_c)
            if mean_dist <= 0.01:
                return 10.0
            normalized = centered / mean_dist
            if not is_convex(normalized):
                return 10.0
            for i in range(n):
                for j in range(i + 1, n):
                    if np.linalg.norm(normalized[i] - normalized[j]) < 1e-10:
                        return 10.0
            scores = np.ones(n)
            for v_idx in range(n):
                v = normalized[v_idx]
                d = np.linalg.norm(normalized - v, axis=1)
                d[v_idx] = np.inf
                d_sorted = np.sort(d)
                if d_sorted[0] < 0.05:
                    scores[v_idx] = 0.0
                    continue
                gaps = np.diff(d_sorted)
                k = int(np.argmin(gaps))
                D = (d_sorted[k] + d_sorted[k + 1]) / 2.0
                if D < 0.1:
                    scores[v_idx] = 0.0
                    continue
                if k == 0:
                    selected = d_sorted[0:4]
                elif k == 17:
                    selected = d_sorted[15:19]
                else:
                    selected = d_sorted[k - 1:k + 3]
                ratios = np.maximum(selected / D, D / selected)
                raw = np.mean(ratios)
                scores[v_idx] = min(1.0, 1.0 / raw) if raw > 0 else 0.0
            sm = soft_min(scores, beta=80.0)
            return -sm if sm > 0 else 5.0
        except Exception:
            return 5.0

    # Initial: regular polygon
    best_params = np.zeros(n_params)
    best_poly = params_to_polygon(best_params)
    best_score = compute_min_score(best_poly)

    # --- Differential Evolution Global Search ---
    np.random.seed(123)
    population_size = 60
    bounds_list = [(-0.15, 0.15)] * n_params
    bounds_lo = np.array([b[0] for b in bounds_list])
    bounds_hi = np.array([b[1] for b in bounds_list])

    population = np.random.uniform(bounds_lo, bounds_hi, (population_size, n_params))
    population[0] = np.zeros(n_params)

    # Use soft objective for DE to get smoother landscape
    scores_arr = np.array([soft_objective(p) for p in population])

    F = 0.8
    CR = 0.7
    max_generations = 200

    for gen in range(max_generations):
        if time.time() - start_time > deadline - 30:
            break

        best_idx = np.argmin(scores_arr)
        current_best_soft = -scores_arr[best_idx]

        # Track best exact score
        bp = population[best_idx].copy()
        exact_sc = compute_min_score(params_to_polygon(bp))
        if exact_sc > best_score:
            best_score = exact_sc
            best_params = bp.copy()
            best_poly = params_to_polygon(bp)

        if best_score > 0.999:
            break

        new_population = population.copy()
        for i in range(population_size):
            if time.time() - start_time > deadline - 30:
                break

            candidates = [j for j in range(population_size) if j != i]
            a, b, c = np.random.choice(candidates, size=3, replace=False)
            trial = population[a] + F * (population[b] - population[c])

            crossover_mask = np.random.random(n_params) < CR
            if not np.any(crossover_mask):
                crossover_mask[np.random.randint(n_params)] = True
            trial = np.where(crossover_mask, trial, population[i])
            trial = np.clip(trial, bounds_lo, bounds_hi)

            trial_score = soft_objective(trial)
            if trial_score <= scores_arr[i]:
                new_population[i] = trial
                scores_arr[i] = trial_score

        population = new_population

    # --- Local Polish with Nelder-Mead on exact objective ---
    if time.time() - start_time < deadline - 15:
        def local_obj(params):
            if time.time() - start_time > deadline - 5:
                return 10.0
            try:
                poly = params_to_polygon(params)
                sc = compute_min_score(poly)
                return -sc if sc > 0 else 5.0
            except Exception:
                return 5.0

        # Multiple starting points
        local_starts = [best_params.copy()]
        for scale in [0.02, 0.05, 0.1]:
            for _ in range(5):
                pert = best_params + np.random.normal(0, scale, n_params)
                pert = np.clip(pert, bounds_lo, bounds_hi)
                local_starts.append(pert)

        for ls in local_starts:
            if time.time() - start_time > deadline - 10:
                break
            try:
                res = minimize(
                    local_obj, ls,
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-11, 'fatol': 1e-11,
                             'adaptive': True}
                )
                sc = -res.fun
                if 0 < sc < 5 and sc > best_score:
                    best_score = sc
                    best_params = res.x.copy()
                    best_poly = params_to_polygon(res.x)
            except Exception:
                pass

    # --- Final ultra-tight polish ---
    if time.time() - start_time < deadline - 5:
        try:
            res = minimize(
                local_obj, best_params,
                method='Nelder-Mead',
                options={'maxiter': 8000, 'xatol': 1e-13, 'fatol': 1e-13,
                         'adaptive': True}
            )
            sc = -res.fun
            if 0 < sc < 5 and sc > best_score:
                best_score = sc
                best_params = res.x.copy()
                best_poly = params_to_polygon(res.x)
        except Exception:
            pass

    # Final validation
    final_poly = params_to_polygon(best_params)
    final_score = compute_min_score(final_poly)
    if final_score > best_score:
        best_poly = final_poly
        best_score = final_score

    if not is_convex(best_poly):
        best_poly = params_to_polygon(np.zeros(n_params))

    return best_poly


# EVOLVE-BLOCK-END
