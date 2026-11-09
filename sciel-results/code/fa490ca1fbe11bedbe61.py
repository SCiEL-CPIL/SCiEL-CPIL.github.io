# EVOLVE-BLOCK-START
import numpy as np
import time


def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex 20-gon that maximizes the minimum vertex equidistance score.

    Strategy:
      - Parameterize vertices by angle deviations from the regular 20-gon.
      - Use a low-order Fourier series for the angle perturbations to keep
        the polygon close to regular (hence convex) while allowing enough
        freedom to tune chord-length distributions.
      - Use a smooth soft-min surrogate for optimization.
      - Optimize with differential evolution followed by Nelder-Mead polish.
      - Deadline guard: stop before 630 s and return the best valid polygon.
    """
    start_time = time.time()
    deadline = 630.0
    safety_margin = 10.0  # seconds before the hard limit

    n = 20
    base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)

    # Fourier series for angle perturbations
    M = 4
    n_params = 2 * M

    def params_to_polygon(params):
        """Convert Fourier coefficients to a (20, 2) polygon in cyclic order."""
        phi = base_angles
        delta = np.zeros(n)
        for m in range(1, M + 1):
            delta += params[2 * (m - 1)] * np.cos(m * phi)
            delta += params[2 * (m - 1) + 1] * np.sin(m * phi)
        thetas = base_angles + delta
        # Ensure cyclic order: thetas should be monotonically increasing mod 2*pi
        # For small perturbations, this holds. Verify and fix if needed.
        # Unwrap to handle wrap-around
        thetas_unwrapped = np.unwrap(thetas)
        # Normalize to start near 0
        thetas_unwrapped = thetas_unwrapped - thetas_unwrapped[0]
        # If the range exceeds 2*pi, something is wrong; clip
        if thetas_unwrapped[-1] - thetas_unwrapped[0] > 2.0 * np.pi:
            # Fallback: sort
            thetas_unwrapped = np.sort(thetas) % (2.0 * np.pi)
            thetas_unwrapped = np.sort(thetas_unwrapped)
        x = np.cos(thetas_unwrapped)
        y = np.sin(thetas_unwrapped)
        return np.stack([x, y], axis=1)

    def is_convex(polygon):
        """Check if polygon is strictly convex."""
        n = len(polygon)
        for i in range(n):
            p0 = polygon[i]
            p1 = polygon[(i + 1) % n]
            p2 = polygon[(i + 2) % n]
            v1 = p1 - p0
            v2 = p2 - p1
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            if abs(cross) < 1e-12:
                return False
            if i == 0:
                sign0 = np.sign(cross)
            else:
                if np.sign(cross) != sign0:
                    return False
        return True

    def vertex_scores(polygon):
        """Compute per-vertex scores. Returns array of 20 scores."""
        centroid = np.mean(polygon, axis=0)
        centered = polygon - centroid
        dists_to_centroid = np.linalg.norm(centered, axis=1)
        mean_dist = np.mean(dists_to_centroid)
        if mean_dist <= 0.01:
            return np.zeros(n)
        normalized = centered / mean_dist

        if not is_convex(normalized):
            return np.zeros(n)

        # Check no repeated points
        for i in range(len(normalized)):
            for j in range(i + 1, len(normalized)):
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

    def compute_score(polygon):
        """Compute the minimum vertex score."""
        scores = vertex_scores(polygon)
        return np.min(scores)

    def soft_min(scores, beta=50.0):
        """Soft minimum via log-sum-exp."""
        scores = np.asarray(scores)
        if np.any(scores <= 0):
            return 0.0
        # -log(mean(exp(-beta * scores))) / beta approximates min(scores)
        shifted = -beta * scores
        max_shifted = np.max(shifted)
        log_sum = max_shifted + np.log(np.mean(np.exp(shifted - max_shifted)))
        return -log_sum / beta

    def soft_metric(params):
        """Smooth surrogate for the optimization objective (to minimize)."""
        try:
            poly = params_to_polygon(params)
            scores = vertex_scores(poly)
            if np.any(scores <= 0):
                return 1.0
            return -soft_min(scores, beta=50.0)
        except Exception:
            return 1.0

    def objective(params):
        """Negative exact score for minimization."""
        if time.time() - start_time > deadline - safety_margin:
            return 1.0
        try:
            poly = params_to_polygon(params)
            return -compute_score(poly)
        except Exception:
            return 1.0

    # Initial best: regular polygon
    best_params = np.zeros(n_params)
    best_poly = params_to_polygon(best_params)
    best_score = compute_score(best_poly)

    # Differential Evolution global search
    np.random.seed(42)
    population_size = 50
    bounds = [(-0.25, 0.25)] * n_params

    population = np.random.uniform(
        [b[0] for b in bounds],
        [b[1] for b in bounds],
        (population_size, n_params)
    )
    population[0] = np.zeros(n_params)
    population[1] = best_params.copy()

    scores_arr = np.array([objective(p) for p in population])

    F = 0.7
    CR = 0.8
    max_generations = 300

    for gen in range(max_generations):
        if time.time() - start_time > deadline - safety_margin:
            break

        best_idx = np.argmin(scores_arr)
        best_params = population[best_idx].copy()
        best_score = -scores_arr[best_idx]
        if best_score > 0.999:
            break

        new_population = population.copy()
        for i in range(population_size):
            if time.time() - start_time > deadline - safety_margin:
                break

            candidates = [j for j in range(population_size) if j != i]
            a, b, c = np.random.choice(candidates, size=3, replace=False)

            trial = population[a] + F * (population[b] - population[c])

            crossover_mask = np.random.random(n_params) < CR
            if not np.any(crossover_mask):
                crossover_mask[np.random.randint(n_params)] = True
            trial = np.where(crossover_mask, trial, population[i])

            trial = np.clip(trial, [b[0] for b in bounds], [b[1] for b in bounds])

            trial_score = objective(trial)

            if trial_score <= scores_arr[i]:
                new_population[i] = trial
                scores_arr[i] = trial_score

        population = new_population

        best_idx = np.argmin(scores_arr)
        if -scores_arr[best_idx] > best_score:
            best_params = population[best_idx].copy()
            best_score = -scores_arr[best_idx]
            best_poly = params_to_polygon(best_params)

    # Local refinement with Nelder-Mead
    if time.time() - start_time < deadline - safety_margin:
        from scipy.optimize import minimize

        def local_objective(params):
            if time.time() - start_time > deadline - safety_margin:
                return 1.0
            try:
                poly = params_to_polygon(params)
                return -compute_score(poly)
            except Exception:
                return 1.0

        local_starts = [best_params]
        # Add diverse perturbations
        for scale in [0.03, 0.08, 0.15]:
            for _ in range(3):
                pert = best_params + np.random.normal(0, scale, n_params)
                pert = np.clip(pert, [b[0] for b in bounds], [b[1] for b in bounds])
                local_starts.append(pert)

        for ls in local_starts:
            if time.time() - start_time > deadline - safety_margin:
                break
            try:
                res = minimize(
                    local_objective,
                    ls,
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10,
                             'adaptive': True}
                )
                if res.fun < -best_score:
                    best_params = res.x
                    best_score = -res.fun
                    best_poly = params_to_polygon(best_params)
            except Exception:
                pass

    # Final high-precision polish
    if time.time() - start_time < deadline - safety_margin:
        from scipy.optimize import minimize

        def final_objective(params):
            if time.time() - start_time > deadline - safety_margin:
                return 1.0
            try:
                poly = params_to_polygon(params)
                return -compute_score(poly)
            except Exception:
                return 1.0

        try:
            res = minimize(
                final_objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12,
                         'adaptive': True}
            )
            if res.fun < -best_score:
                best_params = res.x
                best_score = -res.fun
                best_poly = params_to_polygon(best_params)
        except Exception:
            pass

    # Final validation
    final_poly = params_to_polygon(best_params)
    final_score = compute_score(final_poly)
    if final_score > best_score:
        best_poly = final_poly
        best_score = final_score

    if not is_convex(best_poly):
        best_poly = params_to_polygon(np.zeros(n_params))

    return best_poly


# EVOLVE-BLOCK-END
